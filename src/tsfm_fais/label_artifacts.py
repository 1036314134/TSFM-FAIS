"""Validation and deterministic merging of completed teacher-label artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from tsfm_fais.artifacts import utc_now

_COMPATIBLE_MANIFEST_FIELDS = (
    "origin_partition",
    "selected_candidates",
    "max_train_episodes_per_dataset",
    "max_teacher_blocks_per_episode",
    "max_teacher_candidates_per_episode",
    "max_pair_labels_per_episode",
    "csdi_num_samples",
)

_UNARY_KEY_FIELDS = (
    "forecaster_id",
    "episode_id",
    "block_id",
    "candidate_id",
)


@dataclass(frozen=True)
class _SourceArtifact:
    root: Path
    teacher_labels: Path
    pair_labels: Path
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    forecasters: tuple[str, ...]
    unary_rows: int
    pair_rows: int
    ranking_groups: int
    teacher_sha256: str
    pair_sha256: str
    episodes: Mapping[str, Mapping[str, tuple[str, str]]]
    blocks: Mapping[str, Mapping[str, frozenset[str]]]


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid {description} at {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _nonempty_string(value: Any, field: str, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{path}:{line_number}: field {field!r} must be a non-empty string"
        )
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"label file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL row at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{line_number}: each JSONL row must be an object"
                    )
                yield line_number, row
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"could not read label file {path}: {type(error).__name__}: {error}"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolved_path(value: Any, field: str, source: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: labels_manifest field {field!r} is invalid")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"{source}: {field} directory does not exist: {path}")
    return path


def _path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _manifest_forecasters(manifest: Mapping[str, Any], source: Path) -> tuple[str, ...]:
    raw = manifest.get("forecasters")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: labels_manifest must list at least one forecaster")
    values = tuple(str(value) for value in raw)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{source}: labels_manifest forecasters must be unique strings")
    return values


def _manifest_count(manifest: Mapping[str, Any], field: str, source: Path) -> int:
    value = manifest.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source}: labels_manifest field {field!r} must be positive")
    return value


def _record_episode(
    episodes: dict[str, dict[str, tuple[str, str]]],
    forecaster_id: str,
    episode_id: str,
    dataset_id: str,
    family_id: str,
    path: Path,
    line_number: int,
) -> None:
    model_episodes = episodes.setdefault(forecaster_id, {})
    metadata = (dataset_id, family_id)
    previous = model_episodes.setdefault(episode_id, metadata)
    if previous != metadata:
        raise ValueError(
            f"{path}:{line_number}: episode {episode_id!r} has conflicting "
            "dataset/family metadata"
        )


def _inspect_source(
    root: Path,
    unary_keys: set[tuple[str, str, str, str]],
    pair_keys: set[tuple[str, str, tuple[str, str], tuple[str, str]]],
) -> _SourceArtifact:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"label artifact input must be a directory: {root}")
    stage_manifest = _read_json_object(root / "stage_manifest.json", "stage manifest")
    if stage_manifest.get("stage") != "labels" or stage_manifest.get("status") != "completed":
        raise ValueError(f"label artifact is not a completed labels stage: {root}")
    manifest = _read_json_object(root / "labels_manifest.json", "labels manifest")
    resolved_config = _read_json_object(root / "resolved_config.json", "resolved config")
    config = resolved_config.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"resolved config does not contain a config object: {root}")

    forecasters = _manifest_forecasters(manifest, root)
    declared_forecasters = set(forecasters)
    expected_unary_rows = _manifest_count(manifest, "unary_rows", root)
    expected_pair_rows = _manifest_count(manifest, "pair_rows", root)
    expected_groups = _manifest_count(manifest, "ranking_groups", root)
    teacher_path = root / "teacher_labels.jsonl"
    pair_path = root / "pair_labels.jsonl"

    episodes: dict[str, dict[str, tuple[str, str]]] = {}
    blocks: dict[str, dict[str, set[str]]] = {}
    observed_unary_forecasters: set[str] = set()
    groups: set[str] = set()
    local_unary_keys: set[tuple[str, str, str, str]] = set()
    unary_rows = 0
    for line_number, row in _iter_jsonl(teacher_path):
        values = tuple(
            _nonempty_string(row.get(field), field, teacher_path, line_number)
            for field in _UNARY_KEY_FIELDS
        )
        forecaster_id, episode_id, block_id, candidate_id = values
        dataset_id = _nonempty_string(
            row.get("dataset_id"), "dataset_id", teacher_path, line_number
        )
        family_id = _nonempty_string(
            row.get("family_id"), "family_id", teacher_path, line_number
        )
        group_id = _nonempty_string(
            row.get("group_id"), "group_id", teacher_path, line_number
        )
        expected_group = f"{forecaster_id}::{episode_id}::{block_id}"
        if group_id != expected_group:
            raise ValueError(
                f"{teacher_path}:{line_number}: group_id is inconsistent with its row key"
            )
        key = (forecaster_id, episode_id, block_id, candidate_id)
        if key in unary_keys:
            raise ValueError(f"duplicate unary label key: {key}")
        unary_keys.add(key)
        local_unary_keys.add(key)
        groups.add(group_id)
        observed_unary_forecasters.add(forecaster_id)
        _record_episode(
            episodes,
            forecaster_id,
            episode_id,
            dataset_id,
            family_id,
            teacher_path,
            line_number,
        )
        blocks.setdefault(forecaster_id, {}).setdefault(episode_id, set()).add(block_id)
        unary_rows += 1

    observed_pair_forecasters: set[str] = set()
    pair_rows = 0
    for line_number, row in _iter_jsonl(pair_path):
        forecaster_id = _nonempty_string(
            row.get("forecaster_id"), "forecaster_id", pair_path, line_number
        )
        episode_id = _nonempty_string(
            row.get("episode_id"), "episode_id", pair_path, line_number
        )
        dataset_id = _nonempty_string(
            row.get("dataset_id"), "dataset_id", pair_path, line_number
        )
        family_id = _nonempty_string(
            row.get("family_id"), "family_id", pair_path, line_number
        )
        left = (
            _nonempty_string(row.get("left_block"), "left_block", pair_path, line_number),
            _nonempty_string(
                row.get("left_candidate"), "left_candidate", pair_path, line_number
            ),
        )
        right = (
            _nonempty_string(row.get("right_block"), "right_block", pair_path, line_number),
            _nonempty_string(
                row.get("right_candidate"), "right_candidate", pair_path, line_number
            ),
        )
        first, second = sorted((left, right))
        pair_key = (forecaster_id, episode_id, first, second)
        if pair_key in pair_keys:
            raise ValueError(f"duplicate pair label key: {pair_key}")
        pair_keys.add(pair_key)
        for block_id, candidate_id in (left, right):
            unary_key = (forecaster_id, episode_id, block_id, candidate_id)
            if unary_key not in local_unary_keys:
                raise ValueError(
                    f"{pair_path}:{line_number}: pair endpoint has no matching unary label: "
                    f"{unary_key}"
                )
        observed_pair_forecasters.add(forecaster_id)
        _record_episode(
            episodes,
            forecaster_id,
            episode_id,
            dataset_id,
            family_id,
            pair_path,
            line_number,
        )
        pair_rows += 1

    if observed_unary_forecasters != declared_forecasters:
        raise ValueError(
            f"{root}: unary forecasters do not match labels_manifest: "
            f"observed={sorted(observed_unary_forecasters)}, "
            f"declared={sorted(declared_forecasters)}"
        )
    if observed_pair_forecasters != declared_forecasters:
        raise ValueError(
            f"{root}: pair forecasters do not match labels_manifest: "
            f"observed={sorted(observed_pair_forecasters)}, "
            f"declared={sorted(declared_forecasters)}"
        )
    if unary_rows != expected_unary_rows or pair_rows != expected_pair_rows:
        raise ValueError(
            f"{root}: labels_manifest row counts do not match JSONL files "
            f"(unary {expected_unary_rows}!={unary_rows} or "
            f"pair {expected_pair_rows}!={pair_rows})"
        )
    if len(groups) != expected_groups:
        raise ValueError(
            f"{root}: labels_manifest ranking_groups={expected_groups} "
            f"does not match {len(groups)} observed groups"
        )

    frozen_blocks = {
        model_id: {
            episode_id: frozenset(block_ids)
            for episode_id, block_ids in model_blocks.items()
        }
        for model_id, model_blocks in blocks.items()
    }
    return _SourceArtifact(
        root=root,
        teacher_labels=teacher_path,
        pair_labels=pair_path,
        manifest=manifest,
        config=config,
        forecasters=forecasters,
        unary_rows=unary_rows,
        pair_rows=pair_rows,
        ranking_groups=len(groups),
        teacher_sha256=_sha256(teacher_path),
        pair_sha256=_sha256(pair_path),
        episodes=episodes,
        blocks=frozen_blocks,
    )


def _compatibility_difference(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
) -> str | None:
    for field in _COMPATIBLE_MANIFEST_FIELDS:
        if reference.get(field) != current.get(field):
            return field
    return None


def _describe_set_difference(reference: set[str], current: set[str]) -> str:
    missing = sorted(reference - current)[:5]
    extra = sorted(current - reference)[:5]
    return f"missing={missing}, extra={extra}"


def _validate_cross_source_compatibility(sources: Sequence[_SourceArtifact]) -> None:
    reference = sources[0]
    reference_config = _canonical_json(reference.config)
    reference_lineage = _path_identity(
        _resolved_path(reference.manifest.get("imputer_artifacts"), "imputer_artifacts", reference.root)
    )
    all_forecasters: set[str] = set()
    episodes_by_model: dict[str, Mapping[str, tuple[str, str]]] = {}
    blocks_by_model: dict[str, Mapping[str, frozenset[str]]] = {}

    for source in sources:
        if _canonical_json(source.config) != reference_config:
            raise ValueError(
                f"resolved config mismatch between {reference.root} and {source.root}"
            )
        lineage = _path_identity(
            _resolved_path(source.manifest.get("imputer_artifacts"), "imputer_artifacts", source.root)
        )
        if lineage != reference_lineage:
            raise ValueError(
                f"imputer_artifacts lineage mismatch between {reference.root} and {source.root}"
            )
        incompatible_field = _compatibility_difference(reference.manifest, source.manifest)
        if incompatible_field is not None:
            raise ValueError(
                f"labels manifest field {incompatible_field!r} differs between "
                f"{reference.root} and {source.root}"
            )
        overlap = all_forecasters.intersection(source.forecasters)
        if overlap:
            raise ValueError(
                "forecaster labels occur in more than one source: " + ", ".join(sorted(overlap))
            )
        all_forecasters.update(source.forecasters)
        episodes_by_model.update(source.episodes)
        blocks_by_model.update(source.blocks)

    reference_model = sorted(episodes_by_model)[0]
    reference_episodes = episodes_by_model[reference_model]
    reference_episode_ids = set(reference_episodes)
    for model_id, episodes in episodes_by_model.items():
        episode_ids = set(episodes)
        if episode_ids != reference_episode_ids:
            difference = _describe_set_difference(reference_episode_ids, episode_ids)
            raise ValueError(
                f"episode compatibility mismatch for forecaster {model_id!r}: {difference}"
            )
        for episode_id in reference_episode_ids:
            if episodes[episode_id] != reference_episodes[episode_id]:
                raise ValueError(
                    f"dataset compatibility mismatch for episode {episode_id!r} "
                    f"under forecaster {model_id!r}"
                )

    reference_blocks = blocks_by_model[reference_model]
    for model_id, model_blocks in blocks_by_model.items():
        for episode_id in reference_episode_ids:
            if model_blocks.get(episode_id) != reference_blocks.get(episode_id):
                raise ValueError(
                    f"block compatibility mismatch for episode {episode_id!r} "
                    f"under forecaster {model_id!r}"
                )


def _copy_checked(source: Path, destination: BinaryIO, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    last_byte = b""
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            destination.write(chunk)
            last_byte = chunk[-1:]
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"source label file changed during merge: {source}")
    if last_byte not in {b"\n", b"\r"}:
        destination.write(b"\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def merge_label_artifacts(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and merge completed, mutually compatible labels-stage directories.

    The output directory must not already exist. All source validation is completed
    before the directory is created, so incompatible inputs do not leave an output.
    """

    if len(inputs) < 2:
        raise ValueError("at least two label artifact directories are required")
    roots = tuple(Path(value).resolve() for value in inputs)
    if len({_path_identity(path) for path in roots}) != len(roots):
        raise ValueError("label artifact input directories must be unique")

    unary_keys: set[tuple[str, str, str, str]] = set()
    pair_keys: set[tuple[str, str, tuple[str, str], tuple[str, str]]] = set()
    sources = tuple(_inspect_source(root, unary_keys, pair_keys) for root in roots)
    _validate_cross_source_compatibility(sources)
    ordered_sources = tuple(sorted(sources, key=lambda source: source.forecasters))

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"label merge output directory already exists: {output}")
    if any(output == source.root or output.is_relative_to(source.root) for source in sources):
        raise ValueError("label merge output must not be inside an input artifact directory")
    output.mkdir(parents=True, exist_ok=False)

    teacher_output = output / "teacher_labels.jsonl"
    pair_output = output / "pair_labels.jsonl"
    with teacher_output.with_suffix(".jsonl.tmp").open("wb") as handle:
        for source in ordered_sources:
            _copy_checked(source.teacher_labels, handle, source.teacher_sha256)
    teacher_output.with_suffix(".jsonl.tmp").replace(teacher_output)
    with pair_output.with_suffix(".jsonl.tmp").open("wb") as handle:
        for source in ordered_sources:
            _copy_checked(source.pair_labels, handle, source.pair_sha256)
    pair_output.with_suffix(".jsonl.tmp").replace(pair_output)

    reference = ordered_sources[0]
    forecasters = sorted(
        model_id for source in ordered_sources for model_id in source.forecasters
    )
    reference_episodes = reference.episodes[reference.forecasters[0]]
    datasets = sorted({metadata[0] for metadata in reference_episodes.values()})
    families = sorted({metadata[1] for metadata in reference_episodes.values()})
    resolved_lineage = _resolved_path(
        reference.manifest.get("imputer_artifacts"), "imputer_artifacts", reference.root
    )
    source_records = [
        {
            "directory": str(source.root),
            "forecasters": list(source.forecasters),
            "unary_rows": source.unary_rows,
            "pair_rows": source.pair_rows,
            "ranking_groups": source.ranking_groups,
            "artifact_loading": source.manifest.get("artifact_loading"),
            "teacher_labels_sha256": source.teacher_sha256,
            "pair_labels_sha256": source.pair_sha256,
        }
        for source in ordered_sources
    ]
    episode_execution_count = sum(
        int(source.manifest.get("episode_count", 0)) for source in ordered_sources
    )
    episode_sampling = reference.manifest.get("episode_sampling")
    if isinstance(episode_sampling, Mapping):
        episode_sampling = dict(episode_sampling)
        episode_sampling["episode_execution_count"] = episode_execution_count
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "merged_teacher_labels",
        "created_at": utc_now(),
        "teacher_labels": str(teacher_output),
        "pair_labels": str(pair_output),
        "forecasters": forecasters,
        "imputer_artifacts": str(resolved_lineage),
        "source_directories": [record["directory"] for record in source_records],
        "sources": source_records,
        "config_sha256": hashlib.sha256(
            _canonical_json(reference.config).encode("utf-8")
        ).hexdigest(),
        "origin_partition": reference.manifest.get("origin_partition"),
        "dataset_ids": datasets,
        "family_ids": families,
        "episode_count": len(reference_episodes),
        "episode_execution_count": episode_execution_count,
        "episode_sampling": episode_sampling,
        "ranking_groups": sum(source.ranking_groups for source in ordered_sources),
        "unary_rows": sum(source.unary_rows for source in ordered_sources),
        "pair_rows": sum(source.pair_rows for source in ordered_sources),
        "selected_candidates": reference.manifest.get("selected_candidates"),
        "max_train_episodes_per_dataset": reference.manifest.get(
            "max_train_episodes_per_dataset"
        ),
        "max_teacher_blocks_per_episode": reference.manifest.get(
            "max_teacher_blocks_per_episode"
        ),
        "max_teacher_candidates_per_episode": reference.manifest.get(
            "max_teacher_candidates_per_episode"
        ),
        "max_pair_labels_per_episode": reference.manifest.get(
            "max_pair_labels_per_episode"
        ),
        "csdi_num_samples": reference.manifest.get("csdi_num_samples"),
    }
    _write_json(output / "labels_manifest.json", summary)
    _write_json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "sources": [str(source.root / "resolved_config.json") for source in ordered_sources],
            "config": reference.config,
        },
    )
    return summary


__all__ = ["merge_label_artifacts"]
