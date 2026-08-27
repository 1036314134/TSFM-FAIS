"""Migrate preserved reconstruction labels to strict episode sidecars.

The numerical label rows, deterministic episode plans, expectations, and
outcomes are copied without alteration.  The new artifact adds the row hashes
and counts required by :class:`LabelProgressStore` and records complete source
and target lineage.  Existing artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsfm_fais.label_resume import (  # noqa: E402
    LABEL_PROGRESS_SCHEMA_VERSION,
    LABEL_SIDECAR_SCHEMA_VERSION,
    LabelEpisodeExpectation,
    LabelProgressStore,
    canonical_sha256,
    validate_label_rows,
)

MIGRATION_SCHEMA_VERSION = 1
MIGRATION_PROTOCOL_ID = "b_fais_r2_strict_reconstruction_sidecars_v001"
EXPECTED_EPISODE_COUNT = 2430
EXPECTED_DATASET_COUNT = 27
EXPECTED_FAMILY_COUNT = 17
EXPECTED_EPISODES_PER_DATASET = 90
STATIC_ROOT_FILES = (
    "candidate_status.json",
    "experiment_protocol.json",
    "repository_state.json",
    "resolved_config.json",
    "seeds.json",
    "software_versions.json",
)
DERIVED_METADATA_FILES = {
    "migration_failure.json",
    "migration_manifest.json",
    "migration_validation.json",
}


@dataclass(frozen=True)
class SourceEpisode:
    key: str
    expectation: LabelEpisodeExpectation
    outcome: Literal["labeled", "no_labels"]
    unary_rows: tuple[dict[str, Any], ...]
    pair_rows: tuple[dict[str, Any], ...]
    validated: dict[str, Any]
    source_sidecar_sha256: str
    source_sidecar_created_at: str | None


@dataclass(frozen=True)
class ValidatedSource:
    root: Path
    stage: dict[str, Any]
    manifest: dict[str, Any]
    progress: dict[str, Any]
    plans: dict[str, dict[str, Any]]
    episodes: tuple[SourceEpisode, ...]
    signatures: dict[str, Any]
    tree_signature: dict[str, Any]
    unary_rows: int
    pair_rows: int
    ranking_groups: int
    labeled_episodes: int
    no_label_episodes: int
    family_ids: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} at {path}: {error}") from error
    return rows


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _atomic_write_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    _atomic_write_bytes(path, _json_bytes(payload), replace=replace)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]], *, replace: bool = False) -> None:
    _atomic_write_bytes(path, _jsonl_bytes(rows), replace=replace)


def _tree_signature(root: Path, *, excluded_names: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded_names or set()
    files = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if Path(relative).name in excluded:
            continue
        files.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError(f"artifact has no files: {root}")
    return {
        "path": str(root),
        "file_count": len(files),
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def _non_negative_int(payload: dict[str, Any], field: str, description: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} field {field!r} must be a non-negative integer")
    return value


def _replace_source_paths(value: Any, source: Path, target: Path) -> Any:
    source_text = str(source)
    target_text = str(target)
    if isinstance(value, dict):
        return {
            str(key): _replace_source_paths(item, source, target) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_source_paths(item, source, target) for item in value]
    if isinstance(value, str):
        return value.replace(source_text, target_text)
    return value


def _validate_plan(
    dataset_id: str,
    plan: Any,
    *,
    expected_episodes_per_dataset: int,
    require_complete_sampling: bool,
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("dataset_id") != dataset_id:
        raise ValueError(f"dataset plan {dataset_id!r} is invalid")
    selection_summary = plan.get("selection_summary")
    episode_ids = plan.get("episode_ids")
    if not isinstance(selection_summary, dict):
        raise ValueError(f"dataset plan {dataset_id!r} has no selection summary")
    if (
        not isinstance(episode_ids, list)
        or len(episode_ids) != expected_episodes_per_dataset
        or any(not isinstance(item, str) or not item for item in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError(f"dataset plan {dataset_id!r} episode IDs differ")
    unsigned = {
        "dataset_id": dataset_id,
        "selection_summary": selection_summary,
        "episode_ids": episode_ids,
    }
    if plan.get("sha256") != canonical_sha256(unsigned):
        raise ValueError(f"dataset plan {dataset_id!r} signature differs")
    if require_complete_sampling:
        consistency = selection_summary.get("revision_sampling_consistency")
        if (
            selection_summary.get("selected_episode_count") != expected_episodes_per_dataset
            or selection_summary.get("mechanism_rate_seed_full_coverage_achieved") is not True
            or not isinstance(consistency, dict)
            or consistency.get("status") != "complete"
            or consistency.get("expected_mechanism_rate_seed_cells")
            != expected_episodes_per_dataset
            or consistency.get("selected_mechanism_rate_seed_cells")
            != expected_episodes_per_dataset
            or consistency.get("missing_mechanism_rate_seed_cells") not in ([], "")
        ):
            raise ValueError(f"dataset plan {dataset_id!r} sampling coverage differs")
    return copy.deepcopy(plan)


def _validate_source(
    root: Path,
    *,
    expected_episode_count: int,
    expected_dataset_count: int,
    expected_family_count: int,
    expected_episodes_per_dataset: int,
    require_complete_sampling: bool,
) -> ValidatedSource:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"source label artifact does not exist: {root}")
    required_files = {
        *STATIC_ROOT_FILES,
        "labels_manifest.json",
        "labels_progress.json",
        "pair_labels.jsonl",
        "stage_manifest.json",
        "teacher_labels.jsonl",
    }
    missing = sorted(name for name in required_files if not (root / name).is_file())
    if missing:
        raise ValueError(f"source label artifact is missing files: {missing}")
    source_tree = _tree_signature(root)
    stage = _read_json(root / "stage_manifest.json", "source stage manifest")
    manifest = _read_json(root / "labels_manifest.json", "source labels manifest")
    progress = _read_json(root / "labels_progress.json", "source labels progress")
    if (
        stage.get("stage") != "labels"
        or stage.get("status") != "completed"
        or stage.get("run_id") != root.name
    ):
        raise ValueError("source stage manifest is not a completed labels stage")
    if (
        progress.get("schema_version") != LABEL_PROGRESS_SCHEMA_VERSION
        or progress.get("status") != "rebuilt"
    ):
        raise ValueError("source progress is not rebuilt with schema version 1")
    if progress.get("resume_count") != 0 or progress.get("repair_count") != 0:
        raise ValueError("source progress has a nonzero resume or repair count")
    if (
        manifest.get("episode_count") != expected_episode_count
        or manifest.get("expected_episode_count") != expected_episode_count
        or progress.get("completed_count") != expected_episode_count
    ):
        raise ValueError("source episode counts differ from the required deterministic plan")
    expected_protocol = {
        "routing_target_protocol": "sequence_imputation_quality_v1",
        "target_protocol": "masked_context_reconstruction_asmape_v1",
        "forecasters": ["imputation"],
        "active_mask_partition": "train",
        "active_mask_seeds": [1101, 1102, 1103],
        "resume_count": 0,
        "repair_count": 0,
    }
    for field, expected in expected_protocol.items():
        if manifest.get(field) != expected:
            raise ValueError(f"source reconstruction protocol differs at {field}")
    selected_candidates = manifest.get("selected_candidates")
    if (
        not isinstance(selected_candidates, list)
        or not selected_candidates
        or len(set(map(str, selected_candidates))) != len(selected_candidates)
    ):
        raise ValueError("source selected candidates are invalid")

    teacher_path = root / "teacher_labels.jsonl"
    pair_path = root / "pair_labels.jsonl"
    if _sha256(teacher_path) != manifest.get("teacher_labels_sha256"):
        raise ValueError("source teacher-label signature differs")
    if _sha256(pair_path) != manifest.get("pair_labels_sha256"):
        raise ValueError("source pair-label signature differs")
    if Path(str(manifest.get("teacher_labels", ""))).resolve() != teacher_path:
        raise ValueError("source teacher-label path differs")
    if Path(str(manifest.get("pair_labels", ""))).resolve() != pair_path:
        raise ValueError("source pair-label path differs")
    if Path(str(manifest.get("progress", ""))).resolve() != root / "labels_progress.json":
        raise ValueError("source progress path differs")

    plans_payload = progress.get("dataset_plans")
    entries = progress.get("entries")
    identity = progress.get("identity")
    if (
        not isinstance(plans_payload, dict)
        or not isinstance(entries, dict)
        or not isinstance(identity, dict)
    ):
        raise ValueError("source progress plans, entries, or identity are invalid")
    if len(plans_payload) != expected_dataset_count:
        raise ValueError("source dataset-plan count differs")
    plans: dict[str, dict[str, Any]] = {}
    planned_episode_ids: set[str] = set()
    for dataset_id in sorted(plans_payload):
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("source dataset-plan IDs must be non-empty strings")
        plan = _validate_plan(
            dataset_id,
            plans_payload[dataset_id],
            expected_episodes_per_dataset=expected_episodes_per_dataset,
            require_complete_sampling=require_complete_sampling,
        )
        duplicates = planned_episode_ids.intersection(plan["episode_ids"])
        if duplicates:
            raise ValueError(
                f"source episode IDs occur in multiple plans: {sorted(duplicates)[:5]}"
            )
        planned_episode_ids.update(plan["episode_ids"])
        plans[dataset_id] = plan
    if len(planned_episode_ids) != expected_episode_count:
        raise ValueError("source plans do not contain the expected number of episodes")

    expected_keys = {f"{index:08d}" for index in range(expected_episode_count)}
    if set(entries) != expected_keys:
        raise ValueError("source progress does not have a contiguous episode index")
    sidecar_root = root / "label_episode_records"
    if not sidecar_root.is_dir():
        raise ValueError("source sidecar directory is missing")
    sidecar_files = {
        path.relative_to(root).as_posix() for path in sidecar_root.rglob("*.json") if path.is_file()
    }
    expected_sidecars = {
        f"label_episode_records/{index:08d}.json" for index in range(expected_episode_count)
    }
    if sidecar_files != expected_sidecars:
        raise ValueError("source sidecar files do not match the deterministic episode index")

    episodes: list[SourceEpisode] = []
    all_unary: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    episode_identity: set[tuple[str, str, str]] = set()
    labeled = 0
    no_labels = 0
    omitted_strict_fields = 0
    for index in range(expected_episode_count):
        key = f"{index:08d}"
        entry = entries[key]
        if not isinstance(entry, dict) or entry.get("artifact_index") != index:
            raise ValueError(f"source progress entry {key} is invalid")
        expectation_payload = entry.get("expectation")
        if not isinstance(expectation_payload, dict):
            raise ValueError(f"source progress entry {key} has no expectation")
        expectation = LabelEpisodeExpectation.from_payload(expectation_payload)
        if expectation.key != key or expectation.forecaster_id != "imputation":
            raise ValueError(f"source progress entry {key} identity differs")
        episode_plan = plans.get(expectation.dataset_id)
        if (
            episode_plan is None
            or expectation.dataset_plan_sha256 != episode_plan["sha256"]
            or expectation.episode_id not in episode_plan["episode_ids"]
            or expectation.episode_id not in planned_episode_ids
        ):
            raise ValueError(f"source progress entry {key} is absent from its signed plan")
        identity_key = (
            expectation.forecaster_id,
            expectation.dataset_id,
            expectation.episode_id,
        )
        if identity_key in episode_identity:
            raise ValueError(f"source progress contains duplicate episode identity: {identity_key}")
        episode_identity.add(identity_key)
        outcome = entry.get("outcome")
        if outcome not in {"labeled", "no_labels"}:
            raise ValueError(f"source progress entry {key} has an invalid outcome")
        relative = str(entry.get("sidecar_file", ""))
        expected_relative = f"label_episode_records/{key}.json"
        if relative != expected_relative:
            raise ValueError(f"source progress entry {key} sidecar path differs")
        sidecar_path = (root / relative).resolve()
        if not sidecar_path.is_relative_to(root) or not sidecar_path.is_file():
            raise ValueError(f"source progress entry {key} sidecar is unsafe or missing")
        source_sidecar_sha256 = _sha256(sidecar_path)
        if source_sidecar_sha256 != entry.get("sidecar_sha256"):
            raise ValueError(f"source progress entry {key} sidecar signature differs")
        sidecar = _read_json(sidecar_path, f"source sidecar {key}")
        if (
            sidecar.get("schema_version") != LABEL_SIDECAR_SCHEMA_VERSION
            or sidecar.get("expectation") != expectation.to_payload()
            or sidecar.get("outcome") != outcome
        ):
            raise ValueError(f"source sidecar {key} identity or outcome differs")
        unary_rows = sidecar.get("unary_rows")
        pair_rows = sidecar.get("pair_rows")
        if not isinstance(unary_rows, list) or not isinstance(pair_rows, list):
            raise ValueError(f"source sidecar {key} rows are invalid")
        validated = validate_label_rows(
            expectation,
            unary_rows,
            pair_rows,
            outcome=outcome,
        )
        source_counts = {
            "unary_rows": validated["unary_row_count"],
            "pair_rows": validated["pair_row_count"],
            "ranking_groups": validated["ranking_group_count"],
        }
        for field, expected in source_counts.items():
            if entry.get(field) != expected:
                raise ValueError(f"source progress entry {key} field {field!r} differs")
        strict_sidecar_fields = {
            "unary_rows_sha256": validated["unary_rows_sha256"],
            "pair_rows_sha256": validated["pair_rows_sha256"],
            "unary_row_count": validated["unary_row_count"],
            "pair_row_count": validated["pair_row_count"],
            "ranking_group_count": validated["ranking_group_count"],
        }
        for field, expected in strict_sidecar_fields.items():
            if field not in sidecar:
                omitted_strict_fields += 1
            elif sidecar.get(field) != expected:
                raise ValueError(f"source sidecar {key} field {field!r} differs")
        for field in ("unary_rows_sha256", "pair_rows_sha256"):
            if field not in entry:
                omitted_strict_fields += 1
            elif entry.get(field) != validated[field]:
                raise ValueError(f"source progress entry {key} field {field!r} differs")
        normalized_unary = tuple(validated["unary_rows"])
        normalized_pairs = tuple(validated["pair_rows"])
        all_unary.extend(normalized_unary)
        all_pairs.extend(normalized_pairs)
        family_ids.add(expectation.family_id)
        labeled += outcome == "labeled"
        no_labels += outcome == "no_labels"
        episodes.append(
            SourceEpisode(
                key=key,
                expectation=expectation,
                outcome=outcome,
                unary_rows=normalized_unary,
                pair_rows=normalized_pairs,
                validated=validated,
                source_sidecar_sha256=source_sidecar_sha256,
                source_sidecar_created_at=(
                    str(sidecar["created_at"]) if sidecar.get("created_at") is not None else None
                ),
            )
        )

    if omitted_strict_fields == 0:
        raise ValueError("source artifact already contains complete strict sidecar metadata")
    if len(family_ids) != expected_family_count:
        raise ValueError("source family count differs")
    teacher_rows = _read_jsonl(teacher_path, "source teacher labels")
    pair_rows = _read_jsonl(pair_path, "source pair labels")
    if teacher_rows != all_unary or pair_rows != all_pairs:
        raise ValueError("source JSONL rows differ from the ordered episode sidecars")
    unique_unary_keys = {
        (
            str(row["forecaster_id"]),
            str(row["episode_id"]),
            str(row["block_id"]),
            str(row["candidate_id"]),
        )
        for row in all_unary
    }
    if len(unique_unary_keys) != len(all_unary):
        raise ValueError("source teacher labels contain duplicate composite keys")
    if any(
        row.get("forecaster_id") != "imputation" or row.get("label_scope") != "whole_series"
        for row in all_unary
    ):
        raise ValueError("source teacher labels contain an invalid scope")
    counts = {
        "unary_rows": len(all_unary),
        "pair_rows": len(all_pairs),
        "ranking_groups": len({str(row["group_id"]) for row in all_unary}),
        "labeled_episode_count": labeled,
        "no_label_episode_count": no_labels,
    }
    for field, observed in counts.items():
        if manifest.get(field) != observed:
            raise ValueError(f"source labels manifest field {field!r} differs")
    for field in ("unary_rows", "pair_rows", "ranking_groups"):
        if field in progress and progress[field] != counts[field]:
            raise ValueError(f"source progress field {field!r} differs")
    stage_outputs = stage.get("outputs")
    if not isinstance(stage_outputs, dict):
        raise ValueError("source stage manifest has no output summary")
    for field in (
        "episode_count",
        "expected_episode_count",
        "labeled_episode_count",
        "no_label_episode_count",
        "unary_rows",
        "pair_rows",
        "ranking_groups",
        "teacher_labels_sha256",
        "pair_labels_sha256",
        "routing_target_protocol",
        "target_protocol",
    ):
        if stage_outputs.get(field) != manifest.get(field):
            raise ValueError(f"source stage and labels manifests differ at {field}")
    signatures = {
        "stage_manifest_sha256": _sha256(root / "stage_manifest.json"),
        "labels_manifest_sha256": _sha256(root / "labels_manifest.json"),
        "labels_progress_sha256": _sha256(root / "labels_progress.json"),
        "teacher_labels_sha256": _sha256(teacher_path),
        "pair_labels_sha256": _sha256(pair_path),
        "ordered_unary_rows_sha256": canonical_sha256(all_unary),
        "ordered_pair_rows_sha256": canonical_sha256(all_pairs),
        "dataset_plans_sha256": canonical_sha256(plans),
        "expectations_sha256": canonical_sha256(
            [episode.expectation.to_payload() for episode in episodes]
        ),
        "outcomes_sha256": canonical_sha256([episode.outcome for episode in episodes]),
        "omitted_strict_field_count": omitted_strict_fields,
    }
    return ValidatedSource(
        root=root,
        stage=stage,
        manifest=manifest,
        progress=progress,
        plans=plans,
        episodes=tuple(episodes),
        signatures=signatures,
        tree_signature=source_tree,
        unary_rows=counts["unary_rows"],
        pair_rows=counts["pair_rows"],
        ranking_groups=counts["ranking_groups"],
        labeled_episodes=labeled,
        no_label_episodes=no_labels,
        family_ids=tuple(sorted(family_ids)),
    )


def _build_strict_progress(
    source: ValidatedSource,
    target: Path,
    migration_started_at: str,
) -> tuple[LabelProgressStore, list[dict[str, Any]]]:
    identity = source.progress["identity"]
    store = LabelProgressStore.create(target, identity)
    for dataset_id, plan in sorted(source.plans.items()):
        plan_sha256 = store.register_dataset_plan(
            dataset_id,
            plan["selection_summary"],
            plan["episode_ids"],
        )
        if plan_sha256 != plan["sha256"] or store.payload["dataset_plans"][dataset_id] != plan:
            raise ValueError(f"target dataset plan {dataset_id!r} differs from source")

    integrity_rows: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    for episode in source.episodes:
        validated = episode.validated
        sidecar = {
            "schema_version": LABEL_SIDECAR_SCHEMA_VERSION,
            "outcome": episode.outcome,
            "expectation": episode.expectation.to_payload(),
            "unary_rows": list(episode.unary_rows),
            "pair_rows": list(episode.pair_rows),
            "unary_rows_sha256": validated["unary_rows_sha256"],
            "pair_rows_sha256": validated["pair_rows_sha256"],
            "unary_row_count": validated["unary_row_count"],
            "pair_row_count": validated["pair_row_count"],
            "ranking_group_count": validated["ranking_group_count"],
            "artifact_loading_delta": {},
            "created_at": migration_started_at,
        }
        sidecar_path = target / episode.expectation.sidecar_relative_path
        _atomic_write_json(sidecar_path, sidecar)
        target_sidecar_sha256 = _sha256(sidecar_path)
        entry = {
            "artifact_index": episode.expectation.artifact_index,
            "expectation": episode.expectation.to_payload(),
            "outcome": episode.outcome,
            "sidecar_file": episode.expectation.sidecar_relative_path.as_posix(),
            "sidecar_sha256": target_sidecar_sha256,
            "unary_rows_sha256": validated["unary_rows_sha256"],
            "pair_rows_sha256": validated["pair_rows_sha256"],
            "unary_rows": validated["unary_row_count"],
            "pair_rows": validated["pair_row_count"],
            "ranking_groups": validated["ranking_group_count"],
            "completed_at": migration_started_at,
        }
        entries[episode.key] = entry
        integrity_rows.append(
            {
                "artifact_index": episode.expectation.artifact_index,
                "key": episode.key,
                "forecaster_id": episode.expectation.forecaster_id,
                "dataset_id": episode.expectation.dataset_id,
                "family_id": episode.expectation.family_id,
                "episode_id": episode.expectation.episode_id,
                "outcome": episode.outcome,
                "expectation_sha256": canonical_sha256(episode.expectation.to_payload()),
                "unary_rows_sha256": validated["unary_rows_sha256"],
                "pair_rows_sha256": validated["pair_rows_sha256"],
                "unary_row_count": validated["unary_row_count"],
                "pair_row_count": validated["pair_row_count"],
                "ranking_group_count": validated["ranking_group_count"],
                "source_sidecar_sha256": episode.source_sidecar_sha256,
                "source_sidecar_created_at": episode.source_sidecar_created_at,
                "target_sidecar_sha256": target_sidecar_sha256,
            }
        )
    updated = copy.deepcopy(store.payload)
    updated["entries"] = entries
    updated.update(
        {
            "status": "running",
            "completed_count": len(entries),
            "unary_rows": sum(entry["unary_rows"] for entry in entries.values()),
            "pair_rows": sum(entry["pair_rows"] for entry in entries.values()),
            "ranking_groups": sum(entry["ranking_groups"] for entry in entries.values()),
            "resume_count": 0,
            "repair_count": 0,
            "created_at": migration_started_at,
            "updated_at": migration_started_at,
        }
    )
    _atomic_write_json(store.progress_path, updated, replace=True)
    store.payload = updated
    return store, integrity_rows


def _validate_target_store(
    store: LabelProgressStore,
    *,
    expected_episode_count: int,
) -> int:
    entries = store.payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("target progress entries are invalid")
    validated_count = 0
    for index in range(expected_episode_count):
        key = f"{index:08d}"
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"target progress entry {key} is missing")
        expectation = LabelEpisodeExpectation.from_payload(entry["expectation"])
        result = store.validate_episode(expectation)
        if result.status != "valid":
            raise ValueError(f"target sidecar {key} is invalid: {result.reason}")
        validated_count += 1
    return validated_count


def _write_target_manifests(
    source: ValidatedSource,
    target: Path,
    rebuilt: dict[str, Any],
    migration_started_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_manifest = _replace_source_paths(copy.deepcopy(source.manifest), source.root, target)
    labels_manifest.update(
        {
            "teacher_labels": rebuilt["teacher_labels"],
            "pair_labels": rebuilt["pair_labels"],
            "teacher_labels_sha256": rebuilt["teacher_labels_sha256"],
            "pair_labels_sha256": rebuilt["pair_labels_sha256"],
            "episode_count": rebuilt["episode_count"],
            "expected_episode_count": rebuilt["episode_count"],
            "labeled_episode_count": rebuilt["labeled_episode_count"],
            "no_label_episode_count": rebuilt["no_label_episode_count"],
            "unary_rows": rebuilt["unary_rows"],
            "pair_rows": rebuilt["pair_rows"],
            "ranking_groups": rebuilt["ranking_groups"],
            "progress": str((target / "labels_progress.json").resolve()),
            "episodes_executed_last_invocation": 0,
            "episodes_reused_last_invocation": rebuilt["episode_count"],
            "resume_count": 0,
            "repair_count": 0,
            "artifact_derivation": {
                "protocol_id": MIGRATION_PROTOCOL_ID,
                "kind": "deterministic_integrity_metadata_migration",
                "source_artifact": str(source.root),
                "source_run_id": source.root.name,
                "source_signatures": source.signatures,
                "numerical_rows_changed": False,
                "dataset_plans_changed": False,
                "episode_expectations_changed": False,
                "episode_outcomes_changed": False,
                "migration_manifest": str((target / "migration_manifest.json").resolve()),
            },
        }
    )
    _atomic_write_json(target / "labels_manifest.json", labels_manifest)

    stage_manifest = _replace_source_paths(copy.deepcopy(source.stage), source.root, target)
    stage_manifest.update(
        {
            "run_id": target.name,
            "status": "completed",
            "created_at": migration_started_at,
            "updated_at": _utc_now(),
            "execution_started": False,
            "message": "deterministic strict-sidecar migration completed; numerical rows unchanged",
            "outputs": labels_manifest,
            "migration": {
                "protocol_id": MIGRATION_PROTOCOL_ID,
                "source_artifact": str(source.root),
                "source_run_id": source.root.name,
                "numerical_execution_repeated": False,
            },
        }
    )
    stage_manifest["protocol_artifact"] = str((target / "experiment_protocol.json").resolve())
    _atomic_write_json(target / "stage_manifest.json", stage_manifest)
    return labels_manifest, stage_manifest


def _validate_completed_target(
    target: Path,
    source: ValidatedSource,
    migration_manifest: dict[str, Any],
    *,
    expected_episode_count: int,
) -> dict[str, Any]:
    manifest = _read_json(target / "labels_manifest.json", "target labels manifest")
    progress = _read_json(target / "labels_progress.json", "target labels progress")
    stage = _read_json(target / "stage_manifest.json", "target stage manifest")
    if (
        stage.get("status") != "completed"
        or stage.get("stage") != "labels"
        or stage.get("run_id") != target.name
        or progress.get("status") != "rebuilt"
    ):
        raise ValueError("target stage or progress completion state differs")
    if progress.get("resume_count") != 0 or progress.get("repair_count") != 0:
        raise ValueError("target progress has a nonzero resume or repair count")
    if (
        manifest.get("expected_episode_count") != expected_episode_count
        or progress.get("completed_count") != expected_episode_count
    ):
        raise ValueError("target episode count differs")
    if progress.get("identity") != source.progress.get("identity"):
        raise ValueError("target progress identity differs from source")
    if progress.get("dataset_plans") != source.plans:
        raise ValueError("target dataset plans differ from source")
    target_store = LabelProgressStore(target, progress)
    sidecars_validated = _validate_target_store(
        target_store,
        expected_episode_count=expected_episode_count,
    )
    teacher_sha256 = _sha256(target / "teacher_labels.jsonl")
    pair_sha256 = _sha256(target / "pair_labels.jsonl")
    if (
        teacher_sha256 != source.signatures["teacher_labels_sha256"]
        or pair_sha256 != source.signatures["pair_labels_sha256"]
        or teacher_sha256 != manifest.get("teacher_labels_sha256")
        or pair_sha256 != manifest.get("pair_labels_sha256")
    ):
        raise ValueError("target JSONL signatures differ from source or manifest")
    target_tree = _tree_signature(target, excluded_names=DERIVED_METADATA_FILES)
    expected_tree = migration_manifest.get("target", {}).get("content_signature")
    if not isinstance(expected_tree, dict) or target_tree["tree_sha256"] != expected_tree.get(
        "tree_sha256"
    ):
        raise ValueError("target content signature differs from the migration manifest")
    unsigned_manifest = copy.deepcopy(migration_manifest)
    digest = unsigned_manifest.pop("canonical_sha256", None)
    if digest != canonical_sha256(unsigned_manifest):
        raise ValueError("migration manifest canonical signature differs")
    return {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "status": "verified",
        "protocol_id": MIGRATION_PROTOCOL_ID,
        "artifact": str(target),
        "episode_count": expected_episode_count,
        "sidecars_validated": sidecars_validated,
        "dataset_count": len(source.plans),
        "family_count": len(source.family_ids),
        "unary_rows": source.unary_rows,
        "pair_rows": source.pair_rows,
        "ranking_groups": source.ranking_groups,
        "teacher_labels_sha256": teacher_sha256,
        "pair_labels_sha256": pair_sha256,
        "target_content_tree_sha256": target_tree["tree_sha256"],
        "source_content_tree_sha256": source.tree_signature["tree_sha256"],
        "resume_count": 0,
        "repair_count": 0,
        "numerical_rows_changed": False,
    }


def migrate_label_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_episode_count: int = EXPECTED_EPISODE_COUNT,
    expected_dataset_count: int = EXPECTED_DATASET_COUNT,
    expected_family_count: int = EXPECTED_FAMILY_COUNT,
    expected_episodes_per_dataset: int = EXPECTED_EPISODES_PER_DATASET,
    require_complete_sampling: bool = True,
) -> dict[str, Any]:
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if source_path == output_path or source_path.parent != output_path.parent:
        raise ValueError("source and target must be distinct sibling artifact directories")
    validated_source = _validate_source(
        source_path,
        expected_episode_count=expected_episode_count,
        expected_dataset_count=expected_dataset_count,
        expected_family_count=expected_family_count,
        expected_episodes_per_dataset=expected_episodes_per_dataset,
        require_complete_sampling=require_complete_sampling,
    )
    started_at = _utc_now()
    output_path.mkdir(parents=False, exist_ok=False)
    try:
        for name in STATIC_ROOT_FILES:
            _atomic_write_bytes(output_path / name, (source_path / name).read_bytes())
        store, integrity_rows = _build_strict_progress(
            validated_source,
            output_path,
            started_at,
        )
        first_validation_count = _validate_target_store(
            store,
            expected_episode_count=expected_episode_count,
        )
        rebuilt = store.rebuild_outputs(
            output_path / "teacher_labels.jsonl",
            output_path / "pair_labels.jsonl",
            expected_episode_count=expected_episode_count,
        )
        if (
            rebuilt["teacher_labels_sha256"] != validated_source.signatures["teacher_labels_sha256"]
            or rebuilt["pair_labels_sha256"] != validated_source.signatures["pair_labels_sha256"]
        ):
            raise ValueError("rebuilt JSONL bytes differ from the preserved source")
        if (
            rebuilt["unary_rows"] != validated_source.unary_rows
            or rebuilt["pair_rows"] != validated_source.pair_rows
            or rebuilt["ranking_groups"] != validated_source.ranking_groups
            or rebuilt["labeled_episode_count"] != validated_source.labeled_episodes
            or rebuilt["no_label_episode_count"] != validated_source.no_label_episodes
        ):
            raise ValueError("rebuilt target counts differ from the preserved source")
        _write_target_manifests(validated_source, output_path, rebuilt, started_at)
        integrity_path = output_path / "episode_integrity_map.jsonl"
        _atomic_write_jsonl(integrity_path, integrity_rows)
        source_post_signature = _tree_signature(source_path)
        if source_post_signature["tree_sha256"] != validated_source.tree_signature["tree_sha256"]:
            raise ValueError("source artifact changed during migration")
        target_content_signature = _tree_signature(
            output_path,
            excluded_names=DERIVED_METADATA_FILES,
        )
        implementation_files = {
            "migration_script": Path(__file__).resolve(),
            "label_resume": SOURCE_ROOT / "tsfm_fais" / "label_resume.py",
            "stage_execution": SOURCE_ROOT / "tsfm_fais" / "stage_execution.py",
        }
        migration_manifest: dict[str, Any] = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "protocol_id": MIGRATION_PROTOCOL_ID,
            "status": "completed",
            "created_at": started_at,
            "completed_at": _utc_now(),
            "source": {
                "path": str(source_path),
                "run_id": source_path.name,
                "signatures": validated_source.signatures,
                "content_signature": validated_source.tree_signature,
            },
            "target": {
                "path": str(output_path),
                "run_id": output_path.name,
                "content_signature": target_content_signature,
            },
            "implementation_signatures": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
                for name, path in sorted(implementation_files.items())
            },
            "episode_integrity": {
                "path": str(integrity_path.resolve()),
                "file_sha256": _sha256(integrity_path),
                "canonical_rows_sha256": canonical_sha256(integrity_rows),
                "row_count": len(integrity_rows),
            },
            "counts": {
                "episodes": expected_episode_count,
                "datasets": expected_dataset_count,
                "families": expected_family_count,
                "labeled_episodes": validated_source.labeled_episodes,
                "no_label_episodes": validated_source.no_label_episodes,
                "unary_rows": validated_source.unary_rows,
                "pair_rows": validated_source.pair_rows,
                "ranking_groups": validated_source.ranking_groups,
                "strict_sidecars_validated_before_rebuild": first_validation_count,
            },
            "invariants": {
                "source_unchanged": True,
                "numerical_rows_changed": False,
                "teacher_labels_byte_identical": True,
                "pair_labels_byte_identical": True,
                "dataset_plans_changed": False,
                "episode_expectations_changed": False,
                "episode_outcomes_changed": False,
                "resume_count": 0,
                "repair_count": 0,
            },
        }
        migration_manifest["canonical_sha256"] = canonical_sha256(migration_manifest)
        migration_manifest_path = output_path / "migration_manifest.json"
        _atomic_write_json(migration_manifest_path, migration_manifest)
        validation = _validate_completed_target(
            output_path,
            validated_source,
            migration_manifest,
            expected_episode_count=expected_episode_count,
        )
        validation.update(
            {
                "validated_at": _utc_now(),
                "migration_manifest_sha256": _sha256(migration_manifest_path),
                "episode_integrity_map_sha256": _sha256(integrity_path),
            }
        )
        validation["canonical_sha256"] = canonical_sha256(validation)
        _atomic_write_json(output_path / "migration_validation.json", validation)
        final_source_signature = _tree_signature(source_path)
        if final_source_signature["tree_sha256"] != validated_source.tree_signature["tree_sha256"]:
            raise ValueError("source artifact changed before final validation completed")
        return validation
    except Exception as error:
        failure = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "protocol_id": MIGRATION_PROTOCOL_ID,
            "status": "failed",
            "failed_at": _utc_now(),
            "source": str(source_path),
            "output": str(output_path),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        failure["canonical_sha256"] = canonical_sha256(failure)
        try:
            _atomic_write_json(output_path / "migration_failure.json", failure)
        except Exception:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = migrate_label_artifact(arguments.source, arguments.output)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
