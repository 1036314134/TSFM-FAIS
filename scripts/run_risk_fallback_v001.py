#!/usr/bin/env python3
"""Versioned B-FAIS R2 risk scoring, selection, derivation, and validation."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsfm_fais.contracts import CandidateStatus, MissingBlock, SeriesBatch  # noqa: E402
from tsfm_fais.imputers.artifacts import load_dataset_imputer_artifacts  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.label_resume import (  # noqa: E402
    LabelEpisodeExpectation,
    LabelProgressStore,
)
from tsfm_fais.risk_fallback import (  # noqa: E402
    RiskScoreProtocol,
    RiskThreshold,
    ThresholdEpisode,
    apply_whole_episode_fallback,
    channel_proxy_mae,
    derived_pipeline_runtime,
    deterministic_pseudo_observed_mask,
    score_block,
    score_episode,
    select_robust_candidate,
    select_threshold,
    source_routing_actual_ids,
)
from tsfm_fais.routing.blocks import detect_missing_blocks  # noqa: E402

SCRIPT_SCHEMA_VERSION = 1
SCRIPT_PROTOCOL_ID = "b_fais_r2_episode_proxy_regret_v001"
ARTIFACTS_ROOT = (REPOSITORY_ROOT / "artifacts" / "iclr27-r2").resolve()
SOURCE_FILE_PATHS = {
    "contracts.py": SOURCE_ROOT / "tsfm_fais" / "contracts.py",
    "imputers/base.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "base.py",
    "imputers/classical.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "classical.py",
    "imputers/pypots.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "pypots.py",
    "imputers/registry.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "registry.py",
    "imputers/runner.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "runner.py",
    "imputers/structured.py": SOURCE_ROOT / "tsfm_fais" / "imputers" / "structured.py",
    "pipeline.py": SOURCE_ROOT / "tsfm_fais" / "pipeline.py",
    "stage_execution.py": SOURCE_ROOT / "tsfm_fais" / "stage_execution.py",
}
HISTORICAL_ETT_EXEMPT_SOURCE_FILES = {"pipeline.py", "stage_execution.py"}
FROZEN_ETT_EVALUATIONS = {
    "chronos2": {
        "run_name": "r2-target-ett-full-candidate-chronos2-ms2101x3-rs4101-eval-v002",
        "manifest_sha256": "4934887309108e6d840d9412aa00948e1985dd24269c45d560822f63d5934b25",
        "metrics_sha256": "3c1941982cb9ac47e995c0c6cc191f47b67b7e3349680d2418634e99967e9cc5",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    },
    "timesfm2p5": {
        "run_name": "r2-target-ett-full-candidate-timesfm2p5-ms2101x3-rs4101-eval-v002",
        "manifest_sha256": "d3ced83df4298628ec09484782feafec406df8ff3eee7760974bba498874d603",
        "metrics_sha256": "f98be3a9b6beaadb78e4338efe6f64b7efb3858927dfd83562a939f80f652468",
        "revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    },
}


@dataclass(frozen=True)
class SourceEpisode:
    key: str
    index: int
    entry: dict[str, Any]
    assignment: dict[str, Any]
    npz_path: Path
    assignment_path: Path


@dataclass(frozen=True)
class ValidatedSource:
    root: Path
    manifest: dict[str, Any]
    progress: dict[str, Any]
    episodes: tuple[SourceEpisode, ...]
    lineage_mode: str
    signatures: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


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


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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


def _atomic_write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, replace: bool = False
) -> None:
    _atomic_write_bytes(path, _jsonl_bytes(rows), replace=replace)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256(source) != _sha256(target):
        raise RuntimeError(f"copied output differs from its source: {target}")


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} contains an empty line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {label} row at {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(row)
    return rows


def _safe_relative(value: Any, label: str) -> Path:
    relative = Path(str(value))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {label} path: {relative}")
    return relative


def _new_output_root(value: str | Path) -> Path:
    output = Path(value).resolve()
    if not output.is_relative_to(ARTIFACTS_ROOT):
        raise ValueError(f"new R2 outputs must stay under {ARTIFACTS_ROOT}")
    if output.exists():
        raise FileExistsError(f"output already exists; refusing overwrite or resume: {output}")
    output.mkdir(parents=True)
    return output


def _npz_scalar(archive: Any, key: str) -> Any:
    if key not in archive:
        raise ValueError(f"NPZ is missing scalar field {key!r}")
    values = np.asarray(archive[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"NPZ field {key!r} must contain one value")
    value = values[0]
    return value.item() if hasattr(value, "item") else value


def _forecast_mode(value: Any) -> Literal["joint_multivariate", "independent_univariate"]:
    mode = str(value)
    if mode not in {"joint_multivariate", "independent_univariate"}:
        raise ValueError(f"unsupported forecast mode: {mode!r}")
    if mode == "joint_multivariate":
        return "joint_multivariate"
    return "independent_univariate"


def _repository_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    tracked_diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest(),
        "risk_module": {
            "path": str(SOURCE_ROOT / "tsfm_fais" / "risk_fallback.py"),
            "sha256": _sha256(SOURCE_ROOT / "tsfm_fais" / "risk_fallback.py"),
        },
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
    }


def _validate_sha(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def _validate_recorded_content_signature(
    signature: Mapping[str, Any], label: str, candidate_paths: Sequence[Path]
) -> None:
    expected = _validate_sha(signature.get("sha256"), f"{label} SHA-256")
    expected_size = int(signature.get("size_bytes", -1))
    recorded_path = signature.get("path")
    paths = [Path(str(recorded_path))] if recorded_path else list(candidate_paths)
    matches = [
        path
        for path in paths
        if path.is_file() and path.stat().st_size == expected_size and _sha256(path) == expected
    ]
    if not matches:
        raise FileNotFoundError(f"no local {label} matches its recorded content signature")


def _validate_generation_identity(identity: Mapping[str, Any], lineage_mode: str) -> None:
    candidate_specs = identity.get("candidate_specs")
    if not isinstance(candidate_specs, Mapping):
        raise ValueError("candidate generation identity has no candidate specifications")
    if _canonical_sha256(candidate_specs) != identity.get("candidate_specs_sha256"):
        raise ValueError("candidate specification signature differs")
    known_paths = {
        "audit_artifact": (REPOSITORY_ROOT / "artifacts" / "data-audit-main-seq96-opt9-v1.json",),
        "imputer_artifact_manifest": tuple(
            sorted(ARTIFACTS_ROOT.glob("*/imputer_artifacts/manifest.json"))
        ),
        "imputer_registry": (REPOSITORY_ROOT / "configs" / "imputers" / "pool.yaml",),
    }
    for field in ("audit_artifact", "imputer_artifact_manifest", "imputer_registry"):
        signature = identity.get(field)
        if not isinstance(signature, Mapping):
            raise ValueError(f"candidate generation identity has no {field}")
        _validate_recorded_content_signature(signature, field, known_paths[field])
    recorded_sources = identity.get("source_files")
    if not isinstance(recorded_sources, Mapping):
        raise ValueError("candidate generation identity has no source file signatures")
    expected_names = set(SOURCE_FILE_PATHS)
    if set(recorded_sources) != expected_names:
        raise ValueError("candidate generation source file set differs")
    exemptions = (
        HISTORICAL_ETT_EXEMPT_SOURCE_FILES if lineage_mode == "historical-ett-v002" else set()
    )
    for name, path in SOURCE_FILE_PATHS.items():
        recorded = recorded_sources[name]
        if not isinstance(recorded, Mapping):
            raise ValueError(f"invalid source signature for {name}")
        _validate_sha(recorded.get("sha256"), f"source {name}")
        if name not in exemptions:
            if _sha256(path) != recorded.get("sha256") or path.stat().st_size != int(
                recorded.get("size_bytes", -1)
            ):
                raise ValueError(f"current source differs from signed source: {name}")


def _validate_npz_episode(path: Path, entry: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        for field in ("episode_id", "dataset_id", "family_id", "item_id", "forecaster_id"):
            if str(_npz_scalar(archive, field)) != str(entry[field]):
                raise ValueError(f"NPZ identity differs for {entry['index']:08d}.{field}")
        values = np.asarray(archive["values"], dtype=float)
        observed = np.asarray(archive["observed_mask"], dtype=bool)
        clean = np.asarray(archive["clean_context"], dtype=float)
        candidate_ids = tuple(str(value) for value in np.asarray(archive["candidate_ids"]).tolist())
        candidates = np.asarray(archive["candidate_values"], dtype=float)
        native = np.asarray(archive["candidate_native_valid"], dtype=bool)
        statuses = np.asarray(archive["candidate_status"]).reshape(-1)
        runtimes = np.asarray(archive["candidate_runtime_seconds"], dtype=float).reshape(-1)
        memories = np.asarray(archive["candidate_peak_memory_bytes"], dtype=np.int64).reshape(-1)
        if values.shape != observed.shape or clean.shape != values.shape:
            raise ValueError("NPZ assembled, mask, and context shapes differ")
        expected_shape = (len(candidate_ids), *values.shape)
        if candidates.shape != expected_shape or native.shape != expected_shape:
            raise ValueError("NPZ candidate tensor shapes differ")
        if not (len(statuses) == len(runtimes) == len(memories) == len(candidate_ids)):
            raise ValueError("NPZ candidate metadata lengths differ")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("NPZ contains duplicate candidate IDs")
        if tuple(entry.get("candidate_ids", ())) != candidate_ids:
            raise ValueError("NPZ candidate IDs differ from progress")
        if not np.isfinite(values).all() or not np.isfinite(candidates).all():
            raise ValueError("NPZ completed values must be finite")
        if not np.array_equal(values[observed], clean[observed]):
            raise ValueError("assembled output changed an observed value")
        if not np.all(candidates[:, observed] == clean[observed]):
            raise ValueError("candidate output changed an observed value")
        if not np.isfinite(runtimes).all() or np.any(runtimes < 0) or np.any(memories < 0):
            raise ValueError("candidate resource metadata is invalid")
        for status in statuses:
            CandidateStatus(str(status))


def _validate_imputation_source(root: str | Path, lineage_mode: str) -> ValidatedSource:
    source = Path(root).resolve()
    manifest_path = source / "imputation_manifest.json"
    progress_path = source / "imputation_progress.json"
    assignments_path = source / "routing_assignments.jsonl"
    manifest = _read_json(manifest_path, "imputation manifest")
    progress = _read_json(progress_path, "imputation progress")
    if progress.get("status") != "completed":
        raise ValueError("imputation progress is not completed")
    if not manifest.get("save_all_candidate_outputs"):
        raise ValueError("imputation source did not save every candidate output")
    episode_count = int(manifest.get("episode_count", -1))
    if episode_count < 1 or episode_count != int(progress.get("expected_episode_count", -2)):
        raise ValueError("imputation source episode counts differ")
    if int(progress.get("completed_count", -1)) != episode_count:
        raise ValueError("imputation progress is incomplete")
    if int(progress.get("resume_count", 0)) != 0 or int(progress.get("repair_count", 0)) != 0:
        raise ValueError("formal imputation source has a nonzero resume or repair count")
    if progress.get("imputation_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("imputation manifest signature differs from progress")
    if progress.get("routing_assignments_sha256") != _sha256(assignments_path):
        raise ValueError("routing assignment signature differs from progress")
    identity = manifest.get("candidate_generation_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("imputation manifest has no candidate generation identity")
    if manifest.get("candidate_generation_identity_sha256") != _canonical_sha256(identity):
        raise ValueError("candidate generation identity signature differs")
    _validate_generation_identity(identity, lineage_mode)

    raw_entries = progress.get("entries")
    if not isinstance(raw_entries, Mapping) or len(raw_entries) != episode_count:
        raise ValueError("imputation progress entries are incomplete")
    jsonl_rows = _read_jsonl(assignments_path, "routing assignments")
    if len(jsonl_rows) != episode_count:
        raise ValueError("routing assignment row count differs")
    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in jsonl_rows:
        index = int(row.get("artifact_index", -1))
        if index in rows_by_index:
            raise ValueError("routing assignments contain a duplicate artifact index")
        rows_by_index[index] = row

    episodes: list[SourceEpisode] = []
    for index in range(episode_count):
        key = f"{index:08d}"
        entry = raw_entries.get(key)
        if not isinstance(entry, dict) or int(entry.get("index", -1)) != index:
            raise ValueError(f"invalid imputation progress entry {key}")
        npz_relative = _safe_relative(entry.get("file"), "imputation")
        assignment_relative = _safe_relative(entry.get("assignment_file"), "assignment")
        npz_path = source / "imputations" / npz_relative
        assignment_path = source / assignment_relative
        if _sha256(npz_path) != _validate_sha(entry.get("npz_sha256"), "NPZ signature"):
            raise ValueError(f"NPZ signature differs at {key}")
        if _sha256(assignment_path) != _validate_sha(
            entry.get("assignment_sha256"), "assignment signature"
        ):
            raise ValueError(f"assignment signature differs at {key}")
        assignment = _read_json(assignment_path, "assignment sidecar")
        if rows_by_index.get(index) != assignment:
            raise ValueError(f"assignment JSONL and sidecar differ at {key}")
        for field in ("episode_id", "dataset_id", "family_id", "item_id", "forecaster_id"):
            if str(assignment.get(field)) != str(entry.get(field)):
                raise ValueError(f"assignment and progress differ at {key}.{field}")
        if str(assignment.get("file")) != str(entry.get("file")):
            raise ValueError(f"assignment and progress file differ at {key}")
        _validate_npz_episode(npz_path, entry)
        with np.load(npz_path, allow_pickle=False) as archive:
            observed = np.asarray(archive["observed_mask"], dtype=bool)[None, ...]
        blocks = detect_missing_blocks(observed)
        assignments = assignment.get("assignments")
        if not isinstance(assignments, Mapping) or set(assignments) != {
            block.block_id for block in blocks
        }:
            raise ValueError(f"assignment block set differs from reconstructed blocks at {key}")
        episodes.append(SourceEpisode(key, index, entry, assignment, npz_path, assignment_path))
    signatures = {
        "manifest_sha256": _sha256(manifest_path),
        "progress_sha256": _sha256(progress_path),
        "routing_assignments_sha256": _sha256(assignments_path),
        "candidate_generation_identity_sha256": _canonical_sha256(identity),
        "episode_npz_map_sha256": _canonical_sha256(
            {episode.key: episode.entry["npz_sha256"] for episode in episodes}
        ),
    }
    return ValidatedSource(
        source,
        manifest,
        progress,
        tuple(episodes),
        lineage_mode,
        signatures,
    )


def _validate_label_artifact(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(root / "labels_manifest.json", "labels manifest")
    progress = _read_json(root / "labels_progress.json", "labels progress")
    stage = _read_json(root / "stage_manifest.json", "label stage manifest")
    if stage.get("status") != "completed" or progress.get("status") != "rebuilt":
        raise ValueError("label artifact is not completed and rebuilt")
    if int(progress.get("resume_count", -1)) != 0 or int(progress.get("repair_count", -1)) != 0:
        raise ValueError("formal label artifact has a nonzero resume or repair count")
    expected_count = int(manifest.get("expected_episode_count", -1))
    if expected_count != 2430 or int(progress.get("completed_count", -1)) != expected_count:
        raise ValueError("formal reconstruction artifact must contain 2430 episodes")
    expected_protocol = {
        "routing_target_protocol": "sequence_imputation_quality_v1",
        "target_protocol": "masked_context_reconstruction_asmape_v1",
        "forecasters": ["imputation"],
        "active_mask_seeds": [1101, 1102, 1103],
    }
    for field, expected in expected_protocol.items():
        if manifest.get(field) != expected:
            raise ValueError(f"reconstruction label protocol differs at {field}")
    if _sha256(root / "teacher_labels.jsonl") != manifest.get("teacher_labels_sha256"):
        raise ValueError("teacher label signature differs")
    if _sha256(root / "pair_labels.jsonl") != manifest.get("pair_labels_sha256"):
        raise ValueError("pair label signature differs")
    entries = progress.get("entries")
    plans = progress.get("dataset_plans")
    if not isinstance(entries, dict) or not isinstance(plans, dict):
        raise ValueError("label progress has invalid plans or entries")
    store = LabelProgressStore(root, progress)
    families: set[str] = set()
    episode_total = 0
    for key in sorted(entries):
        entry = entries[key]
        expectation = LabelEpisodeExpectation.from_payload(entry["expectation"])
        validation = store.validate_episode(expectation)
        if validation.status != "valid":
            raise ValueError(f"invalid label sidecar {key}: {validation.reason}")
        families.add(expectation.family_id)
    for plan in plans.values():
        episode_total += len(plan["episode_ids"])
    if episode_total != expected_count or len(families) != 17:
        raise ValueError("reconstruction label plans must cover 2430 episodes and 17 families")
    rows = _read_jsonl(root / "teacher_labels.jsonl", "teacher labels")
    unique: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("forecaster_id") != "imputation" or row.get("label_scope") != "whole_series":
            raise ValueError("reconstruction row has an invalid label scope")
        key = str(row["dataset_id"]), str(row["episode_id"]), str(row["candidate_id"])
        if key in unique:
            raise ValueError("reconstruction labels contain a conflicting duplicate key")
        unique.add(key)
    summary = {
        "manifest_sha256": _sha256(root / "labels_manifest.json"),
        "progress_sha256": _sha256(root / "labels_progress.json"),
        "teacher_labels_sha256": _sha256(root / "teacher_labels.jsonl"),
        "pair_labels_sha256": _sha256(root / "pair_labels.jsonl"),
        "episode_count": expected_count,
        "dataset_count": len(plans),
        "family_count": len(families),
        "row_count": len(rows),
        "candidate_ids": list(manifest.get("selected_candidates", ())),
        "dataset_episode_ids": {
            dataset_id: list(plan["episode_ids"]) for dataset_id, plan in sorted(plans.items())
        },
    }
    return rows, summary


def _load_signed_selection(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "robust_selection.json"
    payload = _read_json(source, "robust selection")
    digest = payload.pop("canonical_sha256", None)
    if digest != _canonical_sha256(payload):
        raise ValueError("robust selection canonical signature differs")
    payload["canonical_sha256"] = digest
    return payload


def _select_robust(arguments: argparse.Namespace) -> dict[str, Any]:
    labels_root = Path(arguments.labels_artifact).resolve()
    rows, label_summary = _validate_label_artifact(labels_root)
    selection_a = select_robust_candidate(
        rows,
        label_summary["dataset_episode_ids"],
        label_summary["candidate_ids"],
        availability_floor=0.95,
    )
    selection_b = select_robust_candidate(
        rows,
        label_summary["dataset_episode_ids"],
        label_summary["candidate_ids"],
        availability_floor=0.95,
    )
    if asdict(selection_a) != asdict(selection_b):
        raise RuntimeError("robust selection did not reproduce exactly")
    output = _new_output_root(arguments.output_dir)
    payload: dict[str, Any] = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "protocol_id": SCRIPT_PROTOCOL_ID,
        "operation": "select-robust",
        "created_at": _utc_now(),
        "labels_artifact": str(labels_root),
        "label_source": label_summary,
        "selection": asdict(selection_a),
        "deterministic_second_computation": True,
        "resume_count": 0,
        "repair_count": 0,
        "repository_state": _repository_state(),
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    _atomic_write_json(output / "robust_selection.json", payload)
    validation = {
        "schema_version": 1,
        "status": "verified",
        "robust_selection_sha256": _sha256(output / "robust_selection.json"),
        "selected_candidate_id": selection_a.selected_candidate_id,
        "candidate_count": len(selection_a.candidates),
        "episode_count": label_summary["episode_count"],
        "sidecars_validated": label_summary["episode_count"],
    }
    _atomic_write_json(output / "validation.json", validation)
    return {"output": str(output), **validation}


def _selection_candidate_id(selection: Mapping[str, Any]) -> str | None:
    raw = selection.get("selection")
    if not isinstance(raw, Mapping):
        raise ValueError("robust selection payload has no selection object")
    candidate_id = raw.get("selected_candidate_id")
    return None if candidate_id is None else str(candidate_id)


def _candidate_params(source: ValidatedSource, candidate_id: str) -> dict[str, Any]:
    specs = source.manifest["candidate_generation_identity"]["candidate_specs"]
    raw = specs.get(candidate_id)
    if not isinstance(raw, Mapping):
        raise ValueError(f"source has no candidate specification for {candidate_id}")
    params = raw.get("execution_params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"source candidate parameters are invalid for {candidate_id}")
    return dict(params)


def _episode_arrays(episode: SourceEpisode) -> dict[str, np.ndarray]:
    with np.load(episode.npz_path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    numeric = left_array.dtype.kind in "fc" and right_array.dtype.kind in "fc"
    return bool(
        np.array_equal(left_array, right_array, equal_nan=True)
        if numeric
        else np.array_equal(left_array, right_array)
    )


def _source_candidate_index(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {
        str(candidate_id): index
        for index, candidate_id in enumerate(np.asarray(arrays["candidate_ids"]).tolist())
    }


def _candidate_native_on_missing(
    arrays: Mapping[str, np.ndarray], candidate_id: str | None
) -> bool:
    if candidate_id is None:
        return False
    by_id = _source_candidate_index(arrays)
    if candidate_id not in by_id:
        return False
    index = by_id[candidate_id]
    observed = np.asarray(arrays["observed_mask"], dtype=bool)
    missing = ~observed
    status = str(np.asarray(arrays["candidate_status"])[index])
    values = np.asarray(arrays["candidate_values"], dtype=float)[index]
    native = np.asarray(arrays["candidate_native_valid"], dtype=bool)[index]
    return bool(
        missing.any()
        and status not in {"failed", "unavailable"}
        and native[missing].all()
        and np.isfinite(values[missing]).all()
    )


def _direct_native_block(
    arrays: Mapping[str, np.ndarray], candidate_id: str | None, block: MissingBlock
) -> bool:
    if candidate_id is None:
        return False
    by_id = _source_candidate_index(arrays)
    if candidate_id not in by_id:
        return False
    index = by_id[candidate_id]
    status = str(np.asarray(arrays["candidate_status"])[index])
    native = np.asarray(arrays["candidate_native_valid"], dtype=bool)[index]
    values = np.asarray(arrays["candidate_values"], dtype=float)[index]
    selected_native = native[block.start : block.end, block.channel]
    selected_values = values[block.start : block.end, block.channel]
    return bool(
        status not in {"failed", "unavailable"}
        and selected_native.all()
        and np.isfinite(selected_values).all()
    )


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _source_support_summary(source: ValidatedSource) -> dict[str, Any]:
    support_names = (
        "resolved_config.json",
        "experiment_protocol.json",
        "repository_state.json",
        "seeds.json",
        "stage_manifest.json",
    )
    support = {name: _read_json(source.root / name, name) for name in support_names}
    stage = support["stage_manifest.json"]
    if stage.get("status") != "completed" or stage.get("stage") != "impute":
        raise ValueError("source stage manifest is not a completed imputation stage")
    outputs = stage.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("source stage manifest has no outputs")
    for field in (
        "episode_count",
        "forecaster_id",
        "candidate_generation_identity_sha256",
        "resume_identity_sha256",
    ):
        if outputs.get(field) != source.manifest.get(field):
            raise ValueError(f"source stage output differs from imputation manifest at {field}")
    resolved = support["resolved_config.json"].get("config")
    protocol_artifact = support["experiment_protocol.json"].get("protocol")
    if not isinstance(resolved, Mapping) or not isinstance(protocol_artifact, Mapping):
        raise ValueError("source resolved configuration or protocol artifact is invalid")
    resolved_protocol = resolved.get("protocol")
    if (
        resolved_protocol != protocol_artifact
        or stage.get("experiment_protocol") != protocol_artifact
    ):
        raise ValueError("source protocol copies differ")
    seeds = support["seeds.json"]
    if seeds.get("active_mask_partition") != protocol_artifact.get("active_mask_partition"):
        raise ValueError("source seed partition differs from the protocol")
    return {
        "run_id": stage.get("run_id"),
        "config_source": support["resolved_config.json"].get("source"),
        "protocol": protocol_artifact,
        "router_config": resolved.get("registries", {}).get("router_config"),
        "experiment_seeds": seeds.get("experiment_seeds"),
        "active_mask_partition": seeds.get("active_mask_partition"),
        "support_sha256": {name: _sha256(source.root / name) for name in support_names},
    }


def _validate_ett_score_source(source: ValidatedSource) -> dict[str, Any]:
    summary = _source_support_summary(source)
    forecaster_id = str(source.manifest.get("forecaster_id"))
    if forecaster_id not in {"chronos2", "timesfm2p5"}:
        raise ValueError("ETT score source has an unexpected forecaster")
    expected_run_id = f"r2-target-ett-full-candidate-{forecaster_id}-ms2101x3-rs4101-impute-v002"
    if summary["run_id"] != expected_run_id:
        raise ValueError("ETT score source run ID differs from the frozen input")
    protocol = summary["protocol"]
    expected_protocol_fields = {
        "active_mask_partition": "development",
        "family_split_policy": "ett_development_rolling_v1",
        "protocol_id": "iclr27-r2-target-full-candidate-v1",
        "target_protocol": "full_candidate_forecast_loss_v2",
        "teacher_forecaster_ids": ["chronos2", "timesfm2p5"],
    }
    for field, expected in expected_protocol_fields.items():
        if protocol.get(field) != expected:
            raise ValueError(f"ETT score source protocol differs at {field}")
    if summary["experiment_seeds"] != [2101, 2102, 2103]:
        raise ValueError("ETT score source mask seeds differ")
    if Path(str(summary["config_source"])).name != "full_candidate_ett_development.yaml":
        raise ValueError("ETT score source configuration is not full-candidate development")
    if Path(str(summary["router_config"])).name != "block_fais_full_candidate.yaml":
        raise ValueError("ETT score source router configuration differs")
    datasets = Counter(str(episode.entry["dataset_id"]) for episode in source.episodes)
    if datasets != Counter({"ETTh1": 90, "ETTh2": 90, "ETTm1": 90, "ETTm2": 90}):
        raise ValueError("ETT score source dataset coverage differs")
    if {str(episode.entry["family_id"]) for episode in source.episodes} != {"ett"}:
        raise ValueError("ETT score source contains a non-ETT family")
    if {int(episode.assignment["seed"]) for episode in source.episodes} != {2101, 2102, 2103}:
        raise ValueError("ETT score assignment seeds differ")
    expected_blend = {"global_prior": 0.0, "proxy": 0.0, "r0": 0.0, "r1": 1.0}
    for episode in source.episodes:
        blend = episode.assignment.get("routing_metadata", {}).get("evidence_blend")
        if blend == expected_blend:
            continue
        if blend is None and not episode.assignment.get("assignments"):
            continue
        raise ValueError("ETT score source is not the learned-only evidence setting")
    stage = _read_json(source.root / "stage_manifest.json", "source stage manifest")
    router_root = Path(str(stage.get("inputs", {}).get("router_artifact", "")))
    if router_root.parent.name != "r2-target-rolling-full-candidate-ms1101x3-rs4101-router-v002":
        raise ValueError("ETT score source does not use the frozen full-candidate router")
    for name in ("manifest.json", "router_bundle.joblib"):
        if not (router_root / name).is_file():
            raise FileNotFoundError(f"frozen router file is missing: {router_root / name}")
    summary["router_artifact"] = str(router_root)
    summary["router_artifact_sha256"] = {
        name: _sha256(router_root / name) for name in ("manifest.json", "router_bundle.joblib")
    }
    return summary


def _score(arguments: argparse.Namespace) -> dict[str, Any]:
    source = _validate_imputation_source(arguments.source_imputation, arguments.lineage_mode)
    source_protocol = (
        _validate_ett_score_source(source)
        if arguments.lineage_mode == "historical-ett-v002"
        else _source_support_summary(source)
    )
    selection = _load_signed_selection(arguments.robust_selection)
    robust_id = _selection_candidate_id(selection)
    artifact_root = Path(arguments.imputer_artifacts).resolve()
    output = _new_output_root(arguments.output_dir)
    (output / "imputations").mkdir()
    (output / "assignment_records").mkdir()
    started = time.perf_counter()
    by_dataset: dict[str, list[SourceEpisode]] = defaultdict(list)
    for episode in source.episodes:
        by_dataset[str(episode.entry["dataset_id"])].append(episode)
    score_records_by_index: dict[int, dict[str, Any]] = {}
    derivation_candidates: dict[str, dict[str, Any]] = {}
    runner = CandidateRunner(DEFAULT_REGISTRY)

    for dataset_id in sorted(by_dataset):
        episode_cache: dict[int, dict[str, Any]] = {}
        candidate_episode_indices: dict[str, list[int]] = defaultdict(list)
        for episode in by_dataset[dataset_id]:
            arrays = _episode_arrays(episode)
            observed = np.asarray(arrays["observed_mask"], dtype=bool)[None, ...]
            clean = np.asarray(arrays["clean_context"], dtype=float)[None, ...]
            blocks = detect_missing_blocks(observed)
            protocol = RiskScoreProtocol(
                _forecast_mode(episode.assignment["forecast_mode"]),
                tuple(int(value) for value in arguments.target_indices),
            )
            pseudo = deterministic_pseudo_observed_mask(
                observed,
                int(episode.assignment["episode_seed"]),
                max_blocks=protocol.max_pseudo_blocks,
                target_blocks=blocks,
                priority_channels=protocol.target_indices,
            )
            shortlist = tuple(str(value) for value in episode.assignment.get("shortlist", ()))
            for candidate_id in shortlist:
                candidate_episode_indices[candidate_id].append(episode.index)
            episode_cache[episode.index] = {
                "episode": episode,
                "arrays": arrays,
                "observed": observed,
                "clean": clean,
                "blocks": blocks,
                "protocol": protocol,
                "pseudo": pseudo,
                "shortlist": shortlist,
                "proxy": defaultdict(dict),
            }

        for candidate_id in sorted(candidate_episode_indices):
            params = _candidate_params(source, candidate_id)
            artifacts, _, _, failures = load_dataset_imputer_artifacts(
                artifact_root,
                dataset_id,
                DEFAULT_REGISTRY,
                candidate_ids=(candidate_id,),
                adapter_params={candidate_id: params},
            )
            candidate_started = time.perf_counter()
            runtime_sum = 0.0
            peak_memory = 0
            status_counts: dict[str, int] = defaultdict(int)
            for index in candidate_episode_indices[candidate_id]:
                cached = episode_cache[index]
                channels = sorted(
                    {
                        block.channel
                        for block in cached["blocks"]
                        if block.channel in cached["protocol"].target_indices
                        or cached["protocol"].forecast_mode == "joint_multivariate"
                    }
                )
                if candidate_id in failures:
                    for channel in channels:
                        cached["proxy"][candidate_id][channel] = None
                    status_counts["artifact_load_failed"] += 1
                    continue
                batch = SeriesBatch(
                    cached["clean"],
                    cached["pseudo"],
                    item_ids=(str(cached["episode"].entry["item_id"]),),
                    metadata={"period": int(np.asarray(cached["arrays"]["period"]).reshape(-1)[0])},
                )
                result = runner.run(
                    candidate_id,
                    batch,
                    artifacts.get(candidate_id),
                    seed=int(cached["episode"].assignment["episode_seed"]),
                    params=params,
                )
                runtime_sum += result.runtime_seconds
                peak_memory = max(peak_memory, result.peak_memory_bytes)
                status_counts[result.status.value] += 1
                for channel in channels:
                    cached["proxy"][candidate_id][channel] = channel_proxy_mae(
                        cached["clean"],
                        result.values,
                        result.native_valid_mask,
                        cached["observed"],
                        cached["pseudo"],
                        channel=channel,
                        candidate_status=result.status.value,
                    )
            derivation_candidates[f"{dataset_id}/{candidate_id}"] = {
                "episode_count": len(candidate_episode_indices[candidate_id]),
                "artifact_load_failure": failures.get(candidate_id),
                "candidate_runtime_seconds": runtime_sum,
                "wall_seconds": time.perf_counter() - candidate_started,
                "peak_memory_bytes": peak_memory,
                "status_counts": dict(sorted(status_counts.items())),
            }
            artifacts.clear()
            _clear_accelerator_cache()

        for index in sorted(episode_cache):
            cached = episode_cache[index]
            current_episode: SourceEpisode = cached["episode"]
            arrays = cached["arrays"]
            assignments = current_episode.assignment["assignments"]
            fallback_records = current_episode.assignment.get("fallback_records", {})
            fallback_blocks = set(
                str(value) for value in current_episode.assignment.get("fallback_blocks", ())
            )
            scores = []
            for block in cached["blocks"]:
                selected_id = str(assignments[block.block_id])
                proxy_by_candidate = {
                    candidate_id: cached["proxy"].get(candidate_id, {}).get(block.channel)
                    for candidate_id in cached["shortlist"]
                }
                scores.append(
                    score_block(
                        block,
                        cached["protocol"],
                        selected_candidate_id=selected_id,
                        shortlist=cached["shortlist"],
                        proxy_mae_by_candidate=proxy_by_candidate,
                        direct_native_valid=_direct_native_block(arrays, selected_id, block),
                        safety_result_used=(
                            block.block_id in fallback_records or block.block_id in fallback_blocks
                        ),
                    )
                )
            episode_score = score_episode(scores)
            robust_ready = _candidate_native_on_missing(arrays, robust_id)
            score_record = {
                "schema_version": SCRIPT_SCHEMA_VERSION,
                "protocol_id": SCRIPT_PROTOCOL_ID,
                "artifact_index": current_episode.index,
                "episode_id": current_episode.entry["episode_id"],
                "dataset_id": current_episode.entry["dataset_id"],
                "family_id": current_episode.entry["family_id"],
                "item_id": current_episode.entry["item_id"],
                "forecast_origin": current_episode.assignment["forecast_origin"],
                "forecaster_id": current_episode.entry["forecaster_id"],
                "forecast_mode": current_episode.assignment["forecast_mode"],
                "mechanism": current_episode.assignment["mechanism"],
                "seed": current_episode.assignment["seed"],
                "target_indices": list(arguments.target_indices),
                "mask_seed": current_episode.assignment["mask_seed"],
                "mask_realization_id": current_episode.assignment["mask_realization_id"],
                "episode_seed": current_episode.assignment["episode_seed"],
                "target_missing_rate": current_episode.assignment["target_missing_rate"],
                "global_missing_rate": current_episode.assignment["global_missing_rate"],
                "local_missing_rate": current_episode.assignment["local_missing_rate"],
                "mask_protocol": current_episode.assignment["mask_protocol"],
                "pseudo_observed_mask_sha256": hashlib.sha256(
                    np.asarray(cached["pseudo"], dtype=bool).tobytes(order="C")
                ).hexdigest(),
                "newly_hidden_count": int(np.sum(cached["observed"] & ~cached["pseudo"])),
                "shortlist": list(cached["shortlist"]),
                "proxy_mae_by_candidate_and_channel": {
                    candidate_id: {
                        str(channel): value for channel, value in sorted(channels.items())
                    }
                    for candidate_id, channels in sorted(cached["proxy"].items())
                },
                "block_scores": [asdict(value) for value in scores],
                "episode_score": episode_score.score,
                "episode_score_status": episode_score.status,
                "scored_block_ids": list(episode_score.scored_block_ids),
                "unavailable_block_ids": list(episode_score.unavailable_block_ids),
                "robust_candidate_id": robust_id,
                "robust_candidate_native_finite": robust_ready,
                "configured_threshold": {"kind": "disabled", "value": None},
                "action_status": "kept",
                "action_reason": "disabled",
                "source_npz_sha256": current_episode.entry["npz_sha256"],
                "source_assignment_sha256": current_episode.entry["assignment_sha256"],
            }
            score_records_by_index[index] = score_record

    score_rows = [score_records_by_index[index] for index in range(len(source.episodes))]
    score_path = output / "risk_scores.jsonl"
    _atomic_write_jsonl(score_path, score_rows)
    entries: dict[str, Any] = {}
    assignment_rows: list[dict[str, Any]] = []
    for episode in source.episodes:
        record = score_records_by_index[episode.index]
        npz_relative = _safe_relative(episode.entry["file"], "imputation")
        assignment_relative = _safe_relative(episode.entry["assignment_file"], "assignment")
        target_npz = output / "imputations" / npz_relative
        _atomic_copy(episode.npz_path, target_npz)
        assignment = copy.deepcopy(episode.assignment)
        assignment["risk_fallback"] = record
        target_assignment = output / assignment_relative
        _atomic_write_json(target_assignment, assignment)
        entry = copy.deepcopy(episode.entry)
        entry["npz_sha256"] = _sha256(target_npz)
        entry["assignment_sha256"] = _sha256(target_assignment)
        entry["completed_at"] = _utc_now()
        entries[episode.key] = entry
        assignment_rows.append(assignment)
    assignments_path = output / "routing_assignments.jsonl"
    _atomic_write_jsonl(assignments_path, assignment_rows)
    score_manifest = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "protocol_id": SCRIPT_PROTOCOL_ID,
        "operation": "score",
        "created_at": _utc_now(),
        "source_imputation": str(source.root),
        "source_signatures": source.signatures,
        "source_protocol": source_protocol,
        "source_lineage_mode": source.lineage_mode,
        "imputer_artifacts": str(artifact_root),
        "robust_selection": {
            "path": str(Path(arguments.robust_selection).resolve()),
            "canonical_sha256": selection["canonical_sha256"],
            "candidate_id": robust_id,
        },
        "episode_count": len(score_rows),
        "risk_scores": str(score_path),
        "risk_scores_sha256": _sha256(score_path),
        "score_only_npz_byte_identity": True,
        "forecast_call_count": 0,
        "derivation_runtime_seconds": time.perf_counter() - started,
        "derivation_candidates": derivation_candidates,
        "resume_count": 0,
        "repair_count": 0,
        "repository_state": _repository_state(),
    }
    score_manifest["canonical_sha256"] = _canonical_sha256(score_manifest)
    _atomic_write_json(output / "risk_score_manifest.json", score_manifest)
    imputation_manifest = copy.deepcopy(source.manifest)
    imputation_manifest.update(
        {
            "imputations": str(output / "imputations"),
            "routing_assignments": str(assignments_path),
            "progress_manifest": str(output / "imputation_progress.json"),
            "risk_fallback": {
                "operation": "score",
                "manifest": str(output / "risk_score_manifest.json"),
                "manifest_sha256": _sha256(output / "risk_score_manifest.json"),
                "forecast_call_count": 0,
            },
        }
    )
    _atomic_write_json(output / "imputation_manifest.json", imputation_manifest)
    progress = copy.deepcopy(source.progress)
    progress.update(
        {
            "status": "completed",
            "entries": entries,
            "completed_count": len(entries),
            "expected_episode_count": len(entries),
            "resume_count": 0,
            "repair_count": 0,
            "routing_assignments_sha256": _sha256(assignments_path),
            "imputation_manifest_sha256": _sha256(output / "imputation_manifest.json"),
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
        }
    )
    _atomic_write_json(output / "imputation_progress.json", progress)
    validation = _validate_scored_artifact(output)
    _atomic_write_json(output / "validation.json", validation)
    return {"output": str(output), **validation}


def _recompute_score_record(record: Mapping[str, Any], assignment: Mapping[str, Any]) -> None:
    protocol = RiskScoreProtocol(
        _forecast_mode(record["forecast_mode"]),
        tuple(int(value) for value in record["target_indices"]),
    )
    scores = []
    for raw in record["block_scores"]:
        block_id = str(raw["block_id"])
        parts = block_id.split(":")
        batch_index = int(parts[0][1:])
        channel = int(parts[1][1:])
        start, end = (int(value) for value in parts[2].split("-"))
        block = MissingBlock(block_id, batch_index, channel, start, end)
        proxy = {
            candidate_id: channels.get(str(channel))
            for candidate_id, channels in record["proxy_mae_by_candidate_and_channel"].items()
        }
        rebuilt = score_block(
            block,
            protocol,
            selected_candidate_id=raw.get("selected_candidate_id"),
            shortlist=record["shortlist"],
            proxy_mae_by_candidate=proxy,
            direct_native_valid=raw["status"] == "direct",
            safety_result_used=raw["status"] == "safety",
        )
        if _canonical_bytes(asdict(rebuilt)) != _canonical_bytes(raw):
            raise ValueError(f"risk block score does not reproduce: {block_id}")
        scores.append(rebuilt)
    episode = score_episode(scores)
    if episode.score != record.get("episode_score") or episode.status != record.get(
        "episode_score_status"
    ):
        raise ValueError("episode risk score does not reproduce")
    if assignment.get("risk_fallback") != record:
        raise ValueError("assignment risk record differs from score JSONL")


def _validate_score_record_identity(
    record: Mapping[str, Any],
    scored: SourceEpisode,
    original: SourceEpisode,
    *,
    robust_candidate_id: str,
) -> None:
    expected = {
        "artifact_index": original.index,
        "episode_id": original.entry["episode_id"],
        "dataset_id": original.entry["dataset_id"],
        "family_id": original.entry["family_id"],
        "item_id": original.entry["item_id"],
        "forecaster_id": original.entry["forecaster_id"],
        "forecast_origin": original.assignment["forecast_origin"],
        "forecast_mode": original.assignment["forecast_mode"],
        "mechanism": original.assignment["mechanism"],
        "seed": original.assignment["seed"],
        "mask_seed": original.assignment["mask_seed"],
        "mask_realization_id": original.assignment["mask_realization_id"],
        "episode_seed": original.assignment["episode_seed"],
        "target_missing_rate": original.assignment["target_missing_rate"],
        "global_missing_rate": original.assignment["global_missing_rate"],
        "local_missing_rate": original.assignment["local_missing_rate"],
        "mask_protocol": original.assignment["mask_protocol"],
        "source_npz_sha256": original.entry["npz_sha256"],
        "source_assignment_sha256": original.entry["assignment_sha256"],
        "robust_candidate_id": robust_candidate_id,
        "configured_threshold": {"kind": "disabled", "value": None},
        "action_status": "kept",
        "action_reason": "disabled",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"risk score identity differs at {field}")
    if scored.index != original.index or scored.key != original.key:
        raise ValueError("score artifact episode order differs from its source")


def _validate_scored_artifact(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "risk_score_manifest.json", "risk score manifest")
    digest = manifest.pop("canonical_sha256", None)
    if digest != _canonical_sha256(manifest):
        raise ValueError("risk score manifest canonical signature differs")
    manifest["canonical_sha256"] = digest
    source = _validate_imputation_source(root, str(manifest["source_lineage_mode"]))
    rows = _read_jsonl(root / "risk_scores.jsonl", "risk scores")
    if _sha256(root / "risk_scores.jsonl") != manifest.get("risk_scores_sha256"):
        raise ValueError("risk score JSONL signature differs")
    if len(rows) != len(source.episodes):
        raise ValueError("risk score row count differs")
    source_root = Path(str(manifest["source_imputation"]))
    source_validated = _validate_imputation_source(
        source_root, str(manifest["source_lineage_mode"])
    )
    expected_protocol = (
        _validate_ett_score_source(source_validated)
        if manifest["source_lineage_mode"] == "historical-ett-v002"
        else _source_support_summary(source_validated)
    )
    if manifest.get("source_signatures") != source_validated.signatures:
        raise ValueError("risk score source signatures differ")
    if manifest.get("source_protocol") != expected_protocol:
        raise ValueError("risk score source protocol signatures differ")
    robust_selection = _load_signed_selection(manifest["robust_selection"]["path"])
    if robust_selection.get("canonical_sha256") != manifest["robust_selection"].get(
        "canonical_sha256"
    ):
        raise ValueError("risk score robust selection signature differs")
    robust_candidate_id = _selection_candidate_id(robust_selection)
    if robust_candidate_id is None or robust_candidate_id != manifest["robust_selection"].get(
        "candidate_id"
    ):
        raise ValueError("risk score robust candidate differs from its frozen selection")
    for episode, original, record in zip(
        source.episodes, source_validated.episodes, rows, strict=True
    ):
        if _sha256(episode.npz_path) != _sha256(original.npz_path):
            raise ValueError("score-only NPZ differs from source")
        _validate_score_record_identity(
            record,
            episode,
            original,
            robust_candidate_id=robust_candidate_id,
        )
        _recompute_score_record(record, episode.assignment)
    return {
        "schema_version": 1,
        "status": "verified",
        "operation": "score",
        "episode_count": len(rows),
        "npz_byte_identity_count": len(rows),
        "score_recomputation_count": len(rows),
        "forecast_call_count": 0,
        "risk_score_manifest_sha256": _sha256(root / "risk_score_manifest.json"),
        "imputation_manifest_sha256": _sha256(root / "imputation_manifest.json"),
        "imputation_progress_sha256": _sha256(root / "imputation_progress.json"),
    }


def _load_threshold_selection(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "threshold_selection.json"
    payload = _read_json(source, "threshold selection")
    digest = payload.pop("canonical_sha256", None)
    if digest != _canonical_sha256(payload):
        raise ValueError("threshold selection canonical signature differs")
    payload["canonical_sha256"] = digest
    return payload


def _threshold_from_payload(payload: Mapping[str, Any]) -> RiskThreshold:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("threshold payload has no selection")
    raw = selection.get("selected_threshold")
    if not isinstance(raw, Mapping):
        raise ValueError("threshold payload has no selected threshold")
    return RiskThreshold(str(raw["kind"]), raw.get("value"))  # type: ignore[arg-type]


def _derive(arguments: argparse.Namespace) -> dict[str, Any]:
    score_root = Path(arguments.score_artifact).resolve()
    score_manifest = _read_json(score_root / "risk_score_manifest.json", "risk score manifest")
    _validate_scored_artifact(score_root)
    scored = _validate_imputation_source(score_root, str(score_manifest["source_lineage_mode"]))
    score_rows = _read_jsonl(score_root / "risk_scores.jsonl", "risk scores")
    threshold_payload = _load_threshold_selection(arguments.threshold_selection)
    threshold = _threshold_from_payload(threshold_payload)
    robust_id = str(threshold_payload["robust_candidate_id"])
    output = _new_output_root(arguments.output_dir)
    (output / "imputations").mkdir()
    (output / "assignment_records").mkdir()
    entries: dict[str, Any] = {}
    assignment_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    switched_count = 0
    pipeline_runtime_total = 0.0
    for episode, score_record in zip(scored.episodes, score_rows, strict=True):
        arrays = _episode_arrays(episode)
        by_id = _source_candidate_index(arrays)
        robust_values = (
            None
            if robust_id not in by_id
            else np.asarray(arrays["candidate_values"], dtype=float)[by_id[robust_id]]
        )
        robust_native = (
            None
            if robust_id not in by_id
            else np.asarray(arrays["candidate_native_valid"], dtype=bool)[by_id[robust_id]]
        )
        action = apply_whole_episode_fallback(
            np.asarray(arrays["values"], dtype=float),
            np.asarray(arrays["observed_mask"], dtype=bool),
            robust_values,
            robust_native,
            score=score_record.get("episode_score"),
            threshold=threshold,
        )
        valid_ids = tuple(str(value) for value in np.asarray(arrays["candidate_ids"]).tolist())
        source_actual = source_routing_actual_ids(
            episode.assignment.get("shortlist", ()),
            episode.assignment.get("fallback_records", {}),
            valid_ids,
        )
        robust_runtime = (
            None
            if robust_id not in by_id
            else float(np.asarray(arrays["candidate_runtime_seconds"])[by_id[robust_id]])
        )
        source_runtime = float(np.asarray(arrays["pipeline_runtime_seconds"]).reshape(-1)[0])
        runtime = derived_pipeline_runtime(
            source_runtime,
            switched=action.switched,
            robust_candidate_id=robust_id,
            robust_candidate_runtime_seconds=robust_runtime,
            source_actual_candidate_ids=source_actual,
        )
        pipeline_runtime_total += runtime
        npz_relative = _safe_relative(episode.entry["file"], "imputation")
        target_npz = output / "imputations" / npz_relative
        if action.switched:
            arrays["values"] = action.values
            arrays["pipeline_runtime_seconds"] = np.asarray(runtime, dtype=np.float64)
            _atomic_write_npz(target_npz, arrays)
            switched_count += 1
        else:
            _atomic_copy(episode.npz_path, target_npz)
        action_record = copy.deepcopy(score_record)
        action_record.update(
            {
                "configured_threshold": asdict(threshold),
                "action_status": "switched" if action.switched else "kept",
                "action_reason": action.reason,
                "robust_candidate_id": robust_id,
                "robust_candidate_native_finite": action.robust_native_valid,
                "source_routing_actual_ids": list(source_actual),
                "source_pipeline_runtime_seconds": source_runtime,
                "derived_pipeline_runtime_seconds": runtime,
            }
        )
        assignment = copy.deepcopy(episode.assignment)
        assignment["pipeline_runtime_seconds"] = runtime
        assignment["risk_fallback"] = action_record
        assignment_relative = _safe_relative(episode.entry["assignment_file"], "assignment")
        target_assignment = output / assignment_relative
        _atomic_write_json(target_assignment, assignment)
        entry = copy.deepcopy(episode.entry)
        entry["npz_sha256"] = _sha256(target_npz)
        entry["assignment_sha256"] = _sha256(target_assignment)
        entry["completed_at"] = _utc_now()
        entries[episode.key] = entry
        assignment_rows.append(assignment)
        action_rows.append(action_record)
    assignments_path = output / "routing_assignments.jsonl"
    actions_path = output / "risk_actions.jsonl"
    _atomic_write_jsonl(assignments_path, assignment_rows)
    _atomic_write_jsonl(actions_path, action_rows)
    derive_manifest = {
        "schema_version": 1,
        "protocol_id": SCRIPT_PROTOCOL_ID,
        "operation": "derive",
        "created_at": _utc_now(),
        "score_artifact": str(score_root),
        "score_manifest_sha256": _sha256(score_root / "risk_score_manifest.json"),
        "source_lineage_mode": score_manifest["source_lineage_mode"],
        "threshold_selection": str(Path(arguments.threshold_selection).resolve()),
        "threshold_selection_canonical_sha256": threshold_payload["canonical_sha256"],
        "threshold": asdict(threshold),
        "robust_candidate_id": robust_id,
        "episode_count": len(action_rows),
        "switched_count": switched_count,
        "kept_count": len(action_rows) - switched_count,
        "risk_actions_sha256": _sha256(actions_path),
        "forecast_call_count": 0,
        "resume_count": 0,
        "repair_count": 0,
        "repository_state": _repository_state(),
    }
    derive_manifest["canonical_sha256"] = _canonical_sha256(derive_manifest)
    _atomic_write_json(output / "risk_derive_manifest.json", derive_manifest)
    imputation_manifest = copy.deepcopy(scored.manifest)
    imputation_manifest.update(
        {
            "imputations": str(output / "imputations"),
            "routing_assignments": str(assignments_path),
            "progress_manifest": str(output / "imputation_progress.json"),
            "pipeline_runtime_total_seconds": pipeline_runtime_total,
            "risk_fallback": {
                "operation": "derive",
                "manifest": str(output / "risk_derive_manifest.json"),
                "manifest_sha256": _sha256(output / "risk_derive_manifest.json"),
                "threshold": asdict(threshold),
                "robust_candidate_id": robust_id,
                "switched_count": switched_count,
                "forecast_call_count": 0,
            },
        }
    )
    _atomic_write_json(output / "imputation_manifest.json", imputation_manifest)
    progress = copy.deepcopy(scored.progress)
    progress.update(
        {
            "status": "completed",
            "entries": entries,
            "completed_count": len(entries),
            "expected_episode_count": len(entries),
            "resume_count": 0,
            "repair_count": 0,
            "pipeline_runtime_total_seconds": pipeline_runtime_total,
            "routing_assignments_sha256": _sha256(assignments_path),
            "imputation_manifest_sha256": _sha256(output / "imputation_manifest.json"),
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
        }
    )
    _atomic_write_json(output / "imputation_progress.json", progress)
    validation = _validate_derived_artifact(output)
    _atomic_write_json(output / "validation.json", validation)
    return {"output": str(output), **validation}


def _validate_derived_artifact(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "risk_derive_manifest.json", "risk derive manifest")
    digest = manifest.pop("canonical_sha256", None)
    if digest != _canonical_sha256(manifest):
        raise ValueError("risk derive manifest canonical signature differs")
    manifest["canonical_sha256"] = digest
    derived = _validate_imputation_source(root, str(manifest["source_lineage_mode"]))
    score_root = Path(str(manifest["score_artifact"]))
    _validate_scored_artifact(score_root)
    if _sha256(score_root / "risk_score_manifest.json") != manifest.get("score_manifest_sha256"):
        raise ValueError("derived score manifest signature differs")
    score_manifest = _read_json(score_root / "risk_score_manifest.json", "risk score manifest")
    score_source = _validate_imputation_source(
        score_root, str(score_manifest["source_lineage_mode"])
    )
    score_rows = _read_jsonl(score_root / "risk_scores.jsonl", "risk scores")
    action_rows = _read_jsonl(root / "risk_actions.jsonl", "risk actions")
    if _sha256(root / "risk_actions.jsonl") != manifest.get("risk_actions_sha256"):
        raise ValueError("derived risk action signature differs")
    if len(action_rows) != int(manifest.get("episode_count", -1)):
        raise ValueError("derived risk action count differs")
    threshold_payload = _load_threshold_selection(manifest["threshold_selection"])
    if threshold_payload["canonical_sha256"] != manifest["threshold_selection_canonical_sha256"]:
        raise ValueError("derived threshold selection signature differs")
    threshold = _threshold_from_payload(threshold_payload)
    robust_id = str(manifest["robust_candidate_id"])
    if robust_id != str(threshold_payload["robust_candidate_id"]):
        raise ValueError("derived robust candidate differs from threshold selection")
    switched = 0
    for result, source, score_record, action in zip(
        derived.episodes, score_source.episodes, score_rows, action_rows, strict=True
    ):
        result_arrays = _episode_arrays(result)
        source_arrays = _episode_arrays(source)
        by_id = _source_candidate_index(source_arrays)
        robust_values = (
            None
            if robust_id not in by_id
            else np.asarray(source_arrays["candidate_values"], dtype=float)[by_id[robust_id]]
        )
        robust_native = (
            None
            if robust_id not in by_id
            else np.asarray(source_arrays["candidate_native_valid"], dtype=bool)[by_id[robust_id]]
        )
        expected_action = apply_whole_episode_fallback(
            np.asarray(source_arrays["values"], dtype=float),
            np.asarray(source_arrays["observed_mask"], dtype=bool),
            robust_values,
            robust_native,
            score=action.get("episode_score"),
            threshold=threshold,
        )
        if (
            action.get("action_status") != ("switched" if expected_action.switched else "kept")
            or action.get("action_reason") != expected_action.reason
        ):
            raise ValueError("derived action does not reproduce")
        valid_ids = tuple(
            str(value) for value in np.asarray(source_arrays["candidate_ids"]).tolist()
        )
        source_actual = source_routing_actual_ids(
            source.assignment.get("shortlist", ()),
            source.assignment.get("fallback_records", {}),
            valid_ids,
        )
        robust_runtime = (
            None
            if robust_id not in by_id
            else float(np.asarray(source_arrays["candidate_runtime_seconds"])[by_id[robust_id]])
        )
        source_runtime = float(np.asarray(source_arrays["pipeline_runtime_seconds"]).reshape(-1)[0])
        expected_runtime = derived_pipeline_runtime(
            source_runtime,
            switched=expected_action.switched,
            robust_candidate_id=robust_id,
            robust_candidate_runtime_seconds=robust_runtime,
            source_actual_candidate_ids=source_actual,
        )
        expected_record = copy.deepcopy(score_record)
        expected_record.update(
            {
                "configured_threshold": asdict(threshold),
                "action_status": "switched" if expected_action.switched else "kept",
                "action_reason": expected_action.reason,
                "robust_candidate_id": robust_id,
                "robust_candidate_native_finite": expected_action.robust_native_valid,
                "source_routing_actual_ids": list(source_actual),
                "source_pipeline_runtime_seconds": source_runtime,
                "derived_pipeline_runtime_seconds": expected_runtime,
            }
        )
        if _canonical_bytes(action) != _canonical_bytes(expected_record):
            raise ValueError("derived action record does not reproduce")
        if result.assignment.get("risk_fallback") != action:
            raise ValueError("derived assignment risk record differs from action JSONL")
        if float(result.assignment.get("pipeline_runtime_seconds", -1.0)) != expected_runtime:
            raise ValueError("derived assignment runtime does not reproduce")
        changed_fields = {
            name
            for name in result_arrays
            if not _arrays_equal(result_arrays[name], source_arrays[name])
        }
        if action["action_status"] == "switched":
            switched += 1
            if changed_fields != {"values", "pipeline_runtime_seconds"}:
                raise ValueError("switched NPZ changed an unauthorized field")
            robust_id = str(action["robust_candidate_id"])
            robust_index = _source_candidate_index(source_arrays)[robust_id]
            observed = np.asarray(source_arrays["observed_mask"], dtype=bool)
            missing = ~observed
            robust = np.asarray(source_arrays["candidate_values"])[robust_index]
            if not np.array_equal(np.asarray(result_arrays["values"])[missing], robust[missing]):
                raise ValueError("switched NPZ differs from the robust candidate")
            if not np.array_equal(
                np.asarray(result_arrays["values"])[observed],
                np.asarray(source_arrays["values"])[observed],
            ):
                raise ValueError("switched NPZ changed an observed value")
        else:
            if _sha256(result.npz_path) != _sha256(source.npz_path):
                raise ValueError("kept NPZ is not byte-identical to the score source")
    if switched != int(manifest["switched_count"]):
        raise ValueError("derived switch count differs")
    return {
        "schema_version": 1,
        "status": "verified",
        "operation": "derive",
        "episode_count": len(action_rows),
        "switched_count": switched,
        "kept_count": len(action_rows) - switched,
        "forecast_call_count": 0,
        "risk_derive_manifest_sha256": _sha256(root / "risk_derive_manifest.json"),
        "imputation_manifest_sha256": _sha256(root / "imputation_manifest.json"),
        "imputation_progress_sha256": _sha256(root / "imputation_progress.json"),
    }


def _validated_evaluation(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "evaluation_manifest.json"
    metrics_path = root / "episode_metrics.jsonl"
    manifest = _read_json(manifest_path, "evaluation manifest")
    if manifest.get("status") != "completed":
        raise ValueError("evaluation artifact is not completed")
    forecaster_id = str(manifest.get("forecaster_id"))
    frozen = FROZEN_ETT_EVALUATIONS.get(forecaster_id)
    if frozen is None or root.name != frozen["run_name"]:
        raise ValueError("evaluation artifact is not a frozen ETT threshold input")
    manifest_sha256 = _sha256(manifest_path)
    metrics_sha256 = _sha256(metrics_path)
    if manifest_sha256 != frozen["manifest_sha256"] or metrics_sha256 != frozen["metrics_sha256"]:
        raise ValueError("evaluation artifact differs from its preregistered ETT input")
    if manifest.get("episode_metrics_jsonl_sha256") != metrics_sha256:
        raise ValueError("evaluation metric signature differs")
    rows = _read_jsonl(metrics_path, "evaluation metrics")
    if len(rows) != int(manifest.get("total_rows", -1)):
        raise ValueError("evaluation metric count differs")
    signature = manifest.get("evaluation_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("evaluation manifest has no evaluation signature")
    expected_signature = {
        "forecaster_id": forecaster_id,
        "forecast_call_protocol": "batched_common_contexts_v1",
        "forecast_num_samples": 20,
        "context_length": 96,
        "horizon": 96,
        "mask_protocol": "sequence_mask_v2",
        "target_indices": [0, 1],
    }
    for field, expected in expected_signature.items():
        if signature.get(field) != expected:
            raise ValueError(f"evaluation signature differs at {field}")
    if Path(str(signature.get("forecaster_artifact"))).name != frozen["revision"]:
        raise ValueError("evaluation forecaster revision differs")
    if manifest.get("forecaster_artifact") != signature.get("forecaster_artifact"):
        raise ValueError("evaluation forecaster artifact copies differ")
    if manifest.get("forecast_call_protocol") != signature.get("forecast_call_protocol"):
        raise ValueError("evaluation forecast call protocol copies differ")
    if manifest.get("forecast_num_samples") != signature.get("forecast_num_samples"):
        raise ValueError("evaluation sample-count copies differ")
    if (
        int(manifest.get("episodes_seen", -1)) != 360
        or int(manifest.get("episodes_evaluated", -1)) != 360
        or len(rows) != 7920
        or bool(manifest.get("resume"))
    ):
        raise ValueError("evaluation ETT counts or resume state differ")
    identity_fields = (
        "dataset_id",
        "family_id",
        "item_id",
        "forecast_origin",
        "mechanism",
        "seed",
        "mask_seed",
        "mask_realization_id",
        "target_missing_rate",
        "global_missing_rate",
        "local_missing_rate",
        "mask_protocol",
        "forecast_seed",
        "forecaster_id",
        "routing_forecaster_id",
        "routing_artifact_forecaster_id",
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_keys: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("forecaster_id")) != forecaster_id:
            raise ValueError("evaluation row forecaster differs from its manifest")
        key = str(row.get("episode_id")), str(row.get("method"))
        if key in unique_keys:
            raise ValueError("evaluation contains a duplicate episode-method key")
        unique_keys.add(key)
        groups[key[0]].append(row)
    if len(groups) != 360 or any(len(group) != 22 for group in groups.values()):
        raise ValueError("evaluation episode or method coverage differs")
    representatives = []
    for group in groups.values():
        representative = group[0]
        representatives.append(representative)
        for row in group[1:]:
            for field in identity_fields:
                if row.get(field) != representative.get(field):
                    raise ValueError(f"evaluation method identity differs at {field}")
    if Counter(str(row["dataset_id"]) for row in representatives) != Counter(
        {"ETTh1": 90, "ETTh2": 90, "ETTm1": 90, "ETTm2": 90}
    ):
        raise ValueError("evaluation dataset coverage differs")
    return rows, {
        "path": str(root),
        "manifest_sha256": manifest_sha256,
        "metrics_sha256": metrics_sha256,
        "forecaster_id": forecaster_id,
        "forecaster_revision": frozen["revision"],
        "impute_artifact": signature.get("impute_artifact"),
        "impute_content": signature.get("impute_content"),
        "row_count": len(rows),
    }


def _select_threshold(arguments: argparse.Namespace) -> dict[str, Any]:
    selection = _load_signed_selection(arguments.robust_selection)
    robust_id = _selection_candidate_id(selection)
    if robust_id is None:
        raise ValueError("threshold selection requires a frozen robust candidate")
    if len(arguments.score_artifact) != 2 or len(arguments.evaluation_artifact) != 2:
        raise ValueError("threshold selection requires exactly two score and evaluation artifacts")
    score_sources: list[dict[str, Any]] = []
    scores_by_forecaster: dict[str, dict[str, dict[str, Any]]] = {}
    score_manifests_by_forecaster: dict[str, dict[str, Any]] = {}
    for value in arguments.score_artifact:
        root = Path(value).resolve()
        validation = _validate_scored_artifact(root)
        manifest = _read_json(root / "risk_score_manifest.json", "risk score manifest")
        rows = _read_jsonl(root / "risk_scores.jsonl", "risk scores")
        forecasters = {str(row["forecaster_id"]) for row in rows}
        if len(forecasters) != 1:
            raise ValueError("each score artifact must contain one forecaster")
        forecaster = next(iter(forecasters))
        if forecaster in scores_by_forecaster:
            raise ValueError("score artifacts contain the same forecaster")
        episode_ids = [str(row["episode_id"]) for row in rows]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("score artifact contains duplicate episode IDs")
        scores_by_forecaster[forecaster] = dict(zip(episode_ids, rows, strict=True))
        score_manifests_by_forecaster[forecaster] = manifest
        score_sources.append(
            {
                "path": str(root),
                "manifest_sha256": _sha256(root / "risk_score_manifest.json"),
                "risk_scores_sha256": _sha256(root / "risk_scores.jsonl"),
                "validation": validation,
                "source_imputation": manifest["source_imputation"],
            }
        )
    evaluations: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for value in arguments.evaluation_artifact:
        root = Path(value).resolve()
        rows, summary = _validated_evaluation(root)
        forecaster = str(summary["forecaster_id"])
        if forecaster in evaluations:
            raise ValueError("evaluation artifacts contain the same forecaster")
        if forecaster not in score_manifests_by_forecaster:
            raise ValueError("evaluation has no matching score artifact")
        score_manifest = score_manifests_by_forecaster[forecaster]
        score_source = Path(str(score_manifest["source_imputation"])).resolve()
        if Path(str(summary["impute_artifact"])).resolve() != score_source:
            raise ValueError("evaluation and score artifacts use different imputations")
        impute_content = summary.get("impute_content")
        if not isinstance(impute_content, Mapping):
            raise ValueError("evaluation signature has no imputation content")
        expected_content = {
            "assembled_method_id": "b_fais",
            "imputation_manifest_sha256": score_manifest["source_signatures"]["manifest_sha256"],
            "imputation_progress_sha256": score_manifest["source_signatures"]["progress_sha256"],
            "routing_assignments_sha256": score_manifest["source_signatures"][
                "routing_assignments_sha256"
            ],
        }
        for field, expected in expected_content.items():
            if impute_content.get(field) != expected:
                raise ValueError(f"evaluation imputation content differs at {field}")
        evaluations[forecaster] = rows, summary
    if set(evaluations) != set(scores_by_forecaster):
        raise ValueError("score and evaluation forecaster sets differ")

    threshold_episodes: list[ThresholdEpisode] = []
    evaluation_sources: list[dict[str, Any]] = []
    join_audit: list[dict[str, Any]] = []
    for forecaster in sorted(scores_by_forecaster):
        rows, summary = evaluations[forecaster]
        evaluation_sources.append(summary)
        required: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            method = str(row.get("method"))
            if method not in {"b_fais", "clean", robust_id}:
                continue
            episode_id = str(row["episode_id"])
            if method in required[episode_id]:
                raise ValueError("evaluation contains a duplicate required method row")
            required[episode_id][method] = row
        score_map = scores_by_forecaster[forecaster]
        if set(required) != set(score_map):
            raise ValueError("score and evaluation episode sets differ")
        for episode_id in sorted(score_map):
            methods = required[episode_id]
            if set(methods) != {"b_fais", "clean", robust_id}:
                raise ValueError("evaluation is missing a required threshold method")
            b_fais = methods["b_fais"]
            clean = methods["clean"]
            robust = methods[robust_id]
            identity_fields = (
                "dataset_id",
                "family_id",
                "item_id",
                "forecast_origin",
                "mechanism",
                "seed",
                "mask_protocol",
                "mask_seed",
                "mask_realization_id",
                "target_missing_rate",
                "global_missing_rate",
                "local_missing_rate",
                "forecast_seed",
                "forecaster_id",
                "routing_forecaster_id",
                "routing_artifact_forecaster_id",
            )
            for field in identity_fields:
                if not (b_fais.get(field) == clean.get(field) == robust.get(field)):
                    raise ValueError(f"threshold evaluation join identity differs at {field}")
            score = score_map[episode_id]
            for field in (
                "dataset_id",
                "family_id",
                "item_id",
                "forecast_origin",
                "mechanism",
                "seed",
                "mask_seed",
                "mask_realization_id",
                "target_missing_rate",
                "global_missing_rate",
                "local_missing_rate",
                "mask_protocol",
                "forecaster_id",
            ):
                if score.get(field) != b_fais.get(field):
                    raise ValueError(f"score and evaluation identity differ at {field}")
            for field in ("routing_forecaster_id", "routing_artifact_forecaster_id"):
                if b_fais.get(field) != forecaster:
                    raise ValueError(f"evaluation routing identity differs at {field}")
            action_ready = bool(score["robust_candidate_native_finite"])
            if action_ready and (
                not bool(robust.get("native_valid")) or not bool(robust.get("metric_eligible"))
            ):
                raise ValueError("native robust action has no eligible evaluation row")
            threshold_episodes.append(
                ThresholdEpisode(
                    episode_id=f"{forecaster}/{episode_id}",
                    forecaster_id=forecaster,
                    dataset_id=str(b_fais["dataset_id"]),
                    score=score.get("episode_score"),
                    action_ready=action_ready,
                    b_fais_mase=float(b_fais["mase"]),
                    robust_mase=float(robust["mase"]),
                    clean_mase=float(clean["mase"]),
                )
            )
            join_audit.append(
                {
                    "forecaster_id": forecaster,
                    "episode_id": episode_id,
                    "dataset_id": b_fais["dataset_id"],
                    "forecast_seed": b_fais["forecast_seed"],
                    "mask_realization_id": b_fais["mask_realization_id"],
                    "action_ready": action_ready,
                }
            )
    cells = sorted({(episode.forecaster_id, episode.dataset_id) for episode in threshold_episodes})
    if len(threshold_episodes) != 720 or len(cells) != 8:
        raise ValueError("ETT threshold input must contain 720 episodes and eight cells")
    result_a = select_threshold(threshold_episodes, expected_cells=cells)
    result_b = select_threshold(threshold_episodes, expected_cells=cells)
    if asdict(result_a) != asdict(result_b):
        raise RuntimeError("threshold selection did not reproduce exactly")
    output = _new_output_root(arguments.output_dir)
    join_path = output / "join_audit.jsonl"
    _atomic_write_jsonl(join_path, join_audit)
    payload: dict[str, Any] = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "protocol_id": SCRIPT_PROTOCOL_ID,
        "operation": "select-threshold",
        "created_at": _utc_now(),
        "robust_selection": str(Path(arguments.robust_selection).resolve()),
        "robust_selection_canonical_sha256": selection["canonical_sha256"],
        "robust_candidate_id": robust_id,
        "score_sources": score_sources,
        "evaluation_sources": evaluation_sources,
        "join_audit": str(join_path),
        "join_audit_sha256": _sha256(join_path),
        "selection": asdict(result_a),
        "deterministic_second_computation": True,
        "forecast_call_count": 0,
        "resume_count": 0,
        "repair_count": 0,
        "repository_state": _repository_state(),
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    _atomic_write_json(output / "threshold_selection.json", payload)
    validation = {
        "schema_version": 1,
        "status": "verified",
        "episode_count": result_a.episode_count,
        "cell_count": result_a.cell_count,
        "selected_threshold": asdict(result_a.selected_threshold),
        "delta_ett": result_a.delta_ett,
        "candidate_threshold_count": len(result_a.candidates),
        "forecast_call_count": 0,
        "threshold_selection_sha256": _sha256(output / "threshold_selection.json"),
    }
    _atomic_write_json(output / "validation.json", validation)
    return {"output": str(output), **validation}


def _validate(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.artifact).resolve()
    if (root / "risk_score_manifest.json").is_file():
        result = _validate_scored_artifact(root)
    elif (root / "risk_derive_manifest.json").is_file():
        result = _validate_derived_artifact(root)
    elif (root / "robust_selection.json").is_file():
        payload = _load_signed_selection(root)
        result = {
            "schema_version": 1,
            "status": "verified",
            "operation": "select-robust",
            "selected_candidate_id": _selection_candidate_id(payload),
            "artifact_sha256": _sha256(root / "robust_selection.json"),
        }
    elif (root / "threshold_selection.json").is_file():
        payload = _load_threshold_selection(root)
        result = {
            "schema_version": 1,
            "status": "verified",
            "operation": "select-threshold",
            "selected_threshold": asdict(_threshold_from_payload(payload)),
            "artifact_sha256": _sha256(root / "threshold_selection.json"),
        }
    else:
        raise ValueError(f"cannot identify risk fallback artifact: {root}")
    return {"artifact": str(root), **result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    robust = subparsers.add_parser("select-robust")
    robust.add_argument("--labels-artifact", required=True)
    robust.add_argument("--output-dir", required=True)
    robust.set_defaults(function=_select_robust)

    score = subparsers.add_parser("score")
    score.add_argument("--source-imputation", required=True)
    score.add_argument("--imputer-artifacts", required=True)
    score.add_argument("--robust-selection", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument(
        "--lineage-mode",
        choices=("normal", "historical-ett-v002"),
        default="normal",
    )
    score.add_argument("--target-indices", nargs="+", type=int, default=(0, 1))
    score.set_defaults(function=_score)

    derive = subparsers.add_parser("derive")
    derive.add_argument("--score-artifact", required=True)
    derive.add_argument("--threshold-selection", required=True)
    derive.add_argument("--output-dir", required=True)
    derive.set_defaults(function=_derive)

    threshold = subparsers.add_parser("select-threshold")
    threshold.add_argument("--score-artifact", action="append", required=True)
    threshold.add_argument("--evaluation-artifact", action="append", required=True)
    threshold.add_argument("--robust-selection", required=True)
    threshold.add_argument("--output-dir", required=True)
    threshold.set_defaults(function=_select_threshold)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--artifact", required=True)
    validate.set_defaults(function=_validate)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = arguments.function(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
