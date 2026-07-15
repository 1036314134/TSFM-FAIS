"""Strict episode-level transactions for resumable teacher-label generation.

This module is intentionally independent from the stage executor.  The labels
stage can construct deterministic episode expectations, hand completed rows to
``LabelProgressStore``, and later rebuild its existing JSONL outputs without
coupling resource-loading policy to recovery semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from tsfm_fais.artifacts import utc_now

LABEL_PROGRESS_SCHEMA_VERSION = 1
LABEL_SIDECAR_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64


class LabelResumeError(ValueError):
    """Raised when persisted recovery state cannot be trusted."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelResumeError(f"{field} must be a non-negative integer")
    return value


def _normalize_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def canonical_sha256(payload: Any) -> str:
    normalized = _normalize_json(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_signature(path: str | Path) -> dict[str, Any]:
    """Return a deterministic content signature for one file or directory tree."""

    source = Path(path).resolve()
    if source.is_file():
        return {
            "kind": "file",
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
        }
    if not source.is_dir():
        raise FileNotFoundError(f"signature source does not exist: {source}")
    files = []
    for candidate in sorted(
        (entry for entry in source.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(source).as_posix(),
    ):
        files.append(
            {
                "relative_path": candidate.relative_to(source).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": _file_sha256(candidate),
            }
        )
    if not files:
        raise ValueError(f"signature directory contains no files: {source}")
    return {
        "kind": "directory",
        "path": str(source),
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def build_label_resume_identity(
    *,
    resolved_config: Mapping[str, Any],
    audit_artifact: str | Path,
    imputer_manifest: str | Path,
    checkpoint: str | Path,
    forecaster_id: str,
    forecaster_mode: str,
    forecaster_spec: Mapping[str, Any],
    selected_candidates: Sequence[str],
    source_artifacts: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Bind one label run to every input that can change teacher targets."""

    candidates = tuple(selected_candidates)
    if not forecaster_id or not forecaster_mode:
        raise ValueError("forecaster ID and mode must be non-empty")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("selected candidates must be non-empty and unique")
    sources = {
        str(name): path_signature(path)
        for name, path in sorted((source_artifacts or {}).items())
    }
    normalized_spec = _normalize_json(forecaster_spec)
    return {
        "schema_version": 1,
        "progress_schema_version": LABEL_PROGRESS_SCHEMA_VERSION,
        "sidecar_schema_version": LABEL_SIDECAR_SCHEMA_VERSION,
        "resolved_config_sha256": canonical_sha256(resolved_config),
        "audit_artifact": path_signature(audit_artifact),
        "imputer_manifest": path_signature(imputer_manifest),
        "checkpoint": path_signature(checkpoint),
        "forecaster": {
            "id": forecaster_id,
            "mode": forecaster_mode,
            "spec": normalized_spec,
            "spec_sha256": canonical_sha256(normalized_spec),
        },
        "selected_candidates": list(candidates),
        "source_artifacts": sources,
    }


@dataclass(frozen=True)
class LabelEpisodeExpectation:
    artifact_index: int
    forecaster_id: str
    episode_id: str
    dataset_id: str
    family_id: str
    item_id: str
    forecast_origin: int
    sampling_cell: Mapping[str, Any]
    dataset_plan_sha256: str
    candidate_ids: tuple[str, ...]
    block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.artifact_index < 0 or self.forecast_origin < 0:
            raise ValueError("artifact index and forecast origin must be non-negative")
        strings = (
            self.forecaster_id,
            self.episode_id,
            self.dataset_id,
            self.family_id,
            self.item_id,
            self.dataset_plan_sha256,
        )
        if any(not value for value in strings):
            raise ValueError("episode identity strings must be non-empty")
        if not _is_sha256(self.dataset_plan_sha256):
            raise ValueError("dataset plan signature must be a SHA-256 digest")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block IDs must be unique")
        object.__setattr__(self, "sampling_cell", _normalize_json(self.sampling_cell))

    @property
    def key(self) -> str:
        return f"{self.artifact_index:08d}"

    @property
    def sidecar_relative_path(self) -> Path:
        return Path("label_episode_records") / f"{self.key}.json"

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_index": self.artifact_index,
            "forecaster_id": self.forecaster_id,
            "episode_id": self.episode_id,
            "dataset_id": self.dataset_id,
            "family_id": self.family_id,
            "item_id": self.item_id,
            "forecast_origin": self.forecast_origin,
            "sampling_cell": _normalize_json(self.sampling_cell),
            "dataset_plan_sha256": self.dataset_plan_sha256,
            "candidate_ids": list(self.candidate_ids),
            "block_ids": list(self.block_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LabelEpisodeExpectation:
        try:
            sampling_cell = payload["sampling_cell"]
            candidate_ids = payload["candidate_ids"]
            block_ids = payload["block_ids"]
            if not isinstance(sampling_cell, Mapping):
                raise TypeError("sampling_cell must be a mapping")
            if not isinstance(candidate_ids, list) or not isinstance(block_ids, list):
                raise TypeError("candidate_ids and block_ids must be lists")
            return cls(
                artifact_index=int(payload["artifact_index"]),
                forecaster_id=str(payload["forecaster_id"]),
                episode_id=str(payload["episode_id"]),
                dataset_id=str(payload["dataset_id"]),
                family_id=str(payload["family_id"]),
                item_id=str(payload["item_id"]),
                forecast_origin=int(payload["forecast_origin"]),
                sampling_cell=dict(sampling_cell),
                dataset_plan_sha256=str(payload["dataset_plan_sha256"]),
                candidate_ids=tuple(map(str, candidate_ids)),
                block_ids=tuple(map(str, block_ids)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LabelResumeError(f"invalid persisted episode expectation: {error}") from error


@dataclass(frozen=True)
class LabelEpisodeValidation:
    status: Literal["missing", "valid", "invalid"]
    reason: str | None = None
    sidecar: Mapping[str, Any] | None = None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _validate_feature_values(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_feature_values(item, f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_feature_values(item, f"{field}[{index}]")
        return
    _finite_number(value, field)


def _require_identity(
    row: Mapping[str, Any], expectation: LabelEpisodeExpectation, description: str
) -> None:
    expected = {
        "forecaster_id": expectation.forecaster_id,
        "episode_id": expectation.episode_id,
        "dataset_id": expectation.dataset_id,
        "family_id": expectation.family_id,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{description} has inconsistent {field}")


def validate_label_rows(
    expectation: LabelEpisodeExpectation,
    unary_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    outcome: Literal["labeled", "no_labels"] = "labeled",
) -> dict[str, Any]:
    """Validate row structure and return normalized rows plus stable hashes."""

    unary = [_normalize_json(row) for row in unary_rows]
    pairs = [_normalize_json(row) for row in pair_rows]
    if outcome not in {"labeled", "no_labels"}:
        raise ValueError("episode outcome is invalid")
    if outcome == "no_labels" and (unary or pairs):
        raise ValueError("no_labels episodes cannot contain label rows")
    if outcome == "labeled" and not unary:
        raise ValueError("labeled episodes must contain at least one unary row")
    candidates = set(expectation.candidate_ids)
    blocks = set(expectation.block_ids)
    unary_keys: set[tuple[str, str]] = set()
    group_ids: set[str] = set()
    reference_clean: float | None = None
    reference_anchor: float | None = None
    for index, row in enumerate(unary):
        if not isinstance(row, dict):
            raise ValueError(f"unary row {index} must be an object")
        _require_identity(row, expectation, f"unary row {index}")
        block_id = row.get("block_id")
        candidate_id = row.get("candidate_id")
        if block_id not in blocks or candidate_id not in candidates:
            raise ValueError(f"unary row {index} uses an unexpected block or candidate")
        key = (str(block_id), str(candidate_id))
        if key in unary_keys:
            raise ValueError(f"duplicate unary key: {key}")
        unary_keys.add(key)
        expected_group = (
            f"{expectation.forecaster_id}::{expectation.episode_id}::{block_id}"
        )
        if row.get("group_id") != expected_group:
            raise ValueError(f"unary row {index} has an inconsistent group_id")
        group_ids.add(expected_group)
        prior = row.get("prior_features")
        features = row.get("unary_features")
        if not isinstance(prior, dict) or not isinstance(features, dict):
            raise ValueError(f"unary row {index} feature fields must be objects")
        _validate_feature_values(prior, f"unary[{index}].prior_features")
        _validate_feature_values(features, f"unary[{index}].unary_features")
        forecast_loss = _finite_number(
            row.get("forecast_loss"), f"unary[{index}].forecast_loss"
        )
        clean_loss = _finite_number(
            row.get("clean_loss"), f"unary[{index}].clean_loss"
        )
        anchor_loss = _finite_number(
            row.get("anchor_loss"), f"unary[{index}].anchor_loss"
        )
        degradation = _finite_number(
            row.get("degradation"), f"unary[{index}].degradation"
        )
        if not math.isclose(
            degradation,
            forecast_loss - clean_loss,
            rel_tol=1e-9,
            abs_tol=1e-10,
        ):
            raise ValueError(f"unary row {index} has an inconsistent degradation")
        if reference_clean is None:
            reference_clean = clean_loss
            reference_anchor = anchor_loss
        elif clean_loss != reference_clean or anchor_loss != reference_anchor:
            raise ValueError("unary rows disagree on clean or anchor loss")

    pair_keys: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for index, row in enumerate(pairs):
        if not isinstance(row, dict):
            raise ValueError(f"pair row {index} must be an object")
        _require_identity(row, expectation, f"pair row {index}")
        left = (str(row.get("left_block")), str(row.get("left_candidate")))
        right = (str(row.get("right_block")), str(row.get("right_candidate")))
        if left == right:
            raise ValueError(f"pair row {index} repeats one endpoint")
        for block_id, candidate_id in (left, right):
            if block_id not in blocks or candidate_id not in candidates:
                raise ValueError(f"pair row {index} uses an unexpected endpoint")
            if (block_id, candidate_id) not in unary_keys:
                raise ValueError(f"pair row {index} endpoint has no unary row")
        first, second = sorted((left, right))
        pair_key = (first, second)
        if pair_key in pair_keys:
            raise ValueError(f"duplicate pair key: {pair_key}")
        pair_keys.add(pair_key)
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"pair row {index} features must be an object")
        _validate_feature_values(features, f"pair[{index}].features")
        _finite_number(row.get("interaction"), f"pair[{index}].interaction")

    return {
        "unary_rows": unary,
        "pair_rows": pairs,
        "unary_rows_sha256": canonical_sha256(unary),
        "pair_rows_sha256": canonical_sha256(pairs),
        "unary_row_count": len(unary),
        "pair_row_count": len(pairs),
        "ranking_group_count": len(group_ids),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    normalized = _normalize_json(payload)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            normalized,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _normalize_json(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LabelResumeError(f"invalid {description} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LabelResumeError(f"{description} must be a JSON object: {path}")
    return payload


def _validate_dataset_plans(plans: Mapping[str, Any]) -> None:
    for dataset_id, raw_plan in plans.items():
        if not isinstance(dataset_id, str) or not dataset_id:
            raise LabelResumeError("dataset plan keys must be non-empty strings")
        if not isinstance(raw_plan, Mapping):
            raise LabelResumeError(f"dataset plan {dataset_id!r} must be an object")
        if raw_plan.get("dataset_id") != dataset_id:
            raise LabelResumeError(f"dataset plan {dataset_id!r} has a mismatched ID")
        selection_summary = raw_plan.get("selection_summary")
        episode_ids = raw_plan.get("episode_ids")
        if not isinstance(selection_summary, Mapping):
            raise LabelResumeError(
                f"dataset plan {dataset_id!r} selection summary must be an object"
            )
        if (
            not isinstance(episode_ids, list)
            or not episode_ids
            or any(not isinstance(item, str) or not item for item in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)
        ):
            raise LabelResumeError(
                f"dataset plan {dataset_id!r} requires unique non-empty episode IDs"
            )
        signature = raw_plan.get("sha256")
        unsigned_plan = {
            "dataset_id": dataset_id,
            "selection_summary": selection_summary,
            "episode_ids": episode_ids,
        }
        if not _is_sha256(signature) or signature != canonical_sha256(unsigned_plan):
            raise LabelResumeError(f"dataset plan {dataset_id!r} signature is invalid")


def _entry_totals(
    entries: Mapping[str, Any],
    plans: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    totals = {
        "completed_count": len(entries),
        "unary_rows": 0,
        "pair_rows": 0,
        "ranking_groups": 0,
    }
    episode_keys: set[tuple[str, str, str]] = set()
    for key, raw_entry in entries.items():
        if not isinstance(key, str) or not isinstance(raw_entry, Mapping):
            raise LabelResumeError("progress entries must use string keys and objects")
        expectation_payload = raw_entry.get("expectation")
        if not isinstance(expectation_payload, Mapping):
            raise LabelResumeError(f"progress entry {key!r} expectation is invalid")
        expectation = LabelEpisodeExpectation.from_payload(expectation_payload)
        if key != expectation.key:
            raise LabelResumeError(f"progress entry {key!r} has a mismatched index key")
        artifact_index = _non_negative_int(
            raw_entry.get("artifact_index"), f"progress entry {key!r} artifact_index"
        )
        if artifact_index != expectation.artifact_index:
            raise LabelResumeError(
                f"progress entry {key!r} artifact index differs from its expectation"
            )
        episode_key = (
            expectation.forecaster_id,
            expectation.dataset_id,
            expectation.episode_id,
        )
        if episode_key in episode_keys:
            raise LabelResumeError(f"duplicate committed episode key: {episode_key}")
        episode_keys.add(episode_key)
        if plans is not None:
            plan = plans.get(expectation.dataset_id)
            if not isinstance(plan, Mapping):
                raise LabelResumeError(
                    f"progress entry {key!r} has no registered dataset plan"
                )
            if plan.get("sha256") != expectation.dataset_plan_sha256:
                raise LabelResumeError(
                    f"progress entry {key!r} dataset-plan signature differs"
                )
            episode_ids = plan.get("episode_ids")
            if not isinstance(episode_ids, list) or expectation.episode_id not in episode_ids:
                raise LabelResumeError(
                    f"progress entry {key!r} is absent from its dataset plan"
                )
        if raw_entry.get("outcome") not in {"labeled", "no_labels"}:
            raise LabelResumeError(f"progress entry {key!r} outcome is invalid")
        expected_sidecar = expectation.sidecar_relative_path.as_posix()
        if raw_entry.get("sidecar_file") != expected_sidecar:
            raise LabelResumeError(f"progress entry {key!r} sidecar path is invalid")
        for field in (
            "sidecar_sha256",
            "unary_rows_sha256",
            "pair_rows_sha256",
        ):
            if not _is_sha256(raw_entry.get(field)):
                raise LabelResumeError(
                    f"progress entry {key!r} field {field!r} is not a SHA-256 digest"
                )
        for field in ("unary_rows", "pair_rows", "ranking_groups"):
            totals[field] += _non_negative_int(
                raw_entry.get(field), f"progress entry {key!r} {field}"
            )
    return totals


class LabelProgressStore:
    """Atomic commit marker and sidecar validator for one forecaster run."""

    def __init__(self, root: Path, payload: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.progress_path = self.root / "labels_progress.json"
        self.payload = payload

    @classmethod
    def create(
        cls, root: str | Path, identity: Mapping[str, Any]
    ) -> LabelProgressStore:
        target = Path(root).resolve()
        target.mkdir(parents=True, exist_ok=True)
        progress_path = target / "labels_progress.json"
        if progress_path.exists():
            raise FileExistsError(f"labels progress already exists: {progress_path}")
        payload = {
            "schema_version": LABEL_PROGRESS_SCHEMA_VERSION,
            "status": "running",
            "identity": _normalize_json(identity),
            "dataset_plans": {},
            "entries": {},
            "completed_count": 0,
            "unary_rows": 0,
            "pair_rows": 0,
            "ranking_groups": 0,
            "resume_count": 0,
            "repair_count": 0,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        _atomic_write_json(progress_path, payload)
        return cls(target, payload)

    @classmethod
    def open_existing(
        cls, root: str | Path, identity: Mapping[str, Any]
    ) -> LabelProgressStore:
        target = Path(root).resolve()
        progress_path = target / "labels_progress.json"
        payload = _read_json_object(progress_path, "labels progress")
        if payload.get("schema_version") != LABEL_PROGRESS_SCHEMA_VERSION:
            raise LabelResumeError("unsupported labels progress schema")
        if payload.get("identity") != _normalize_json(identity):
            raise LabelResumeError("labels resume identity changed")
        entries = payload.get("entries")
        plans = payload.get("dataset_plans")
        if not isinstance(entries, dict) or not isinstance(plans, dict):
            raise LabelResumeError("labels progress entries or dataset plans are invalid")
        _validate_dataset_plans(plans)
        expected_totals = _entry_totals(entries, plans)
        for field, expected in expected_totals.items():
            if payload.get(field) != expected:
                raise LabelResumeError(f"labels progress field {field!r} is inconsistent")
        payload["status"] = "running"
        payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
        payload["updated_at"] = utc_now()
        _atomic_write_json(progress_path, payload)
        return cls(target, payload)

    def register_dataset_plan(
        self,
        dataset_id: str,
        selection_summary: Mapping[str, Any],
        episode_ids: Sequence[str],
    ) -> str:
        if (
            not dataset_id
            or not episode_ids
            or any(not isinstance(item, str) or not item for item in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)
        ):
            raise ValueError("dataset plan requires a dataset and unique episode IDs")
        plan = {
            "dataset_id": dataset_id,
            "selection_summary": _normalize_json(selection_summary),
            "episode_ids": list(episode_ids),
        }
        plan["sha256"] = canonical_sha256(plan)
        existing = self.payload["dataset_plans"].get(dataset_id)
        if existing is not None:
            if existing != plan:
                raise LabelResumeError(f"dataset plan changed for {dataset_id!r}")
            return str(plan["sha256"])
        updated = copy.deepcopy(self.payload)
        updated["dataset_plans"][dataset_id] = plan
        updated["updated_at"] = utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.payload = updated
        return str(plan["sha256"])

    def _check_expectation_plan(self, expectation: LabelEpisodeExpectation) -> None:
        plan = self.payload["dataset_plans"].get(expectation.dataset_id)
        if not isinstance(plan, dict):
            raise LabelResumeError(
                f"dataset plan is not registered for {expectation.dataset_id!r}"
            )
        if plan.get("sha256") != expectation.dataset_plan_sha256:
            raise LabelResumeError("episode dataset-plan signature differs")
        if expectation.episode_id not in plan.get("episode_ids", []):
            raise LabelResumeError("episode is absent from its registered dataset plan")

    def commit_episode(
        self,
        expectation: LabelEpisodeExpectation,
        unary_rows: Sequence[Mapping[str, Any]],
        pair_rows: Sequence[Mapping[str, Any]],
        *,
        outcome: Literal["labeled", "no_labels"] = "labeled",
        artifact_loading_delta: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> Mapping[str, Any]:
        self._check_expectation_plan(expectation)
        existing = self.payload["entries"].get(expectation.key)
        if existing is not None:
            if existing.get("expectation") != expectation.to_payload():
                raise LabelResumeError("cannot replace an episode with different identity")
            if not replace:
                raise FileExistsError(f"episode {expectation.key} is already committed")
        validated = validate_label_rows(
            expectation,
            unary_rows,
            pair_rows,
            outcome=outcome,
        )
        sidecar = {
            "schema_version": LABEL_SIDECAR_SCHEMA_VERSION,
            "outcome": outcome,
            "expectation": expectation.to_payload(),
            **validated,
            "artifact_loading_delta": _normalize_json(artifact_loading_delta or {}),
            "created_at": utc_now(),
        }
        sidecar_path = self.root / expectation.sidecar_relative_path
        _atomic_write_json(sidecar_path, sidecar)
        entry = {
            "artifact_index": expectation.artifact_index,
            "expectation": expectation.to_payload(),
            "outcome": outcome,
            "sidecar_file": expectation.sidecar_relative_path.as_posix(),
            "sidecar_sha256": _file_sha256(sidecar_path),
            "unary_rows_sha256": validated["unary_rows_sha256"],
            "pair_rows_sha256": validated["pair_rows_sha256"],
            "unary_rows": validated["unary_row_count"],
            "pair_rows": validated["pair_row_count"],
            "ranking_groups": validated["ranking_group_count"],
            "completed_at": utc_now(),
        }
        updated = copy.deepcopy(self.payload)
        updated["entries"][expectation.key] = entry
        updated.update(_entry_totals(updated["entries"]))
        if existing is not None:
            updated["repair_count"] = int(updated.get("repair_count", 0)) + 1
        updated["updated_at"] = utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.payload = updated
        return entry

    def _sidecar_path(self, entry: Mapping[str, Any]) -> Path:
        relative = Path(str(entry.get("sidecar_file", "")))
        expected = Path("label_episode_records") / f"{int(entry['artifact_index']):08d}.json"
        if relative != expected or relative.is_absolute() or ".." in relative.parts:
            raise LabelResumeError("progress contains an unsafe sidecar path")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise LabelResumeError("sidecar path escapes the run directory")
        return path

    def validate_episode(
        self, expectation: LabelEpisodeExpectation
    ) -> LabelEpisodeValidation:
        self._check_expectation_plan(expectation)
        entry = self.payload["entries"].get(expectation.key)
        if entry is None:
            return LabelEpisodeValidation("missing", "episode is not committed")
        if not isinstance(entry, dict):
            raise LabelResumeError("progress entry must be an object")
        if entry.get("expectation") != expectation.to_payload():
            raise LabelResumeError("progress episode identity differs from the current plan")
        try:
            sidecar_path = self._sidecar_path(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise LabelResumeError(f"invalid progress sidecar entry: {error}") from error
        if not sidecar_path.is_file():
            return LabelEpisodeValidation("invalid", "committed sidecar is missing")
        if _file_sha256(sidecar_path) != entry.get("sidecar_sha256"):
            return LabelEpisodeValidation("invalid", "sidecar hash mismatch")
        try:
            sidecar = _read_json_object(sidecar_path, "label episode sidecar")
            if sidecar.get("schema_version") != LABEL_SIDECAR_SCHEMA_VERSION:
                raise ValueError("sidecar schema mismatch")
            if sidecar.get("expectation") != expectation.to_payload():
                raise ValueError("sidecar episode identity differs")
            outcome = sidecar.get("outcome")
            if outcome not in {"labeled", "no_labels"}:
                raise ValueError("sidecar outcome is invalid")
            if entry.get("outcome") != outcome:
                raise ValueError("sidecar and progress outcomes differ")
            unary_rows = sidecar.get("unary_rows")
            pair_rows = sidecar.get("pair_rows")
            if not isinstance(unary_rows, list) or not isinstance(pair_rows, list):
                raise ValueError("sidecar label rows must be lists")
            validated = validate_label_rows(
                expectation,
                unary_rows,
                pair_rows,
                outcome=outcome,
            )
            comparisons = {
                "unary_rows_sha256": validated["unary_rows_sha256"],
                "pair_rows_sha256": validated["pair_rows_sha256"],
                "unary_rows": validated["unary_row_count"],
                "pair_rows": validated["pair_row_count"],
                "ranking_groups": validated["ranking_group_count"],
            }
            for field, expected in comparisons.items():
                sidecar_field = field.replace("_rows", "_row_count")
                if field.endswith("_sha256"):
                    sidecar_field = field
                elif field == "ranking_groups":
                    sidecar_field = "ranking_group_count"
                if sidecar.get(sidecar_field) != expected or entry.get(field) != expected:
                    raise ValueError(f"sidecar or progress {field} differs")
        except (LabelResumeError, TypeError, ValueError) as error:
            return LabelEpisodeValidation("invalid", str(error))
        return LabelEpisodeValidation("valid", sidecar=sidecar)

    def rebuild_outputs(
        self,
        teacher_labels_path: str | Path,
        pair_labels_path: str | Path,
        *,
        expected_episode_count: int,
    ) -> dict[str, Any]:
        if expected_episode_count < 1:
            raise ValueError("expected episode count must be positive")
        entries = self.payload["entries"]
        expected_keys = {f"{index:08d}" for index in range(expected_episode_count)}
        if set(entries) != expected_keys:
            raise LabelResumeError(
                "progress does not match the complete deterministic episode index"
            )
        unary_rows: list[Mapping[str, Any]] = []
        pair_rows: list[Mapping[str, Any]] = []
        datasets: set[str] = set()
        families: set[str] = set()
        forecasters: set[str] = set()
        labeled = 0
        no_labels = 0
        artifact_loading: list[Mapping[str, Any]] = []
        for index in range(expected_episode_count):
            entry = entries[f"{index:08d}"]
            expectation = LabelEpisodeExpectation.from_payload(entry["expectation"])
            validation = self.validate_episode(expectation)
            if validation.status != "valid" or validation.sidecar is None:
                raise LabelResumeError(
                    f"cannot rebuild invalid episode {index:08d}: {validation.reason}"
                )
            sidecar = validation.sidecar
            unary_rows.extend(sidecar["unary_rows"])
            pair_rows.extend(sidecar["pair_rows"])
            datasets.add(expectation.dataset_id)
            families.add(expectation.family_id)
            forecasters.add(expectation.forecaster_id)
            if sidecar["outcome"] == "labeled":
                labeled += 1
            else:
                no_labels += 1
            loading_delta = sidecar.get("artifact_loading_delta")
            if isinstance(loading_delta, Mapping):
                artifact_loading.append(loading_delta)
        teacher_path = Path(teacher_labels_path).resolve()
        pair_path = Path(pair_labels_path).resolve()
        _atomic_write_jsonl(teacher_path, unary_rows)
        _atomic_write_jsonl(pair_path, pair_rows)
        summary = {
            "episode_count": expected_episode_count,
            "labeled_episode_count": labeled,
            "no_label_episode_count": no_labels,
            "unary_rows": len(unary_rows),
            "pair_rows": len(pair_rows),
            "ranking_groups": len(
                {str(row["group_id"]) for row in unary_rows}
            ),
            "forecasters": sorted(forecasters),
            "dataset_ids": sorted(datasets),
            "family_ids": sorted(families),
            "teacher_labels": str(teacher_path),
            "pair_labels": str(pair_path),
            "teacher_labels_sha256": _file_sha256(teacher_path),
            "pair_labels_sha256": _file_sha256(pair_path),
            "artifact_loading_deltas": artifact_loading,
        }
        updated = copy.deepcopy(self.payload)
        updated["status"] = "rebuilt"
        updated["final_outputs"] = {
            key: value
            for key, value in summary.items()
            if key not in {"artifact_loading_deltas"}
        }
        updated["updated_at"] = utc_now()
        _atomic_write_json(self.progress_path, updated)
        self.payload = updated
        return summary


__all__ = [
    "LABEL_PROGRESS_SCHEMA_VERSION",
    "LABEL_SIDECAR_SCHEMA_VERSION",
    "LabelEpisodeExpectation",
    "LabelEpisodeValidation",
    "LabelProgressStore",
    "LabelResumeError",
    "build_label_resume_identity",
    "canonical_sha256",
    "path_signature",
    "validate_label_rows",
]
