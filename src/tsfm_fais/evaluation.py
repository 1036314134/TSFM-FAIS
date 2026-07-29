"""Streaming downstream forecast evaluation and grouped result summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tsfm_fais.config import AppConfig
from tsfm_fais.contracts import ForecastResult, ForecastSpec, SeriesBatch
from tsfm_fais.data import stable_seed
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.routing.sequence_protocol import (
    SELECTOR_INDEPENDENT_FORECAST_MODE,
    SELECTOR_INDEPENDENT_FORECASTER_ID,
    is_independent_sequence_routing_metadata,
)

EVALUATION_FIELDS: tuple[str, ...] = (
    "schema_version",
    "episode_id",
    "dataset_id",
    "family_id",
    "forecaster_id",
    "routing_forecaster_id",
    "routing_artifact_forecaster_id",
    "item_id",
    "forecast_origin",
    "mechanism",
    "missing_rate",
    "seed",
    "mask_protocol",
    "mask_seed",
    "mask_realization_id",
    "target_missing_rate",
    "global_missing_rate",
    "local_missing_rate",
    "contains_missing",
    "mase_scale_lag",
    "method",
    "method_role",
    "oracle_source",
    "oracle_eligible",
    "native_valid",
    "metric_eligible",
    "ineligibility_reason",
    "candidate_status",
    "mase",
    "mae",
    "rmse",
    "imputation_mae",
    "imputation_rmse",
    "degradation_vs_clean_mase",
    "relative_degradation_vs_clean",
    "regret_mase",
    "relative_regret",
    "forecast_seed",
    "runtime_seconds",
    "rss_delta_bytes",
    "runtime_scope",
)

SUMMARY_METRICS: tuple[str, ...] = (
    "mase",
    "mae",
    "rmse",
    "imputation_mae",
    "imputation_rmse",
    "degradation_vs_clean_mase",
    "relative_degradation_vs_clean",
    "regret_mase",
    "relative_regret",
    "runtime_seconds",
    "rss_delta_bytes",
)

DEFAULT_GROUP_BY: tuple[str, ...] = (
    "dataset_id",
    "family_id",
    "forecaster_id",
    "mechanism",
    "missing_rate",
    "method",
)

FORECAST_CALL_PROTOCOL = "batched_common_contexts_v1"


class ForecastPredictor(Protocol):
    def predict(self, contexts: np.ndarray, forecast_spec: ForecastSpec) -> ForecastResult: ...


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _impute_content_signature(
    root: Path,
    assignments: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the completed imputation ledger and identify every NPZ payload."""

    manifest_path = root / "imputation_manifest.json"
    progress_path = root / "imputation_progress.json"
    if not manifest_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError(
            "impute artifact must contain imputation_manifest.json and imputation_progress.json"
        )
    manifest_method: str | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read the imputation manifest or progress ledger") from error
    if not isinstance(manifest, Mapping) or not isinstance(progress, Mapping):
        raise ValueError("imputation manifest and progress ledger must be objects")
    if progress.get("status") != "completed":
        raise ValueError("imputation progress ledger is not completed")
    manifest_sha256 = _file_sha256(manifest_path)
    assignments_sha256 = _file_sha256(assignments)
    if progress.get("imputation_manifest_sha256") != manifest_sha256:
        raise ValueError("imputation manifest hash differs from the completed progress ledger")
    if progress.get("routing_assignments_sha256") != assignments_sha256:
        raise ValueError("routing assignments hash differs from the completed progress ledger")
    raw_method = manifest.get("assembled_method_id")
    if raw_method is not None:
        manifest_method = str(raw_method).strip()
        if not manifest_method:
            raise ValueError("imputation manifest assembled_method_id must be non-empty")
    raw_entries = progress.get("entries")
    if not isinstance(raw_entries, Mapping):
        raise ValueError("imputation progress ledger has no entries")
    episode_count = int(manifest.get("episode_count", -1))
    if episode_count < 1 or len(raw_entries) != episode_count:
        raise ValueError("imputation progress entry count differs from the manifest")
    npz_hashes: dict[str, str] = {}
    for raw_entry in raw_entries.values():
        if not isinstance(raw_entry, Mapping):
            raise ValueError("imputation progress contains an invalid entry")
        relative = Path(str(raw_entry.get("file", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe imputation progress path: {relative}")
        normalized = relative.as_posix()
        digest = raw_entry.get("npz_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("imputation progress contains an invalid NPZ hash")
        if normalized in npz_hashes:
            raise ValueError(f"imputation progress contains a duplicate file: {normalized}")
        npz_hashes[normalized] = digest
    npz_manifest_sha256 = hashlib.sha256(
        json.dumps(npz_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    signature = {
        "routing_assignments_sha256": _file_sha256(assignments),
        "imputation_manifest_sha256": manifest_sha256,
        "imputation_progress_sha256": _file_sha256(progress_path),
        "npz_manifest_sha256": npz_manifest_sha256,
        "assembled_method_id": manifest_method,
    }
    return signature, npz_hashes


def _verify_imputation_npz_integrity(imputations: Path, npz_hashes: Mapping[str, str]) -> None:
    for relative, expected in npz_hashes.items():
        path = imputations / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"imputation NPZ listed in progress does not exist: {path}")
        if _file_sha256(path) != expected:
            raise ValueError(f"imputation NPZ hash differs from progress: {relative}")


def parse_ids(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(","))
    else:
        parts = tuple(str(part).strip() for part in value)
    if not parts or any(not part for part in parts):
        raise ValueError("IDs must be non-empty comma-separated values")
    if len(set(parts)) != len(parts):
        raise ValueError("IDs must be unique")
    return parts


def _episode_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    episode_id = str(record["episode_id"])
    mechanism = record.get("mechanism")
    missing_rate = record.get("missing_rate")
    seed = record.get("seed")
    if mechanism is None or missing_rate is None or seed is None:
        parts = episode_id.rsplit("__", 4)
        if len(parts) == 5:
            _, _, parsed_mechanism, parsed_rate, parsed_seed = parts
            mechanism = mechanism if mechanism is not None else parsed_mechanism
            if missing_rate is None:
                try:
                    missing_rate = float(parsed_rate)
                except ValueError:
                    missing_rate = None
            if seed is None:
                try:
                    seed = int(parsed_seed)
                except ValueError:
                    seed = None
    return {
        "episode_id": episode_id,
        "dataset_id": str(record.get("dataset_id", "")),
        "family_id": str(record.get("family_id", "")),
        "forecaster_id": str(record.get("forecaster_id", "")),
        "routing_forecaster_id": str(record.get("forecaster_id", "")),
        "routing_artifact_forecaster_id": str(record.get("forecaster_id", "")),
        "item_id": str(record.get("item_id", "")),
        "forecast_origin": int(record.get("forecast_origin", -1)),
        "mechanism": mechanism,
        "missing_rate": missing_rate,
        "seed": seed,
        "mask_protocol": record.get("mask_protocol"),
        "mask_seed": record.get("mask_seed"),
        "mask_realization_id": record.get("mask_realization_id"),
        "target_missing_rate": record.get("target_missing_rate", missing_rate),
        "global_missing_rate": record.get("global_missing_rate"),
        "local_missing_rate": record.get("local_missing_rate"),
        "mase_scale_lag": record.get("mase_scale_lag"),
    }


def forecast_metrics(
    clean_context: np.ndarray,
    clean_future: np.ndarray,
    forecast: ForecastResult,
    *,
    seasonality: int,
    mase_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return one MASE/MAE/RMSE value for each forecast batch member."""

    context = np.asarray(clean_context, dtype=float)
    future = np.asarray(clean_future, dtype=float)
    if context.ndim != 2 or future.ndim != 2:
        raise ValueError("clean context and future must have shapes [L,D] and [H,D]")
    targets = tuple(forecast.target_indices)
    if not targets or any(target >= context.shape[1] for target in targets):
        raise ValueError("forecast target indices are incompatible with the episode")
    prediction = np.asarray(forecast.point, dtype=float)
    if prediction.shape[1:] != (future.shape[0], len(targets)):
        raise ValueError("forecast point shape does not match future and target indices")
    truth = future[:, targets][None, :, :]
    error = prediction - truth
    absolute = np.abs(error)
    mae_values = np.mean(absolute, axis=(1, 2))
    rmse_values = np.sqrt(np.mean(error**2, axis=(1, 2)))

    if mase_scale is None:
        requested_lag = max(1, int(seasonality))
        lag = requested_lag if context.shape[0] > requested_lag else 1
        history = context[:, targets]
        differences = np.abs(history[lag:] - history[:-lag])
        scales = np.maximum(np.mean(differences, axis=0), 1e-8)
    else:
        supplied = np.asarray(mase_scale, dtype=float).reshape(-1)
        if supplied.shape == (context.shape[1],):
            scales = supplied[np.asarray(targets)]
        elif supplied.shape == (len(targets),):
            scales = supplied
        else:
            raise ValueError("mase_scale must align with all variates or forecast targets")
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("mase_scale must contain finite positive values")
    mase_values = np.mean(np.mean(absolute, axis=1) / scales[None, :], axis=1)
    return {"mase": mase_values, "mae": mae_values, "rmse": rmse_values}


def _imputation_metrics(
    clean_context: np.ndarray,
    candidate_context: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[float, float]:
    missing = ~np.asarray(observed_mask, dtype=bool)
    if not missing.any():
        return 0.0, 0.0
    error = (
        np.asarray(candidate_context, dtype=float)[missing]
        - np.asarray(clean_context, dtype=float)[missing]
    )
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def _load_saved_candidates(
    archive: Mapping[str, Any],
    observed_mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if "candidate_ids" not in archive or "candidate_values" not in archive:
        return {}
    identifiers = tuple(str(value) for value in np.asarray(archive["candidate_ids"]).tolist())
    values = np.asarray(archive["candidate_values"], dtype=float)
    native = np.asarray(
        archive.get("candidate_native_valid", np.ones_like(values, dtype=bool)),
        dtype=bool,
    )
    statuses = tuple(
        str(value)
        for value in np.asarray(
            archive.get("candidate_status", np.asarray(["unknown"] * len(identifiers)))
        ).tolist()
    )
    runtimes = np.asarray(
        archive.get("candidate_runtime_seconds", np.zeros(len(identifiers))),
        dtype=float,
    )
    rss_deltas = np.asarray(
        archive.get(
            "candidate_rss_delta_bytes",
            archive.get(
                "candidate_peak_memory_bytes",
                np.zeros(len(identifiers)),
            ),
        ),
        dtype=float,
    )
    expected = (len(identifiers), *observed_mask.shape)
    if values.shape != expected or native.shape != expected:
        raise ValueError(f"saved candidate tensors must have shape {expected}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("saved candidate IDs must be unique")
    if len(statuses) != len(identifiers):
        raise ValueError("saved candidate statuses do not align with candidate IDs")
    if runtimes.shape != (len(identifiers),) or rss_deltas.shape != (len(identifiers),):
        raise ValueError("saved candidate resource metrics do not align with candidate IDs")
    missing = ~observed_mask
    result: dict[str, dict[str, Any]] = {}
    for index, identifier in enumerate(identifiers):
        native_valid = bool(native[index][missing].all())
        result[identifier] = {
            "values": values[index],
            "native_valid": native_valid,
            "ineligibility_reason": None if native_valid else "native_invalid",
            "status": statuses[index],
            "runtime_seconds": float(runtimes[index]),
            # CandidateRunner currently measures end RSS minus start RSS.  The
            # persisted key keeps its historical name, but evaluation reports
            # the quantity with its actual semantics.
            "rss_delta_bytes": int(rss_deltas[index]),
        }
    return result


def _complete_stateless_baselines(
    clean_context: np.ndarray,
    observed_mask: np.ndarray,
    candidates: dict[str, dict[str, Any]],
    baseline_ids: Sequence[str],
) -> None:
    batch = SeriesBatch(
        clean_context[None, ...],
        observed_mask[None, ...],
        item_ids=("evaluation",),
    )
    runner = CandidateRunner(DEFAULT_REGISTRY)
    for candidate_id in baseline_ids:
        if candidate_id in candidates:
            continue
        if candidate_id not in DEFAULT_REGISTRY:
            raise KeyError(f"unknown evaluation baseline {candidate_id!r}")
        spec = DEFAULT_REGISTRY.get_spec(candidate_id)
        if spec.fit_scope != "none":
            raise ValueError(
                f"baseline {candidate_id!r} was not saved and requires a fitted artifact"
            )
        result = runner.run(candidate_id, batch, seed=0)
        missing = ~observed_mask
        candidates[candidate_id] = {
            "values": result.values[0],
            "native_valid": bool(result.native_valid_mask[0][missing].all()),
            "ineligibility_reason": (
                None if bool(result.native_valid_mask[0][missing].all()) else "native_invalid"
            ),
            "status": result.status.value,
            "runtime_seconds": float(result.runtime_seconds),
            "rss_delta_bytes": int(result.peak_memory_bytes),
        }


def _forecast_spec(config: AppConfig, model_id: str, dimensions: int) -> ForecastSpec:
    registry = default_forecast_registry()
    adapter = registry.get(model_id)
    targets = (
        tuple(range(dimensions))
        if config.experiment.target_indices == "all"
        else tuple(config.experiment.target_indices)
    )
    return ForecastSpec(
        model_id=model_id,
        mode=adapter.mode,
        horizon=config.experiment.horizon,
        context_length=config.experiment.context_length,
        target_indices=targets,
        num_samples=config.experiment.forecast_num_samples,
    )


def _resolve_forecast_device(config: AppConfig) -> str:
    requested = config.runtime.device
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as error:
        if requested == "gpu":
            raise RuntimeError("runtime.device=gpu requires PyTorch with CUDA") from error
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "gpu" and not available:
        raise RuntimeError("runtime.device=gpu was requested but CUDA is unavailable")
    return "cuda" if available else "cpu"


def _set_forecast_seed(seed: int) -> None:
    normalized = int(seed) % (2**32)
    random.seed(normalized)
    np.random.seed(normalized)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def _build_forecast_runner(
    model_id: str,
    artifact: Path,
    *,
    device: str,
    batch_size: int = 128,
) -> ForecastRunner:
    source = artifact.resolve()
    if not source.exists():
        raise FileNotFoundError(f"forecaster artifact does not exist: {source}")
    registry = default_forecast_registry()
    adapter = registry.build(
        model_id,
        model_name=str(source),
        device=device,
        batch_size=batch_size,
    )
    ensure_backend = getattr(adapter, "_ensure_backend", None)
    if callable(ensure_backend):
        ensure_backend()
    return ForecastRunner(registry, adapters={model_id: adapter})


def _resolve_forecaster_artifact(path: str | Path, model_id: str) -> Path:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"forecaster artifact does not exist: {source}")
    if not (source.is_file() and source.suffix.lower() == ".json"):
        return source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("forecaster artifact JSON is invalid") from error
    mapping = payload.get("artifacts") if isinstance(payload, dict) else None
    if mapping is None:
        mapping = payload
    if not isinstance(mapping, Mapping):
        raise ValueError("forecaster artifact JSON must map model IDs to paths")
    raw_path = mapping.get(model_id)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"forecaster artifact mapping has no path for {model_id!r}")
    target = Path(raw_path)
    resolved = target if target.is_absolute() else (source.parent / target).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"forecaster artifact does not exist: {resolved}")
    return resolved


def _recover_completed(path: Path, *, resume: bool) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    if not resume:
        raise FileExistsError(
            f"evaluation output already exists; pass --resume or choose another directory: {path}"
        )
    completed: set[tuple[str, str, str]] = set()
    with path.open("rb+") as handle:
        last_good = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                last_good = handle.tell()
                continue
            try:
                row = json.loads(line.decode("utf-8"))
                completed.add(
                    (
                        str(row["forecaster_id"]),
                        str(row["episode_id"]),
                        str(row["method"]),
                    )
                )
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                handle.truncate(last_good)
                break
            last_good = handle.tell()
    return completed


def _write_csv_from_jsonl(source: Path, target: Path) -> Path:
    temporary = target.with_suffix(target.suffix + ".tmp")
    with (
        source.open("r", encoding="utf-8") as input_handle,
        temporary.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        writer = csv.DictWriter(output_handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            writer.writerow({field: row.get(field) for field in EVALUATION_FIELDS})
    temporary.replace(target)
    return target


def _validate_resume_signature(
    manifest_path: Path,
    jsonl_path: Path,
    signature: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if not jsonl_path.exists() or not resume:
        return
    if not manifest_path.is_file():
        raise ValueError("cannot safely resume evaluation without evaluation_manifest.json")
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read the existing evaluation manifest") from error
    existing_signature = existing.get("evaluation_signature")
    if existing_signature != dict(signature):
        raise ValueError(
            "evaluation resume signature does not match the existing output; "
            "use a new output directory"
        )


@dataclass(frozen=True)
class _SharedEvaluation:
    root: Path
    manifest_sha256: str
    metrics_sha256: str
    rows_by_episode: dict[str, dict[str, dict[str, Any]]]

    def signature(self) -> dict[str, str]:
        return {
            "artifact": str(self.root),
            "evaluation_manifest_sha256": self.manifest_sha256,
            "episode_metrics_sha256": self.metrics_sha256,
        }


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _load_assignment_records(
    assignments: Path,
    npz_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    files: set[str] = set()
    with assignments.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("imputation assignment rows must be JSON objects")
            episode_id = str(payload.get("episode_id", "")).strip()
            if not episode_id or episode_id in records:
                raise ValueError("imputation assignments contain an empty or duplicate episode ID")
            relative = Path(str(payload.get("file", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe imputation artifact path: {relative}")
            normalized = relative.as_posix()
            if normalized not in npz_hashes or normalized in files:
                raise ValueError(
                    "imputation assignments and completed progress entries do not match"
                )
            records[episode_id] = payload
            files.add(normalized)
    if files != set(npz_hashes):
        raise ValueError("imputation assignments and completed progress entries do not match")
    return records


def _arrays_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


_SELECTOR_SPECIFIC_ARCHIVE_FIELDS = frozenset(
    {
        "values",
        "assembled_method_id",
        "pipeline_runtime_seconds",
        "pipeline_rss_before_bytes",
        "pipeline_rss_after_bytes",
        "pipeline_rss_delta_bytes",
        "pipeline_peak_memory_bytes",
    }
)

_ROUTING_IDENTITY_ARCHIVE_FIELDS = frozenset(
    {
        "forecaster_id",
        "forecast_mode",
        "forecaster_independent_selection",
    }
)

_SHARED_ASSIGNMENT_FIELDS = (
    "schema_version",
    "episode_id",
    "dataset_id",
    "family_id",
    "item_id",
    "forecast_origin",
    "mechanism",
    "missing_rate",
    "seed",
    "episode_seed",
    "mask_protocol",
    "mask_seed",
    "mask_realization_id",
    "target_missing_rate",
    "global_missing_rate",
    "local_missing_rate",
    "mase_scale",
    "mase_scale_lag",
    "candidate_ids",
    "file",
    "forecaster_id",
    "forecast_mode",
    "forecaster_independent_selection",
)


def _validate_shared_assignment_content(
    target_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    *,
    episode_id: str,
    allow_routing_identity_difference: bool,
) -> None:
    for field in _SHARED_ASSIGNMENT_FIELDS:
        if allow_routing_identity_difference and field in {
            "forecaster_id",
            "forecast_mode",
            "forecaster_independent_selection",
        }:
            continue
        target_has_field = field in target_record
        source_has_field = field in source_record
        if target_has_field != source_has_field or (
            target_has_field
            and json.dumps(target_record[field], sort_keys=True, separators=(",", ":"))
            != json.dumps(source_record[field], sort_keys=True, separators=(",", ":"))
        ):
            raise ValueError(
                f"shared imputation assignment differs for episode {episode_id}: {field}"
            )


def _validate_shared_imputation_content(
    target_archive: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    *,
    episode_id: str,
    allow_routing_identity_difference: bool,
) -> None:
    """Require exact equality for every field unrelated to selector assembly."""

    ignored_fields = _SELECTOR_SPECIFIC_ARCHIVE_FIELDS
    if allow_routing_identity_difference:
        ignored_fields = ignored_fields | _ROUTING_IDENTITY_ARCHIVE_FIELDS
    target_fields = set(target_archive) - ignored_fields
    source_fields = set(source_archive) - ignored_fields
    if target_fields != source_fields:
        raise ValueError(
            f"shared imputation content differs for episode {episode_id}: archive fields"
        )
    for field in sorted(target_fields):
        if not _arrays_equal(target_archive[field], source_archive[field]):
            raise ValueError(f"shared imputation content differs for episode {episode_id}: {field}")


def _load_shared_metric_rows(
    metrics_path: Path,
    *,
    forecaster_id: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    rows_by_episode: dict[str, dict[str, dict[str, Any]]] = {}
    row_count = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("shared evaluation metrics must contain JSON objects")
            if str(payload.get("forecaster_id", "")) != forecaster_id:
                raise ValueError("shared evaluation metrics contain a different forecaster ID")
            episode_id = str(payload.get("episode_id", "")).strip()
            method_id = str(payload.get("method", "")).strip()
            if not episode_id or not method_id:
                raise ValueError("shared evaluation metrics contain an empty episode or method ID")
            episode_rows = rows_by_episode.setdefault(episode_id, {})
            if method_id in episode_rows:
                raise ValueError("shared evaluation metrics contain duplicate episode/method rows")
            episode_rows[method_id] = payload
            row_count += 1
    if row_count == 0:
        raise ValueError("shared evaluation metrics are empty")
    return rows_by_episode


def _validate_shared_rows_for_episode(
    rows: Mapping[str, Mapping[str, Any]],
    source_record: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    *,
    config: AppConfig,
    forecaster_id: str,
    baseline_ids: Sequence[str],
) -> None:
    episode_id = str(source_record["episode_id"])
    observed_mask = np.asarray(source_archive["observed_mask"], dtype=bool)
    clean_context = np.asarray(source_archive["clean_context"], dtype=float)
    candidates = _load_saved_candidates(source_archive, observed_mask)
    _complete_stateless_baselines(clean_context, observed_mask, candidates, baseline_ids)
    period = int(np.asarray(source_archive.get("period", 1)).reshape(-1)[0])
    has_tail_missing = bool((~observed_mask[-1]).any())
    for candidate_id, candidate_metadata in candidates.items():
        candidate_spec = DEFAULT_REGISTRY.get_spec(candidate_id)
        if has_tail_missing and not candidate_spec.supports_tail:
            candidate_metadata["native_valid"] = False
        elif candidate_spec.requires_period and period < 2:
            candidate_metadata["native_valid"] = False
    assembled_method = _assembled_method_id(source_record, source_archive)
    expected = {"clean", assembled_method, "oracle", *candidates}
    if set(rows) != expected:
        raise ValueError(
            f"shared evaluation method set differs from its imputation artifact: {episode_id}"
        )
    expected_seed = stable_seed(config.seed, forecaster_id, episode_id, "evaluation")
    for method_id, row in rows.items():
        if str(row.get("episode_id")) != episode_id:
            raise ValueError("shared evaluation metrics contain inconsistent episode metadata")
        if int(row.get("forecast_seed", -1)) != expected_seed:
            raise ValueError("shared evaluation forecast seed differs from the requested spec")
        if str(row.get("routing_artifact_forecaster_id", "")) != str(
            source_record.get("forecaster_id", "")
        ):
            raise ValueError(
                "shared evaluation routing provenance differs from its imputation artifact"
            )
        if method_id in candidates:
            expected_valid = bool(candidates[method_id]["native_valid"])
            if bool(row.get("native_valid")) != expected_valid:
                raise ValueError("shared candidate validity differs from its imputation artifact")
    eligible = [
        method_id for method_id, candidate in candidates.items() if candidate["native_valid"]
    ]
    if not eligible:
        raise ValueError("shared evaluation episode has no valid oracle candidate")
    expected_oracle = min(
        eligible,
        key=lambda method_id: (float(rows[method_id]["mase"]), method_id),
    )
    oracle = rows["oracle"]
    if oracle.get("oracle_source") != expected_oracle:
        raise ValueError("shared evaluation oracle is inconsistent with candidate rows")
    for metric in ("mase", "mae", "rmse", "imputation_mae", "imputation_rmse"):
        if oracle.get(metric) != rows[expected_oracle].get(metric):
            raise ValueError("shared evaluation oracle metrics differ from its source candidate")


def _prepare_shared_evaluation(
    shared_artifact: str | Path,
    *,
    target_root: Path,
    target_assignments: Path,
    target_npz_hashes: Mapping[str, str],
    expected_spec: Mapping[str, Any],
    config: AppConfig,
    forecaster_id: str,
    baseline_ids: Sequence[str],
    shared_reference_only: bool,
) -> _SharedEvaluation:
    shared_root = Path(shared_artifact).resolve()
    if shared_root.is_file() and shared_root.name == "evaluation_manifest.json":
        shared_root = shared_root.parent
    manifest_path = shared_root / "evaluation_manifest.json"
    metrics_path = shared_root / "episode_metrics.jsonl"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "shared evaluation artifact must contain evaluation_manifest.json and "
            "episode_metrics.jsonl"
        )
    manifest_sha256 = _file_sha256(manifest_path)
    metrics_sha256 = _file_sha256(metrics_path)
    manifest = _read_json_object(manifest_path, description="shared evaluation manifest")
    if manifest.get("status") != "completed":
        raise ValueError("shared evaluation artifact is not completed")
    recorded_metrics_sha256 = manifest.get("episode_metrics_jsonl_sha256")
    if not isinstance(recorded_metrics_sha256, str) or not recorded_metrics_sha256.strip():
        raise ValueError("shared evaluation manifest has no episode_metrics.jsonl SHA-256")
    if recorded_metrics_sha256.lower() != metrics_sha256:
        raise ValueError(
            "shared evaluation episode_metrics.jsonl SHA-256 differs from its manifest"
        )
    recorded_metrics_size = manifest.get("episode_metrics_jsonl_size_bytes")
    if recorded_metrics_size is not None:
        try:
            expected_metrics_size = int(recorded_metrics_size)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "shared evaluation manifest has an invalid episode_metrics.jsonl size"
            ) from error
        if expected_metrics_size < 0 or metrics_path.stat().st_size != expected_metrics_size:
            raise ValueError(
                "shared evaluation episode_metrics.jsonl size differs from its manifest"
            )
    if manifest.get("forecaster_id") != forecaster_id:
        raise ValueError("shared evaluation forecaster ID differs from the requested forecaster")
    signature = manifest.get("evaluation_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("shared evaluation manifest has no evaluation signature")
    for field, expected in expected_spec.items():
        if signature.get(field) != expected:
            raise ValueError(f"shared evaluation spec differs for {field}")
    if manifest.get("forecaster_artifact") != expected_spec["forecaster_artifact"]:
        raise ValueError(
            "shared evaluation forecaster artifact differs from the requested artifact"
        )

    source_value = manifest.get("impute_artifact")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError("shared evaluation manifest has no source imputation artifact")
    source_root = Path(source_value).resolve()
    if signature.get("impute_artifact") != str(source_root):
        raise ValueError("shared evaluation source path differs from its evaluation signature")
    source_assignments = source_root / "routing_assignments.jsonl"
    source_imputations = source_root / "imputations"
    if not source_assignments.is_file() or not source_imputations.is_dir():
        raise FileNotFoundError("shared evaluation source imputation artifact is unavailable")
    source_content, source_hashes = _impute_content_signature(source_root, source_assignments)
    _verify_imputation_npz_integrity(source_imputations, source_hashes)
    if signature.get("impute_content") != source_content:
        raise ValueError("shared evaluation source content differs from its evaluation signature")

    target_records = _load_assignment_records(target_assignments, target_npz_hashes)
    source_records = _load_assignment_records(source_assignments, source_hashes)
    rows_by_episode = _load_shared_metric_rows(metrics_path, forecaster_id=forecaster_id)
    episode_ids = set(target_records)
    if episode_ids != set(source_records) or episode_ids != set(rows_by_episode):
        raise ValueError(
            "shared evaluation episode set differs from the target imputation artifact"
        )
    if int(manifest.get("total_rows", -1)) != sum(len(rows) for rows in rows_by_episode.values()):
        raise ValueError("shared evaluation row count differs from its completed manifest")

    target_imputations = target_root / "imputations"
    for episode_id in sorted(episode_ids):
        target_record = target_records[episode_id]
        source_record = source_records[episode_id]
        _validate_shared_assignment_content(
            target_record,
            source_record,
            episode_id=episode_id,
            allow_routing_identity_difference=shared_reference_only,
        )
        target_file = Path(str(target_record["file"]))
        source_file = Path(str(source_record["file"]))
        with (
            np.load(target_imputations / target_file, allow_pickle=False) as target_archive,
            np.load(source_imputations / source_file, allow_pickle=False) as source_archive,
        ):
            _validate_shared_imputation_content(
                target_archive,
                source_archive,
                episode_id=episode_id,
                allow_routing_identity_difference=shared_reference_only,
            )
            _validate_shared_rows_for_episode(
                rows_by_episode[episode_id],
                source_record,
                source_archive,
                config=config,
                forecaster_id=forecaster_id,
                baseline_ids=baseline_ids,
            )

    if (
        _file_sha256(manifest_path) != manifest_sha256
        or _file_sha256(metrics_path) != metrics_sha256
    ):
        raise ValueError("shared evaluation artifact changed during validation")

    return _SharedEvaluation(
        root=shared_root,
        manifest_sha256=manifest_sha256,
        metrics_sha256=metrics_sha256,
        rows_by_episode=rows_by_episode,
    )


def _assembled_method_id(
    record: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> str:
    """Resolve the assembled selector identity from new or legacy artifacts."""

    record_value = record.get("assembled_method_id")
    archive_value = (
        np.asarray(archive["assembled_method_id"]).reshape(-1)[0]
        if "assembled_method_id" in archive
        else None
    )
    resolved: list[str] = []
    for source, value in (("record", record_value), ("archive", archive_value)):
        if value is None:
            continue
        method_id = str(value).strip()
        if not method_id:
            raise ValueError(f"{source} assembled_method_id must be non-empty")
        resolved.append(method_id)
    if len(set(resolved)) > 1:
        raise ValueError("record and archive assembled_method_id values differ")
    method_id = resolved[0] if resolved else "b_fais"
    if method_id in {"clean", "oracle"}:
        raise ValueError(f"assembled_method_id is reserved: {method_id!r}")
    return method_id


def _optional_archive_scalar(archive: Mapping[str, Any], field: str) -> Any | None:
    if field not in archive:
        return None
    values = np.asarray(archive[field]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"imputation archive field {field!r} must be scalar")
    value = values[0]
    return value.item() if hasattr(value, "item") else value


def _routing_artifact_forecaster_id(
    record: Mapping[str, Any],
    archive: Mapping[str, Any],
    *,
    requested_forecaster_id: str,
) -> str:
    """Validate dependent or selector-independent routing identity."""

    registry = default_forecast_registry()
    requested_forecaster = registry.get(requested_forecaster_id)
    record_id = record.get("forecaster_id")
    archive_id = _optional_archive_scalar(archive, "forecaster_id")
    ids = [str(value).strip() for value in (record_id, archive_id) if value not in {None, ""}]
    if len(set(ids)) > 1:
        raise ValueError("record and archive routing forecaster IDs differ")
    routing_model = ids[0] if ids else ""

    record_mode = record.get("forecast_mode")
    archive_mode = _optional_archive_scalar(archive, "forecast_mode")
    modes = [str(value).strip() for value in (record_mode, archive_mode) if value not in {None, ""}]
    if len(set(modes)) > 1:
        raise ValueError("record and archive routing forecast modes differ")
    routing_mode = modes[0] if modes else ""

    record_independent = record.get("forecaster_independent_selection")
    archive_independent = _optional_archive_scalar(archive, "forecaster_independent_selection")
    routing_metadata = record.get("routing_metadata")
    metadata_independent = is_independent_sequence_routing_metadata(routing_metadata)
    claims_independent = (
        routing_model == SELECTOR_INDEPENDENT_FORECASTER_ID
        or routing_mode == SELECTOR_INDEPENDENT_FORECAST_MODE
        or record_independent is True
        or archive_independent is True
        or metadata_independent
    )
    if claims_independent:
        if (
            record_id != SELECTOR_INDEPENDENT_FORECASTER_ID
            or archive_id != SELECTOR_INDEPENDENT_FORECASTER_ID
            or record_mode != SELECTOR_INDEPENDENT_FORECAST_MODE
            or archive_mode != SELECTOR_INDEPENDENT_FORECAST_MODE
            or record_independent is not True
            or archive_independent is not True
            or not metadata_independent
        ):
            raise ValueError(
                "selector-independent routing identity lacks strict sequence protocol metadata"
            )
        assembled_method = _assembled_method_id(record, archive)
        assert isinstance(routing_metadata, Mapping)
        if assembled_method != str(routing_metadata["selector_method"]):
            raise ValueError("selector-independent assembled method differs from routing metadata")
        return SELECTOR_INDEPENDENT_FORECASTER_ID

    if not routing_model:
        return ""
    try:
        routing_forecaster = registry.get(routing_model)
    except KeyError as error:
        raise ValueError(
            f"episode {record.get('episode_id')} records unknown routing "
            f"forecaster {routing_model!r}"
        ) from error
    if routing_mode and routing_mode != routing_forecaster.mode:
        raise ValueError(
            f"episode {record.get('episode_id')} records routing mode "
            f"{routing_mode!r}, expected {routing_forecaster.mode!r}"
        )
    if routing_forecaster.mode != requested_forecaster.mode:
        raise ValueError(
            f"episode {record.get('episode_id')} was routed for "
            f"{routing_model!r} ({routing_forecaster.mode}), which is "
            f"incompatible with {requested_forecaster_id!r} "
            f"({requested_forecaster.mode})"
        )
    return routing_model


def _evaluate_episode(
    record: Mapping[str, Any],
    archive: Mapping[str, Any],
    config: AppConfig,
    model_id: str,
    predictor: ForecastPredictor,
    baseline_ids: Sequence[str],
) -> list[dict[str, Any]]:
    required = (
        "schema_version",
        "values",
        "observed_mask",
        "clean_context",
        "clean_future",
        "mask_protocol",
        "mask_seed",
        "mask_realization_id",
        "target_missing_rate",
        "global_missing_rate",
        "local_missing_rate",
        "mase_scale",
        "mase_scale_lag",
    )
    missing_fields = [field for field in required if field not in archive]
    if missing_fields:
        raise ValueError(
            "imputation artifact is not evaluation-ready; missing: " + ", ".join(missing_fields)
        )
    assembled = np.asarray(archive["values"], dtype=float)
    observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
    clean_context = np.asarray(archive["clean_context"], dtype=float)
    clean_future = np.asarray(archive["clean_future"], dtype=float)
    schema_version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
    mask_protocol = str(np.asarray(archive["mask_protocol"]).reshape(-1)[0])
    if schema_version != 3 or mask_protocol != "sequence_mask_v2":
        raise ValueError("imputation artifact does not use sequence_mask_v2 schema 3")
    mase_scale = np.asarray(archive["mase_scale"], dtype=float).reshape(-1)
    if not (
        assembled.shape == observed_mask.shape == clean_context.shape
        and clean_future.ndim == 2
        and clean_future.shape[1] == clean_context.shape[1]
    ):
        raise ValueError("imputation artifact contains incompatible episode shapes")
    if (
        not np.isfinite(assembled).all()
        or not np.isfinite(clean_context).all()
        or not np.isfinite(clean_future).all()
    ):
        raise ValueError("evaluation contexts must be finite")

    period = int(np.asarray(archive.get("period", 1)).reshape(-1)[0])
    candidates = _load_saved_candidates(archive, observed_mask)
    _complete_stateless_baselines(clean_context, observed_mask, candidates, baseline_ids)
    has_tail_missing = bool((~observed_mask[-1]).any())
    for candidate_id, candidate_metadata in candidates.items():
        candidate_spec = DEFAULT_REGISTRY.get_spec(candidate_id)
        if has_tail_missing and not candidate_spec.supports_tail:
            candidate_metadata["native_valid"] = False
            candidate_metadata["ineligibility_reason"] = "unsupported_tail"
        elif candidate_spec.requires_period and period < 2:
            candidate_metadata["native_valid"] = False
            candidate_metadata["ineligibility_reason"] = "missing_period"
    candidate_ids = tuple(sorted(candidates))
    assembled_method_id = _assembled_method_id(record, archive)
    if assembled_method_id in candidates:
        raise ValueError(
            f"assembled_method_id collides with a saved candidate ID: {assembled_method_id!r}"
        )
    routing_metadata = record.get("routing_metadata")
    assembled_native_valid = not (
        isinstance(routing_metadata, Mapping)
        and routing_metadata.get("paper_native_valid") is False
    )
    assembled_ineligibility_reason = (
        None
        if assembled_native_valid or not isinstance(routing_metadata, Mapping)
        else str(
            routing_metadata.get(
                "paper_ineligibility_reason",
                "paper_method_required_a_safety_fallback",
            )
        )
    )
    method_ids = ("clean", assembled_method_id, *candidate_ids)
    valid_candidate_ids = tuple(
        candidate_id for candidate_id in candidate_ids if candidates[candidate_id]["native_valid"]
    )
    forecast_method_ids = (
        "clean",
        *((assembled_method_id,) if assembled_native_valid else ()),
        *valid_candidate_ids,
    )
    contexts = np.stack(
        (
            clean_context,
            *((assembled,) if assembled_native_valid else ()),
            *(candidates[key]["values"] for key in valid_candidate_ids),
        )
    )
    spec = _forecast_spec(config, model_id, clean_context.shape[1])
    forecast_seed = stable_seed(
        config.seed,
        model_id,
        str(record["episode_id"]),
        "evaluation",
    )
    _set_forecast_seed(forecast_seed)
    forecast = predictor.predict(contexts, spec)
    if forecast.point.shape[0] != len(forecast_method_ids):
        raise ValueError("forecaster did not return one prediction per evaluation method")
    metrics = forecast_metrics(
        clean_context,
        clean_future,
        forecast,
        seasonality=max(1, period),
        mase_scale=mase_scale,
    )

    metadata = _episode_metadata(record)
    metadata.update(
        {
            "mask_protocol": mask_protocol,
            "mask_seed": int(np.asarray(archive["mask_seed"]).reshape(-1)[0]),
            "mask_realization_id": str(np.asarray(archive["mask_realization_id"]).reshape(-1)[0]),
            "target_missing_rate": float(np.asarray(archive["target_missing_rate"]).reshape(-1)[0]),
            "global_missing_rate": float(np.asarray(archive["global_missing_rate"]).reshape(-1)[0]),
            "local_missing_rate": float(np.asarray(archive["local_missing_rate"]).reshape(-1)[0]),
            "contains_missing": bool((~observed_mask).any()),
            "mase_scale_lag": int(np.asarray(archive["mase_scale_lag"]).reshape(-1)[0]),
        }
    )
    metrics_by_method = {
        method_id: {
            metric_name: float(metric_values[index])
            for metric_name, metric_values in metrics.items()
        }
        for index, method_id in enumerate(forecast_method_ids)
    }
    clean_mase = metrics_by_method["clean"]["mase"]
    clean_denominator = max(abs(clean_mase), 1e-8)
    pipeline_runtime = float(
        np.asarray(archive.get("pipeline_runtime_seconds", 0.0)).reshape(-1)[0]
    )
    pipeline_rss_delta = int(np.asarray(archive.get("pipeline_rss_delta_bytes", 0)).reshape(-1)[0])
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        candidate = candidates.get(method_id)
        if method_id == "clean":
            method_role = "reference"
            candidate_status = "reference"
        elif method_id == assembled_method_id:
            method_role = "method" if assembled_method_id == "b_fais" else "selector_baseline"
            candidate_status = "assembled" if assembled_native_valid else "assembled_fallback"
        else:
            if candidate is None:  # pragma: no cover - method_ids construction guard
                raise RuntimeError(f"missing candidate metadata for {method_id!r}")
            method_role = "missing_anchor" if method_id == "locf" else "baseline"
            candidate_status = candidate["status"]
        if method_id == "clean":
            runtime_seconds = 0.0
            rss_delta_bytes = 0
            runtime_scope = "reference"
        elif method_id == assembled_method_id:
            runtime_seconds = pipeline_runtime
            rss_delta_bytes = pipeline_rss_delta
            runtime_scope = "end_to_end_imputation"
        else:
            assert candidate is not None
            runtime_seconds = float(candidate["runtime_seconds"])
            rss_delta_bytes = int(candidate["rss_delta_bytes"])
            runtime_scope = "single_imputer"
        metric_eligible = (
            assembled_native_valid
            if method_id == assembled_method_id
            else candidate is None or bool(candidate["native_valid"])
        )
        method_metrics = metrics_by_method.get(method_id)
        imputation_mae: float | None
        imputation_rmse: float | None
        degradation: float | None
        relative_degradation: float | None
        if metric_eligible:
            context = (
                clean_context
                if method_id == "clean"
                else assembled
                if method_id == assembled_method_id
                else candidates[method_id]["values"]
            )
            imputation_mae, imputation_rmse = _imputation_metrics(
                clean_context, context, observed_mask
            )
            assert method_metrics is not None
            degradation = method_metrics["mase"] - clean_mase
            relative_degradation = degradation / clean_denominator
        else:
            imputation_mae = imputation_rmse = None
            degradation = relative_degradation = None
        rows.append(
            {
                "schema_version": 1,
                **metadata,
                "forecaster_id": model_id,
                "routing_forecaster_id": (
                    metadata["routing_artifact_forecaster_id"]
                    if method_id == assembled_method_id
                    else model_id
                ),
                "method": method_id,
                "method_role": method_role,
                "oracle_source": None,
                "oracle_eligible": bool(candidate and candidate["native_valid"]),
                "native_valid": (
                    assembled_native_valid
                    if method_id == assembled_method_id
                    else True
                    if candidate is None
                    else candidate["native_valid"]
                ),
                "metric_eligible": metric_eligible,
                "ineligibility_reason": (
                    assembled_ineligibility_reason
                    if method_id == assembled_method_id
                    else None
                    if candidate is None
                    else candidate["ineligibility_reason"]
                ),
                "candidate_status": candidate_status,
                "mase": None if method_metrics is None else method_metrics["mase"],
                "mae": None if method_metrics is None else method_metrics["mae"],
                "rmse": None if method_metrics is None else method_metrics["rmse"],
                "imputation_mae": imputation_mae,
                "imputation_rmse": imputation_rmse,
                "degradation_vs_clean_mase": degradation,
                "relative_degradation_vs_clean": relative_degradation,
                "regret_mase": 0.0,
                "relative_regret": 0.0,
                "runtime_seconds": runtime_seconds,
                "rss_delta_bytes": rss_delta_bytes,
                "runtime_scope": runtime_scope,
                "forecast_seed": forecast_seed,
            }
        )

    eligible = [
        row
        for row in rows
        if row["method_role"] in {"baseline", "missing_anchor"} and row["oracle_eligible"]
    ]
    if not eligible:
        raise ValueError("episode has no natively valid single-imputer oracle candidate")
    oracle_source = min(eligible, key=lambda row: (row["mase"], row["method"]))
    oracle_mase = float(oracle_source["mase"])
    denominator = max(abs(oracle_mase), 1e-8)
    for row in rows:
        row["oracle_source"] = oracle_source["method"]
        if row["mase"] is None:
            row["regret_mase"] = None
            row["relative_regret"] = None
        else:
            regret = float(row["mase"] - oracle_mase)
            row["regret_mase"] = regret
            row["relative_regret"] = regret / denominator
    oracle = dict(oracle_source)
    oracle.update(
        {
            "method": "oracle",
            "method_role": "oracle",
            "oracle_source": oracle_source["method"],
            "oracle_eligible": True,
            "metric_eligible": True,
            "candidate_status": "oracle",
            "regret_mase": 0.0,
            "relative_regret": 0.0,
        }
    )
    rows.append(oracle)
    return rows


def _evaluate_episode_with_shared_rows(
    record: Mapping[str, Any],
    archive: Mapping[str, Any],
    config: AppConfig,
    model_id: str,
    predictor: ForecastPredictor | None,
    baseline_ids: Sequence[str],
    shared_rows: Mapping[str, Mapping[str, Any]],
    *,
    shared_reference_only: bool,
) -> list[dict[str, Any]]:
    """Evaluate only the assembled context and reuse common forecast rows."""

    required = (
        "schema_version",
        "values",
        "observed_mask",
        "clean_context",
        "clean_future",
        "mask_protocol",
        "mask_seed",
        "mask_realization_id",
        "target_missing_rate",
        "global_missing_rate",
        "local_missing_rate",
        "mase_scale",
        "mase_scale_lag",
    )
    missing_fields = [field for field in required if field not in archive]
    if missing_fields:
        raise ValueError(
            "imputation artifact is not evaluation-ready; missing: " + ", ".join(missing_fields)
        )
    assembled = np.asarray(archive["values"], dtype=float)
    observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
    clean_context = np.asarray(archive["clean_context"], dtype=float)
    clean_future = np.asarray(archive["clean_future"], dtype=float)
    schema_version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
    mask_protocol = str(np.asarray(archive["mask_protocol"]).reshape(-1)[0])
    if schema_version != 3 or mask_protocol != "sequence_mask_v2":
        raise ValueError("imputation artifact does not use sequence_mask_v2 schema 3")
    mase_scale = np.asarray(archive["mase_scale"], dtype=float).reshape(-1)
    if not (
        assembled.shape == observed_mask.shape == clean_context.shape
        and clean_future.ndim == 2
        and clean_future.shape[1] == clean_context.shape[1]
    ):
        raise ValueError("imputation artifact contains incompatible episode shapes")
    if (
        not np.isfinite(assembled).all()
        or not np.isfinite(clean_context).all()
        or not np.isfinite(clean_future).all()
    ):
        raise ValueError("evaluation contexts must be finite")

    period = int(np.asarray(archive.get("period", 1)).reshape(-1)[0])
    candidates = _load_saved_candidates(archive, observed_mask)
    _complete_stateless_baselines(clean_context, observed_mask, candidates, baseline_ids)
    has_tail_missing = bool((~observed_mask[-1]).any())
    for candidate_id, candidate_metadata in candidates.items():
        candidate_spec = DEFAULT_REGISTRY.get_spec(candidate_id)
        if has_tail_missing and not candidate_spec.supports_tail:
            candidate_metadata["native_valid"] = False
            candidate_metadata["ineligibility_reason"] = "unsupported_tail"
        elif candidate_spec.requires_period and period < 2:
            candidate_metadata["native_valid"] = False
            candidate_metadata["ineligibility_reason"] = "missing_period"

    assembled_method_id = _assembled_method_id(record, archive)
    if assembled_method_id in candidates:
        raise ValueError(
            f"assembled_method_id collides with a saved candidate ID: {assembled_method_id!r}"
        )
    routing_metadata = record.get("routing_metadata")
    assembled_native_valid = not (
        isinstance(routing_metadata, Mapping)
        and routing_metadata.get("paper_native_valid") is False
    )
    assembled_ineligibility_reason = (
        None
        if assembled_native_valid or not isinstance(routing_metadata, Mapping)
        else str(
            routing_metadata.get(
                "paper_ineligibility_reason",
                "paper_method_required_a_safety_fallback",
            )
        )
    )

    episode_id = str(record["episode_id"])
    forecast_seed = stable_seed(config.seed, model_id, episode_id, "evaluation")
    assembled_metrics: dict[str, float] | None = None
    if assembled_native_valid:
        if predictor is None:
            raise RuntimeError("shared evaluation requires a predictor for a valid assembled row")
        spec = _forecast_spec(config, model_id, clean_context.shape[1])
        valid_candidate_ids = tuple(
            candidate_id
            for candidate_id in sorted(candidates)
            if candidates[candidate_id]["native_valid"]
        )
        # Keep the target method in the same batch position and batch shape used by
        # a complete evaluation. Some forecasters have small batch-size-dependent
        # numerical differences, so a singleton call would make shared and full
        # evaluations incomparable even when their assembled contexts are equal.
        aligned_contexts = np.stack(
            (
                clean_context,
                assembled,
                *(candidates[key]["values"] for key in valid_candidate_ids),
            )
        )
        _set_forecast_seed(forecast_seed)
        forecast = predictor.predict(aligned_contexts, spec)
        if forecast.point.shape[0] != len(aligned_contexts):
            raise ValueError("forecaster did not return one prediction per aligned context")
        metric_arrays = forecast_metrics(
            clean_context,
            clean_future,
            forecast,
            seasonality=max(1, period),
            mase_scale=mase_scale,
        )
        assembled_metrics = {
            metric_name: float(metric_values[1])
            for metric_name, metric_values in metric_arrays.items()
        }

    metadata = _episode_metadata(record)
    metadata.update(
        {
            "mask_protocol": mask_protocol,
            "mask_seed": int(np.asarray(archive["mask_seed"]).reshape(-1)[0]),
            "mask_realization_id": str(np.asarray(archive["mask_realization_id"]).reshape(-1)[0]),
            "target_missing_rate": float(np.asarray(archive["target_missing_rate"]).reshape(-1)[0]),
            "global_missing_rate": float(np.asarray(archive["global_missing_rate"]).reshape(-1)[0]),
            "local_missing_rate": float(np.asarray(archive["local_missing_rate"]).reshape(-1)[0]),
            "contains_missing": bool((~observed_mask).any()),
            "mase_scale_lag": int(np.asarray(archive["mase_scale_lag"]).reshape(-1)[0]),
        }
    )
    shared_candidate_ids = tuple(
        sorted(
            method_id
            for method_id, row in shared_rows.items()
            if row.get("method_role") in {"baseline", "missing_anchor"}
        )
    )
    candidate_ids = shared_candidate_ids if shared_reference_only else tuple(sorted(candidates))
    expected_shared = {"clean", "oracle", *candidate_ids}
    available_shared = {
        method_id
        for method_id, row in shared_rows.items()
        if row.get("method_role") in {"reference", "baseline", "missing_anchor", "oracle"}
    }
    if available_shared != expected_shared:
        raise ValueError("shared evaluation common method rows are incomplete")

    common_rows: dict[str, dict[str, Any]] = {}
    for method_id in expected_shared:
        row = dict(shared_rows[method_id])
        if not shared_reference_only:
            row.update(metadata)
            row.update(
                {
                    "forecaster_id": model_id,
                    "routing_forecaster_id": model_id,
                    "routing_artifact_forecaster_id": metadata["routing_artifact_forecaster_id"],
                    "forecast_seed": forecast_seed,
                }
            )
        common_rows[method_id] = row

    clean_mase = float(common_rows["clean"]["mase"])
    clean_denominator = max(abs(clean_mase), 1e-8)
    oracle_mase = float(common_rows["oracle"]["mase"])
    oracle_denominator = max(abs(oracle_mase), 1e-8)
    imputation_mae: float | None
    imputation_rmse: float | None
    degradation: float | None
    relative_degradation: float | None
    regret: float | None
    relative_regret: float | None
    if assembled_native_valid:
        assert assembled_metrics is not None
        imputation_mae, imputation_rmse = _imputation_metrics(
            clean_context,
            assembled,
            observed_mask,
        )
        degradation = assembled_metrics["mase"] - clean_mase
        relative_degradation = degradation / clean_denominator
        regret = assembled_metrics["mase"] - oracle_mase
        relative_regret = regret / oracle_denominator
    else:
        imputation_mae = imputation_rmse = None
        degradation = relative_degradation = None
        regret = relative_regret = None

    pipeline_runtime = float(
        np.asarray(archive.get("pipeline_runtime_seconds", 0.0)).reshape(-1)[0]
    )
    pipeline_rss_delta = int(np.asarray(archive.get("pipeline_rss_delta_bytes", 0)).reshape(-1)[0])
    assembled_row = {
        "schema_version": 1,
        **metadata,
        "forecaster_id": model_id,
        "routing_forecaster_id": metadata["routing_artifact_forecaster_id"],
        "method": assembled_method_id,
        "method_role": "method" if assembled_method_id == "b_fais" else "selector_baseline",
        "oracle_source": common_rows["oracle"]["oracle_source"],
        "oracle_eligible": False,
        "native_valid": assembled_native_valid,
        "metric_eligible": assembled_native_valid,
        "ineligibility_reason": assembled_ineligibility_reason,
        "candidate_status": "assembled" if assembled_native_valid else "assembled_fallback",
        "mase": None if assembled_metrics is None else assembled_metrics["mase"],
        "mae": None if assembled_metrics is None else assembled_metrics["mae"],
        "rmse": None if assembled_metrics is None else assembled_metrics["rmse"],
        "imputation_mae": imputation_mae,
        "imputation_rmse": imputation_rmse,
        "degradation_vs_clean_mase": degradation,
        "relative_degradation_vs_clean": relative_degradation,
        "regret_mase": regret,
        "relative_regret": relative_regret,
        "runtime_seconds": pipeline_runtime,
        "rss_delta_bytes": pipeline_rss_delta,
        "runtime_scope": "end_to_end_imputation",
        "forecast_seed": forecast_seed,
    }
    return [
        common_rows["clean"],
        assembled_row,
        *(common_rows[method_id] for method_id in candidate_ids),
        common_rows["oracle"],
    ]


def evaluate_imputations(
    *,
    config: AppConfig,
    impute_artifact: str | Path,
    forecaster_id: str,
    forecaster_artifact: str | Path | None,
    output_dir: str | Path,
    baseline_ids: Sequence[str] = ("locf", "linear_interp"),
    resume: bool = False,
    forecast_runner: ForecastPredictor | None = None,
    shared_evaluation_artifact: str | Path | None = None,
    shared_reference_only: bool = False,
) -> dict[str, Any]:
    """Evaluate imputed contexts with one frozen TSFM and stream durable rows."""

    if shared_reference_only and shared_evaluation_artifact is None:
        raise ValueError("shared_reference_only requires shared_evaluation_artifact")
    baseline_tuple = parse_ids(tuple(baseline_ids))
    registry = default_forecast_registry()
    requested_forecaster = registry.get(forecaster_id)
    resolved_device = _resolve_forecast_device(config)
    root = Path(impute_artifact).resolve()
    if root.is_file() and root.name == "imputation_manifest.json":
        root = root.parent
    assignments = root / "routing_assignments.jsonl"
    imputations = root / "imputations"
    if not assignments.is_file() or not imputations.is_dir():
        raise FileNotFoundError(
            f"impute artifact must contain routing_assignments.jsonl and imputations/: {root}"
        )
    predictor = forecast_runner
    if predictor is None and forecaster_artifact is None:
        raise ValueError("a local forecaster artifact is required for real evaluation")
    resolved_forecaster_artifact = (
        None
        if forecaster_artifact is None
        else _resolve_forecaster_artifact(forecaster_artifact, forecaster_id)
    )
    if predictor is None and resolved_forecaster_artifact is not None:
        if not resolved_forecaster_artifact.exists():
            raise FileNotFoundError(
                f"forecaster artifact does not exist: {resolved_forecaster_artifact}"
            )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "episode_metrics.jsonl"
    csv_path = output / "episode_metrics.csv"
    manifest_path = output / "evaluation_manifest.json"
    target_signature: str | list[int] = (
        "all"
        if config.experiment.target_indices == "all"
        else list(config.experiment.target_indices)
    )
    impute_content, imputation_npz_hashes = _impute_content_signature(root, assignments)
    _verify_imputation_npz_integrity(imputations, imputation_npz_hashes)
    evaluation_spec = {
        "forecaster_id": forecaster_id,
        "forecaster_artifact": (
            None if resolved_forecaster_artifact is None else str(resolved_forecaster_artifact)
        ),
        "baseline_ids": list(baseline_tuple),
        "context_length": config.experiment.context_length,
        "horizon": config.experiment.horizon,
        "target_indices": target_signature,
        "forecast_num_samples": config.experiment.forecast_num_samples,
        "forecast_batch_size": config.experiment.forecast_batch_size,
        "seed": config.seed,
        "mask_protocol": "sequence_mask_v2",
        "resolved_device": resolved_device,
        "forecast_call_protocol": FORECAST_CALL_PROTOCOL,
    }
    evaluation_signature: dict[str, Any] = {
        "schema_version": 2,
        "impute_artifact": str(root),
        "impute_content": impute_content,
        **evaluation_spec,
    }
    shared_evaluation = (
        None
        if shared_evaluation_artifact is None
        else _prepare_shared_evaluation(
            shared_evaluation_artifact,
            target_root=root,
            target_assignments=assignments,
            target_npz_hashes=imputation_npz_hashes,
            expected_spec=evaluation_spec,
            config=config,
            forecaster_id=forecaster_id,
            baseline_ids=baseline_tuple,
            shared_reference_only=shared_reference_only,
        )
    )
    if shared_evaluation is not None:
        evaluation_signature["shared_evaluation"] = shared_evaluation.signature()
    if shared_reference_only:
        evaluation_signature["shared_reference_only"] = True
    _validate_resume_signature(
        manifest_path,
        jsonl_path,
        evaluation_signature,
        resume=resume,
    )
    completed = _recover_completed(jsonl_path, resume=resume)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "impute_artifact": str(root),
        "forecaster_id": forecaster_id,
        "forecaster_artifact": (
            None if resolved_forecaster_artifact is None else str(resolved_forecaster_artifact)
        ),
        "forecaster_mode": requested_forecaster.mode,
        "routing_forecaster_ids": [],
        "runtime_device": config.runtime.device,
        "resolved_device": resolved_device,
        "forecast_call_protocol": FORECAST_CALL_PROTOCOL,
        "baseline_ids": list(baseline_tuple),
        "shared_evaluation_artifact": (
            None if shared_evaluation is None else str(shared_evaluation.root)
        ),
        "shared_evaluation_signature": (
            None if shared_evaluation is None else shared_evaluation.signature()
        ),
        "shared_reference_only": bool(shared_reference_only),
        "forecast_reuse_mode": (
            "none"
            if shared_evaluation is None
            else "shared_reference_only"
            if shared_reference_only
            else "common_rows_from_completed_evaluation"
        ),
        "resume": bool(resume),
        "existing_rows": len(completed),
        "metric_definition": {
            "mase": "target-macro MASE using a frozen historical-prefix scale",
            "mase_scale": (
                "stored per variate from the fit prefix; seasonal lag when available, "
                "otherwise lag one"
            ),
            "missing_anchor": "LOCF completion of the same corrupted context",
            "oracle": "lowest-MASE natively valid saved single-imputer candidate",
            "degradation_vs_clean_mase": "method_mase - clean_context_mase",
            "relative_degradation_vs_clean": (
                "(method_mase - clean_context_mase) / max(abs(clean_context_mase), 1e-8)"
            ),
            "relative_regret": "(method_mase - oracle_mase) / max(abs(oracle_mase), 1e-8)",
            "runtime_seconds": (
                "end-to-end imputation for the assembled method; one imputer call for candidates"
            ),
            "rss_delta_bytes": (
                "non-negative process RSS after-minus-before delta; not peak memory"
            ),
        },
        "forecast_num_samples": config.experiment.forecast_num_samples,
        "forecast_batch_size": config.experiment.forecast_batch_size,
        "evaluation_signature": evaluation_signature,
        "metric_eligibility": (
            "candidate fallback rows remain in episode outputs for diagnostics; "
            "grouped metric means include only metric_eligible rows"
        ),
    }
    _write_json(manifest_path, manifest)

    episodes_seen = episodes_evaluated = rows_written = 0
    routing_forecaster_ids: set[str] = set()
    try:
        with (
            assignments.open("r", encoding="utf-8") as assignments_handle,
            jsonl_path.open("a", encoding="utf-8") as output_handle,
        ):
            for line in assignments_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                episodes_seen += 1
                relative = Path(str(record["file"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe imputation artifact path: {relative}")
                artifact_path = imputations / relative
                if relative.as_posix() not in imputation_npz_hashes:
                    raise ValueError(f"imputation assignment is absent from progress: {relative}")
                with np.load(artifact_path, allow_pickle=False) as archive:
                    routing_model = _routing_artifact_forecaster_id(
                        record,
                        archive,
                        requested_forecaster_id=forecaster_id,
                    )
                    if routing_model:
                        routing_forecaster_ids.add(routing_model)
                    assembled_method_id = _assembled_method_id(record, archive)
                    episode_id = str(record["episode_id"])
                    if shared_reference_only:
                        assert shared_evaluation is not None
                        episode_shared_rows = shared_evaluation.rows_by_episode.get(episode_id)
                        if episode_shared_rows is None:  # pragma: no cover - preflight guard
                            raise ValueError(
                                f"shared evaluation has no rows for episode {episode_id}"
                            )
                        expected_methods = {
                            assembled_method_id,
                            *(
                                method_id
                                for method_id, row in episode_shared_rows.items()
                                if row.get("method_role")
                                in {"reference", "baseline", "missing_anchor", "oracle"}
                            ),
                        }
                    else:
                        saved_ids = (
                            tuple(
                                str(value)
                                for value in np.asarray(archive["candidate_ids"]).tolist()
                            )
                            if "candidate_ids" in archive
                            else ()
                        )
                        expected_methods = {
                            "clean",
                            assembled_method_id,
                            "oracle",
                            *saved_ids,
                            *baseline_tuple,
                        }
                    if all(
                        (forecaster_id, episode_id, method_id) in completed
                        for method_id in expected_methods
                    ):
                        continue
                    routing_metadata = record.get("routing_metadata")
                    assembled_native_valid = not (
                        isinstance(routing_metadata, Mapping)
                        and routing_metadata.get("paper_native_valid") is False
                    )
                    needs_predictor = shared_evaluation is None or assembled_native_valid
                    if predictor is None and needs_predictor:
                        assert resolved_forecaster_artifact is not None
                        predictor = _build_forecast_runner(
                            forecaster_id,
                            resolved_forecaster_artifact,
                            device=resolved_device,
                            batch_size=config.experiment.forecast_batch_size,
                        )
                    if shared_evaluation is None:
                        assert predictor is not None
                        rows = _evaluate_episode(
                            record,
                            archive,
                            config,
                            forecaster_id,
                            predictor,
                            baseline_tuple,
                        )
                    else:
                        episode_shared_rows = shared_evaluation.rows_by_episode.get(episode_id)
                        if episode_shared_rows is None:  # pragma: no cover - preflight guard
                            raise ValueError(
                                f"shared evaluation has no rows for episode {episode_id}"
                            )
                        rows = _evaluate_episode_with_shared_rows(
                            record,
                            archive,
                            config,
                            forecaster_id,
                            predictor,
                            baseline_tuple,
                            episode_shared_rows,
                            shared_reference_only=shared_reference_only,
                        )
                pending = [
                    row
                    for row in rows
                    if (
                        row["forecaster_id"],
                        row["episode_id"],
                        row["method"],
                    )
                    not in completed
                ]
                if not pending:
                    continue
                output_handle.write(
                    "".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in pending
                    )
                )
                output_handle.flush()
                for row in pending:
                    completed.add((row["forecaster_id"], row["episode_id"], row["method"]))
                rows_written += len(pending)
                episodes_evaluated += 1
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "episodes_seen": episodes_seen,
                "episodes_evaluated": episodes_evaluated,
                "rows_written": rows_written,
                "total_rows": len(completed),
                "routing_forecaster_ids": sorted(routing_forecaster_ids),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(manifest_path, manifest)
        raise

    _write_csv_from_jsonl(jsonl_path, csv_path)
    metrics_size = jsonl_path.stat().st_size
    metrics_sha256 = _file_sha256(jsonl_path)
    manifest.update(
        {
            "status": "completed",
            "episodes_seen": episodes_seen,
            "episodes_evaluated": episodes_evaluated,
            "rows_written": rows_written,
            "total_rows": len(completed),
            "routing_forecaster_ids": sorted(routing_forecaster_ids),
            "episode_metrics_jsonl": str(jsonl_path),
            "episode_metrics_jsonl_sha256": metrics_sha256,
            "episode_metrics_jsonl_size_bytes": metrics_size,
            "episode_metrics_csv": str(csv_path),
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


@dataclass
class _RunningStat:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("summary metrics must be finite")
        self.count += 1
        delta = number - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (number - self.mean)
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)

    def payload(self) -> dict[str, float | None]:
        if self.count == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        standard_deviation = (
            math.sqrt(max(0.0, self.m2 / (self.count - 1))) if self.count > 1 else 0.0
        )
        return {
            "mean": self.mean,
            "std": standard_deviation,
            "min": self.minimum,
            "max": self.maximum,
        }


def _summary_rows(
    source: Path,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> Iterable[dict[str, Any]]:
    aggregates: dict[tuple[Any, ...], dict[str, _RunningStat]] = {}
    row_counts: dict[tuple[Any, ...], int] = {}
    metric_counts: dict[tuple[Any, ...], int] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row.get(field) for field in group_by)
            if key not in aggregates:
                aggregates[key] = {metric: _RunningStat() for metric in metrics}
                row_counts[key] = 0
                metric_counts[key] = 0
            row_counts[key] += 1
            metric_eligible = bool(row.get("metric_eligible", row.get("native_valid", True)))
            if not metric_eligible:
                continue
            metric_counts[key] += 1
            for metric in metrics:
                aggregates[key][metric].update(float(row[metric]))
    if not aggregates:
        raise ValueError("evaluation metrics file is empty")
    for key in sorted(aggregates, key=lambda values: tuple(map(str, values))):
        result: dict[str, Any] = {field: value for field, value in zip(group_by, key, strict=True)}
        result["count"] = row_counts[key]
        result["metric_count"] = metric_counts[key]
        result["invalid_count"] = row_counts[key] - metric_counts[key]
        result["invalid_rate"] = result["invalid_count"] / row_counts[key]
        for metric, statistic in aggregates[key].items():
            for suffix, value in statistic.payload().items():
                result[f"{metric}_{suffix}"] = value
        yield result


def summarize_evaluation(
    *,
    metrics_path: str | Path,
    output_dir: str | Path,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> dict[str, Any]:
    """Stream JSONL rows into deterministic grouped JSON and CSV summaries."""

    source = Path(metrics_path).resolve()
    if source.is_dir():
        source = source / "episode_metrics.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"evaluation metrics do not exist: {source}")
    groups = parse_ids(tuple(group_by))
    metric_names = parse_ids(tuple(metrics))
    rows = list(_summary_rows(source, groups, metric_names))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "summary.json"
    csv_path = output / "summary.csv"
    payload = {
        "schema_version": 1,
        "source": str(source),
        "group_by": list(groups),
        "metrics": list(metric_names),
        "metric_eligibility": (
            "means and dispersion exclude rows with metric_eligible=false; "
            "count, metric_count, invalid_count, and invalid_rate report coverage"
        ),
        "group_count": len(rows),
        "groups": rows,
    }
    _write_json(json_path, payload)
    fieldnames = tuple(rows[0])
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "group_count": len(rows),
        "group_by": list(groups),
    }


__all__ = [
    "DEFAULT_GROUP_BY",
    "EVALUATION_FIELDS",
    "SUMMARY_METRICS",
    "evaluate_imputations",
    "forecast_metrics",
    "parse_ids",
    "summarize_evaluation",
]
