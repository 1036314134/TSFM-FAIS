"""Streaming downstream forecast evaluation and grouped result summaries."""

from __future__ import annotations

import csv
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

EVALUATION_FIELDS: tuple[str, ...] = (
    "schema_version",
    "episode_id",
    "dataset_id",
    "family_id",
    "forecaster_id",
    "routing_forecaster_id",
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


class ForecastPredictor(Protocol):
    def predict(
        self, contexts: np.ndarray, forecast_spec: ForecastSpec
    ) -> ForecastResult: ...


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


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
    error = np.asarray(candidate_context, dtype=float)[missing] - np.asarray(
        clean_context, dtype=float
    )[missing]
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
    if runtimes.shape != (len(identifiers),) or rss_deltas.shape != (
        len(identifiers),
    ):
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
                None
                if bool(result.native_valid_mask[0][missing].all())
                else "native_invalid"
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
) -> ForecastRunner:
    source = artifact.resolve()
    if not source.exists():
        raise FileNotFoundError(f"forecaster artifact does not exist: {source}")
    registry = default_forecast_registry()
    adapter = registry.build(model_id, model_name=str(source), device=device)
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
    with source.open("r", encoding="utf-8") as input_handle, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
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
        raise ValueError(
            "cannot safely resume evaluation without evaluation_manifest.json"
        )
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
            "imputation artifact is not evaluation-ready; missing: "
            + ", ".join(missing_fields)
        )
    b_fais = np.asarray(archive["values"], dtype=float)
    observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
    clean_context = np.asarray(archive["clean_context"], dtype=float)
    clean_future = np.asarray(archive["clean_future"], dtype=float)
    schema_version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
    mask_protocol = str(np.asarray(archive["mask_protocol"]).reshape(-1)[0])
    if schema_version != 3 or mask_protocol != "sequence_mask_v2":
        raise ValueError("imputation artifact does not use sequence_mask_v2 schema 3")
    mase_scale = np.asarray(archive["mase_scale"], dtype=float).reshape(-1)
    if not (
        b_fais.shape == observed_mask.shape == clean_context.shape
        and clean_future.ndim == 2
        and clean_future.shape[1] == clean_context.shape[1]
    ):
        raise ValueError("imputation artifact contains incompatible episode shapes")
    if (
        not np.isfinite(b_fais).all()
        or not np.isfinite(clean_context).all()
        or not np.isfinite(clean_future).all()
    ):
        raise ValueError("evaluation contexts must be finite")

    period = int(np.asarray(archive.get("period", 1)).reshape(-1)[0])
    candidates = _load_saved_candidates(archive, observed_mask)
    _complete_stateless_baselines(
        clean_context, observed_mask, candidates, baseline_ids
    )
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
    method_ids = ("clean", "b_fais", *candidate_ids)
    valid_candidate_ids = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if candidates[candidate_id]["native_valid"]
    )
    forecast_method_ids = ("clean", "b_fais", *valid_candidate_ids)
    contexts = np.stack(
        (
            clean_context,
            b_fais,
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
            "mask_realization_id": str(
                np.asarray(archive["mask_realization_id"]).reshape(-1)[0]
            ),
            "target_missing_rate": float(
                np.asarray(archive["target_missing_rate"]).reshape(-1)[0]
            ),
            "global_missing_rate": float(
                np.asarray(archive["global_missing_rate"]).reshape(-1)[0]
            ),
            "local_missing_rate": float(
                np.asarray(archive["local_missing_rate"]).reshape(-1)[0]
            ),
            "contains_missing": bool((~observed_mask).any()),
            "mase_scale_lag": int(
                np.asarray(archive["mase_scale_lag"]).reshape(-1)[0]
            ),
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
    pipeline_rss_delta = int(
        np.asarray(archive.get("pipeline_rss_delta_bytes", 0)).reshape(-1)[0]
    )
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        candidate = candidates.get(method_id)
        if method_id == "clean":
            method_role = "reference"
            candidate_status = "reference"
        elif method_id == "b_fais":
            method_role = "method"
            candidate_status = "assembled"
        else:
            if candidate is None:  # pragma: no cover - method_ids construction guard
                raise RuntimeError(f"missing candidate metadata for {method_id!r}")
            method_role = "missing_anchor" if method_id == "locf" else "baseline"
            candidate_status = candidate["status"]
        if method_id == "clean":
            runtime_seconds = 0.0
            rss_delta_bytes = 0
            runtime_scope = "reference"
        elif method_id == "b_fais":
            runtime_seconds = pipeline_runtime
            rss_delta_bytes = pipeline_rss_delta
            runtime_scope = "end_to_end_imputation"
        else:
            assert candidate is not None
            runtime_seconds = float(candidate["runtime_seconds"])
            rss_delta_bytes = int(candidate["rss_delta_bytes"])
            runtime_scope = "single_imputer"
        metric_eligible = candidate is None or bool(candidate["native_valid"])
        method_metrics = metrics_by_method.get(method_id)
        imputation_mae: float | None
        imputation_rmse: float | None
        degradation: float | None
        relative_degradation: float | None
        if metric_eligible:
            context = (
                clean_context
                if method_id == "clean"
                else b_fais
                if method_id == "b_fais"
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
                "method": method_id,
                "method_role": method_role,
                "oracle_source": None,
                "oracle_eligible": bool(candidate and candidate["native_valid"]),
                "native_valid": True if candidate is None else candidate["native_valid"],
                "metric_eligible": metric_eligible,
                "ineligibility_reason": (
                    None if candidate is None else candidate["ineligibility_reason"]
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
        if row["method_role"] in {"baseline", "missing_anchor"}
        and row["oracle_eligible"]
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
) -> dict[str, Any]:
    """Evaluate imputed contexts with one frozen TSFM and stream durable rows."""

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
    evaluation_signature = {
        "schema_version": 1,
        "impute_artifact": str(root),
        "forecaster_id": forecaster_id,
        "forecaster_artifact": (
            None
            if resolved_forecaster_artifact is None
            else str(resolved_forecaster_artifact)
        ),
        "baseline_ids": list(baseline_tuple),
        "context_length": config.experiment.context_length,
        "horizon": config.experiment.horizon,
        "target_indices": target_signature,
        "forecast_num_samples": config.experiment.forecast_num_samples,
        "seed": config.seed,
        "mask_protocol": "sequence_mask_v2",
        "resolved_device": resolved_device,
    }
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
            None
            if resolved_forecaster_artifact is None
            else str(resolved_forecaster_artifact)
        ),
        "forecaster_mode": requested_forecaster.mode,
        "routing_forecaster_ids": [],
        "runtime_device": config.runtime.device,
        "resolved_device": resolved_device,
        "baseline_ids": list(baseline_tuple),
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
                "(method_mase - clean_context_mase) / "
                "max(abs(clean_context_mase), 1e-8)"
            ),
            "relative_regret": "(method_mase - oracle_mase) / max(abs(oracle_mase), 1e-8)",
            "runtime_seconds": (
                "end-to-end imputation for b_fais; one imputer call for candidates"
            ),
            "rss_delta_bytes": (
                "non-negative process RSS after-minus-before delta; not peak memory"
            ),
        },
        "forecast_num_samples": config.experiment.forecast_num_samples,
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
        with assignments.open("r", encoding="utf-8") as assignments_handle, jsonl_path.open(
            "a", encoding="utf-8"
        ) as output_handle:
            for line in assignments_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                episodes_seen += 1
                recorded_model = record.get("forecaster_id")
                if recorded_model not in {None, ""}:
                    routing_model = str(recorded_model)
                    try:
                        routing_forecaster = registry.get(routing_model)
                    except KeyError as error:
                        raise ValueError(
                            f"episode {record.get('episode_id')} records unknown routing "
                            f"forecaster {routing_model!r}"
                        ) from error
                    if routing_forecaster.mode != requested_forecaster.mode:
                        raise ValueError(
                            f"episode {record.get('episode_id')} was routed for "
                            f"{routing_model!r} ({routing_forecaster.mode}), which is "
                            f"incompatible with {forecaster_id!r} "
                            f"({requested_forecaster.mode})"
                        )
                    routing_forecaster_ids.add(routing_model)
                relative = Path(str(record["file"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe imputation artifact path: {relative}")
                artifact_path = imputations / relative
                with np.load(artifact_path, allow_pickle=False) as archive:
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
                        "b_fais",
                        "oracle",
                        *saved_ids,
                        *baseline_tuple,
                    }
                    episode_id = str(record["episode_id"])
                    if all(
                        (forecaster_id, episode_id, method_id) in completed
                        for method_id in expected_methods
                    ):
                        continue
                    if predictor is None:
                        assert resolved_forecaster_artifact is not None
                        predictor = _build_forecast_runner(
                            forecaster_id,
                            resolved_forecaster_artifact,
                            device=resolved_device,
                        )
                    rows = _evaluate_episode(
                        record,
                        archive,
                        config,
                        forecaster_id,
                        predictor,
                        baseline_tuple,
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
                    completed.add(
                        (row["forecaster_id"], row["episode_id"], row["method"])
                    )
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
    manifest.update(
        {
            "status": "completed",
            "episodes_seen": episodes_seen,
            "episodes_evaluated": episodes_evaluated,
            "rows_written": rows_written,
            "total_rows": len(completed),
            "routing_forecaster_ids": sorted(routing_forecaster_ids),
            "episode_metrics_jsonl": str(jsonl_path),
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
            math.sqrt(max(0.0, self.m2 / (self.count - 1)))
            if self.count > 1
            else 0.0
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
            metric_eligible = bool(
                row.get("metric_eligible", row.get("native_valid", True))
            )
            if not metric_eligible:
                continue
            metric_counts[key] += 1
            for metric in metrics:
                aggregates[key][metric].update(float(row[metric]))
    if not aggregates:
        raise ValueError("evaluation metrics file is empty")
    for key in sorted(aggregates, key=lambda values: tuple(map(str, values))):
        result: dict[str, Any] = {
            field: value for field, value in zip(group_by, key, strict=True)
        }
        result["count"] = row_counts[key]
        result["metric_count"] = metric_counts[key]
        result["invalid_count"] = row_counts[key] - metric_counts[key]
        result["invalid_rate"] = (
            result["invalid_count"] / row_counts[key]
        )
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
