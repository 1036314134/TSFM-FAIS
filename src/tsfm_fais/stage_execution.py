"""Opt-in implementations of the four experiment stages.

The CLI imports this module only for ``run --execute``. Merely validating a
configuration or preparing a stage never loads data, fits a model, or invokes
a forecaster.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import joblib
import numpy as np
from pandas.tseries.frequencies import to_offset

from tsfm_fais.config import AppConfig
from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.data import (
    MaskingSpec,
    audit_dataset,
    build_episode,
    fit_prefix_end,
    load_dataset,
    load_manifest,
    mask_time_series,
    rolling_origins,
    stable_seed,
)
from tsfm_fais.experiment_sampling import (
    candidate_subset,
    connected_subset,
    deterministic_subset,
    evenly_spaced_subset,
)
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry
from tsfm_fais.imputers import (
    DEFAULT_REGISTRY,
    CandidateRunner,
    DatasetImputerArtifactStore,
    ImputerRegistry,
    failed_candidate_result,
)
from tsfm_fais.label_resume import (
    LabelEpisodeExpectation,
    LabelProgressStore,
    LabelResumeError,
    build_label_resume_identity,
)
from tsfm_fais.pipeline import BlockwiseFAIS, FAISResult, RoutePlan
from tsfm_fais.routing.blocks import build_block_graph
from tsfm_fais.routing.features import (
    block_features,
    candidate_features,
    merge_features,
    pair_features,
    proxy_features,
)
from tsfm_fais.routing.models import RouterBundle, RouterTrainer
from tsfm_fais.routing.teacher import TeacherBuilder
from tsfm_fais.stages import (
    StageInputs,
    StagePreparation,
    parse_forecaster_ids,
    utc_now,
)

_IMPUTATION_SCHEMA_VERSION = 3
_IMPUTATION_PROGRESS_SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _write_npz_atomic(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _append_jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    handle.write("\n")


def _accepted_datasets(audit_path: Path) -> dict[str, str | None]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        str(entry["dataset_id"]): entry.get("content_sha256")
        for entry in payload.get("datasets", [])
        if isinstance(entry, dict) and entry.get("accepted") is True
    }


def _datasets(config: AppConfig, audit_path: Path):
    accepted = _accepted_datasets(audit_path)
    manifest = load_manifest(config.registries.data_manifest)
    for spec in manifest.datasets:
        if not spec.enabled or spec.dataset_id not in accepted:
            continue
        items = load_dataset(spec)
        report = audit_dataset(spec, items)
        if not report.accepted:
            raise ValueError(
                f"dataset {spec.dataset_id} changed after its audit artifact: "
                f"{[issue.code for issue in report.issues]}"
            )
        expected_hash = accepted[spec.dataset_id]
        if expected_hash is not None and report.content_sha256 != expected_hash:
            raise ValueError(
                f"dataset {spec.dataset_id} content hash differs from its audit artifact"
            )
        selected_items = deterministic_subset(
            items,
            config.experiment.max_items_per_dataset,
            config.seed,
            spec.dataset_id,
            "items",
        )
        yield spec, selected_items


def _sequence_mask_seed(
    dataset_id: str,
    item_id: str,
    masking: MaskingSpec,
    configured_seed: int,
) -> int:
    """Derive a mask seed that is independent of every forecast origin."""

    return stable_seed(
        dataset_id,
        item_id,
        masking.mechanism,
        f"{masking.missing_rate:.17g}",
        int(configured_seed),
        "sequence_mask_v2",
    )


def _training_prefix_end(
    length: int,
    context_length: int,
    horizon: int,
    fraction: float,
) -> int:
    if length < context_length:
        raise ValueError("series is shorter than the imputer context")
    if length >= context_length + horizon:
        return fit_prefix_end(length, context_length, horizon, fraction)
    return min(length, max(context_length, int(np.floor(fraction * length))))


def _training_batch(
    items: Iterable[TimeSeriesItem],
    context_length: int,
    horizon: int,
    max_windows: int | None = None,
    *,
    dataset_id: str = "dataset",
    masking_specs: Iterable[MaskingSpec] | None = None,
    configured_seeds: Iterable[int] = (20260710,),
    fit_fraction: float = 0.2,
    training_stride: int = 24,
) -> SeriesBatch:
    selected = [item for item in items if len(item.values) >= context_length]
    if not selected:
        raise ValueError("no item has enough history for an imputer training window")
    dimensions = {item.values.shape[1] for item in selected}
    if len(dimensions) != 1:
        raise ValueError("training items must share the same variate dimension")
    specs = tuple(masking_specs or (MaskingSpec("independent_block", 0.2),))
    seeds = tuple(map(int, configured_seeds))
    if not specs or not seeds:
        raise ValueError("imputer training requires masking specs and seeds")
    if training_stride < 1:
        raise ValueError("training_stride must be positive")
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    window_ids: list[str] = []
    for item in selected:
        fit_end = _training_prefix_end(
            len(item.values), context_length, horizon, fit_fraction
        )
        starts = list(range(0, fit_end - context_length + 1, training_stride))
        final_start = fit_end - context_length
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        calibration = item.values[:fit_end]
        for spec in specs:
            for configured_seed in seeds:
                realization = mask_time_series(
                    item.values,
                    spec,
                    _sequence_mask_seed(
                        dataset_id, item.item_id, spec, configured_seed
                    ),
                    calibration_values=calibration,
                )
                for start in starts:
                    stop = start + context_length
                    windows.append(realization.values[start:stop])
                    masks.append(realization.observed_mask[start:stop])
                    window_ids.append(
                        f"{item.item_id}@{start}|{spec.mechanism}|"
                        f"{spec.missing_rate:g}|{configured_seed}"
                    )
    selected_windows = evenly_spaced_subset(
        tuple(zip(windows, masks, window_ids, strict=True)), max_windows
    )
    windows = [window for window, _, _ in selected_windows]
    masks = [mask for _, mask, _ in selected_windows]
    window_ids = [window_id for _, _, window_id in selected_windows]
    values = np.stack(windows)
    return SeriesBatch(
        values,
        np.stack(masks),
        item_ids=tuple(window_ids),
        metadata={"mask_protocol": "sequence_mask_v2"},
    )


def _training_statistics(batch: SeriesBatch) -> tuple[np.ndarray, np.ndarray]:
    matrix = batch.values.reshape(-1, batch.shape[2])
    medians = np.nanmedian(matrix, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("at least one variate has no observed training value")
    filled = np.where(np.isfinite(matrix), matrix, medians[None, :])
    centered = filled - np.mean(filled, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(
        centered,
        norms[None, :],
        out=np.zeros_like(centered),
        where=norms[None, :] > 0,
    )
    correlation = normalized.T @ normalized
    valid = np.flatnonzero(norms > 0)
    correlation[valid, valid] = 1.0
    return medians, np.clip(correlation, -1.0, 1.0)


def _training_mase_scale(
    values: np.ndarray,
    period: int,
) -> tuple[np.ndarray, int]:
    """Freeze one per-variate MASE scale from a historical training prefix."""

    history = np.asarray(values, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2 or not np.isfinite(history).all():
        raise ValueError("MASE scaling requires a complete [T,D] training prefix")
    requested = max(1, int(period))
    lag = requested if history.shape[0] > requested else 1
    scale = np.mean(np.abs(history[lag:] - history[:-lag]), axis=0)
    return np.maximum(scale, 1e-8), lag


def _selected_candidate_ids(config: AppConfig) -> tuple[str, ...]:
    requested = config.experiment.candidate_ids
    if requested == "all":
        return DEFAULT_REGISTRY.ids
    unknown = tuple(
        candidate_id for candidate_id in requested if candidate_id not in DEFAULT_REGISTRY
    )
    if unknown:
        raise ValueError("configured candidates have no runtime adapter: " + ", ".join(unknown))
    return tuple(requested)


def _selected_imputer_registry(config: AppConfig) -> ImputerRegistry:
    selected = set(_selected_candidate_ids(config))
    return ImputerRegistry(spec for spec in DEFAULT_REGISTRY.specs() if spec.imputer_id in selected)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _torch_device(config: AppConfig) -> str:
    if config.runtime.device in {"auto", "gpu"} and _cuda_available():
        return "cuda"
    return "cpu"


def _allowed_devices(config: AppConfig) -> tuple[str, ...]:
    if _torch_device(config) == "cuda":
        return ("cpu", "gpu")
    return ("cpu",)


def _empty_cuda_cache(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    torch.cuda.empty_cache()


def _resident_memory_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    try:
        return int(psutil.Process().memory_info().rss)
    except OSError:
        return 0


def _pypots_params(config: AppConfig, spec: Any) -> dict[str, Any] | None:
    if not str(spec.factory).startswith("tsfm_fais.imputers.pypots:"):
        return None
    params: dict[str, Any] = {
        "epochs": config.experiment.deep_imputer_epochs,
        "batch_size": config.experiment.deep_imputer_batch_size,
        "num_samples": config.experiment.csdi_num_samples,
    }
    if spec.device == "any":
        params["device"] = _torch_device(config)
    elif spec.device == "cpu":
        params["device"] = "cpu"
    return params


def _fit_candidate_params(config: AppConfig, spec: Any) -> dict[str, Any] | None:
    if spec.imputer_id == "missforest":
        return {"n_jobs": config.experiment.missforest_n_jobs}
    return _pypots_params(config, spec)


def _execution_metadata(config: AppConfig) -> dict[str, Any]:
    limits = {
        name: getattr(config.experiment, name)
        for name in (
            "max_items_per_dataset",
            "max_training_windows_per_dataset",
            "max_train_origins_per_item",
            "max_eval_origins_per_item",
            "max_train_episodes_per_dataset",
            "max_eval_episodes_per_dataset",
            "max_teacher_blocks_per_episode",
            "max_teacher_candidates_per_episode",
            "max_pair_labels_per_episode",
        )
    }
    limits.update(
        {
            "candidate_ids": list(_selected_candidate_ids(config)),
            "deep_imputer_epochs": config.experiment.deep_imputer_epochs,
            "deep_imputer_batch_size": config.experiment.deep_imputer_batch_size,
            "csdi_num_samples": config.experiment.csdi_num_samples,
            "missforest_n_jobs": config.experiment.missforest_n_jobs,
            "forecast_num_samples": config.experiment.forecast_num_samples,
            "save_all_candidate_outputs": config.experiment.save_all_candidate_outputs,
        }
    )
    return {
        "sampling_limits": limits,
        "device_resolution": {
            "requested": config.runtime.device,
            "torch_device": _torch_device(config),
            "cuda_available": _cuda_available(),
            "allowed_candidate_devices": list(_allowed_devices(config)),
        },
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required signature file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _router_content_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    files: list[Path] = []
    if resolved.is_file():
        files.append(resolved)
        sibling_manifest = resolved.with_name("manifest.json")
        if sibling_manifest.is_file():
            files.append(sibling_manifest)
    else:
        folds_path = resolved / "folds.json"
        if folds_path.is_file():
            files.append(folds_path)
            try:
                folds_payload = json.loads(folds_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot sign invalid router fold manifest: {error}"
                ) from error
            folds = folds_payload.get("folds") if isinstance(folds_payload, dict) else None
            if not isinstance(folds, dict) or not folds:
                raise ValueError("cannot sign router folds without a non-empty folds map")
            for held_out, raw_target in sorted(folds.items()):
                target = Path(str(raw_target))
                if not target.is_absolute():
                    target = (folds_path.parent / target).resolve()
                bundle = target / "router_bundle.joblib" if target.is_dir() else target
                if not bundle.is_file():
                    raise FileNotFoundError(
                        f"router fold {held_out!r} bundle does not exist: {bundle}"
                    )
                files.append(bundle)
                fold_manifest = target / "manifest.json" if target.is_dir() else None
                if fold_manifest is not None and fold_manifest.is_file():
                    files.append(fold_manifest)
        else:
            bundle = resolved / "router_bundle.joblib"
            if not bundle.is_file():
                raise FileNotFoundError(f"router bundle does not exist: {bundle}")
            files.append(bundle)
            manifest = resolved / "manifest.json"
            if manifest.is_file():
                files.append(manifest)
    unique = {str(file.resolve()): file.resolve() for file in files}
    return {
        "path": str(resolved),
        "files": [_file_signature(unique[key]) for key in sorted(unique)],
    }


def _impute_resume_identity(
    config: AppConfig,
    inputs: StageInputs,
    registry: ImputerRegistry,
    model_id: str,
) -> dict[str, Any]:
    if inputs.audit_artifact is None or inputs.imputer_artifacts is None:
        raise ValueError("impute resume identity requires audit and imputer artifacts")
    if inputs.router_artifact is None:
        raise ValueError("impute resume identity requires a router artifact")
    imputer_manifest = inputs.imputer_artifacts.resolve() / "manifest.json"
    if not imputer_manifest.is_file():
        raise FileNotFoundError(
            f"imputer artifact manifest does not exist: {imputer_manifest}"
        )
    forecast_registry = default_forecast_registry()
    forecast_adapter = forecast_registry.get(model_id)
    registry_files = {
        name: _file_signature(getattr(config.registries, name))
        for name in (
            "data_manifest",
            "imputer_registry",
            "forecaster_registry",
            "router_config",
        )
    }
    raw_forecaster_spec = asdict(forecast_adapter)
    forecaster_spec = json.loads(
        json.dumps(raw_forecaster_spec, ensure_ascii=False, sort_keys=True, default=str)
    )
    return {
        "schema_version": 1,
        "artifact_schema_version": _IMPUTATION_SCHEMA_VERSION,
        "resolved_config_sha256": _canonical_sha256(config.model_dump(mode="json")),
        "root_seed": config.seed,
        "audit_artifact": _file_signature(inputs.audit_artifact),
        "imputer_artifacts": {
            "path": str(inputs.imputer_artifacts.resolve()),
            "manifest": _file_signature(imputer_manifest),
        },
        "router_artifact": _router_content_signature(inputs.router_artifact),
        "registries": registry_files,
        "forecaster": {
            "id": model_id,
            "mode": forecast_adapter.mode,
            "spec_sha256": _canonical_sha256(forecaster_spec),
            "spec": forecaster_spec,
        },
        "selected_candidates": list(registry.ids),
    }


def _labels_resume_identity(
    config: AppConfig,
    inputs: StageInputs,
    model_id: str,
    checkpoint: Path,
) -> dict[str, Any]:
    if inputs.audit_artifact is None or inputs.imputer_artifacts is None:
        raise ValueError("labels resume identity requires audit and imputer artifacts")
    imputer_manifest = inputs.imputer_artifacts.resolve() / "manifest.json"
    if not imputer_manifest.is_file():
        raise FileNotFoundError(
            f"imputer artifact manifest does not exist: {imputer_manifest}"
        )
    adapter_spec = default_forecast_registry().get(model_id)
    forecaster_spec = json.loads(
        json.dumps(
            asdict(adapter_spec),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    package_root = Path(__file__).resolve().parent
    source_artifacts = {
        "data_manifest": config.registries.data_manifest,
        "imputer_registry": config.registries.imputer_registry,
        "forecaster_registry": config.registries.forecaster_registry,
        "router_config": config.registries.router_config,
        "block_graph_source": package_root / "routing" / "blocks.py",
        "routing_features_source": package_root / "routing" / "features.py",
        "teacher_source": package_root / "routing" / "teacher.py",
        "sampling_source": package_root / "experiment_sampling.py",
    }
    return build_label_resume_identity(
        resolved_config=config.model_dump(mode="json"),
        audit_artifact=inputs.audit_artifact,
        imputer_manifest=imputer_manifest,
        checkpoint=checkpoint,
        forecaster_id=model_id,
        forecaster_mode=adapter_spec.mode,
        forecaster_spec=forecaster_spec,
        selected_candidates=_selected_candidate_ids(config),
        source_artifacts=source_artifacts,
    )


def _training_batch_summary(batch: SeriesBatch) -> dict[str, Any]:
    digest = hashlib.sha256()
    values = np.ascontiguousarray(batch.values, dtype="<f8")
    observed = np.ascontiguousarray(batch.observed_mask, dtype=np.uint8)
    digest.update(values.tobytes())
    digest.update(observed.tobytes())
    digest.update(
        json.dumps(list(batch.item_ids), ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return {
        "shape": list(batch.shape),
        "item_ids": list(batch.item_ids),
        "content_sha256": digest.hexdigest(),
    }


def _fit_resume_identity(
    config: AppConfig,
    audit_artifact: Path,
    registry: ImputerRegistry,
) -> dict[str, Any]:
    raw_candidate_specs = {
        spec.imputer_id: {
            "spec": asdict(spec),
            "effective_fit_params": _fit_candidate_params(config, spec) or {},
        }
        for spec in registry.specs()
    }
    candidate_specs = json.loads(
        json.dumps(raw_candidate_specs, ensure_ascii=False, sort_keys=True, default=str)
    )
    return {
        "schema_version": 1,
        "resolved_config_sha256": _canonical_sha256(config.model_dump(mode="json")),
        "audit_artifact_sha256": _file_sha256(audit_artifact),
        "root_seed": config.seed,
        "candidate_specs_sha256": _canonical_sha256(candidate_specs),
        "candidate_specs": candidate_specs,
    }


def _artifact_target(candidate_id: str, directory: Path, adapter: Any) -> tuple[str, Path]:
    if hasattr(adapter, "save_artifact"):
        return "adapter", directory / candidate_id
    return "joblib", directory / f"{candidate_id}.joblib"


def _load_artifact_entry(
    candidate_id: str,
    entry: Mapping[str, Any],
    directory: Path,
    registry: ImputerRegistry,
    config: AppConfig,
) -> None:
    root = directory.resolve()
    path = (directory / str(entry.get("path", ""))).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes its dataset directory: {path}")
    if not path.exists():
        raise ValueError(f"fitted artifact is missing for {candidate_id}: {path}")
    serializer = entry.get("serializer")
    if serializer == "joblib":
        artifact = joblib.load(path)
    elif serializer == "adapter":
        spec = registry.get_spec(candidate_id)
        adapter = registry.create(candidate_id, **(_fit_candidate_params(config, spec) or {}))
        loader = getattr(adapter, "load_artifact", None)
        if not callable(loader):
            raise TypeError(f"candidate {candidate_id!r} has no artifact loader")
        artifact = loader(path)
    else:
        raise ValueError(f"unknown artifact serializer for {candidate_id}: {serializer!r}")
    del artifact
    _empty_cuda_cache(_torch_device(config))


def _recover_existing_artifact(
    candidate_id: str,
    directory: Path,
    registry: ImputerRegistry,
    config: AppConfig,
) -> dict[str, Any] | None:
    spec = registry.get_spec(candidate_id)
    adapter = registry.create(candidate_id, **(_fit_candidate_params(config, spec) or {}))
    serializer, target = _artifact_target(candidate_id, directory, adapter)
    if not target.exists():
        return None
    entry = {
        "status": "fitted",
        "serializer": serializer,
        "path": target.name,
        "recovered_without_progress_entry": True,
    }
    _load_artifact_entry(candidate_id, entry, directory, registry, config)
    return entry


def _save_imputer_artifact(
    candidate_id: str,
    artifact: Any,
    directory: Path,
    registry: ImputerRegistry,
    config: AppConfig,
) -> tuple[str, str]:
    spec = registry.get_spec(candidate_id)
    adapter = registry.create(candidate_id, **(_fit_candidate_params(config, spec) or {}))
    serializer, target = _artifact_target(candidate_id, directory, adapter)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate artifact: {target}")
    temporary = directory / f".{target.name}.{uuid4().hex}.tmp"
    if serializer == "adapter":
        saver = getattr(adapter, "save_artifact", None)
        if not callable(saver):
            raise TypeError(f"candidate {candidate_id!r} has no artifact saver")
        saver(artifact, temporary)
    else:
        joblib.dump(artifact, temporary)
    temporary.rename(target)
    return serializer, target.name


def _write_training_statistics(
    path: Path,
    medians: np.ndarray,
    correlation: np.ndarray,
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, medians=medians, correlation=correlation)
    temporary.rename(path)


def _validate_training_statistics(
    path: Path,
    medians: np.ndarray,
    correlation: np.ndarray,
) -> None:
    try:
        with np.load(path) as stored:
            stored_medians = np.asarray(stored["medians"], dtype=float)
            stored_correlation = np.asarray(stored["correlation"], dtype=float)
    except Exception as error:
        raise ValueError(
            f"cannot resume with invalid training statistics {path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not np.array_equal(stored_medians, medians) or not np.array_equal(
        stored_correlation, correlation
    ):
        raise ValueError(f"training statistics differ from recomputed values: {path}")


def execute_fit_imputers(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if inputs.audit_artifact is None:
        raise ValueError("fit-imputers requires an audit artifact")
    output = preparation.store.root / "imputer_artifacts"
    output.mkdir(parents=True, exist_ok=preparation.resuming)
    registry = _selected_imputer_registry(config)
    runner = CandidateRunner(registry)
    allowed_devices = _allowed_devices(config)
    resume_identity = _fit_resume_identity(config, inputs.audit_artifact, registry)
    manifest_path = output / "manifest.json"
    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "datasets": {},
        "selected_candidates": list(registry.ids),
        "torch_device": _torch_device(config),
        "deep_imputer_epochs": config.experiment.deep_imputer_epochs,
        "deep_imputer_batch_size": config.experiment.deep_imputer_batch_size,
        "csdi_num_samples": config.experiment.csdi_num_samples,
        "missforest_n_jobs": config.experiment.missforest_n_jobs,
        "max_items_per_dataset": config.experiment.max_items_per_dataset,
        "max_training_windows_per_dataset": (config.experiment.max_training_windows_per_dataset),
        "mask_protocol": "sequence_mask_v2",
        "fit_prefix_fraction": config.experiment.fit_prefix_fraction,
        "training_window_stride": config.experiment.training_window_stride,
        "missing_block_lengths": list(config.experiment.missing_block_lengths),
        "resume": resume_identity,
    }
    if preparation.resuming and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
            raise ValueError("cannot resume: imputer manifest is invalid")
        stored_identity = manifest.get("resume")
        if stored_identity is not None and stored_identity != resume_identity:
            raise ValueError(
                "cannot resume: audit, seed, resolved config, or candidate specs changed"
            )
        selected = manifest.get("selected_candidates")
        if selected is not None and selected != list(registry.ids):
            raise ValueError("cannot resume: selected candidates changed")
        if stored_identity is None:
            manifest["legacy_manifest_migrated"] = True
            manifest["resume"] = resume_identity
        for key, value in base_manifest.items():
            if key not in {"datasets", "resume"}:
                manifest.setdefault(key, value)
        manifest["status"] = "running"
    elif preparation.resuming:
        manifest = base_manifest
        if any(output.iterdir()):
            manifest["legacy_partial_run_recovered"] = True
    else:
        manifest = base_manifest
    _write_json(manifest_path, manifest)

    seen_datasets: set[str] = set()
    for dataset, items in _datasets(config, inputs.audit_artifact):
        seen_datasets.add(dataset.dataset_id)
        batch = _training_batch(
            items,
            config.experiment.context_length,
            config.experiment.horizon,
            config.experiment.max_training_windows_per_dataset,
            dataset_id=dataset.dataset_id,
            masking_specs=(
                MaskingSpec(
                    mechanism,
                    rate,
                    config.experiment.missing_block_lengths,
                )
                for mechanism in config.experiment.missing_mechanisms
                for rate in config.experiment.missing_rates
            ),
            configured_seeds=config.experiment.seeds,
            fit_fraction=config.experiment.fit_prefix_fraction,
            training_stride=config.experiment.training_window_stride,
        )
        dataset_dir = output / dataset.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=preparation.resuming)
        medians, correlation = _training_statistics(batch)
        training_summary = _training_batch_summary(batch)
        dataset_entry = manifest["datasets"].get(dataset.dataset_id)
        if dataset_entry is None:
            dataset_entry = {
                "training_windows": batch.shape[0],
                "dimension": batch.shape[2],
                "context_length": batch.shape[1],
                "mask_protocol": "sequence_mask_v2",
                "training_summary": training_summary,
                "statistics": "training_statistics.npz",
                "candidates": {},
            }
            manifest["datasets"][dataset.dataset_id] = dataset_entry
        else:
            expected_shape = (
                int(dataset_entry.get("training_windows", -1)),
                int(dataset_entry.get("context_length", -1)),
                int(dataset_entry.get("dimension", -1)),
            )
            if expected_shape != batch.shape:
                raise ValueError(
                    f"cannot resume {dataset.dataset_id}: training batch shape changed"
                )
            stored_summary = dataset_entry.get("training_summary")
            if stored_summary is not None and stored_summary != training_summary:
                raise ValueError(
                    f"cannot resume {dataset.dataset_id}: training batch summary changed"
                )
            dataset_entry["training_summary"] = training_summary
            if not isinstance(dataset_entry.get("candidates"), dict):
                raise ValueError(f"cannot resume {dataset.dataset_id}: invalid candidates map")
        statistics_path = dataset_dir / str(dataset_entry["statistics"])
        if statistics_path.exists():
            _validate_training_statistics(statistics_path, medians, correlation)
        else:
            _write_training_statistics(statistics_path, medians, correlation)
        entries = dataset_entry["candidates"]
        _write_json(manifest_path, manifest)

        for spec in registry.specs():
            existing = entries.get(spec.imputer_id)
            if isinstance(existing, dict) and existing.get("status") == "fitted":
                _load_artifact_entry(
                    spec.imputer_id,
                    existing,
                    dataset_dir,
                    registry,
                    config,
                )
                existing["verified_at"] = utc_now()
                _write_json(manifest_path, manifest)
                continue
            if isinstance(existing, dict) and existing.get("status") == "unavailable":
                continue
            if spec.fit_scope == "none":
                entries[spec.imputer_id] = {"status": "stateless"}
                _write_json(manifest_path, manifest)
                continue
            if spec.device != "any" and spec.device not in allowed_devices:
                entries[spec.imputer_id] = {
                    "status": "unavailable",
                    "reason": f"device {spec.device!r} is excluded by runtime config",
                }
                _write_json(manifest_path, manifest)
                continue
            availability = registry.availability(spec.imputer_id)
            if not availability.available:
                entries[spec.imputer_id] = {
                    "status": "unavailable",
                    "missing_dependencies": list(availability.missing),
                }
                _write_json(manifest_path, manifest)
                continue
            recovered = _recover_existing_artifact(
                spec.imputer_id,
                dataset_dir,
                registry,
                config,
            )
            if recovered is not None:
                recovered["verified_at"] = utc_now()
                entries[spec.imputer_id] = recovered
                _write_json(manifest_path, manifest)
                continue
            attempts = int(existing.get("attempts", 0)) if isinstance(existing, dict) else 0
            entries[spec.imputer_id] = {
                "status": "running",
                "attempts": attempts + 1,
                "started_at": utc_now(),
            }
            _write_json(manifest_path, manifest)
            artifact: Any = None
            try:
                artifact = runner.fit(
                    spec.imputer_id,
                    batch,
                    {"period": dataset.period, "dataset_id": dataset.dataset_id},
                    params=_fit_candidate_params(config, spec),
                )
                serializer, relative_path = _save_imputer_artifact(
                    spec.imputer_id,
                    artifact,
                    dataset_dir,
                    registry,
                    config,
                )
            except Exception as error:
                entries[spec.imputer_id] = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "attempts": attempts + 1,
                    "completed_at": utc_now(),
                }
                _write_json(manifest_path, manifest)
                if config.runtime.fail_fast:
                    raise
            else:
                entries[spec.imputer_id] = {
                    "status": "fitted",
                    "serializer": serializer,
                    "path": relative_path,
                    "attempts": attempts + 1,
                    "completed_at": utc_now(),
                }
                _write_json(manifest_path, manifest)
            finally:
                del artifact
                _empty_cuda_cache(_torch_device(config))
    if not manifest["datasets"]:
        raise ValueError("audit artifact contains no enabled dataset from the manifest")
    extra_datasets = set(manifest["datasets"]).difference(seen_datasets)
    if extra_datasets:
        raise ValueError(
            "cannot resume: manifest contains datasets absent from current audited config: "
            + ", ".join(sorted(extra_datasets))
        )
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    _write_json(manifest_path, manifest)
    return {"imputer_artifacts": str(output), "manifest": str(manifest_path)}


class _LabelArtifactManager:
    """Load only one label episode's fitted artifacts with a narrow cache."""

    _CACHE_IDS = frozenset(
        {"knn_multivariate", "mice", "missforest", "softimpute"}
    )

    def __init__(
        self,
        store: DatasetImputerArtifactStore,
        registry: ImputerRegistry,
        config: AppConfig,
    ) -> None:
        self.store = store
        self.registry = registry
        self.config = config
        self._cache: dict[str, Any] = {}
        self._cached_failures: dict[str, str] = {}
        self._totals: Counter[str] = Counter()
        self._candidate_totals: dict[str, Counter[str]] = {}
        self._load_mode_counts: Counter[str] = Counter()
        self._candidate_load_modes: dict[str, str] = {}
        self._load_seconds = 0.0
        self._max_deep_load_batch = 0

    def _counts(self, candidate_id: str) -> Counter[str]:
        return self._candidate_totals.setdefault(candidate_id, Counter())

    def _is_deep(self, candidate_id: str) -> bool:
        return str(self.registry.get_spec(candidate_id).factory).startswith(
            "tsfm_fais.imputers.pypots:"
        )

    def candidate_pool(self, allowed_devices: tuple[str, ...]) -> tuple[str, ...]:
        """Build eligibility from manifest state without loading an artifact."""

        return tuple(
            spec.imputer_id
            for spec in self.registry.specs()
            if self.registry.availability(spec.imputer_id).available
            and (spec.device == "any" or spec.device in allowed_devices)
            and (
                spec.fit_scope == "none"
                or self.store.status(spec.imputer_id) == "fitted"
            )
        )

    def acquire(
        self,
        candidate_ids: tuple[str, ...],
    ) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
        """Load fitted artifacts needed by one selected candidate subset."""

        artifacts: dict[str, Any] = {}
        failures: dict[str, str] = {}
        to_load: list[str] = []
        for candidate_id in candidate_ids:
            spec = self.registry.get_spec(candidate_id)
            if spec.fit_scope == "none":
                continue
            self._totals["artifact_request_count"] += 1
            self._counts(candidate_id)["artifact_request_count"] += 1
            if candidate_id in self._cache:
                artifacts[candidate_id] = self._cache[candidate_id]
                self._totals["cache_hit_count"] += 1
                self._counts(candidate_id)["cache_hit_count"] += 1
            elif candidate_id in self._cached_failures:
                failures[candidate_id] = self._cached_failures[candidate_id]
                self._totals["cached_failure_hit_count"] += 1
                self._counts(candidate_id)["cached_failure_hit_count"] += 1
            else:
                to_load.append(candidate_id)

        if not to_load:
            return artifacts, failures, ()
        adapter_params = {
            candidate_id: params
            for candidate_id in to_load
            if (
                params := _pypots_params(
                    self.config,
                    self.registry.get_spec(candidate_id),
                )
            )
            is not None
        }
        result = self.store.load_artifacts(to_load, adapter_params=adapter_params)
        self._max_deep_load_batch = max(
            self._max_deep_load_batch,
            sum(self._is_deep(candidate_id) for candidate_id in to_load),
        )
        self._load_seconds += result.load_seconds
        self._totals["load_call_count"] += int(bool(to_load))
        for candidate_id, mode in result.load_modes.items():
            self._load_mode_counts[mode] += 1
            self._candidate_load_modes[candidate_id] = mode
        for candidate_id in result.attempted_ids:
            self._totals["deserialization_attempt_count"] += 1
            self._counts(candidate_id)["deserialization_attempt_count"] += 1
        for candidate_id, artifact in result.artifacts.items():
            self._totals["load_success_count"] += 1
            self._counts(candidate_id)["load_success_count"] += 1
            if candidate_id in self._CACHE_IDS:
                self._cache[candidate_id] = artifact
                self._totals["cache_store_count"] += 1
                self._counts(candidate_id)["cache_store_count"] += 1
            artifacts[candidate_id] = artifact
        for candidate_id, reason in result.failures.items():
            self._totals["load_failure_count"] += 1
            self._counts(candidate_id)["load_failure_count"] += 1
            if candidate_id in self._CACHE_IDS:
                self._cached_failures[candidate_id] = reason
                self._totals["failure_cache_store_count"] += 1
                self._counts(candidate_id)["failure_cache_store_count"] += 1
            failures[candidate_id] = reason
        ephemeral = tuple(
            candidate_id
            for candidate_id in result.attempted_ids
            if candidate_id not in self._CACHE_IDS
        )
        return artifacts, failures, ephemeral

    def release(
        self,
        artifacts: dict[str, Any],
        ephemeral_ids: tuple[str, ...],
    ) -> None:
        """Drop episode-scoped artifacts and release deep-model device memory."""

        deep_candidates = tuple(
            candidate_id
            for candidate_id in ephemeral_ids
            if self._is_deep(candidate_id)
        )
        deep_released = bool(deep_candidates)
        for candidate_id in deep_candidates:
            self._totals["deep_cleanup_count"] += 1
            self._counts(candidate_id)["deep_cleanup_count"] += 1
        ephemeral = set(ephemeral_ids)
        for candidate_id in tuple(artifacts):
            artifacts.pop(candidate_id, None)
            if candidate_id not in ephemeral:
                continue
            self._totals["evict_count"] += 1
            self._counts(candidate_id)["evict_count"] += 1
            if self._is_deep(candidate_id):
                deep_released = True
                self._totals["deep_evict_count"] += 1
                self._counts(candidate_id)["deep_evict_count"] += 1
        if deep_released:
            gc.collect()
            _empty_cuda_cache(_torch_device(self.config))

    def close(self) -> None:
        """Evict dataset-scoped structured artifacts."""

        for candidate_id in tuple(self._cache):
            self._cache.pop(candidate_id, None)
            self._totals["evict_count"] += 1
            self._totals["dataset_cache_evict_count"] += 1
            self._counts(candidate_id)["evict_count"] += 1
            self._counts(candidate_id)["dataset_cache_evict_count"] += 1
        self._cached_failures.clear()
        gc.collect()
        self._totals["dataset_cache_cleanup_count"] += 1

    def audit(self) -> dict[str, Any]:
        totals = {
            key: int(self._totals.get(key, 0))
            for key in (
                "artifact_request_count",
                "load_call_count",
                "deserialization_attempt_count",
                "load_success_count",
                "load_failure_count",
                "cache_store_count",
                "cache_hit_count",
                "failure_cache_store_count",
                "cached_failure_hit_count",
                "evict_count",
                "deep_evict_count",
                "deep_cleanup_count",
                "dataset_cache_evict_count",
                "dataset_cache_cleanup_count",
            )
        }
        by_candidate = {
            candidate_id: {
                **{key: int(counts.get(key, 0)) for key in totals},
                "load_mode": self._candidate_load_modes.get(candidate_id),
            }
            for candidate_id, counts in sorted(self._candidate_totals.items())
        }
        return {
            **totals,
            "load_seconds": self._load_seconds,
            "load_mode_counts": dict(sorted(self._load_mode_counts.items())),
            "max_deep_load_batch": self._max_deep_load_batch,
            "cache_policy": {"dataset_scoped": sorted(self._CACHE_IDS)},
            "by_candidate": by_candidate,
    }


def _run_label_candidate_pairs(
    manager: _LabelArtifactManager,
    runner: CandidateRunner,
    candidate_ids: tuple[str, ...],
    real_batch: SeriesBatch,
    pseudo_batch: SeriesBatch,
    *,
    seed: int,
    params: Mapping[str, Mapping[str, Any]],
    budget: BudgetSpec,
) -> tuple[dict[str, CandidateResult], dict[str, CandidateResult]]:
    """Run real and pseudo inputs consecutively, one loaded candidate at a time."""

    real_results: dict[str, CandidateResult] = {}
    pseudo_results: dict[str, CandidateResult] = {}
    for candidate_id in candidate_ids:
        artifacts, artifact_failures, ephemeral = manager.acquire((candidate_id,))
        try:
            real_results.update(
                runner.run_many(
                    (candidate_id,),
                    real_batch,
                    artifacts,
                    seed=seed,
                    params=params,
                    artifact_failures=artifact_failures,
                    budget=budget,
                )
            )
            pseudo_results.update(
                runner.run_many(
                    (candidate_id,),
                    pseudo_batch,
                    artifacts,
                    seed=seed,
                    params=params,
                    artifact_failures=artifact_failures,
                    budget=budget,
                )
            )
        finally:
            manager.release(artifacts, ephemeral)
            artifact_failures.clear()
    return (
        {candidate_id: real_results[candidate_id] for candidate_id in candidate_ids},
        {candidate_id: pseudo_results[candidate_id] for candidate_id in candidate_ids},
    )


_LABEL_ARTIFACT_COUNT_FIELDS = (
    "artifact_request_count",
    "load_call_count",
    "deserialization_attempt_count",
    "load_success_count",
    "load_failure_count",
    "cache_store_count",
    "cache_hit_count",
    "failure_cache_store_count",
    "cached_failure_hit_count",
    "evict_count",
    "deep_evict_count",
    "deep_cleanup_count",
    "dataset_cache_evict_count",
    "dataset_cache_cleanup_count",
)


def _label_artifact_audit_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the additive portion of a cumulative artifact-manager audit."""

    result: dict[str, Any] = {
        field: int(after.get(field, 0)) - int(before.get(field, 0))
        for field in _LABEL_ARTIFACT_COUNT_FIELDS
    }
    result["load_seconds"] = float(after.get("load_seconds", 0.0)) - float(
        before.get("load_seconds", 0.0)
    )
    before_modes = before.get("load_mode_counts", {})
    after_modes = after.get("load_mode_counts", {})
    before_modes = before_modes if isinstance(before_modes, Mapping) else {}
    after_modes = after_modes if isinstance(after_modes, Mapping) else {}
    result["load_mode_counts"] = {
        str(mode): int(after_modes.get(mode, 0)) - int(before_modes.get(mode, 0))
        for mode in sorted(set(before_modes) | set(after_modes))
        if int(after_modes.get(mode, 0)) != int(before_modes.get(mode, 0))
    }
    previous_max = int(before.get("max_deep_load_batch", 0))
    current_max = int(after.get("max_deep_load_batch", 0))
    result["max_deep_load_batch"] = current_max if current_max > previous_max else 0
    result["cache_policy"] = dict(after.get("cache_policy", {}))
    before_candidates = before.get("by_candidate", {})
    after_candidates = after.get("by_candidate", {})
    before_candidates = (
        before_candidates if isinstance(before_candidates, Mapping) else {}
    )
    after_candidates = after_candidates if isinstance(after_candidates, Mapping) else {}
    by_candidate: dict[str, Any] = {}
    for candidate_id in sorted(set(before_candidates) | set(after_candidates)):
        previous = before_candidates.get(candidate_id, {})
        current = after_candidates.get(candidate_id, {})
        previous = previous if isinstance(previous, Mapping) else {}
        current = current if isinstance(current, Mapping) else {}
        candidate_delta: dict[str, Any] = {
            field: int(current.get(field, 0)) - int(previous.get(field, 0))
            for field in _LABEL_ARTIFACT_COUNT_FIELDS
        }
        candidate_delta["load_mode"] = current.get("load_mode")
        if any(candidate_delta[field] for field in _LABEL_ARTIFACT_COUNT_FIELDS):
            by_candidate[str(candidate_id)] = candidate_delta
    result["by_candidate"] = by_candidate
    return result


def _merge_label_artifact_loading_deltas(
    deltas: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for delta in deltas:
        dataset_id = delta.get("dataset_id")
        forecaster_id = delta.get("forecaster_id")
        audit = delta.get("audit")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or not isinstance(forecaster_id, str)
            or not forecaster_id
            or not isinstance(audit, Mapping)
        ):
            raise LabelResumeError("persisted artifact-loading delta is invalid")
        current = records.setdefault(dataset_id, {}).setdefault(
            forecaster_id,
            {
                **{field: 0 for field in _LABEL_ARTIFACT_COUNT_FIELDS},
                "load_seconds": 0.0,
                "load_mode_counts": {},
                "max_deep_load_batch": 0,
                "cache_policy": {},
                "by_candidate": {},
            },
        )
        for field in _LABEL_ARTIFACT_COUNT_FIELDS:
            current[field] += int(audit.get(field, 0))
        current["load_seconds"] += float(audit.get("load_seconds", 0.0))
        modes = audit.get("load_mode_counts", {})
        if isinstance(modes, Mapping):
            for mode, count in modes.items():
                current["load_mode_counts"][str(mode)] = int(
                    current["load_mode_counts"].get(str(mode), 0)
                ) + int(count)
        current["max_deep_load_batch"] = max(
            int(current["max_deep_load_batch"]),
            int(audit.get("max_deep_load_batch", 0)),
        )
        policy = audit.get("cache_policy")
        if isinstance(policy, Mapping):
            current["cache_policy"] = dict(policy)
        candidates = audit.get("by_candidate", {})
        if isinstance(candidates, Mapping):
            for candidate_id, candidate_audit in candidates.items():
                if not isinstance(candidate_audit, Mapping):
                    raise LabelResumeError(
                        "persisted candidate artifact-loading delta is invalid"
                    )
                target = current["by_candidate"].setdefault(
                    str(candidate_id),
                    {
                        **{field: 0 for field in _LABEL_ARTIFACT_COUNT_FIELDS},
                        "load_mode": candidate_audit.get("load_mode"),
                    },
                )
                for field in _LABEL_ARTIFACT_COUNT_FIELDS:
                    target[field] += int(candidate_audit.get(field, 0))
                if candidate_audit.get("load_mode") is not None:
                    target["load_mode"] = candidate_audit.get("load_mode")
    return records


def _artifact_loading_manifest(
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    count_fields = _LABEL_ARTIFACT_COUNT_FIELDS
    overall: Counter[str] = Counter()
    overall_seconds = 0.0
    overall_load_modes: Counter[str] = Counter()
    overall_max_deep_load_batch = 0
    datasets: dict[str, Any] = {}
    for dataset_id, forecasters in sorted(records.items()):
        totals: Counter[str] = Counter()
        load_seconds = 0.0
        load_modes: Counter[str] = Counter()
        max_deep_load_batch = 0
        for audit in forecasters.values():
            totals.update({field: int(audit.get(field, 0)) for field in count_fields})
            load_seconds += float(audit.get("load_seconds", 0.0))
            raw_modes = audit.get("load_mode_counts", {})
            if isinstance(raw_modes, Mapping):
                load_modes.update(
                    {str(mode): int(count) for mode, count in raw_modes.items()}
                )
            max_deep_load_batch = max(
                max_deep_load_batch,
                int(audit.get("max_deep_load_batch", 0)),
            )
        overall.update(totals)
        overall_seconds += load_seconds
        overall_load_modes.update(load_modes)
        overall_max_deep_load_batch = max(
            overall_max_deep_load_batch,
            max_deep_load_batch,
        )
        datasets[dataset_id] = {
            **{field: int(totals[field]) for field in count_fields},
            "load_seconds": load_seconds,
            "load_mode_counts": dict(sorted(load_modes.items())),
            "max_deep_load_batch": max_deep_load_batch,
            "forecasters": dict(forecasters),
        }
    return {
        "strategy": "candidate_filtered_with_structured_dataset_cache_v1",
        "dataset_count": len(datasets),
        **{field: int(overall[field]) for field in count_fields},
        "load_seconds": overall_seconds,
        "load_mode_counts": dict(sorted(overall_load_modes.items())),
        "max_deep_load_batch": overall_max_deep_load_batch,
        "datasets": datasets,
    }


def _partition_origins(
    origins: tuple[int, ...],
    partition: Literal["all", "train", "eval"],
    split: str = "rolling_origin",
) -> tuple[int, ...]:
    if split == "leave_family_out":
        return origins
    if partition == "all":
        return origins
    if len(origins) < 2:
        return origins if partition == "train" else ()
    boundary = min(len(origins) - 1, max(1, int(np.ceil(0.7 * len(origins)))))
    return origins[:boundary] if partition == "train" else origins[boundary:]


def _capped_origins(
    config: AppConfig,
    origins: tuple[int, ...],
    partition: Literal["all", "train", "eval"],
) -> tuple[int, ...]:
    if config.experiment.split == "leave_family_out":
        limit = (
            config.experiment.max_train_origins_per_item
            if partition == "train"
            else config.experiment.max_eval_origins_per_item
            if partition == "eval"
            else None
        )
        return evenly_spaced_subset(origins, limit)
    training = evenly_spaced_subset(
        _partition_origins(origins, "train", config.experiment.split),
        config.experiment.max_train_origins_per_item,
    )
    evaluation = evenly_spaced_subset(
        _partition_origins(origins, "eval", config.experiment.split),
        config.experiment.max_eval_origins_per_item,
    )
    if partition == "train":
        return training
    if partition == "eval":
        return evaluation
    return training + evaluation


@dataclass(frozen=True)
class _EpisodeDescriptor:
    """An episode grid entry that can be sampled without reading series values."""

    item: TimeSeriesItem
    forecast_origin: int
    masking: MaskingSpec
    configured_seed: int
    source_index: int


def _episode_descriptor_grid(
    config: AppConfig,
    items: tuple[TimeSeriesItem, ...],
    partition: Literal["all", "train", "eval"],
) -> tuple[_EpisodeDescriptor, ...]:
    length = config.experiment.context_length
    descriptors: list[_EpisodeDescriptor] = []
    for item in items:
        if len(item.values) < length + config.experiment.horizon:
            continue
        first_origin = fit_prefix_end(
            len(item.values),
            length,
            config.experiment.horizon,
            config.experiment.fit_prefix_fraction,
        )
        origins = rolling_origins(
            len(item.values),
            length,
            config.experiment.horizon,
            config.experiment.forecast_stride or config.experiment.horizon,
            start=first_origin,
        )
        for origin in _capped_origins(config, origins, partition):
            for mechanism in config.experiment.missing_mechanisms:
                for rate in config.experiment.missing_rates:
                    for configured_seed in config.experiment.seeds:
                        descriptors.append(
                            _EpisodeDescriptor(
                                item=item,
                                forecast_origin=origin,
                                masking=MaskingSpec(
                                    mechanism,
                                    rate,
                                    config.experiment.missing_block_lengths,
                                ),
                                configured_seed=int(configured_seed),
                                source_index=len(descriptors),
                            )
                        )
    return tuple(descriptors)


def _balanced_episode_subset(
    descriptors: tuple[_EpisodeDescriptor, ...],
    limit: int | None,
    *seed_parts: object,
) -> tuple[_EpisodeDescriptor, ...]:
    """Select a deterministic, coverage-seeking subset of an episode grid.

    The greedy score first avoids repeating a mechanism/rate stratum, then
    balances mechanism and rate marginals, followed by item, configured seed,
    and origin. A stable hash resolves the remaining ties. Selected entries are
    returned in source order so an episode keeps its pre-cap identifier and
    execution ordering relative to other selected entries.
    """

    if limit is None or len(descriptors) <= limit:
        return descriptors
    if limit < 1:
        raise ValueError("episode limit must be positive")

    combination_counts: Counter[tuple[str, float]] = Counter()
    mechanism_counts: Counter[str] = Counter()
    rate_counts: Counter[float] = Counter()
    item_counts: Counter[str] = Counter()
    seed_counts: Counter[int] = Counter()
    origin_counts: Counter[tuple[str, int]] = Counter()
    remaining = set(range(len(descriptors)))
    selected: list[int] = []

    while len(selected) < limit:
        def score(index: int) -> tuple[int, int, int, int, int, int, int, int]:
            descriptor = descriptors[index]
            mechanism = descriptor.masking.mechanism
            rate = descriptor.masking.missing_rate
            item_id = descriptor.item.item_id
            origin_key = (item_id, descriptor.forecast_origin)
            tie_break = stable_seed(
                *seed_parts,
                item_id,
                descriptor.forecast_origin,
                mechanism,
                f"{rate:.17g}",
                descriptor.configured_seed,
                "episode_descriptor",
            )
            return (
                combination_counts[(mechanism, rate)],
                mechanism_counts[mechanism],
                rate_counts[rate],
                item_counts[item_id],
                seed_counts[descriptor.configured_seed],
                origin_counts[origin_key],
                tie_break,
                descriptor.source_index,
            )

        selected_index = min(remaining, key=score)
        remaining.remove(selected_index)
        selected.append(selected_index)
        descriptor = descriptors[selected_index]
        mechanism = descriptor.masking.mechanism
        rate = descriptor.masking.missing_rate
        item_id = descriptor.item.item_id
        combination_counts[(mechanism, rate)] += 1
        mechanism_counts[mechanism] += 1
        rate_counts[rate] += 1
        item_counts[item_id] += 1
        seed_counts[descriptor.configured_seed] += 1
        origin_counts[(item_id, descriptor.forecast_origin)] += 1

    return tuple(descriptors[index] for index in sorted(selected))


def _coverage_counts(
    eligible_values: Iterable[str],
    selected_values: Iterable[str],
) -> dict[str, Any]:
    eligible = Counter(eligible_values)
    selected = Counter(selected_values)
    ordered_keys = tuple(sorted(eligible))
    selected_with_zeros = {key: int(selected.get(key, 0)) for key in ordered_keys}
    counts = tuple(selected_with_zeros.values())
    return {
        "eligible_level_count": len(ordered_keys),
        "selected_level_count": sum(count > 0 for count in counts),
        "eligible_counts": {key: int(eligible[key]) for key in ordered_keys},
        "selected_counts": selected_with_zeros,
        "selected_count_min": min(counts, default=0),
        "selected_count_max": max(counts, default=0),
    }


def _episode_selection_summary(
    eligible: tuple[_EpisodeDescriptor, ...],
    selected: tuple[_EpisodeDescriptor, ...],
    partition: Literal["all", "train", "eval"],
    limit: int | None,
) -> dict[str, Any]:
    def values(descriptors: tuple[_EpisodeDescriptor, ...], field: str) -> tuple[str, ...]:
        if field == "mechanism_rate":
            return tuple(
                f"{entry.masking.mechanism}|{entry.masking.missing_rate:g}"
                for entry in descriptors
            )
        if field == "mechanism":
            return tuple(entry.masking.mechanism for entry in descriptors)
        if field == "rate":
            return tuple(f"{entry.masking.missing_rate:g}" for entry in descriptors)
        if field == "item":
            return tuple(entry.item.item_id for entry in descriptors)
        if field == "seed":
            return tuple(str(entry.configured_seed) for entry in descriptors)
        if field == "origin":
            return tuple(
                f"{entry.item.item_id}|{entry.forecast_origin}" for entry in descriptors
            )
        raise ValueError(f"unknown episode coverage field: {field}")

    coverage = {
        field: _coverage_counts(values(eligible, field), values(selected, field))
        for field in ("mechanism_rate", "mechanism", "rate", "item", "seed", "origin")
    }
    strata = coverage["mechanism_rate"]
    full_coverage_required = limit is not None and limit >= strata["eligible_level_count"]
    return {
        "partition": partition,
        "cap_per_dataset": limit,
        "eligible_episode_count": len(eligible),
        "selected_episode_count": len(selected),
        "truncated": len(selected) < len(eligible),
        "selection_algorithm": "deterministic_stratified_greedy_v1",
        "mechanism_rate_full_coverage_required": full_coverage_required,
        "mechanism_rate_full_coverage_achieved": (
            strata["selected_level_count"] == strata["eligible_level_count"]
        ),
        "coverage": coverage,
    }


def _episode_plan(
    config: AppConfig,
    dataset: Any,
    items: Iterable[TimeSeriesItem],
    partition: Literal["all", "train", "eval"] = "all",
) -> tuple[tuple[_EpisodeDescriptor, ...], dict[str, Any]]:
    item_tuple = tuple(items)
    train_cap = config.experiment.max_train_episodes_per_dataset
    eval_cap = config.experiment.max_eval_episodes_per_dataset
    if partition == "all" and (train_cap is not None or eval_cap is not None):
        train, train_summary = _episode_plan(config, dataset, item_tuple, "train")
        evaluation, eval_summary = _episode_plan(config, dataset, item_tuple, "eval")
        selected = train + evaluation
        summary = {
            "partition": "all",
            "cap_per_dataset": {"train": train_cap, "eval": eval_cap},
            "eligible_episode_count": (
                train_summary["eligible_episode_count"]
                + eval_summary["eligible_episode_count"]
            ),
            "selected_episode_count": len(selected),
            "truncated": train_summary["truncated"] or eval_summary["truncated"],
            "partitions": {"train": train_summary, "eval": eval_summary},
        }
        return selected, summary

    eligible = _episode_descriptor_grid(config, item_tuple, partition)
    limit = train_cap if partition == "train" else eval_cap if partition == "eval" else None
    selected = _balanced_episode_subset(
        eligible,
        limit,
        config.seed,
        dataset.dataset_id,
        partition,
    )
    return selected, _episode_selection_summary(eligible, selected, partition, limit)


def _episode_iter(
    config: AppConfig,
    dataset: Any,
    items: Iterable[TimeSeriesItem],
    partition: Literal["all", "train", "eval"] = "all",
    selection_summary: dict[str, Any] | None = None,
):
    descriptors, summary = _episode_plan(config, dataset, items, partition)
    if selection_summary is not None:
        selection_summary.clear()
        selection_summary.update(summary)
    length = config.experiment.context_length
    realization_cache: dict[tuple[str, str, float, int], Any] = {}
    for descriptor in descriptors:
        cache_key = (
            descriptor.item.item_id,
            descriptor.masking.mechanism,
            descriptor.masking.missing_rate,
            descriptor.configured_seed,
        )
        realization = realization_cache.get(cache_key)
        if realization is None:
            fit_end = fit_prefix_end(
                len(descriptor.item.values),
                length,
                config.experiment.horizon,
                config.experiment.fit_prefix_fraction,
            )
            realization = mask_time_series(
                descriptor.item.values,
                descriptor.masking,
                _sequence_mask_seed(
                    dataset.dataset_id,
                    descriptor.item.item_id,
                    descriptor.masking,
                    descriptor.configured_seed,
                ),
                calibration_values=descriptor.item.values[:fit_end],
            )
            realization_cache[cache_key] = realization
        episode = build_episode(
            descriptor.item,
            dataset.dataset_id,
            realization,
            descriptor.forecast_origin,
            length,
            config.experiment.horizon,
        )
        episode_id = (
            f"{dataset.dataset_id}__{descriptor.item.item_id}__"
            f"{descriptor.forecast_origin}__{descriptor.masking.mechanism}__"
            f"{descriptor.masking.missing_rate:g}__{descriptor.configured_seed}"
        )
        yield episode_id, episode


def _episode_parameters(episode_id: str) -> tuple[str | None, float | None, int | None]:
    """Recover the configured masking tuple from the stable episode identifier."""

    parts = episode_id.rsplit("__", 4)
    if len(parts) != 5:
        return None, None, None
    mechanism = parts[2]
    try:
        rate = float(parts[3])
        seed = int(parts[4])
    except ValueError:
        return mechanism, None, None
    return mechanism, rate, seed


def _record_episode_sampling(
    records: dict[str, dict[str, Any]],
    dataset_id: str,
    summary: dict[str, Any],
) -> None:
    existing = records.get(dataset_id)
    if existing is not None and existing != summary:
        raise RuntimeError(f"episode selection changed within the run for {dataset_id}")
    records[dataset_id] = summary


def _episode_sampling_manifest(
    partition: Literal["train", "eval"],
    cap_per_dataset: int | None,
    records: Mapping[str, Mapping[str, Any]],
    execution_count: int,
) -> dict[str, Any]:
    selected_count = sum(
        int(summary.get("selected_episode_count", 0)) for summary in records.values()
    )
    eligible_count = sum(
        int(summary.get("eligible_episode_count", 0)) for summary in records.values()
    )
    return {
        "partition": partition,
        "cap_per_dataset": cap_per_dataset,
        "dataset_count": len(records),
        "eligible_episode_count": eligible_count,
        "selected_episode_count": selected_count,
        "episode_execution_count": execution_count,
        "datasets": dict(records),
    }


def _forecaster_artifacts(inputs: StageInputs) -> tuple[tuple[str, Path], ...]:
    if inputs.forecaster_id is None or inputs.forecaster_artifact is None:
        raise ValueError("forecaster ID and artifact are required")
    model_ids = parse_forecaster_ids(inputs.forecaster_id)
    source = inputs.forecaster_artifact.resolve()
    if not source.exists():
        raise ValueError(f"forecaster artifact does not exist: {source}")

    resolved: dict[str, Path] = {}
    if source.is_file() and source.suffix.lower() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("forecaster artifact JSON is invalid") from error
        mapping = payload.get("artifacts") if isinstance(payload, dict) else None
        if mapping is None:
            mapping = payload
        if not isinstance(mapping, dict):
            raise ValueError("forecaster artifact JSON must map model IDs to paths")
        for model_id in model_ids:
            raw_path = mapping.get(model_id)
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"forecaster artifact mapping has no path for {model_id!r}")
            target = Path(raw_path)
            resolved[model_id] = (
                target if target.is_absolute() else (source.parent / target).resolve()
            )
    elif len(model_ids) == 1:
        resolved[model_ids[0]] = source
    elif source.is_dir():
        resolved = {model_id: source / model_id for model_id in model_ids}
    else:
        raise ValueError(
            "multiple forecasters require a checkpoint directory or JSON mapping"
        )
    missing = [model_id for model_id, path in resolved.items() if not path.exists()]
    if missing:
        raise ValueError("forecaster artifacts do not exist for: " + ", ".join(sorted(missing)))
    return tuple((model_id, resolved[model_id]) for model_id in model_ids)


def _preflight_forecaster(
    registry: Any,
    model_id: str,
    artifact: Path,
    *,
    device: str,
    batch_size: int = 8,
):
    adapter = registry.build(
        model_id,
        model_name=str(artifact),
        device=device,
        batch_size=batch_size,
    )
    ensure_backend = getattr(adapter, "_ensure_backend", None)
    if callable(ensure_backend):
        ensure_backend()
    return adapter


def _pair_label_requests(
    edges: Iterable[Any],
    eligible: Mapping[str, tuple[str, ...]],
    candidate_ids: tuple[str, ...],
    seed: int,
    limit: int = 64,
) -> tuple[tuple[Any, str, str], ...]:
    """Select deterministic, coverage-seeking edge/candidate pairs."""

    if limit < 1:
        return ()
    edge_list = [edge for edge in edges if eligible.get(edge.left) and eligible.get(edge.right)]
    if not edge_list:
        return ()
    rng = np.random.default_rng(seed)
    edge_list = [edge_list[index] for index in rng.permutation(len(edge_list))]
    candidates = [candidate_ids[index] for index in rng.permutation(len(candidate_ids))]
    selected: list[tuple[Any, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(edge: Any, left_candidate: str, right_candidate: str) -> None:
        key = (edge.left, edge.right, left_candidate, right_candidate)
        if len(selected) >= limit or key in seen:
            return
        if left_candidate not in eligible[edge.left] or right_candidate not in eligible[edge.right]:
            return
        seen.add(key)
        selected.append((edge, left_candidate, right_candidate))

    for candidate_id in candidates:
        matching_left = [edge for edge in edge_list if candidate_id in eligible[edge.left]]
        if matching_left:
            edge = matching_left[int(rng.integers(len(matching_left)))]
            right_values = eligible[edge.right]
            add(
                edge,
                candidate_id,
                right_values[int(rng.integers(len(right_values)))],
            )
        matching_right = [edge for edge in edge_list if candidate_id in eligible[edge.right]]
        if matching_right:
            edge = matching_right[int(rng.integers(len(matching_right)))]
            left_values = eligible[edge.left]
            add(
                edge,
                left_values[int(rng.integers(len(left_values)))],
                candidate_id,
            )
        if len(selected) >= limit:
            return tuple(selected)

    for edge in edge_list:
        left_values = eligible[edge.left]
        right_values = eligible[edge.right]
        add(
            edge,
            left_values[int(rng.integers(len(left_values)))],
            right_values[int(rng.integers(len(right_values)))],
        )
        if len(selected) >= limit:
            return tuple(selected)

    attempts = 0
    max_attempts = max(256, limit * 20)
    while len(selected) < limit and attempts < max_attempts:
        attempts += 1
        edge = edge_list[int(rng.integers(len(edge_list)))]
        left_values = eligible[edge.left]
        right_values = eligible[edge.right]
        add(
            edge,
            left_values[int(rng.integers(len(left_values)))],
            right_values[int(rng.integers(len(right_values)))],
        )
    return tuple(selected)


def _forecast_spec(config: AppConfig, model_id: str, dimensions: int) -> ForecastSpec:
    adapter_spec = default_forecast_registry().get(model_id)
    targets = (
        tuple(range(dimensions))
        if config.experiment.target_indices == "all"
        else tuple(config.experiment.target_indices)
    )
    return ForecastSpec(
        model_id=model_id,
        mode=adapter_spec.mode,
        horizon=config.experiment.horizon,
        context_length=config.experiment.context_length,
        target_indices=targets,
        num_samples=config.experiment.forecast_num_samples,
    )


def _candidate_is_eligible(
    block: Any,
    candidate_id: str,
    _result: Any,
    *,
    eligible: Mapping[str, tuple[str, ...]],
) -> bool:
    return candidate_id in eligible[block.block_id]


def _supplement_candidate_outputs(
    config: AppConfig,
    registry: ImputerRegistry,
    artifacts: Mapping[str, Any],
    batch: SeriesBatch,
    candidates: dict[str, Any],
    seed: int,
) -> tuple[str, ...]:
    """Run evaluation-only candidates without changing the routed assignment."""

    if not config.experiment.save_all_candidate_outputs:
        return ()
    allowed_devices = _allowed_devices(config)
    remaining = tuple(
        spec.imputer_id
        for spec in registry.specs()
        if (
            spec.imputer_id not in candidates
            and registry.availability(spec.imputer_id).available
            and (spec.device == "any" or spec.device in allowed_devices)
            and (spec.fit_scope == "none" or spec.imputer_id in artifacts)
        )
    )
    if not remaining:
        return ()
    params = {
        candidate_id: candidate_params
        for candidate_id in remaining
        if (
            candidate_params := _pypots_params(
                config,
                registry.get_spec(candidate_id),
            )
        )
        is not None
    }
    supplemental = CandidateRunner(registry).run_many(
        remaining,
        batch,
        artifacts,
        seed=seed,
        params=params,
        budget=BudgetSpec(
            max_candidates=len(remaining),
            allowed_devices=allowed_devices,
        ),
    )
    candidates.update(supplemental)
    return remaining


@dataclass(frozen=True)
class _LabelEpisodeRows:
    candidate_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    unary_rows: tuple[Mapping[str, Any], ...]
    pair_rows: tuple[Mapping[str, Any], ...]

    @property
    def outcome(self) -> Literal["labeled", "no_labels"]:
        return "labeled" if self.unary_rows else "no_labels"


def _label_sampling_cell(episode_id: str, episode: Any) -> dict[str, Any]:
    mechanism, missing_rate, configured_seed = _episode_parameters(episode_id)
    if mechanism is None or missing_rate is None or configured_seed is None:
        raise ValueError(f"cannot recover sampling cell from episode ID {episode_id!r}")
    return {
        "mechanism": mechanism,
        "missing_rate": missing_rate,
        "configured_seed": configured_seed,
        "episode_seed": int(episode.seed),
        "mask_seed": int(episode.mask_seed),
        "mask_realization_id": str(episode.mask_realization_id),
        "global_missing_rate": float(episode.global_missing_rate),
        "local_missing_rate": float(episode.local_missing_rate),
    }


def _validate_label_expectation_core(
    expectation: LabelEpisodeExpectation,
    *,
    artifact_index: int,
    model_id: str,
    episode_id: str,
    dataset: Any,
    episode: Any,
    sampling_cell: Mapping[str, Any],
    dataset_plan_sha256: str,
    selected_candidate_ids: tuple[str, ...],
) -> None:
    expected = {
        "artifact_index": artifact_index,
        "forecaster_id": model_id,
        "episode_id": episode_id,
        "dataset_id": dataset.dataset_id,
        "family_id": dataset.family_id,
        "item_id": episode.item_id,
        "forecast_origin": episode.forecast_origin,
        "sampling_cell": dict(sampling_cell),
        "dataset_plan_sha256": dataset_plan_sha256,
    }
    actual = {
        "artifact_index": expectation.artifact_index,
        "forecaster_id": expectation.forecaster_id,
        "episode_id": expectation.episode_id,
        "dataset_id": expectation.dataset_id,
        "family_id": expectation.family_id,
        "item_id": expectation.item_id,
        "forecast_origin": expectation.forecast_origin,
        "sampling_cell": dict(expectation.sampling_cell),
        "dataset_plan_sha256": expectation.dataset_plan_sha256,
    }
    if actual != expected:
        raise LabelResumeError(
            f"persisted episode expectation changed at index {artifact_index:08d}"
        )
    if not set(expectation.candidate_ids).issubset(selected_candidate_ids):
        raise LabelResumeError("persisted episode contains an unconfigured candidate")
    available_blocks = {block.block_id for block in episode.blocks}
    if not set(expectation.block_ids).issubset(available_blocks):
        raise LabelResumeError("persisted episode contains a missing block that changed")


def _build_label_episode_rows(
    config: AppConfig,
    dataset: Any,
    episode_id: str,
    episode: Any,
    model_id: str,
    correlation: np.ndarray,
    candidate_pool: tuple[str, ...],
    imputer_registry: ImputerRegistry,
    artifact_manager: _LabelArtifactManager,
    candidate_runner: CandidateRunner,
    forecast_runner: ForecastRunner,
    allowed_devices: tuple[str, ...],
) -> _LabelEpisodeRows:
    candidate_ids = candidate_subset(
        candidate_pool,
        config.experiment.max_teacher_candidates_per_episode,
        config.seed,
        dataset.dataset_id,
        episode_id,
    )
    full_graph = build_block_graph(episode.blocks, correlation)
    blocks = connected_subset(
        episode.blocks,
        full_graph.edges,
        config.experiment.max_teacher_blocks_per_episode,
        config.seed,
        dataset.dataset_id,
        episode_id,
        "teacher_blocks",
    )
    block_ids = tuple(block.block_id for block in blocks)
    if not candidate_ids or not blocks:
        return _LabelEpisodeRows(candidate_ids, block_ids, (), ())
    budget = BudgetSpec(
        max_candidates=len(candidate_ids),
        allowed_devices=allowed_devices,
    )
    candidate_params = {
        candidate_id: params
        for candidate_id in candidate_ids
        if (
            params := _pypots_params(
                config,
                imputer_registry.get_spec(candidate_id),
            )
        )
        is not None
    }
    pipeline = BlockwiseFAIS(
        imputer_registry=imputer_registry,
        imputer_artifacts={},
        training_correlation=correlation,
    )
    pseudo = pipeline._pseudo_batch(
        episode.context,
        episode.seed,
        max_blocks=(config.experiment.max_teacher_blocks_per_episode or 8),
    )
    candidates, pseudo_candidates = _run_label_candidate_pairs(
        artifact_manager,
        candidate_runner,
        candidate_ids,
        episode.context,
        pseudo,
        seed=episode.seed,
        params=candidate_params,
        budget=budget,
    )
    try:
        proxy_mask = pseudo.observed_mask | ~episode.context.observed_mask
        graph = build_block_graph(blocks, correlation)
        anchor = candidates["locf"].values
        by_block = {block.block_id: block for block in blocks}
        eligible: dict[str, tuple[str, ...]] = {}
        for block in blocks:
            selector = (
                block.batch_index,
                slice(block.start, block.end),
                block.channel,
            )
            eligible[block.block_id] = tuple(
                candidate_id
                for candidate_id in candidate_ids
                if (
                    (
                        block.end != episode.context.shape[1]
                        or imputer_registry.get_spec(candidate_id).supports_tail
                    )
                    and candidates[candidate_id].native_valid_mask[selector].all()
                )
            )
        spec = _forecast_spec(config, model_id, episode.context.shape[2])
        teacher = TeacherBuilder(
            forecast_runner.predict,
            seasonality=dataset.period,
        )
        unary_labels, clean_loss, anchor_loss = teacher.unary_labels_batched(
            episode_id,
            episode.clean_context[None, ...],
            episode.clean_future[None, ...],
            anchor,
            blocks,
            candidates,
            spec,
            candidate_filter=partial(
                _candidate_is_eligible,
                eligible=eligible,
            ),
        )
        losses = {
            (label.block_id, label.candidate_id): label for label in unary_labels
        }
        unary_rows: list[Mapping[str, Any]] = []
        for block in blocks:
            group_id = f"{model_id}::{episode_id}::{block.block_id}"
            for candidate_id in candidate_ids:
                label = losses.get((block.block_id, candidate_id))
                if label is None:
                    continue
                imputer_spec = imputer_registry.get_spec(candidate_id)
                prior = merge_features(
                    block_features(
                        episode.context,
                        block,
                        dataset.period,
                    ),
                    candidate_features(imputer_spec, spec),
                )
                unary = merge_features(
                    prior,
                    proxy_features(
                        pseudo_candidates[candidate_id],
                        episode.context.values,
                        proxy_mask,
                    ),
                )
                unary_rows.append(
                    {
                        "episode_id": episode_id,
                        "dataset_id": dataset.dataset_id,
                        "family_id": dataset.family_id,
                        "forecaster_id": model_id,
                        "group_id": group_id,
                        "block_id": block.block_id,
                        "candidate_id": candidate_id,
                        "prior_features": prior,
                        "unary_features": unary,
                        "forecast_loss": label.forecast_loss,
                        "clean_loss": clean_loss,
                        "anchor_loss": anchor_loss,
                        "degradation": label.degradation,
                    }
                )
        if not unary_rows:
            return _LabelEpisodeRows(candidate_ids, block_ids, (), ())
        labeled_eligible = {
            block_id: tuple(
                candidate_id
                for candidate_id in candidate_ids
                if (block_id, candidate_id) in losses
            )
            for block_id in eligible
        }
        requests = _pair_label_requests(
            graph.edges,
            labeled_eligible,
            candidate_ids,
            stable_seed(
                episode.seed,
                model_id,
                "pair_labels",
            ),
            limit=config.experiment.max_pair_labels_per_episode,
        )
        batched_requests = tuple(
            (
                by_block[edge.left],
                by_block[edge.right],
                candidates[left_candidate],
                candidates[right_candidate],
            )
            for edge, left_candidate, right_candidate in requests
        )
        interactions = teacher.pair_interactions_batched(
            episode.clean_future[None, ...],
            anchor,
            batched_requests,
            spec,
            anchor_loss=anchor_loss,
            scale_context=episode.clean_context[None, ...],
        )
        pair_rows: list[Mapping[str, Any]] = []
        for (
            edge,
            left_candidate,
            right_candidate,
        ), interaction in zip(requests, interactions, strict=True):
            left = by_block[edge.left]
            right = by_block[edge.right]
            pair_rows.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": dataset.dataset_id,
                    "family_id": dataset.family_id,
                    "forecaster_id": model_id,
                    "left_block": edge.left,
                    "right_block": edge.right,
                    "left_candidate": left_candidate,
                    "right_candidate": right_candidate,
                    "features": pair_features(
                        episode.context,
                        left,
                        right,
                        candidates[left_candidate],
                        candidates[right_candidate],
                        edge_weight=edge.weight,
                    ),
                    "interaction": interaction,
                }
            )
        return _LabelEpisodeRows(
            candidate_ids,
            block_ids,
            tuple(unary_rows),
            tuple(pair_rows),
        )
    finally:
        candidates.clear()
        pseudo_candidates.clear()


def _execute_labels_legacy(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if None in (
        inputs.audit_artifact,
        inputs.imputer_artifacts,
        inputs.forecaster_artifact,
        inputs.forecaster_id,
    ):
        raise ValueError("labels stage is missing a required input")
    assert inputs.audit_artifact is not None
    assert inputs.imputer_artifacts is not None
    assert inputs.forecaster_artifact is not None
    assert inputs.forecaster_id is not None
    forecast_registry = default_forecast_registry()
    selected_forecasters = _forecaster_artifacts(inputs)
    imputer_registry = _selected_imputer_registry(config)
    selected_candidate_ids = _selected_candidate_ids(config)
    candidate_runner = CandidateRunner(imputer_registry)
    allowed_devices = _allowed_devices(config)
    torch_device = _torch_device(config)
    labels_path = preparation.store.root / "teacher_labels.jsonl"
    pairs_path = preparation.store.root / "pair_labels.jsonl"
    group_count = row_count = pair_count = 0
    episode_execution_count = 0
    episode_sampling: dict[str, dict[str, Any]] = {}
    artifact_loading_records: dict[str, dict[str, dict[str, Any]]] = {}
    with (
        labels_path.open("w", encoding="utf-8") as labels_handle,
        pairs_path.open("w", encoding="utf-8") as pairs_handle,
    ):
        for model_id, forecaster_artifact in selected_forecasters:
            adapter = _preflight_forecaster(
                forecast_registry,
                model_id,
                forecaster_artifact,
                device=torch_device,
                batch_size=8,
            )
            forecast_runner = ForecastRunner(forecast_registry, adapters={model_id: adapter})
            teacher: TeacherBuilder | None = None
            artifacts: dict[str, Any] = {}
            pipeline: BlockwiseFAIS | None = None
            try:
                for dataset, items in _datasets(config, inputs.audit_artifact):
                    artifact_store = DatasetImputerArtifactStore(
                        inputs.imputer_artifacts,
                        dataset.dataset_id,
                        imputer_registry,
                    )
                    _, correlation = artifact_store.load_statistics()
                    artifact_manager = _LabelArtifactManager(
                        artifact_store,
                        imputer_registry,
                        config,
                    )
                    candidate_pool = artifact_manager.candidate_pool(allowed_devices)
                    dataset_sampling: dict[str, Any] = {}
                    for episode_id, episode in _episode_iter(
                        config,
                        dataset,
                        items,
                        partition="train",
                        selection_summary=dataset_sampling,
                    ):
                        episode_execution_count += 1
                        candidate_ids = candidate_subset(
                            candidate_pool,
                            config.experiment.max_teacher_candidates_per_episode,
                            config.seed,
                            dataset.dataset_id,
                            episode_id,
                        )
                        full_graph = build_block_graph(episode.blocks, correlation)
                        blocks = connected_subset(
                            episode.blocks,
                            full_graph.edges,
                            config.experiment.max_teacher_blocks_per_episode,
                            config.seed,
                            dataset.dataset_id,
                            episode_id,
                            "teacher_blocks",
                        )
                        if not candidate_ids or not blocks:
                            continue
                        budget = BudgetSpec(
                            max_candidates=len(candidate_ids),
                            allowed_devices=allowed_devices,
                        )
                        candidate_params = {
                            candidate_id: params
                            for candidate_id in candidate_ids
                            if (
                                params := _pypots_params(
                                    config,
                                    imputer_registry.get_spec(candidate_id),
                                )
                            )
                            is not None
                        }
                        pipeline = BlockwiseFAIS(
                            imputer_registry=imputer_registry,
                            imputer_artifacts={},
                            training_correlation=correlation,
                        )
                        pseudo = pipeline._pseudo_batch(
                            episode.context,
                            episode.seed,
                            max_blocks=(config.experiment.max_teacher_blocks_per_episode or 8),
                        )
                        candidates, pseudo_candidates = _run_label_candidate_pairs(
                            artifact_manager,
                            candidate_runner,
                            candidate_ids,
                            episode.context,
                            pseudo,
                            seed=episode.seed,
                            params=candidate_params,
                            budget=budget,
                        )
                        proxy_mask = pseudo.observed_mask | ~episode.context.observed_mask
                        graph = build_block_graph(blocks, correlation)
                        anchor = candidates["locf"].values
                        by_block = {block.block_id: block for block in blocks}
                        eligible: dict[str, tuple[str, ...]] = {}
                        for block in blocks:
                            selector = (
                                block.batch_index,
                                slice(block.start, block.end),
                                block.channel,
                            )
                            eligible[block.block_id] = tuple(
                                candidate_id
                                for candidate_id in candidate_ids
                                if (
                                    (
                                        block.end != episode.context.shape[1]
                                        or imputer_registry.get_spec(candidate_id).supports_tail
                                    )
                                    and candidates[candidate_id].native_valid_mask[selector].all()
                                )
                            )
                        spec = _forecast_spec(config, model_id, episode.context.shape[2])
                        teacher = TeacherBuilder(
                            forecast_runner.predict,
                            seasonality=dataset.period,
                        )
                        unary_labels, clean_loss, anchor_loss = teacher.unary_labels_batched(
                            episode_id,
                            episode.clean_context[None, ...],
                            episode.clean_future[None, ...],
                            anchor,
                            blocks,
                            candidates,
                            spec,
                            candidate_filter=partial(
                                _candidate_is_eligible,
                                eligible=eligible,
                            ),
                        )
                        losses = {
                            (label.block_id, label.candidate_id): label for label in unary_labels
                        }
                        for block in blocks:
                            group_id = f"{model_id}::{episode_id}::{block.block_id}"
                            wrote_group = False
                            for candidate_id in candidate_ids:
                                label = losses.get((block.block_id, candidate_id))
                                if label is None:
                                    continue
                                imputer_spec = imputer_registry.get_spec(candidate_id)
                                prior = merge_features(
                                    block_features(
                                        episode.context,
                                        block,
                                        dataset.period,
                                    ),
                                    candidate_features(imputer_spec, spec),
                                )
                                unary = merge_features(
                                    prior,
                                    proxy_features(
                                        pseudo_candidates[candidate_id],
                                        episode.context.values,
                                        proxy_mask,
                                    ),
                                )
                                _append_jsonl(
                                    labels_handle,
                                    {
                                        "episode_id": episode_id,
                                        "dataset_id": dataset.dataset_id,
                                        "family_id": dataset.family_id,
                                        "forecaster_id": model_id,
                                        "group_id": group_id,
                                        "block_id": block.block_id,
                                        "candidate_id": candidate_id,
                                        "prior_features": prior,
                                        "unary_features": unary,
                                        "forecast_loss": label.forecast_loss,
                                        "clean_loss": clean_loss,
                                        "anchor_loss": anchor_loss,
                                        "degradation": label.degradation,
                                    },
                                )
                                row_count += 1
                                wrote_group = True
                            group_count += int(wrote_group)
                        requests = _pair_label_requests(
                            graph.edges,
                            eligible,
                            candidate_ids,
                            stable_seed(
                                episode.seed,
                                model_id,
                                "pair_labels",
                            ),
                            limit=(config.experiment.max_pair_labels_per_episode),
                        )
                        batched_requests = tuple(
                            (
                                by_block[edge.left],
                                by_block[edge.right],
                                candidates[left_candidate],
                                candidates[right_candidate],
                            )
                            for edge, left_candidate, right_candidate in requests
                        )
                        interactions = teacher.pair_interactions_batched(
                            episode.clean_future[None, ...],
                            anchor,
                            batched_requests,
                            spec,
                            anchor_loss=anchor_loss,
                            scale_context=episode.clean_context[None, ...],
                        )
                        for (
                            edge,
                            left_candidate,
                            right_candidate,
                        ), interaction in zip(requests, interactions, strict=True):
                            left = by_block[edge.left]
                            right = by_block[edge.right]
                            _append_jsonl(
                                pairs_handle,
                                {
                                    "episode_id": episode_id,
                                    "dataset_id": dataset.dataset_id,
                                    "family_id": dataset.family_id,
                                    "forecaster_id": model_id,
                                    "left_block": edge.left,
                                    "right_block": edge.right,
                                    "left_candidate": left_candidate,
                                    "right_candidate": right_candidate,
                                    "features": pair_features(
                                        episode.context,
                                        left,
                                        right,
                                        candidates[left_candidate],
                                        candidates[right_candidate],
                                        edge_weight=edge.weight,
                                    ),
                                    "interaction": interaction,
                                },
                            )
                            pair_count += 1
                        candidates.clear()
                        pseudo_candidates.clear()
                        pipeline = None
                    _record_episode_sampling(
                        episode_sampling,
                        dataset.dataset_id,
                        dataset_sampling,
                    )
                    artifact_manager.close()
                    artifact_loading_records.setdefault(dataset.dataset_id, {})[
                        model_id
                    ] = artifact_manager.audit()
                    pipeline = None
                    artifacts.clear()
                    _empty_cuda_cache(torch_device)
            finally:
                teacher = None
                pipeline = None
                artifacts.clear()
                del forecast_runner
                del adapter
                _empty_cuda_cache(torch_device)
    if row_count == 0:
        raise ValueError("no teacher label rows were generated")
    if pair_count == 0:
        raise ValueError("no pair label rows were generated")
    sampling_manifest = _episode_sampling_manifest(
        "train",
        config.experiment.max_train_episodes_per_dataset,
        episode_sampling,
        episode_execution_count,
    )
    summary = {
        "teacher_labels": str(labels_path),
        "pair_labels": str(pairs_path),
        "forecasters": [model_id for model_id, _ in selected_forecasters],
        "imputer_artifacts": str(inputs.imputer_artifacts.resolve()),
        "origin_partition": "train",
        "episode_count": episode_execution_count,
        "unique_episode_count": sampling_manifest["selected_episode_count"],
        "max_train_episodes_per_dataset": (
            config.experiment.max_train_episodes_per_dataset
        ),
        "episode_sampling": sampling_manifest,
        "artifact_loading": _artifact_loading_manifest(artifact_loading_records),
        "ranking_groups": group_count,
        "unary_rows": row_count,
        "pair_rows": pair_count,
        "selected_candidates": list(selected_candidate_ids),
        "max_teacher_blocks_per_episode": (config.experiment.max_teacher_blocks_per_episode),
        "max_teacher_candidates_per_episode": (
            config.experiment.max_teacher_candidates_per_episode
        ),
        "max_pair_labels_per_episode": (config.experiment.max_pair_labels_per_episode),
        "csdi_num_samples": config.experiment.csdi_num_samples,
        "torch_device": torch_device,
        "forecasters_loaded_sequentially": True,
    }
    _write_json(preparation.store.root / "labels_manifest.json", summary)
    return summary


def _execute_labels_resumable_single(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
    model_id: str,
    forecaster_artifact: Path,
) -> Mapping[str, Any]:
    assert inputs.audit_artifact is not None
    assert inputs.imputer_artifacts is not None
    identity = _labels_resume_identity(
        config,
        inputs,
        model_id,
        forecaster_artifact,
    )
    progress = (
        LabelProgressStore.open_existing(preparation.store.root, identity)
        if preparation.resuming
        else LabelProgressStore.create(preparation.store.root, identity)
    )
    selected_candidate_ids = _selected_candidate_ids(config)
    episode_sampling: dict[str, dict[str, Any]] = {}
    dataset_episode_ids: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    pending: dict[int, Literal["missing", "invalid"]] = {}
    reused_count = 0
    artifact_index = 0

    for dataset, items in _datasets(config, inputs.audit_artifact):
        dataset_sampling: dict[str, Any] = {}
        episodes = tuple(
            _episode_iter(
                config,
                dataset,
                items,
                partition="train",
                selection_summary=dataset_sampling,
            )
        )
        episode_ids = tuple(episode_id for episode_id, _ in episodes)
        if dataset.dataset_id in dataset_episode_ids:
            raise ValueError(f"duplicate dataset ID in label plan: {dataset.dataset_id!r}")
        dataset_episode_ids[dataset.dataset_id] = episode_ids
        plan_sha256 = progress.register_dataset_plan(
            dataset.dataset_id,
            dataset_sampling,
            episode_ids,
        )
        _record_episode_sampling(
            episode_sampling,
            dataset.dataset_id,
            dataset_sampling,
        )
        for episode_id, episode in episodes:
            key = f"{artifact_index:08d}"
            entry = progress.payload["entries"].get(key)
            if entry is None:
                pending[artifact_index] = "missing"
            else:
                if not isinstance(entry, Mapping):
                    raise LabelResumeError(f"progress entry {key} must be an object")
                expectation_payload = entry.get("expectation")
                if not isinstance(expectation_payload, Mapping):
                    raise LabelResumeError(
                        f"progress entry {key} has no valid expectation"
                    )
                expectation = LabelEpisodeExpectation.from_payload(expectation_payload)
                sampling_cell = _label_sampling_cell(episode_id, episode)
                _validate_label_expectation_core(
                    expectation,
                    artifact_index=artifact_index,
                    model_id=model_id,
                    episode_id=episode_id,
                    dataset=dataset,
                    episode=episode,
                    sampling_cell=sampling_cell,
                    dataset_plan_sha256=plan_sha256,
                    selected_candidate_ids=selected_candidate_ids,
                )
                validation = progress.validate_episode(expectation)
                if validation.status == "valid":
                    reused_count += 1
                else:
                    pending[artifact_index] = "invalid"
            artifact_index += 1
        del episodes

    expected_episode_count = artifact_index
    if expected_episode_count == 0:
        raise ValueError("no label episodes were selected")
    expected_keys = {
        f"{index:08d}" for index in range(expected_episode_count)
    }
    extra_entries = set(progress.payload["entries"]).difference(expected_keys)
    if extra_entries:
        raise LabelResumeError(
            "labels progress contains episodes outside the rebuilt plan: "
            + ", ".join(sorted(extra_entries))
        )
    registered_plans = set(progress.payload["dataset_plans"])
    if registered_plans != set(dataset_episode_ids):
        raise LabelResumeError(
            "labels progress dataset plans differ from the rebuilt dataset plan"
        )

    executed_count = 0
    if pending:
        forecast_registry = default_forecast_registry()
        adapter = _preflight_forecaster(
            forecast_registry,
            model_id,
            forecaster_artifact,
            device=_torch_device(config),
            batch_size=8,
        )
        forecast_runner = ForecastRunner(
            forecast_registry,
            adapters={model_id: adapter},
        )
        imputer_registry = _selected_imputer_registry(config)
        candidate_runner = CandidateRunner(imputer_registry)
        allowed_devices = _allowed_devices(config)
        cursor = 0
        try:
            for dataset, items in _datasets(config, inputs.audit_artifact):
                expected_ids = dataset_episode_ids[dataset.dataset_id]
                start = cursor
                stop = start + len(expected_ids)
                pending_indices = tuple(
                    index for index in range(start, stop) if index in pending
                )
                cursor = stop
                if not pending_indices:
                    continue
                second_pass_sampling: dict[str, Any] = {}
                episodes = tuple(
                    _episode_iter(
                        config,
                        dataset,
                        items,
                        partition="train",
                        selection_summary=second_pass_sampling,
                    )
                )
                actual_ids = tuple(episode_id for episode_id, _ in episodes)
                if actual_ids != expected_ids:
                    raise LabelResumeError(
                        f"episode order changed for dataset {dataset.dataset_id!r}"
                    )
                if second_pass_sampling != episode_sampling[dataset.dataset_id]:
                    raise LabelResumeError(
                        f"episode sampling changed for dataset {dataset.dataset_id!r}"
                    )
                artifact_store = DatasetImputerArtifactStore(
                    inputs.imputer_artifacts,
                    dataset.dataset_id,
                    imputer_registry,
                )
                _, correlation = artifact_store.load_statistics()
                artifact_manager = _LabelArtifactManager(
                    artifact_store,
                    imputer_registry,
                    config,
                )
                candidate_pool = artifact_manager.candidate_pool(allowed_devices)
                deferred: tuple[
                    LabelEpisodeExpectation,
                    _LabelEpisodeRows,
                    Mapping[str, Any],
                    bool,
                ] | None = None
                try:
                    for offset, (episode_id, episode) in enumerate(episodes):
                        index = start + offset
                        if index not in pending:
                            continue
                        before_audit = artifact_manager.audit()
                        rows = _build_label_episode_rows(
                            config,
                            dataset,
                            episode_id,
                            episode,
                            model_id,
                            correlation,
                            candidate_pool,
                            imputer_registry,
                            artifact_manager,
                            candidate_runner,
                            forecast_runner,
                            allowed_devices,
                        )
                        plan = progress.payload["dataset_plans"][dataset.dataset_id]
                        expectation = LabelEpisodeExpectation(
                            artifact_index=index,
                            forecaster_id=model_id,
                            episode_id=episode_id,
                            dataset_id=dataset.dataset_id,
                            family_id=dataset.family_id,
                            item_id=episode.item_id,
                            forecast_origin=episode.forecast_origin,
                            sampling_cell=_label_sampling_cell(episode_id, episode),
                            dataset_plan_sha256=str(plan["sha256"]),
                            candidate_ids=rows.candidate_ids,
                            block_ids=rows.block_ids,
                        )
                        validation = progress.validate_episode(expectation)
                        expected_status = pending[index]
                        if validation.status != expected_status:
                            raise LabelResumeError(
                                f"episode {index:08d} changed while labels were executing"
                            )
                        replace = expected_status == "invalid"
                        if index == pending_indices[-1]:
                            deferred = (
                                expectation,
                                rows,
                                before_audit,
                                replace,
                            )
                            continue
                        audit_delta = _label_artifact_audit_delta(
                            before_audit,
                            artifact_manager.audit(),
                        )
                        progress.commit_episode(
                            expectation,
                            rows.unary_rows,
                            rows.pair_rows,
                            outcome=rows.outcome,
                            artifact_loading_delta={
                                "dataset_id": dataset.dataset_id,
                                "forecaster_id": model_id,
                                "audit": audit_delta,
                            },
                            replace=replace,
                        )
                        executed_count += 1
                finally:
                    artifact_manager.close()
                if deferred is None:
                    raise RuntimeError(
                        f"no deferred label episode for dataset {dataset.dataset_id!r}"
                    )
                expectation, rows, deferred_before_audit, replace = deferred
                audit_delta = _label_artifact_audit_delta(
                    deferred_before_audit,
                    artifact_manager.audit(),
                )
                progress.commit_episode(
                    expectation,
                    rows.unary_rows,
                    rows.pair_rows,
                    outcome=rows.outcome,
                    artifact_loading_delta={
                        "dataset_id": dataset.dataset_id,
                        "forecaster_id": model_id,
                        "audit": audit_delta,
                    },
                    replace=replace,
                )
                executed_count += 1
                del episodes
                _empty_cuda_cache(_torch_device(config))
            if cursor != expected_episode_count:
                raise LabelResumeError("second-pass episode count differs from the plan")
        finally:
            del forecast_runner
            del adapter
            _empty_cuda_cache(_torch_device(config))

    labels_path = preparation.store.root / "teacher_labels.jsonl"
    pairs_path = preparation.store.root / "pair_labels.jsonl"
    rebuilt = progress.rebuild_outputs(
        labels_path,
        pairs_path,
        expected_episode_count=expected_episode_count,
    )
    if int(rebuilt["unary_rows"]) == 0:
        raise ValueError("no teacher label rows were generated")
    if int(rebuilt["pair_rows"]) == 0:
        raise ValueError("no pair label rows were generated")
    sampling_manifest = _episode_sampling_manifest(
        "train",
        config.experiment.max_train_episodes_per_dataset,
        episode_sampling,
        expected_episode_count,
    )
    artifact_loading_records = _merge_label_artifact_loading_deltas(
        rebuilt["artifact_loading_deltas"]
    )
    summary = {
        "teacher_labels": str(labels_path),
        "pair_labels": str(pairs_path),
        "teacher_labels_sha256": rebuilt["teacher_labels_sha256"],
        "pair_labels_sha256": rebuilt["pair_labels_sha256"],
        "forecasters": [model_id],
        "imputer_artifacts": str(inputs.imputer_artifacts.resolve()),
        "origin_partition": "train",
        "episode_count": expected_episode_count,
        "unique_episode_count": sampling_manifest["selected_episode_count"],
        "expected_episode_count": expected_episode_count,
        "labeled_episode_count": rebuilt["labeled_episode_count"],
        "no_label_episode_count": rebuilt["no_label_episode_count"],
        "episodes_executed_last_invocation": executed_count,
        "episodes_reused_last_invocation": reused_count,
        "resume_count": int(progress.payload.get("resume_count", 0)),
        "repair_count": int(progress.payload.get("repair_count", 0)),
        "progress": str(progress.progress_path),
        "max_train_episodes_per_dataset": (
            config.experiment.max_train_episodes_per_dataset
        ),
        "episode_sampling": sampling_manifest,
        "artifact_loading": _artifact_loading_manifest(artifact_loading_records),
        "ranking_groups": rebuilt["ranking_groups"],
        "unary_rows": rebuilt["unary_rows"],
        "pair_rows": rebuilt["pair_rows"],
        "selected_candidates": list(selected_candidate_ids),
        "max_teacher_blocks_per_episode": (
            config.experiment.max_teacher_blocks_per_episode
        ),
        "max_teacher_candidates_per_episode": (
            config.experiment.max_teacher_candidates_per_episode
        ),
        "max_pair_labels_per_episode": (
            config.experiment.max_pair_labels_per_episode
        ),
        "csdi_num_samples": config.experiment.csdi_num_samples,
        "torch_device": _torch_device(config),
        "forecasters_loaded_sequentially": True,
    }
    _write_json(preparation.store.root / "labels_manifest.json", summary)
    return summary


def execute_labels(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if None in (
        inputs.audit_artifact,
        inputs.imputer_artifacts,
        inputs.forecaster_artifact,
        inputs.forecaster_id,
    ):
        raise ValueError("labels stage is missing a required input")
    selected_forecasters = _forecaster_artifacts(inputs)
    if preparation.resuming and len(selected_forecasters) != 1:
        raise ValueError("labels resume requires exactly one forecaster ID")
    if len(selected_forecasters) != 1:
        return _execute_labels_legacy(preparation, config, inputs)
    model_id, forecaster_artifact = selected_forecasters[0]
    return _execute_labels_resumable_single(
        preparation,
        config,
        inputs,
        model_id,
        forecaster_artifact,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _matrix(rows: list[dict[str, Any]], field: str, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row[field].get(name, 0.0)) for name in names] for row in rows],
        dtype=float,
    )


def _fit_router_bundle(
    rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    output: Path,
    metadata: Mapping[str, Any],
) -> RouterBundle:
    if not rows or not pair_rows:
        raise ValueError("router fitting requires non-empty unary and pair rows")
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    ordered_rows = [row for group in grouped.values() for row in group]
    feature_names = tuple(
        sorted(set().union(*(set(row["unary_features"]) for row in ordered_rows)))
    )
    prior = _matrix(ordered_rows, "prior_features", feature_names)
    unary = _matrix(ordered_rows, "unary_features", feature_names)
    labels = np.asarray([row["degradation"] for row in ordered_rows], dtype=float)
    groups = tuple(len(group) for group in grouped.values())

    pair_feature_names = tuple(sorted(set().union(*(set(row["features"]) for row in pair_rows))))
    pair_matrix = _matrix(pair_rows, "features", pair_feature_names)
    pair_labels = np.asarray([row["interaction"] for row in pair_rows], dtype=float)
    candidates = tuple(sorted({str(row["candidate_id"]) for row in ordered_rows}))
    bundle = RouterTrainer().fit(
        prior,
        unary,
        labels,
        groups,
        pair_matrix,
        pair_labels,
        feature_names,
        candidates,
        pair_feature_names,
    )
    bundle.metadata.update(
        {
            "created_at": utc_now(),
            "ranking_groups": len(groups),
            "unary_rows": len(ordered_rows),
            "pair_rows": len(pair_rows),
            **dict(metadata),
        }
    )
    bundle.save(output)
    return bundle


def execute_train_router(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if inputs.labels_artifact is None:
        raise ValueError("train-router requires teacher labels")
    rows = _read_jsonl(inputs.labels_artifact)
    if not rows:
        raise ValueError("teacher label file is empty")
    pair_path = inputs.labels_artifact.with_name("pair_labels.jsonl")
    pair_rows = _read_jsonl(pair_path)
    if not pair_rows:
        raise ValueError("pair label file is empty")

    lineage: dict[str, Any] = {}
    labels_manifest = inputs.labels_artifact.with_name("labels_manifest.json")
    if labels_manifest.is_file():
        label_metadata = json.loads(labels_manifest.read_text(encoding="utf-8"))
        artifact_root = label_metadata.get("imputer_artifacts")
        if isinstance(artifact_root, str) and Path(artifact_root).is_dir():
            lineage["imputer_artifacts"] = str(Path(artifact_root).resolve())
        forecasters = label_metadata.get("forecasters")
        if isinstance(forecasters, list):
            lineage["teacher_forecasters"] = list(map(str, forecasters))
    from tsfm_fais.config import load_yaml
    from tsfm_fais.registry_configs import RouterConfig

    router_config = RouterConfig.model_validate(load_yaml(config.registries.router_config))
    lineage.update(
        {
            "beta": router_config.beta,
            "cost_weight": router_config.cost_weight,
            "beta_grid": list(router_config.beta_grid),
            "cost_weight_grid": list(router_config.cost_weight_grid),
        }
    )

    split = config.experiment.split
    if split == "rolling_origin":
        output = preparation.store.root / "router"
        _fit_router_bundle(rows, pair_rows, output, {"split": split, **lineage})
        return {"router_artifact": str(output), "split": split}

    field = "family_id" if split == "leave_family_out" else "forecaster_id"
    held_out_values = tuple(sorted({str(row[field]) for row in rows}))
    if len(held_out_values) < 2:
        raise ValueError(f"{split} requires labels from at least two distinct {field} values")
    root = preparation.store.root / "router_folds"
    folds: dict[str, str] = {}
    fold_index: dict[str, str] = {}
    for held_out in held_out_values:
        training_rows = [row for row in rows if str(row[field]) != held_out]
        training_pairs = [row for row in pair_rows if str(row[field]) != held_out]
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in held_out
        )
        output = root / safe_name
        _fit_router_bundle(
            training_rows,
            training_pairs,
            output,
            {
                "split": split,
                "held_out": held_out,
                "held_out_field": field,
                **lineage,
            },
        )
        folds[held_out] = str(output)
        fold_index[held_out] = safe_name
    manifest = _write_json(
        root / "folds.json",
        {"schema_version": 1, "split": split, "folds": fold_index},
    )
    return {"router_folds": folds, "manifest": str(manifest), "split": split}


def _context_item(item: TimeSeriesItem, episode: Any) -> TimeSeriesItem:
    start_index = episode.forecast_origin - episode.context.shape[1]
    timestamps = (
        None if item.timestamps is None else item.timestamps[start_index : episode.forecast_origin]
    )
    frequency = item.freq
    upper = frequency.upper()
    if upper.endswith("T") and upper[:-1].isdigit():
        frequency = f"{upper[:-1]}min"
    else:
        frequency = {"T": "min", "H": "h", "M": "ME"}.get(upper, frequency)
    context_start = (
        item.start + start_index * to_offset(frequency) if timestamps is None else timestamps[0]
    )
    return TimeSeriesItem(
        item_id=item.item_id,
        values=episode.clean_context,
        variate_names=item.variate_names,
        start=context_start,
        freq=item.freq,
        timestamps=timestamps,
        metadata=item.metadata,
    )


def _router_fold_index(path: Path) -> tuple[str, dict[str, Path]] | None:
    manifest = path / "folds.json" if path.is_dir() else None
    if manifest is None or not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    split = payload.get("split") if isinstance(payload, dict) else None
    folds = payload.get("folds") if isinstance(payload, dict) else None
    if split not in {"leave_family_out", "leave_model_out"}:
        raise ValueError(f"invalid router fold split: {split!r}")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("router fold manifest must contain a non-empty folds mapping")
    resolved: dict[str, Path] = {}
    for held_out, raw_path in folds.items():
        target = Path(str(raw_path))
        if not target.is_absolute():
            target = (manifest.parent / target).resolve()
        resolved[str(held_out)] = target
    return split, resolved


def _validate_router_bundle(
    router: RouterBundle,
    expected_split: str,
    *,
    held_out: str | None = None,
    held_out_field: str | None = None,
) -> None:
    metadata = router.metadata
    if metadata.get("split") != expected_split:
        raise ValueError(
            "router split does not match the experiment configuration: "
            f"{metadata.get('split')!r} != {expected_split!r}"
        )
    if held_out is None:
        return
    if str(metadata.get("held_out")) != held_out:
        raise ValueError(
            f"router held-out value {metadata.get('held_out')!r} does not match {held_out!r}"
        )
    if metadata.get("held_out_field") != held_out_field:
        raise ValueError(
            "router held-out field does not match its fold manifest or evaluation split"
        )


def _load_or_create_imputation_progress(
    preparation: StagePreparation,
    path: Path,
    identity: Mapping[str, Any],
    output: Path,
    assignment_root: Path,
) -> dict[str, Any]:
    if preparation.resuming and path.is_file():
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot resume with invalid imputation progress: {error}"
            ) from error
        if not isinstance(progress, dict):
            raise ValueError("cannot resume: imputation progress must be a JSON object")
        if progress.get("schema_version") != _IMPUTATION_PROGRESS_SCHEMA_VERSION:
            raise ValueError("cannot resume: unsupported imputation progress schema")
        if progress.get("identity") != identity:
            raise ValueError(
                "cannot resume: resolved config or an audit, imputer, router, "
                "registry, or forecaster signature changed"
            )
        if not isinstance(progress.get("entries"), dict):
            raise ValueError("cannot resume: imputation progress entries are invalid")
        declared_count = progress.get("completed_count")
        if declared_count != len(progress["entries"]):
            raise ValueError("cannot resume: imputation progress count is inconsistent")
        progress["status"] = "running"
        progress["resume_count"] = int(progress.get("resume_count", 0)) + 1
        progress["updated_at"] = utc_now()
        _write_json(path, progress)
        return progress

    if preparation.resuming:
        residual_paths = (
            output,
            assignment_root,
            preparation.store.root / "routing_assignments.jsonl",
            preparation.store.root / "imputation_manifest.json",
        )
        has_residual = any(
            candidate.is_file()
            or (candidate.is_dir() and any(candidate.iterdir()))
            for candidate in residual_paths
            if candidate.exists()
        )
        if has_residual:
            raise ValueError(
                "cannot safely resume impute outputs without imputation_progress.json"
            )
    elif path.exists():
        raise FileExistsError(f"imputation progress already exists: {path}")

    progress = {
        "schema_version": _IMPUTATION_PROGRESS_SCHEMA_VERSION,
        "status": "running",
        "identity": dict(identity),
        "entries": {},
        "completed_count": 0,
        "repair_count": 0,
        "resume_count": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    _write_json(path, progress)
    return progress


def _safe_artifact_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe imputation artifact path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"imputation artifact path escapes its root: {relative!r}")
    return resolved


def _npz_scalar(archive: Any, key: str) -> Any:
    if key not in archive:
        raise ValueError(f"imputation archive is missing {key!r}")
    values = np.asarray(archive[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"imputation archive field {key!r} must be scalar")
    return values[0].item() if hasattr(values[0], "item") else values[0]


def _validate_resumable_imputation(
    entry: Mapping[str, Any],
    *,
    expected_index: int,
    episode_id: str,
    dataset_id: str,
    family_id: str,
    item_id: str,
    model_id: str,
    forecast_mode: str,
    relative_file: Path,
    relative_assignment: Path,
    artifact_root: Path,
    episode: Any,
    period: int,
    mase_scale: np.ndarray,
    mase_scale_lag: int,
    allowed_candidate_ids: frozenset[str],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    expected_identity = {
        "index": expected_index,
        "episode_id": episode_id,
        "dataset_id": dataset_id,
        "family_id": family_id,
        "item_id": item_id,
        "forecaster_id": model_id,
        "forecast_mode": forecast_mode,
        "file": str(relative_file),
        "assignment_file": str(relative_assignment),
    }
    for field, expected in expected_identity.items():
        if entry.get(field) != expected:
            raise ValueError(
                "cannot resume: progress entry identity differs at "
                f"{expected_index:08d}.{field}"
            )
    try:
        output_path = _safe_artifact_path(
            artifact_root / "imputations", str(relative_file)
        )
        assignment_path = _safe_artifact_path(
            artifact_root, str(relative_assignment)
        )
    except ValueError as error:
        raise ValueError(f"cannot resume: {error}") from error
    if not output_path.is_file() or not assignment_path.is_file():
        return False, "committed output pair is incomplete", None
    expected_npz_hash = entry.get("npz_sha256")
    expected_assignment_hash = entry.get("assignment_sha256")
    if not isinstance(expected_npz_hash, str) or not isinstance(
        expected_assignment_hash, str
    ):
        return False, "committed output hashes are missing", None
    if _file_sha256(output_path) != expected_npz_hash:
        return False, "imputation NPZ hash mismatch", None
    if _file_sha256(assignment_path) != expected_assignment_hash:
        return False, "assignment JSON hash mismatch", None
    try:
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, f"invalid assignment JSON: {error}", None
    if not isinstance(assignment, dict):
        return False, "assignment JSON must be an object", None
    for field, expected in expected_identity.items():
        assignment_field = "artifact_index" if field == "index" else field
        if assignment.get(assignment_field) != expected:
            return False, f"assignment identity mismatch: {assignment_field}", None
    if assignment.get("schema_version") != _IMPUTATION_SCHEMA_VERSION:
        return False, "assignment schema mismatch", None
    assignment_mask_identity = {
        "mask_protocol": "sequence_mask_v2",
        "mask_seed": int(episode.mask_seed),
        "mask_realization_id": str(episode.mask_realization_id),
    }
    for field, expected in assignment_mask_identity.items():
        if assignment.get(field) != expected:
            return False, f"assignment mask identity mismatch: {field}", None
    for field, expected in (
        ("target_missing_rate", episode.target_missing_rate),
        ("global_missing_rate", episode.global_missing_rate),
        ("local_missing_rate", episode.local_missing_rate),
    ):
        try:
            actual = float(assignment.get(field))
        except (TypeError, ValueError):
            return False, f"assignment mask rate is invalid: {field}", None
        if not np.isclose(actual, float(expected), rtol=0.0, atol=1e-15):
            return False, f"assignment mask rate mismatch: {field}", None
    try:
        assignment_scale = np.asarray(assignment.get("mase_scale"), dtype=float)
        assignment_lag = int(assignment.get("mase_scale_lag"))
    except (TypeError, ValueError):
        return False, "assignment MASE scale is invalid", None
    if not np.array_equal(
        assignment_scale.reshape(-1), np.asarray(mase_scale, dtype=float).reshape(-1)
    ):
        return False, "assignment MASE scale differs", None
    if assignment_lag != int(mase_scale_lag):
        return False, "assignment MASE lag differs", None
    raw_candidate_ids = assignment.get("candidate_ids")
    if not isinstance(raw_candidate_ids, list) or any(
        not isinstance(candidate_id, str) for candidate_id in raw_candidate_ids
    ):
        return False, "assignment candidate IDs are invalid", None
    candidate_ids = tuple(raw_candidate_ids)
    if len(set(candidate_ids)) != len(candidate_ids):
        return False, "assignment candidate IDs are duplicated", None
    if not set(candidate_ids).issubset(allowed_candidate_ids):
        return False, "assignment contains an unconfigured candidate ID", None
    if entry.get("candidate_ids") != list(candidate_ids):
        return False, "progress and assignment candidate IDs differ", None

    required_arrays = {
        "schema_version",
        "episode_id",
        "dataset_id",
        "family_id",
        "item_id",
        "forecaster_id",
        "forecast_mode",
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
        "period",
        "candidate_ids",
        "candidate_values",
        "candidate_native_valid",
        "candidate_status",
        "candidate_runtime_seconds",
        "candidate_peak_memory_bytes",
        "pipeline_runtime_seconds",
        "pipeline_rss_before_bytes",
        "pipeline_rss_after_bytes",
        "pipeline_rss_delta_bytes",
        "pipeline_peak_memory_bytes",
    }
    try:
        with np.load(output_path, allow_pickle=False) as archive:
            missing = required_arrays.difference(archive.files)
            if missing:
                return False, "imputation NPZ is missing: " + ", ".join(sorted(missing)), None
            scalar_expectations = {
                "schema_version": _IMPUTATION_SCHEMA_VERSION,
                "episode_id": episode_id,
                "dataset_id": dataset_id,
                "family_id": family_id,
                "item_id": item_id,
                "forecaster_id": model_id,
                "forecast_mode": forecast_mode,
                "period": int(period),
                "mask_protocol": "sequence_mask_v2",
                "mask_seed": int(episode.mask_seed),
                "mask_realization_id": str(episode.mask_realization_id),
            }
            for field, expected in scalar_expectations.items():
                if _npz_scalar(archive, field) != expected:
                    return False, f"imputation NPZ identity mismatch: {field}", None
            for field, expected in (
                ("target_missing_rate", episode.target_missing_rate),
                ("global_missing_rate", episode.global_missing_rate),
                ("local_missing_rate", episode.local_missing_rate),
            ):
                actual = float(_npz_scalar(archive, field))
                if not np.isclose(actual, float(expected), rtol=0.0, atol=1e-15):
                    return False, f"imputation NPZ mask rate mismatch: {field}", None
            stored_scale = np.asarray(archive["mase_scale"], dtype=float).reshape(-1)
            expected_scale = np.asarray(mase_scale, dtype=float).reshape(-1)
            if not np.array_equal(stored_scale, expected_scale):
                return False, "imputation NPZ MASE scale differs", None
            if int(_npz_scalar(archive, "mase_scale_lag")) != int(mase_scale_lag):
                return False, "imputation NPZ MASE lag differs", None
            values = np.asarray(archive["values"], dtype=float)
            observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
            clean_context = np.asarray(archive["clean_context"], dtype=float)
            clean_future = np.asarray(archive["clean_future"], dtype=float)
            expected_context = np.asarray(episode.clean_context, dtype=float)
            expected_future = np.asarray(episode.clean_future, dtype=float)
            expected_mask = np.asarray(episode.context.observed_mask[0], dtype=bool)
            if not (
                values.shape
                == observed_mask.shape
                == clean_context.shape
                == expected_context.shape
            ):
                return False, "imputation NPZ context shapes differ", None
            if clean_future.shape != expected_future.shape:
                return False, "imputation NPZ future shape differs", None
            if not np.array_equal(observed_mask, expected_mask):
                return False, "imputation NPZ observed mask differs", None
            if not np.array_equal(clean_context, expected_context) or not np.array_equal(
                clean_future, expected_future
            ):
                return False, "imputation NPZ clean episode content differs", None
            if not (
                np.isfinite(values).all()
                and np.isfinite(clean_context).all()
                and np.isfinite(clean_future).all()
            ):
                return False, "imputation NPZ contains non-finite primary values", None
            if not np.array_equal(values[observed_mask], clean_context[observed_mask]):
                return False, "imputation NPZ changed observed values", None
            stored_candidate_ids = tuple(
                str(value) for value in np.asarray(archive["candidate_ids"]).tolist()
            )
            if stored_candidate_ids != candidate_ids:
                return False, "imputation NPZ candidate IDs differ", None
            count = len(candidate_ids)
            candidate_values = np.asarray(archive["candidate_values"], dtype=float)
            candidate_native = np.asarray(
                archive["candidate_native_valid"], dtype=bool
            )
            expected_candidate_shape = (count, *expected_context.shape)
            if candidate_values.shape != expected_candidate_shape or (
                candidate_native.shape != expected_candidate_shape
            ):
                return False, "imputation NPZ candidate shapes differ", None
            if not np.isfinite(candidate_values).all():
                return False, "imputation NPZ candidate values are non-finite", None
            if count and not np.array_equal(
                candidate_values[:, observed_mask],
                np.broadcast_to(clean_context[observed_mask], (count, observed_mask.sum())),
            ):
                return False, "candidate output changed observed values", None
            statuses = np.asarray(archive["candidate_status"]).reshape(-1)
            runtimes = np.asarray(
                archive["candidate_runtime_seconds"], dtype=float
            ).reshape(-1)
            memories = np.asarray(
                archive["candidate_peak_memory_bytes"], dtype=np.int64
            ).reshape(-1)
            if not (len(statuses) == len(runtimes) == len(memories) == count):
                return False, "imputation NPZ candidate metadata lengths differ", None
            if any(not str(status) for status in statuses):
                return False, "imputation NPZ candidate status is empty", None
            if not np.isfinite(runtimes).all() or np.any(runtimes < 0):
                return False, "imputation NPZ candidate runtime is invalid", None
            if np.any(memories < 0):
                return False, "imputation NPZ candidate memory is invalid", None
            pipeline_runtime = float(_npz_scalar(archive, "pipeline_runtime_seconds"))
            pipeline_rss_delta = int(_npz_scalar(archive, "pipeline_rss_delta_bytes"))
            if not np.isfinite(pipeline_runtime) or pipeline_runtime < 0:
                return False, "imputation NPZ pipeline runtime is invalid", None
            resource_fields = (
                "pipeline_rss_before_bytes",
                "pipeline_rss_after_bytes",
                "pipeline_rss_delta_bytes",
                "pipeline_peak_memory_bytes",
            )
            if any(int(_npz_scalar(archive, field)) < 0 for field in resource_fields):
                return False, "imputation NPZ pipeline memory is invalid", None
    except Exception as error:
        return False, f"cannot load imputation NPZ: {type(error).__name__}: {error}", None
    try:
        assignment_runtime = float(assignment.get("pipeline_runtime_seconds", -1.0))
        assignment_memory = int(assignment.get("pipeline_rss_delta_bytes", -1))
    except (TypeError, ValueError) as error:
        return False, f"assignment pipeline resource fields are invalid: {error}", None
    if assignment_runtime != pipeline_runtime:
        return False, "assignment and NPZ pipeline runtime differ", None
    if assignment_memory != pipeline_rss_delta:
        return False, "assignment and NPZ pipeline memory differ", None
    return True, None, assignment


class _ImputeArtifactManager:
    """Single-lease dataset artifact loader for candidate-major imputation."""

    def __init__(
        self,
        store: DatasetImputerArtifactStore,
        registry: ImputerRegistry,
        config: AppConfig,
    ) -> None:
        self.store = store
        self.registry = registry
        self.config = config
        self._active_leases = 0
        self._max_active_leases = 0
        self._load_counts: Counter[str] = Counter()
        self._failures: dict[str, str] = {}
        self._load_modes: dict[str, str] = {}
        self._load_seconds = 0.0
        self._repair_reload_count = 0

    def declared_available(self, candidate_ids: Iterable[str]) -> set[str]:
        return {
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in self.registry
            and self.registry.get_spec(candidate_id).fit_scope != "none"
            and self.store.status(candidate_id) == "fitted"
        }

    def declared_failures(self, candidate_ids: Iterable[str]) -> dict[str, str]:
        requested = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in self.registry
            and self.registry.get_spec(candidate_id).fit_scope != "none"
            and self.store.status(candidate_id) != "fitted"
        )
        if not requested:
            return {}
        result = self.store.load_artifacts(requested)
        self._failures.update(result.failures)
        return dict(result.failures)

    def acquire(
        self,
        candidate_id: str,
        *,
        repair: bool = False,
    ) -> tuple[dict[str, Any], dict[str, str], bool]:
        spec = self.registry.get_spec(candidate_id)
        if spec.fit_scope == "none":
            return {}, {}, False
        if candidate_id in self._failures and self._load_counts[candidate_id] == 0:
            return {}, {candidate_id: self._failures[candidate_id]}, False
        if self._active_leases:
            raise RuntimeError("imputation artifact leases must not overlap")
        previous_loads = self._load_counts[candidate_id]
        if previous_loads and not repair:
            raise RuntimeError(
                f"candidate {candidate_id!r} was already loaded for this dataset"
            )
        params = _pypots_params(self.config, spec)
        result = self.store.load_artifacts(
            (candidate_id,),
            adapter_params={} if params is None else {candidate_id: params},
        )
        self._load_seconds += result.load_seconds
        if result.attempted_ids:
            self._load_counts[candidate_id] += 1
            if previous_loads:
                self._repair_reload_count += 1
        self._load_modes.update(result.load_modes)
        self._failures.update(result.failures)
        leased = bool(result.artifacts)
        if leased:
            self._active_leases += 1
            self._max_active_leases = max(
                self._max_active_leases, self._active_leases
            )
        return result.artifacts, result.failures, leased

    def release(self, artifacts: dict[str, Any], leased: bool) -> None:
        artifacts.clear()
        if not leased:
            return
        self._active_leases -= 1
        if self._active_leases != 0:
            raise RuntimeError("imputation artifact lease accounting is inconsistent")
        gc.collect()
        _empty_cuda_cache(_torch_device(self.config))

    def audit(self) -> dict[str, Any]:
        if self._active_leases:
            raise RuntimeError("cannot audit with an active imputation artifact lease")
        return {
            "load_counts": {
                candidate_id: int(count)
                for candidate_id, count in sorted(self._load_counts.items())
            },
            "load_modes": dict(sorted(self._load_modes.items())),
            "load_seconds": self._load_seconds,
            "load_failures": dict(sorted(self._failures.items())),
            "max_active_leases": self._max_active_leases,
            "repair_reload_count": self._repair_reload_count,
            "common_path_max_loads_per_candidate": max(
                (
                    count
                    for candidate_id, count in self._load_counts.items()
                    if candidate_id not in self._failures
                ),
                default=0,
            ),
        }


@dataclass
class _ImputeEpisodeWork:
    index: int
    episode_id: str
    episode: Any
    item: TimeSeriesItem
    spec: ForecastSpec
    relative: Path
    relative_assignment: Path
    entry_key: str
    invalid_reason: str | None
    pipeline_rss_before: int
    mase_scale: np.ndarray
    mase_scale_lag: int
    plan: RoutePlan | None = None
    prepare_seconds: float = 0.0
    raw_actual: dict[str, CandidateResult] = dataclass_field(default_factory=dict)
    raw_pseudo: dict[str, CandidateResult] = dataclass_field(default_factory=dict)
    actual_modes: dict[str, str] = dataclass_field(default_factory=dict)


def _prepare_impute_work(
    work: _ImputeEpisodeWork,
    pipeline: BlockwiseFAIS,
    available_artifact_ids: set[str],
    artifact_failures: Mapping[str, str],
    allowed_devices: tuple[str, ...],
) -> None:
    started = perf_counter()
    work.plan = pipeline.prepare_route(
        work.item,
        work.episode.context.observed_mask[0],
        work.spec,
        BudgetSpec(
            max_candidates=pipeline.shortlist_size,
            allowed_devices=allowed_devices,
        ),
        seed=work.episode.seed,
        available_artifact_ids=available_artifact_ids,
        artifact_load_failures=artifact_failures,
    )
    work.prepare_seconds += perf_counter() - started


def _fallback_ids(plan: RoutePlan, pipeline: BlockwiseFAIS) -> frozenset[str]:
    identifiers: set[str] = set()
    if any(block.end < plan.batch.shape[1] for block in plan.blocks):
        identifiers.update(pipeline.fallback_internal)
    if any(block.end == plan.batch.shape[1] for block in plan.blocks):
        identifiers.update(pipeline.fallback_tail)
    identifiers.discard("train_median")
    return frozenset(
        candidate_id
        for candidate_id in identifiers
        if candidate_id in pipeline.imputer_registry
        and (
            pipeline.imputer_registry.get_spec(candidate_id).fit_scope == "none"
            or candidate_id in plan.available_artifact_ids
        )
    )


def _evaluation_candidate_ids(
    registry: ImputerRegistry,
    available_artifact_ids: set[str],
    allowed_devices: tuple[str, ...],
) -> frozenset[str]:
    return frozenset(
        spec.imputer_id
        for spec in registry.specs()
        if registry.availability(spec.imputer_id).available
        and (spec.device == "any" or spec.device in allowed_devices)
        and (spec.fit_scope == "none" or spec.imputer_id in available_artifact_ids)
    )


def _desired_actual_mode(
    work: _ImputeEpisodeWork,
    candidate_id: str,
    pipeline: BlockwiseFAIS,
    evaluation_ids: frozenset[str],
    save_all: bool,
) -> str | None:
    if work.plan is None:  # pragma: no cover - preparation invariant
        raise RuntimeError("imputation work has no route plan")
    if candidate_id in work.plan.shortlist or candidate_id in _fallback_ids(
        work.plan, pipeline
    ):
        return "routing"
    if save_all and candidate_id in evaluation_ids:
        return "evaluation"
    return None


def _raw_run_budget(allowed_devices: tuple[str, ...]) -> BudgetSpec:
    return BudgetSpec(max_candidates=1, allowed_devices=allowed_devices)


def _run_raw_impute_candidate(
    work: _ImputeEpisodeWork,
    candidate_id: str,
    mode: str,
    pipeline: BlockwiseFAIS,
    artifacts: Mapping[str, Any],
    artifact_failures: Mapping[str, str],
    config: AppConfig,
    allowed_devices: tuple[str, ...],
    *,
    run_actual: bool,
    run_pseudo: bool,
) -> None:
    params: dict[str, Mapping[str, Any]] = {}
    candidate_params = _pypots_params(
        config, pipeline.imputer_registry.get_spec(candidate_id)
    )
    if candidate_params is not None:
        params[candidate_id] = candidate_params
    if run_actual:
        work.raw_actual.update(
            pipeline.candidate_runner.run_many(
                (candidate_id,),
                work.plan.batch,  # type: ignore[union-attr]
                artifacts,
                seed=work.episode.seed,
                params=params,
                artifact_failures=artifact_failures,
                budget=_raw_run_budget(allowed_devices),
            )
        )
        work.actual_modes[candidate_id] = mode
    if run_pseudo:
        if work.plan is None or work.plan.pseudo_batch is None:
            raise RuntimeError("shortlisted candidate has no pseudo batch")
        work.raw_pseudo.update(
            pipeline.candidate_runner.run_many(
                (candidate_id,),
                work.plan.pseudo_batch,
                artifacts,
                seed=work.episode.seed,
                params=params,
                artifact_failures=artifact_failures,
                budget=_raw_run_budget(allowed_devices),
            )
        )


def _memory_limited_result(
    result: CandidateResult,
    batch: SeriesBatch,
    budget: BudgetSpec,
) -> CandidateResult:
    if budget.max_memory_bytes is None or result.peak_memory_bytes <= budget.max_memory_bytes:
        return result
    native = np.asarray(result.native_valid_mask, dtype=bool).copy()
    native[~batch.observed_mask] = False
    return CandidateResult(
        imputer_id=result.imputer_id,
        values=result.values,
        native_valid_mask=native,
        uncertainty=result.uncertainty,
        runtime_seconds=result.runtime_seconds,
        peak_memory_bytes=result.peak_memory_bytes,
        status=CandidateStatus.FAILED,
        failure_reason=(
            f"peak memory {result.peak_memory_bytes} exceeds budget "
            f"{budget.max_memory_bytes}"
        ),
        metadata=dict(result.metadata),
    )


def _budgeted_route_results(
    work: _ImputeEpisodeWork,
    registry: ImputerRegistry,
) -> tuple[dict[str, CandidateResult], dict[str, CandidateResult]]:
    if work.plan is None:  # pragma: no cover - preparation invariant
        raise RuntimeError("imputation work has no route plan")
    plan = work.plan
    actual: dict[str, CandidateResult] = {}
    elapsed = 0.0
    for candidate_id in plan.shortlist:
        spec = registry.get_spec(candidate_id)
        if spec.device != "any" and spec.device not in plan.budget.allowed_devices:
            result = failed_candidate_result(
                candidate_id,
                plan.batch,
                f"device {spec.device!r} is excluded by the budget",
                status=CandidateStatus.UNAVAILABLE,
            )
        elif (
            plan.budget.max_runtime_seconds is not None
            and elapsed >= plan.budget.max_runtime_seconds
        ):
            result = failed_candidate_result(
                candidate_id,
                plan.batch,
                "runtime budget exhausted before candidate execution",
                status=CandidateStatus.UNAVAILABLE,
            )
        else:
            if candidate_id not in work.raw_actual:
                raise RuntimeError(
                    f"missing raw actual result for {work.episode_id}/{candidate_id}"
                )
            result = _memory_limited_result(
                work.raw_actual[candidate_id], plan.batch, plan.budget
            )
            elapsed += result.runtime_seconds
        actual[candidate_id] = result

    pseudo: dict[str, CandidateResult] = {}
    if plan.pseudo_batch is None:
        return actual, pseudo
    elapsed = sum(result.runtime_seconds for result in actual.values())
    for candidate_id in plan.shortlist:
        spec = registry.get_spec(candidate_id)
        if spec.device != "any" and spec.device not in plan.budget.allowed_devices:
            result = failed_candidate_result(
                candidate_id,
                plan.pseudo_batch,
                f"device {spec.device!r} is excluded by the budget",
                status=CandidateStatus.UNAVAILABLE,
            )
        elif (
            plan.budget.max_runtime_seconds is not None
            and elapsed >= plan.budget.max_runtime_seconds
        ):
            result = failed_candidate_result(
                candidate_id,
                plan.pseudo_batch,
                "runtime budget exhausted before candidate execution",
                status=CandidateStatus.UNAVAILABLE,
            )
        else:
            if candidate_id not in work.raw_pseudo:
                raise RuntimeError(
                    f"missing raw pseudo result for {work.episode_id}/{candidate_id}"
                )
            result = _memory_limited_result(
                work.raw_pseudo[candidate_id], plan.pseudo_batch, plan.budget
            )
            elapsed += result.runtime_seconds
        pseudo[candidate_id] = result
    return actual, pseudo


def _rebuild_impute_plans(
    works: Iterable[_ImputeEpisodeWork],
    pipeline: BlockwiseFAIS,
    available_artifact_ids: set[str],
    artifact_failures: Mapping[str, str],
    allowed_devices: tuple[str, ...],
) -> None:
    for work in works:
        _prepare_impute_work(
            work,
            pipeline,
            available_artifact_ids,
            artifact_failures,
            allowed_devices,
        )


def _candidate_tasks_missing(
    work: _ImputeEpisodeWork,
    candidate_id: str,
    pipeline: BlockwiseFAIS,
    evaluation_ids: frozenset[str],
    save_all: bool,
) -> tuple[str | None, bool, bool]:
    if work.plan is None:  # pragma: no cover - preparation invariant
        raise RuntimeError("imputation work has no route plan")
    mode = _desired_actual_mode(
        work, candidate_id, pipeline, evaluation_ids, save_all
    )
    run_actual = mode is not None and (
        candidate_id not in work.raw_actual
    )
    run_pseudo = (
        work.plan.pseudo_batch is not None
        and candidate_id in work.plan.shortlist
        and candidate_id not in work.raw_pseudo
    )
    return mode, run_actual, run_pseudo


def _run_impute_candidate_sweep(
    works: list[_ImputeEpisodeWork],
    pipeline: BlockwiseFAIS,
    manager: _ImputeArtifactManager,
    config: AppConfig,
    allowed_devices: tuple[str, ...],
    available_artifact_ids: set[str],
    artifact_failures: dict[str, str],
) -> dict[str, Any]:
    save_all = config.experiment.save_all_candidate_outputs
    registry = pipeline.imputer_registry

    def task_candidates() -> tuple[str, ...]:
        evaluation_ids = _evaluation_candidate_ids(
            registry, available_artifact_ids, allowed_devices
        )
        return tuple(
            candidate_id
            for candidate_id in registry.ids
            if any(
                _desired_actual_mode(
                    work,
                    candidate_id,
                    pipeline,
                    evaluation_ids,
                    save_all,
                )
                is not None
                for work in works
            )
        )

    processed: set[str] = set()
    for candidate_id in task_candidates():
        evaluation_ids = _evaluation_candidate_ids(
            registry, available_artifact_ids, allowed_devices
        )
        artifacts, failures, leased = manager.acquire(candidate_id)
        try:
            if failures:
                artifact_failures.update(failures)
                if candidate_id in available_artifact_ids:
                    available_artifact_ids.remove(candidate_id)
                    _rebuild_impute_plans(
                        works,
                        pipeline,
                        available_artifact_ids,
                        artifact_failures,
                        allowed_devices,
                    )
                    evaluation_ids = _evaluation_candidate_ids(
                        registry, available_artifact_ids, allowed_devices
                    )
            for work in works:
                mode, run_actual, run_pseudo = _candidate_tasks_missing(
                    work,
                    candidate_id,
                    pipeline,
                    evaluation_ids,
                    save_all,
                )
                if mode is None and not run_pseudo:
                    continue
                _run_raw_impute_candidate(
                    work,
                    candidate_id,
                    mode or "routing",
                    pipeline,
                    artifacts,
                    failures,
                    config,
                    allowed_devices,
                    run_actual=run_actual,
                    run_pseudo=run_pseudo,
                )
        finally:
            manager.release(artifacts, leased)
        processed.add(candidate_id)

    repair_passes = 0
    while True:
        evaluation_ids = _evaluation_candidate_ids(
            registry, available_artifact_ids, allowed_devices
        )
        missing_candidate: str | None = None
        for candidate_id in registry.ids:
            if any(
                any(
                    _candidate_tasks_missing(
                        work,
                        candidate_id,
                        pipeline,
                        evaluation_ids,
                        save_all,
                    )[1:]
                )
                for work in works
            ):
                missing_candidate = candidate_id
                break
        if missing_candidate is None:
            break
        repair_passes += 1
        if repair_passes > len(registry.ids) * 3:
            raise RuntimeError("imputation candidate repair did not converge")
        candidate_id = missing_candidate
        repair = candidate_id in processed
        artifacts, failures, leased = manager.acquire(candidate_id, repair=repair)
        try:
            if failures:
                artifact_failures.update(failures)
                available_artifact_ids.discard(candidate_id)
                for work in works:
                    work.raw_actual.pop(candidate_id, None)
                    work.raw_pseudo.pop(candidate_id, None)
                    work.actual_modes.pop(candidate_id, None)
                _rebuild_impute_plans(
                    works,
                    pipeline,
                    available_artifact_ids,
                    artifact_failures,
                    allowed_devices,
                )
                evaluation_ids = _evaluation_candidate_ids(
                    registry, available_artifact_ids, allowed_devices
                )
                for work in works:
                    mode, run_actual, run_pseudo = _candidate_tasks_missing(
                        work,
                        candidate_id,
                        pipeline,
                        evaluation_ids,
                        save_all,
                    )
                    if mode is None and not run_pseudo:
                        continue
                    _run_raw_impute_candidate(
                        work,
                        candidate_id,
                        mode or "routing",
                        pipeline,
                        {},
                        failures,
                        config,
                        allowed_devices,
                        run_actual=run_actual,
                        run_pseudo=run_pseudo,
                    )
                processed.add(candidate_id)
                continue
            evaluation_ids = _evaluation_candidate_ids(
                registry, available_artifact_ids, allowed_devices
            )
            for work in works:
                mode, run_actual, run_pseudo = _candidate_tasks_missing(
                    work,
                    candidate_id,
                    pipeline,
                    evaluation_ids,
                    save_all,
                )
                if mode is None and not run_pseudo:
                    continue
                _run_raw_impute_candidate(
                    work,
                    candidate_id,
                    mode or "routing",
                    pipeline,
                    artifacts,
                    {},
                    config,
                    allowed_devices,
                    run_actual=run_actual,
                    run_pseudo=run_pseudo,
                )
        finally:
            manager.release(artifacts, leased)
        processed.add(candidate_id)
    return manager.audit()


def _commit_imputation_output(
    preparation: StagePreparation,
    output: Path,
    dataset: Any,
    model_id: str,
    registry: ImputerRegistry,
    work: _ImputeEpisodeWork,
    result: FAISResult,
    pipeline_runtime_seconds: float,
    pipeline_rss_after: int,
    pipeline_rss_delta: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_ids = tuple(
        candidate_id
        for candidate_id in registry.ids
        if candidate_id in result.candidates
    )
    if candidate_ids:
        candidate_values = np.stack(
            [result.candidates[candidate_id].values[0] for candidate_id in candidate_ids]
        )
        candidate_native_valid = np.stack(
            [
                result.candidates[candidate_id].native_valid_mask[0]
                for candidate_id in candidate_ids
            ]
        )
    else:
        candidate_values = np.empty((0, *result.values.shape), dtype=float)
        candidate_native_valid = np.empty(
            (0, *result.values.shape), dtype=bool
        )
    mechanism, missing_rate, configured_seed = _episode_parameters(
        work.episode_id
    )
    target = output / work.relative
    assignment_target = preparation.store.root / work.relative_assignment
    _write_npz_atomic(
        target,
        schema_version=np.asarray([_IMPUTATION_SCHEMA_VERSION], dtype=np.int64),
        episode_id=np.asarray([work.episode_id], dtype=str),
        dataset_id=np.asarray([dataset.dataset_id], dtype=str),
        family_id=np.asarray([dataset.family_id], dtype=str),
        item_id=np.asarray([work.episode.item_id], dtype=str),
        forecaster_id=np.asarray([model_id], dtype=str),
        forecast_mode=np.asarray([work.spec.mode], dtype=str),
        values=result.values,
        observed_mask=result.observed_mask,
        clean_context=work.episode.clean_context,
        clean_future=work.episode.clean_future,
        mask_protocol=np.asarray(["sequence_mask_v2"], dtype=str),
        mask_seed=np.asarray([work.episode.mask_seed], dtype=np.uint64),
        mask_realization_id=np.asarray(
            [work.episode.mask_realization_id], dtype=str
        ),
        target_missing_rate=np.asarray(
            [work.episode.target_missing_rate], dtype=np.float64
        ),
        global_missing_rate=np.asarray(
            [work.episode.global_missing_rate], dtype=np.float64
        ),
        local_missing_rate=np.asarray(
            [work.episode.local_missing_rate], dtype=np.float64
        ),
        mase_scale=np.asarray(work.mase_scale, dtype=np.float64),
        mase_scale_lag=np.asarray([work.mase_scale_lag], dtype=np.int64),
        period=np.asarray([dataset.period], dtype=np.int64),
        candidate_ids=np.asarray(candidate_ids, dtype=str),
        candidate_values=candidate_values,
        candidate_native_valid=candidate_native_valid,
        candidate_status=np.asarray(
            [
                result.candidates[candidate_id].status.value
                for candidate_id in candidate_ids
            ],
            dtype=str,
        ),
        candidate_runtime_seconds=np.asarray(
            [
                result.candidates[candidate_id].runtime_seconds
                for candidate_id in candidate_ids
            ],
            dtype=float,
        ),
        candidate_peak_memory_bytes=np.asarray(
            [
                result.candidates[candidate_id].peak_memory_bytes
                for candidate_id in candidate_ids
            ],
            dtype=np.int64,
        ),
        pipeline_runtime_seconds=np.asarray(
            pipeline_runtime_seconds, dtype=np.float64
        ),
        pipeline_rss_before_bytes=np.asarray(
            work.pipeline_rss_before, dtype=np.int64
        ),
        pipeline_rss_after_bytes=np.asarray(
            pipeline_rss_after, dtype=np.int64
        ),
        pipeline_rss_delta_bytes=np.asarray(
            pipeline_rss_delta, dtype=np.int64
        ),
        pipeline_peak_memory_bytes=np.asarray(
            pipeline_rss_delta, dtype=np.int64
        ),
    )
    assignment = {
        "schema_version": _IMPUTATION_SCHEMA_VERSION,
        "artifact_index": work.index,
        "episode_id": work.episode_id,
        "dataset_id": dataset.dataset_id,
        "family_id": dataset.family_id,
        "forecaster_id": model_id,
        "forecast_mode": work.spec.mode,
        "item_id": work.episode.item_id,
        "forecast_origin": work.episode.forecast_origin,
        "mechanism": mechanism,
        "missing_rate": missing_rate,
        "seed": configured_seed,
        "episode_seed": work.episode.seed,
        "mask_protocol": "sequence_mask_v2",
        "mask_seed": work.episode.mask_seed,
        "mask_realization_id": work.episode.mask_realization_id,
        "target_missing_rate": work.episode.target_missing_rate,
        "global_missing_rate": work.episode.global_missing_rate,
        "local_missing_rate": work.episode.local_missing_rate,
        "mase_scale": list(map(float, work.mase_scale)),
        "mase_scale_lag": work.mase_scale_lag,
        "file": str(work.relative),
        "assignment_file": str(work.relative_assignment),
        "candidate_ids": list(candidate_ids),
        "assignments": result.routing.assignments,
        "shortlist": result.routing.shortlist,
        "activated_candidates": result.routing.activated_candidates,
        "fallback_blocks": result.routing.fallback_blocks,
        "fallback_records": result.routing.fallback_records,
        "candidate_costs": result.routing.candidate_costs,
        "activated_cost": result.routing.activated_cost,
        "risk_energy": result.routing.risk_energy,
        "cost_energy": result.routing.cost_energy,
        "total_energy": result.routing.total_energy,
        "routing_metadata": result.routing.metadata,
        "pipeline_runtime_seconds": pipeline_runtime_seconds,
        "pipeline_rss_before_bytes": work.pipeline_rss_before,
        "pipeline_rss_after_bytes": pipeline_rss_after,
        "pipeline_rss_delta_bytes": pipeline_rss_delta,
        "pipeline_peak_memory_bytes": pipeline_rss_delta,
    }
    _write_json(assignment_target, assignment)
    progress_entry = {
        "index": work.index,
        "episode_id": work.episode_id,
        "dataset_id": dataset.dataset_id,
        "family_id": dataset.family_id,
        "item_id": work.episode.item_id,
        "forecaster_id": model_id,
        "forecast_mode": work.spec.mode,
        "file": str(work.relative),
        "assignment_file": str(work.relative_assignment),
        "candidate_ids": list(candidate_ids),
        "npz_sha256": _file_sha256(target),
        "assignment_sha256": _file_sha256(assignment_target),
        "completed_at": utc_now(),
    }
    if work.invalid_reason is not None:
        progress_entry["repaired_reason"] = work.invalid_reason
    return assignment, progress_entry


def execute_impute(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if None in (
        inputs.audit_artifact,
        inputs.imputer_artifacts,
        inputs.router_artifact,
        inputs.forecaster_id,
    ):
        raise ValueError("impute stage is missing a required input")
    assert inputs.audit_artifact is not None
    assert inputs.imputer_artifacts is not None
    assert inputs.router_artifact is not None
    assert inputs.forecaster_id is not None
    model_ids = parse_forecaster_ids(inputs.forecaster_id)
    if len(model_ids) != 1:
        raise ValueError("impute requires exactly one forecaster ID")
    model_id = model_ids[0]
    imputer_registry = _selected_imputer_registry(config)
    selected_candidate_ids = _selected_candidate_ids(config)
    resume_identity = _impute_resume_identity(
        config,
        inputs,
        imputer_registry,
        model_id,
    )
    allowed_devices = _allowed_devices(config)
    split = config.experiment.split
    fold_index = _router_fold_index(inputs.router_artifact)
    router_cache: dict[str, RouterBundle] = {}
    direct_router: RouterBundle | None = None
    direct_family: str | None = None
    if fold_index is None:
        direct_router = RouterBundle.load(inputs.router_artifact)
        if split == "rolling_origin":
            _validate_router_bundle(direct_router, split)
        elif split == "leave_model_out":
            _validate_router_bundle(
                direct_router,
                split,
                held_out=model_id,
                held_out_field="forecaster_id",
            )
        else:
            raw_family = direct_router.metadata.get("held_out")
            if not isinstance(raw_family, str) or not raw_family:
                raise ValueError("leave-family-out router has no held-out family metadata")
            direct_family = raw_family
            _validate_router_bundle(
                direct_router,
                split,
                held_out=direct_family,
                held_out_field="family_id",
            )
    else:
        fold_split, fold_paths = fold_index
        if fold_split != split:
            raise ValueError(
                f"router fold split {fold_split!r} does not match configured split {split!r}"
            )
        if split == "leave_model_out" and model_id not in fold_paths:
            raise ValueError(f"no router fold is available for forecaster {model_id!r}")

    def router_for(dataset: Any) -> RouterBundle | None:
        if direct_router is not None:
            if direct_family is not None and str(dataset.family_id) != direct_family:
                return None
            return direct_router
        assert fold_index is not None
        _, fold_paths = fold_index
        held_out_field = "family_id" if split == "leave_family_out" else "forecaster_id"
        held_out = str(dataset.family_id) if split == "leave_family_out" else model_id
        if held_out not in fold_paths:
            raise ValueError(f"no router fold is available for {held_out_field}={held_out!r}")
        if held_out not in router_cache:
            router = RouterBundle.load(fold_paths[held_out])
            _validate_router_bundle(
                router,
                split,
                held_out=held_out,
                held_out_field=held_out_field,
            )
            router_cache[held_out] = router
        return router_cache[held_out]

    output = preparation.store.root / "imputations"
    assignment_root = preparation.store.root / "assignment_records"
    assignments_path = preparation.store.root / "routing_assignments.jsonl"
    progress_path = preparation.store.root / "imputation_progress.json"
    progress = _load_or_create_imputation_progress(
        preparation,
        progress_path,
        resume_identity,
        output,
        assignment_root,
    )
    output.mkdir(parents=True, exist_ok=preparation.resuming)
    assignment_root.mkdir(parents=True, exist_ok=preparation.resuming)
    progress_entries: dict[str, Any] = progress["entries"]
    count = 0
    executed_count = 0
    reused_count = 0
    episode_sampling: dict[str, dict[str, Any]] = {}
    pipeline_runtime_total = 0.0
    pipeline_rss_delta_max = 0
    assignment_records_by_index: dict[int, dict[str, Any]] = {}
    artifact_loading: dict[str, dict[str, Any]] = dict(
        progress.get("artifact_loading", {})
    )
    for dataset, items in _datasets(config, inputs.audit_artifact):
        dataset_sampling: dict[str, Any] = {}
        dataset_episodes = tuple(
            _episode_iter(
                config,
                dataset,
                items,
                partition="eval",
                selection_summary=dataset_sampling,
            )
        )
        _record_episode_sampling(
            episode_sampling,
            dataset.dataset_id,
            dataset_sampling,
        )
        if not dataset_episodes:
            continue
        router = router_for(dataset)
        if router is None:
            continue
        item_lookup = {item.item_id: item for item in items}
        pending: list[_ImputeEpisodeWork] = []
        for episode_id, episode in dataset_episodes:
            source_item = item_lookup[episode.item_id]
            scale_end = fit_prefix_end(
                len(source_item.values),
                config.experiment.context_length,
                config.experiment.horizon,
                config.experiment.fit_prefix_fraction,
            )
            mase_scale, mase_scale_lag = _training_mase_scale(
                source_item.values[:scale_end], dataset.period
            )
            item = _context_item(source_item, episode)
            spec = _forecast_spec(config, model_id, episode.context.shape[2])
            relative = Path(dataset.dataset_id) / f"{count:08d}.npz"
            relative_assignment = (
                Path("assignment_records")
                / dataset.dataset_id
                / f"{count:08d}.json"
            )
            entry_key = f"{count:08d}"
            existing = progress_entries.get(entry_key)
            invalid_reason: str | None = None
            if existing is not None:
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"cannot resume: progress entry {entry_key} is not an object"
                    )
                valid, invalid_reason, recovered = _validate_resumable_imputation(
                    existing,
                    expected_index=count,
                    episode_id=episode_id,
                    dataset_id=dataset.dataset_id,
                    family_id=dataset.family_id,
                    item_id=episode.item_id,
                    model_id=model_id,
                    forecast_mode=spec.mode,
                    relative_file=relative,
                    relative_assignment=relative_assignment,
                    artifact_root=preparation.store.root,
                    episode=episode,
                    period=dataset.period,
                    mase_scale=mase_scale,
                    mase_scale_lag=mase_scale_lag,
                    allowed_candidate_ids=frozenset(imputer_registry.ids),
                )
                if valid:
                    assert recovered is not None
                    assignment_records_by_index[count] = recovered
                    pipeline_runtime = float(recovered["pipeline_runtime_seconds"])
                    pipeline_memory = int(recovered["pipeline_rss_delta_bytes"])
                    pipeline_runtime_total += pipeline_runtime
                    pipeline_rss_delta_max = max(
                        pipeline_rss_delta_max,
                        pipeline_memory,
                    )
                    reused_count += 1
                    count += 1
                    continue

            pending.append(
                _ImputeEpisodeWork(
                    index=count,
                    episode_id=episode_id,
                    episode=episode,
                    item=item,
                    spec=spec,
                    relative=relative,
                    relative_assignment=relative_assignment,
                    entry_key=entry_key,
                    invalid_reason=invalid_reason,
                    pipeline_rss_before=_resident_memory_bytes(),
                    mase_scale=mase_scale,
                    mase_scale_lag=mase_scale_lag,
                )
            )
            count += 1
        if not pending:
            continue

        artifact_store = DatasetImputerArtifactStore(
            inputs.imputer_artifacts,
            dataset.dataset_id,
            imputer_registry,
        )
        medians, correlation = artifact_store.load_statistics()
        artifact_manager = _ImputeArtifactManager(
            artifact_store,
            imputer_registry,
            config,
        )
        available_artifact_ids = artifact_manager.declared_available(
            selected_candidate_ids
        )
        artifact_failures = artifact_manager.declared_failures(
            selected_candidate_ids
        )
        pipeline = BlockwiseFAIS(
            config=config,
            router=router,
            imputer_registry=imputer_registry,
            imputer_artifacts={},
            artifact_load_failures=artifact_failures,
            training_medians=medians,
            training_correlation=correlation,
        )
        for work in pending:
            _prepare_impute_work(
                work,
                pipeline,
                available_artifact_ids,
                artifact_failures,
                allowed_devices,
            )
        loading_audit = _run_impute_candidate_sweep(
            pending,
            pipeline,
            artifact_manager,
            config,
            allowed_devices,
            available_artifact_ids,
            artifact_failures,
        )
        loading_audit.update(
            {
                "pending_episode_count": len(pending),
                "available_artifact_ids": sorted(available_artifact_ids),
            }
        )
        artifact_loading[dataset.dataset_id] = loading_audit
        evaluation_ids = _evaluation_candidate_ids(
            imputer_registry,
            available_artifact_ids,
            allowed_devices,
        )

        for work in pending:
            if work.plan is None:  # pragma: no cover - preparation invariant
                raise RuntimeError("imputation work has no route plan")
            candidates, pseudo_candidates = _budgeted_route_results(
                work,
                imputer_registry,
            )
            finish_started = perf_counter()
            result = pipeline.finish_route(
                work.plan,
                candidates,
                pseudo_candidates,
                fallback_candidates=work.raw_actual,
            )
            finish_seconds = perf_counter() - finish_started
            fallback_runtime_ids = {
                candidate_id
                for record in result.routing.fallback_records.values()
                for candidate_id in record.get("attempts", ())
                if (
                    candidate_id in imputer_registry
                    and candidate_id not in work.plan.shortlist
                    and candidate_id in work.raw_actual
                )
            }
            pipeline_runtime_seconds = (
                work.prepare_seconds
                + sum(value.runtime_seconds for value in candidates.values())
                + sum(value.runtime_seconds for value in pseudo_candidates.values())
                + sum(
                    work.raw_actual[candidate_id].runtime_seconds
                    for candidate_id in fallback_runtime_ids
                )
                + finish_seconds
            )
            pipeline_rss_after = _resident_memory_bytes()
            pipeline_rss_delta = max(
                0,
                pipeline_rss_after - work.pipeline_rss_before,
            )
            pipeline_runtime_total += pipeline_runtime_seconds
            pipeline_rss_delta_max = max(
                pipeline_rss_delta_max,
                pipeline_rss_delta,
            )
            saved_candidates = (
                {
                    candidate_id: candidate_result
                    for candidate_id, candidate_result in work.raw_actual.items()
                    if candidate_id in evaluation_ids
                }
                if config.experiment.save_all_candidate_outputs
                else {}
            )
            saved_candidates.update(result.candidates)
            result.candidates = saved_candidates
            assignment, progress_entry = _commit_imputation_output(
                preparation,
                output,
                dataset,
                model_id,
                imputer_registry,
                work,
                result,
                pipeline_runtime_seconds,
                pipeline_rss_after,
                pipeline_rss_delta,
            )
            if work.invalid_reason is not None:
                progress["repair_count"] = int(progress.get("repair_count", 0)) + 1
            progress_entries[work.entry_key] = progress_entry
            progress["completed_count"] = len(progress_entries)
            progress["last_completed_index"] = work.index
            progress["pipeline_runtime_total_seconds"] = pipeline_runtime_total
            progress["pipeline_rss_delta_max_bytes"] = pipeline_rss_delta_max
            progress["artifact_loading"] = artifact_loading
            progress["updated_at"] = utc_now()
            _write_json(progress_path, progress)
            assignment_records_by_index[work.index] = assignment
            executed_count += 1
            work.raw_actual.clear()
            work.raw_pseudo.clear()
            work.actual_modes.clear()
    if count == 0:
        raise ValueError("no imputation episode was generated")
    expected_entry_keys = {f"{index:08d}" for index in range(count)}
    extra_entries = set(progress_entries).difference(expected_entry_keys)
    missing_entries = expected_entry_keys.difference(progress_entries)
    if extra_entries or missing_entries:
        raise ValueError(
            "cannot finalize impute: progress does not match the deterministic episode plan; "
            f"extra={sorted(extra_entries)}, missing={sorted(missing_entries)}"
        )
    if len(assignment_records_by_index) != count:
        raise ValueError("cannot finalize impute: assignment count is inconsistent")
    assignment_records = [
        assignment_records_by_index[index] for index in range(count)
    ]
    sampling_manifest = _episode_sampling_manifest(
        "eval",
        config.experiment.max_eval_episodes_per_dataset,
        episode_sampling,
        count,
    )
    _write_jsonl_atomic(assignments_path, assignment_records)
    summary = {
        "schema_version": _IMPUTATION_SCHEMA_VERSION,
        "imputations": str(output),
        "routing_assignments": str(assignments_path),
        "assignment_records": str(assignment_root),
        "progress_manifest": str(progress_path),
        "resume_identity_sha256": _canonical_sha256(resume_identity),
        "episode_count": count,
        "episodes_executed": executed_count,
        "episodes_reused": reused_count,
        "repaired_episode_count": int(progress.get("repair_count", 0)),
        "origin_partition": "eval",
        "max_eval_episodes_per_dataset": (
            config.experiment.max_eval_episodes_per_dataset
        ),
        "episode_sampling": sampling_manifest,
        "forecaster_id": model_id,
        "selected_candidates": list(selected_candidate_ids),
        "save_all_candidate_outputs": (config.experiment.save_all_candidate_outputs),
        "csdi_num_samples": config.experiment.csdi_num_samples,
        "torch_device": _torch_device(config),
        "pipeline_runtime_total_seconds": pipeline_runtime_total,
        "pipeline_rss_delta_max_bytes": pipeline_rss_delta_max,
        "pipeline_memory_measurement": "endpoint_rss_delta_approximation",
        "artifact_loading": artifact_loading,
    }
    _write_json(preparation.store.root / "imputation_manifest.json", summary)
    progress.update(
        {
            "status": "completed",
            "completed_count": len(progress_entries),
            "expected_episode_count": count,
            "episodes_executed_last_invocation": executed_count,
            "episodes_reused_last_invocation": reused_count,
            "pipeline_runtime_total_seconds": pipeline_runtime_total,
            "pipeline_rss_delta_max_bytes": pipeline_rss_delta_max,
            "artifact_loading": artifact_loading,
            "episode_sampling": sampling_manifest,
            "routing_assignments_sha256": _file_sha256(assignments_path),
            "imputation_manifest_sha256": _file_sha256(
                preparation.store.root / "imputation_manifest.json"
            ),
            "completed_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    _write_json(progress_path, progress)
    return summary


_EXECUTORS = {
    "fit-imputers": execute_fit_imputers,
    "labels": execute_labels,
    "train-router": execute_train_router,
    "impute": execute_impute,
}


def execute_prepared_stage(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    preparation.manifest.update(_execution_metadata(config))
    if preparation.resuming:
        preparation.manifest["resumed_from_status"] = preparation.manifest.get("status")
        preparation.manifest.pop("outputs", None)
    preparation.manifest.update(
        {"status": "running", "execution_started": True, "updated_at": utc_now()}
    )
    preparation.store.write("stage_manifest.json", preparation.manifest)
    try:
        outputs = dict(_EXECUTORS[preparation.stage](preparation, config, inputs))
    except Exception as error:
        preparation.manifest.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "message": f"{type(error).__name__}: {error}",
            }
        )
        preparation.store.write("stage_manifest.json", preparation.manifest)
        raise
    preparation.manifest.update(
        {
            "status": "completed",
            "updated_at": utc_now(),
            "message": "stage completed",
            "outputs": outputs,
        }
    )
    preparation.store.write("stage_manifest.json", preparation.manifest)
    return outputs


__all__ = [
    "execute_fit_imputers",
    "execute_impute",
    "execute_labels",
    "execute_prepared_stage",
    "execute_train_router",
]
