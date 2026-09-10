"""Versioned development experiments for sequence-level forecast utility."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_fais.contracts import ForecastSpec, SeriesBatch
from tsfm_fais.data import (
    MaskingSpec,
    fit_prefix_end,
    load_dataset,
    load_manifest,
    mask_time_series,
    stable_seed,
)
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry
from tsfm_fais.forecasting.metrics import macro_mase, training_mase_scale
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner, DatasetImputerArtifactStore
from tsfm_fais.routing.utility import UtilitySelector, response_features, sequence_features


class UtilityExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    protocol_id: str = "r3-utility-development-v1"
    evidence_role: Literal["development"] = "development"
    data_manifest: Path
    output_root: Path
    imputer_artifact_roots: tuple[Path, ...] = ()
    forecaster_artifacts: dict[str, Path]
    dataset_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...] = (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "knn_multivariate",
        "saits",
        "timemixerpp",
    )
    context_length: int = Field(default=96, ge=3)
    horizon: int = Field(default=96, ge=1)
    target_indices: tuple[int, ...] = (0, 1)
    fit_prefix_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    temporal_boundary: float = Field(default=0.6, gt=0.5, lt=1)
    max_items: int = Field(default=1, ge=1)
    train_origins: int = Field(default=4, ge=1)
    validation_origins: int = Field(default=2, ge=1)
    mechanisms: tuple[str, ...] = ("random_point", "independent_block", "synchronous_block")
    missing_rates: tuple[float, ...] = (0.1, 0.4)
    mask_seeds: tuple[int, ...] = (6101,)
    block_lengths: tuple[int, ...] = (6, 12, 24, 48)
    device: Literal["cpu", "cuda"] = "cuda"
    forecast_batch_size: int = Field(default=32, ge=1)
    selector_seed: int = 5101

    @model_validator(mode="after")
    def validate_protocol(self):
        for name in (
            "dataset_ids",
            "candidate_ids",
            "target_indices",
            "mechanisms",
            "missing_rates",
            "mask_seeds",
        ):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        if "locf" not in self.candidate_ids or "linear_interp" not in self.candidate_ids:
            raise ValueError("candidate pool requires locf and linear_interp")
        if any(candidate not in DEFAULT_REGISTRY for candidate in self.candidate_ids):
            raise ValueError("unregistered imputer candidate")
        if any(target < 0 for target in self.target_indices):
            raise ValueError("target indices must be nonnegative")
        for mechanism in self.mechanisms:
            for rate in self.missing_rates:
                MaskingSpec(mechanism, rate, self.block_lengths)
        return self


def load_utility_config(path: str | Path) -> UtilityExperimentConfig:
    source = Path(path).resolve()
    config = UtilityExperimentConfig.model_validate(
        yaml.safe_load(source.read_text(encoding="utf-8"))
    )

    def resolve(item: Path) -> Path:
        return (source.parent / item).resolve() if not item.is_absolute() else item

    return config.model_copy(
        update={
            "data_manifest": resolve(config.data_manifest),
            "output_root": resolve(config.output_root),
            "imputer_artifact_roots": tuple(
                resolve(root) for root in config.imputer_artifact_roots
            ),
            "forecaster_artifacts": {
                key: resolve(value) for key, value in config.forecaster_artifacts.items()
            },
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _code_hash() -> str:
    root = Path(__file__).parent
    files = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def purged_origins(
    length: int, config: UtilityExperimentConfig
) -> tuple[int, dict[str, tuple[int, ...]]]:
    prefix = fit_prefix_end(
        length, config.context_length, config.horizon, config.fit_prefix_fraction
    )
    boundary = int(np.floor(length * config.temporal_boundary))
    windows = {}
    for split, begin, end, limit in (
        ("train", prefix + config.context_length, boundary - config.horizon, config.train_origins),
        (
            "validation",
            boundary + config.context_length,
            length - config.horizon,
            config.validation_origins,
        ),
    ):
        pool = tuple(range(begin, end + 1, config.context_length + config.horizon))
        indices = np.linspace(0, len(pool) - 1, min(limit, len(pool)), dtype=int) if pool else ()
        windows[split] = tuple(pool[index] for index in indices)
    return prefix, windows


def _load_frozen_imputers(
    config: UtilityExperimentConfig, dataset_id: str
) -> tuple[dict, dict, dict]:
    needed = tuple(
        candidate
        for candidate in config.candidate_ids
        if DEFAULT_REGISTRY.get_spec(candidate).optional_extra == "deep-imputers"
    )
    if not needed:
        return {}, {}, {}
    for root in config.imputer_artifact_roots:
        try:
            store = DatasetImputerArtifactStore(root, dataset_id)
        except KeyError:
            continue
        loaded = store.load_artifacts(needed)
        return (
            loaded.artifacts,
            loaded.failures,
            {
                "root": str(root),
                "manifest_sha256": file_sha256(root / "manifest.json"),
                "loaded_ids": list(loaded.loaded_ids),
                "failures": loaded.failures,
            },
        )
    raise ValueError(f"no frozen imputer training-prefix artifact for {dataset_id}")


def prepare_utility_episodes(config: UtilityExperimentConfig) -> dict[str, Any]:
    """Cache candidate contexts once, before loading any forecasting model."""
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    config_payload = config.model_dump(mode="json")
    source_hash = _code_hash()
    identity = {"config": config_payload, "source_sha256": source_hash}
    identity_path = root / "preparation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("preparation identity changed; use a new output directory")
    _write_json(identity_path, identity)
    manifest_path = root / "episodes_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in saved["episodes"]:
            if file_sha256(root / record["path"]) != record["sha256"]:
                raise ValueError(f"cached episode hash mismatch: {record['episode_id']}")
        return saved
    manifest = load_manifest(config.data_manifest)
    runner = CandidateRunner()
    records: list[dict[str, Any]] = []
    datasets = []
    started = perf_counter()
    for dataset_id in config.dataset_ids:
        spec = manifest.get(dataset_id)
        items = load_dataset(spec)[: config.max_items]
        artifacts, failures, fit_record = _load_frozen_imputers(config, dataset_id)
        source_paths = sorted(spec.path.rglob("*")) if spec.path.is_dir() else [spec.path]
        dataset_record = {
            "dataset_id": dataset_id,
            "family_id": spec.family_id,
            "fit_artifacts": fit_record,
            "sources": {str(path): file_sha256(path) for path in source_paths if path.is_file()},
            "items": [],
        }
        local_artifacts, local = {}, {}
        for item in items:
            if max(config.target_indices) >= item.values.shape[1]:
                raise ValueError(f"invalid target dimension for {dataset_id}")
            if len(item.values) < config.context_length + config.horizon:
                dataset_record["items"].append(
                    {"item_id": item.item_id, "status": "insufficient_history"}
                )
                continue
            prefix_end, origins = purged_origins(len(item.values), config)
            if not all(origins.values()):
                dataset_record["items"].append(
                    {"item_id": item.item_id, "status": "insufficient_purged_history"}
                )
                continue
            prefix = item.values[:prefix_end]
            scales, lag = training_mase_scale(prefix, spec.period)
            defaults = np.nanmedian(prefix, axis=0)
            if not np.isfinite(defaults).all():
                raise ValueError("training prefix must provide each variate")
            local_artifacts = dict(artifacts)
            params = {"seasonal_lag": {"period": max(2, spec.period)}}
            training_batch = SeriesBatch(
                prefix[None], np.isfinite(prefix[None]), metadata={"period": spec.period}
            )
            for candidate in config.candidate_ids:
                if candidate in artifacts or candidate in failures:
                    continue
                if (
                    candidate == "seasonal_lag"
                    or DEFAULT_REGISTRY.get_spec(candidate).fit_scope == "dataset"
                ):
                    local_artifacts[candidate] = runner.fit(
                        candidate,
                        training_batch,
                        {"period": spec.period},
                        params=params.get(candidate),
                    )
            dataset_record["items"].append(
                {
                    "item_id": item.item_id,
                    "prefix_end": prefix_end,
                    "mase_lag": lag,
                    "origins": origins,
                }
            )
            for split, split_origins in origins.items():
                for mechanism in config.mechanisms:
                    for rate in config.missing_rates:
                        for seed in config.mask_seeds:
                            masking = MaskingSpec(mechanism, rate, config.block_lengths)
                            realization = mask_time_series(
                                item.values,
                                masking,
                                stable_seed(
                                    config.protocol_id,
                                    dataset_id,
                                    item.item_id,
                                    split,
                                    mechanism,
                                    rate,
                                    seed,
                                ),
                                calibration_values=prefix,
                            )
                            for origin in split_origins:
                                context = realization.values[
                                    origin - config.context_length : origin
                                ].copy()
                                truth = item.values[origin : origin + config.horizon].copy()
                                clean = item.values[origin - config.context_length : origin].copy()
                                if (
                                    not np.isfinite(truth[:, config.target_indices]).all()
                                    or not np.isfinite(clean).all()
                                ):
                                    raise ValueError(
                                        "synthetic development episodes require complete source context and target truth"
                                    )
                                batch = SeriesBatch(
                                    context[None],
                                    np.isfinite(context[None]),
                                    metadata={"period": spec.period},
                                )
                                local = dict(local_artifacts)
                                if "seasonal_lag" in local:
                                    seasonal = dict(local["seasonal_lag"])
                                    seasonal["profiles"] = np.roll(
                                        seasonal["profiles"],
                                        -(origin - config.context_length) % seasonal["period"],
                                        axis=1,
                                    )
                                    local["seasonal_lag"] = seasonal
                                candidates = runner.run_many(
                                    config.candidate_ids,
                                    batch,
                                    local,
                                    seed=stable_seed(seed, origin),
                                    params=params,
                                    artifact_failures=failures,
                                )
                                anchor = candidates["locf"].values[0].copy()
                                anchor[~candidates["locf"].native_valid_mask[0]] = np.broadcast_to(
                                    defaults, anchor.shape
                                )[~candidates["locf"].native_valid_mask[0]]
                                outputs, coverages, reconstruction, timings, statuses = (
                                    [],
                                    [],
                                    [],
                                    [],
                                    [],
                                )
                                missing = ~batch.observed_mask[0]
                                for candidate_id in config.candidate_ids:
                                    candidate = candidates[candidate_id]
                                    valid = candidate.native_valid_mask[0].copy()
                                    if not DEFAULT_REGISTRY.get_spec(candidate_id).supports_tail:
                                        for channel in range(context.shape[1]):
                                            observed = np.flatnonzero(~missing[:, channel])
                                            if observed.size:
                                                valid[observed[-1] + 1 :, channel] = False
                                    completed = candidate.values[0].copy()
                                    completed[~valid] = anchor[~valid]
                                    if not np.array_equal(completed[~missing], context[~missing]):
                                        raise ValueError("candidate altered an observed value")
                                    outputs.append(completed)
                                    coverages.append(
                                        float(valid[missing].mean()) if missing.any() else 1.0
                                    )
                                    error = (
                                        np.abs(
                                            completed[:, config.target_indices]
                                            - clean[:, config.target_indices]
                                        )
                                        / scales[list(config.target_indices)]
                                    )
                                    reconstruction.append(
                                        float(error[missing[:, config.target_indices]].mean())
                                        if missing[:, config.target_indices].any()
                                        else 0.0
                                    )
                                    timings.append(candidate.runtime_seconds)
                                    statuses.append(
                                        {
                                            "candidate_id": candidate_id,
                                            "status": candidate.status.value,
                                            "failure_reason": candidate.failure_reason,
                                        }
                                    )
                                origin_id = f"{dataset_id}|{item.item_id}|{origin}"
                                episode_id = f"{origin_id}|{split}|{mechanism}|{rate}|{seed}"
                                key = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
                                path = root / "episodes" / f"{key}.npz"
                                _save_npz(
                                    path,
                                    context=context,
                                    clean_context=clean,
                                    future=truth,
                                    candidate_values=np.stack(outputs),
                                    candidate_ids=np.asarray(config.candidate_ids),
                                    native_coverage=np.asarray(coverages),
                                    reconstruction_loss=np.asarray(reconstruction),
                                    candidate_runtime=np.asarray(timings),
                                    mase_scales=scales,
                                    prefix_defaults=defaults,
                                )
                                records.append(
                                    {
                                        "episode_id": episode_id,
                                        "origin_id": origin_id,
                                        "dataset_id": dataset_id,
                                        "family_id": spec.family_id,
                                        "item_id": item.item_id,
                                        "origin": origin,
                                        "split": split,
                                        "mechanism": mechanism,
                                        "missing_rate": rate,
                                        "mask_seed": seed,
                                        "period": spec.period,
                                        "mase_lag": lag,
                                        "path": str(path.relative_to(root)),
                                        "sha256": file_sha256(path),
                                        "candidate_status": statuses,
                                    }
                                )
            print(
                json.dumps(
                    {
                        "stage": "prepare",
                        "dataset": dataset_id,
                        "episodes": len(records),
                        "elapsed_seconds": round(perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
        datasets.append(dataset_record)
        del artifacts, local_artifacts, local
        gc.collect()
    payload = {
        "evidence_role": "development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "datasets": datasets,
        "episodes": records,
        "elapsed_seconds": perf_counter() - started,
    }
    if not records:
        raise ValueError("no eligible development episodes")
    _write_json(manifest_path, payload)
    return payload


def forecast_utility_episodes(config: UtilityExperimentConfig, model_id: str) -> dict[str, Any]:
    root = config.output_root
    manifest_path = root / "episodes_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity"]["config"] != config.model_dump(mode="json"):
        raise ValueError("forecast configuration differs from prepared episodes")
    checkpoint = config.forecaster_artifacts[model_id]
    if not checkpoint.is_dir():
        raise ValueError(f"local forecast checkpoint unavailable: {checkpoint}")
    registry = default_forecast_registry()
    adapter = registry.build(
        model_id,
        model_name=str(checkpoint),
        device=config.device,
        batch_size=config.forecast_batch_size,
    )
    runner = ForecastRunner(registry, {model_id: adapter})
    native_supported = bool(getattr(adapter.capabilities, "supports_missing_context", False))
    spec = ForecastSpec(
        model_id,
        registry.get(model_id).mode,
        config.horizon,
        context_length=config.context_length,
        target_indices=config.target_indices,
        num_samples=20,
    )
    output = root / model_id
    identity = {
        "config": config.model_dump(mode="json"),
        "episode_manifest_sha256": file_sha256(manifest_path),
        "source_sha256": _code_hash(),
        "checkpoint": str(checkpoint),
        "forecast_spec": asdict(spec),
        "native_missing": native_supported,
    }
    identity_path = output / "forecast_identity.json"
    if identity_path.exists() and json.loads(
        identity_path.read_text(encoding="utf-8")
    ) != json.loads(json.dumps(identity)):
        raise ValueError("forecast identity changed; use a new run directory")
    _write_json(identity_path, identity)
    rows, controls, progress = [], [], []
    started = perf_counter()
    for index, record in enumerate(manifest["episodes"]):
        episode_path = root / record["path"]
        if file_sha256(episode_path) != record["sha256"]:
            raise ValueError("candidate episode changed after preparation")
        destination = output / "predictions" / episode_path.name
        with np.load(episode_path, allow_pickle=False) as episode:
            ids = episode["candidate_ids"].tolist()
            candidate_values = episode["candidate_values"]
            context, clean = episode["context"], episode["clean_context"]
            targets = list(config.target_indices)
            scales = episode["mase_scales"]
            if not destination.exists():
                before = runner.resource_metrics()
                completed = runner.predict(
                    np.concatenate([candidate_values, clean[None]], axis=0), spec
                )
                point, quantiles = completed.point[:-1], completed.quantiles[:-1]
                clean_point = completed.point[-1]
                if native_supported:
                    native = runner.predict_missing(context[None], spec)
                    point = np.concatenate([point, native.point])
                    quantiles = np.concatenate([quantiles, native.quantiles])
                after = runner.resource_metrics()
                _save_npz(
                    destination,
                    point=point,
                    quantiles=quantiles,
                    clean_point=clean_point,
                    candidate_sha256=np.asarray(record["sha256"]),
                    runtime_seconds=np.asarray(
                        after["forecast_runtime_seconds"] - before["forecast_runtime_seconds"]
                    ),
                    forecast_context_count=np.asarray(
                        after["forecast_context_count"] - before["forecast_context_count"]
                    ),
                    forecast_call_count=np.asarray(
                        after["forecast_call_count"] - before["forecast_call_count"]
                    ),
                )
            with np.load(destination, allow_pickle=False) as predicted:
                if str(predicted["candidate_sha256"]) != record["sha256"]:
                    raise ValueError("predictions are bound to a different candidate episode")
                point, quantiles = predicted["point"], predicted["quantiles"]
                clean_point = predicted["clean_point"]
                action_ids = ids + (["native_missing"] if native_supported else [])
                if point.shape != (len(action_ids), config.horizon, len(targets)):
                    raise ValueError("cached prediction shape mismatch")
                truth = episode["future"][:, targets]
                losses = macro_mase(
                    point, np.repeat(truth[None], len(point), axis=0), scales[targets]
                )
                anchor_index = ids.index("locf")
                pool_point = np.median(point, axis=0)
                base = {
                    key: record[key]
                    for key in (
                        "episode_id",
                        "origin_id",
                        "dataset_id",
                        "family_id",
                        "item_id",
                        "origin",
                        "split",
                        "mechanism",
                        "missing_rate",
                        "mask_seed",
                    )
                }
                base["model_id"] = model_id
                for action_index, action in enumerate(action_ids):
                    native_action = action == "native_missing"
                    coverage = (
                        1.0 if native_action else float(episode["native_coverage"][action_index])
                    )
                    features = sequence_features(
                        context,
                        candidate_values[anchor_index if native_action else action_index],
                        candidate_values[anchor_index],
                        scales,
                        targets,
                        period=record["period"],
                        native_coverage=coverage,
                    )
                    features["static.native_action"] = float(native_action)
                    features.update(
                        response_features(
                            point[action_index],
                            point[anchor_index],
                            pool_point,
                            candidate_values[anchor_index, -1, targets],
                            scales[targets],
                            quantiles[action_index],
                        )
                    )
                    rows.append(
                        {
                            **base,
                            "candidate_id": action,
                            "loss": float(losses[action_index]),
                            "reconstruction_loss": None
                            if native_action
                            else float(episode["reconstruction_loss"][action_index]),
                            "native_coverage": coverage,
                            **features,
                        }
                    )
                for method, forecast in (
                    ("clean", clean_point),
                    ("forecast_mean", np.mean(point, axis=0)),
                    ("forecast_median", pool_point),
                ):
                    controls.append(
                        {
                            **base,
                            "method": method,
                            "loss": float(
                                macro_mase(forecast[None], truth[None], scales[targets])[0]
                            ),
                        }
                    )
                controls.append({**base, "method": "oracle", "loss": float(np.min(losses))})
                progress.append(
                    {
                        "episode_id": record["episode_id"],
                        "sha256": file_sha256(destination),
                        "runtime_seconds": float(predicted["runtime_seconds"]),
                        "forecast_context_count": int(predicted["forecast_context_count"]),
                        "forecast_call_count": int(predicted["forecast_call_count"]),
                    }
                )
        if (index + 1) % 20 == 0 or index + 1 == len(manifest["episodes"]):
            _write_json(
                output / "forecast_progress.json",
                {
                    "completed": index + 1,
                    "total": len(manifest["episodes"]),
                    "elapsed_seconds": perf_counter() - started,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "forecast",
                        "model": model_id,
                        "completed": index + 1,
                        "total": len(manifest["episodes"]),
                        "elapsed_seconds": round(perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    pd.DataFrame(rows).to_parquet(output / "utility_rows.parquet", index=False)
    pd.DataFrame(controls).to_parquet(output / "control_rows.parquet", index=False)
    result = {
        "identity": identity,
        "episodes": progress,
        "row_count": len(rows),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "resources_this_execution": runner.resource_metrics(),
    }
    _write_json(output / "forecast_manifest.json", result)
    return result


def analyze_utility_experiment(
    config: UtilityExperimentConfig, model_ids: Sequence[str]
) -> dict[str, Any]:
    """Leave each development family out of both fitting and calibration."""
    import joblib

    root = config.output_root
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    selected_rows, fold_records = [], []
    for model_id in model_ids:
        frame = pd.read_parquet(root / model_id / "utility_rows.parquet")
        controls = pd.read_parquet(root / model_id / "control_rows.parquet")
        families = sorted(frame.family_id.unique())
        if len(families) < 3:
            raise ValueError("family-held-out development needs at least three families")
        for held_family in families:
            training = frame[(frame.family_id != held_family) & (frame.split == "train")].copy()
            calibration = frame[
                (frame.family_id != held_family) & (frame.split == "validation")
            ].copy()
            validation = frame[
                (frame.family_id == held_family) & (frame.split == "validation")
            ].copy()
            if min(len(training), len(calibration), len(validation)) == 0:
                raise ValueError(f"empty partition for {model_id}/{held_family}")
            if set(training.origin_id).intersection(calibration.origin_id):
                raise ValueError("fitting and calibration origins overlap")
            if set(validation.family_id).intersection(training.family_id) or set(
                validation.family_id
            ).intersection(calibration.family_id):
                raise ValueError("held-out family leaked into selector fitting or calibration")
            selectors = {}
            for name, use_response in (("sequence_forecast", False), ("response_utility", True)):
                selector = UtilitySelector(
                    use_response=use_response, seed=config.selector_seed
                ).fit(training)
                selector.calibrate(calibration)
                selectors[name] = selector
                model_dir = output / "selectors" / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(selector, model_dir / f"{held_family}-{name}.joblib")
                fold_records.append(
                    {
                        "model_id": model_id,
                        "held_family": held_family,
                        "method": name,
                        "train_families": sorted(training.family_id.unique()),
                        "calibration_families": sorted(calibration.family_id.unique()),
                        "training_origin_count": int(training.origin_id.nunique()),
                        "calibration_origin_count": int(calibration.origin_id.nunique()),
                        "validation_origin_count": int(validation.origin_id.nunique()),
                        "baseline_id": selector.baseline_id,
                        "feature_names": selector.feature_names,
                        "calibration": selector.calibration,
                    }
                )
            baseline_id = selectors["response_utility"].baseline_id
            baseline = (
                validation[validation.candidate_id == baseline_id].set_index("episode_id").loss
            )
            oracle = validation.groupby("episode_id").loss.min()

            def record_selection(
                method: str,
                selected: pd.DataFrame,
                *,
                baseline_id=baseline_id,
                baseline=baseline,
                oracle=oracle,
            ) -> None:
                keep = [
                    "episode_id",
                    "origin_id",
                    "model_id",
                    "family_id",
                    "dataset_id",
                    "mechanism",
                    "missing_rate",
                    "loss",
                ]
                rows = selected.loc[:, keep].copy()
                rows["method"] = method
                rows["selected_candidate"] = (
                    selected.candidate_id.to_numpy() if "candidate_id" in selected else method
                )
                rows["baseline_id"] = baseline_id
                rows["baseline_loss"] = rows.episode_id.map(baseline)
                rows["oracle_loss"] = rows.episode_id.map(oracle)
                rows["delta"] = rows.loss - rows.baseline_loss
                rows["regret"] = rows.loss - rows.oracle_loss
                rows["switched"] = rows.selected_candidate != baseline_id
                rows["harmful"] = rows.delta > 1e-10
                selected_rows.append(rows)

            record_selection("fixed_train_best", validation[validation.candidate_id == baseline_id])
            for candidate, group in validation.groupby("candidate_id"):
                record_selection(f"fixed_{candidate}", group)
            for name, selector in selectors.items():
                record_selection(name, selector.select(validation))
                record_selection(name + "_gated", selector.select(validation, gated=True))
            # Reconstruction has no defined label for passing NaNs through. Its
            # control uses the common finite imputer subset and is marked as such.
            finite_train = training[training.candidate_id != "native_missing"].copy()
            finite_validation = validation[validation.candidate_id != "native_missing"].copy()
            reconstruction_train = finite_train.assign(loss=finite_train.reconstruction_loss)
            reconstruction = UtilitySelector(use_response=False, seed=config.selector_seed).fit(
                reconstruction_train
            )
            record_selection(
                "sequence_reconstruction_finite", reconstruction.select(finite_validation)
            )
            for name, group in controls[
                (controls.family_id == held_family) & (controls.split == "validation")
            ].groupby("method"):
                record_selection(name, group)
            print(
                json.dumps(
                    {
                        "stage": "analysis",
                        "model": model_id,
                        "held_family": held_family,
                        "baseline": baseline_id,
                    }
                ),
                flush=True,
            )
    results = pd.concat(selected_rows, ignore_index=True)
    if results.duplicated(["model_id", "episode_id", "method"]).any():
        raise ValueError("duplicate analysis outputs")
    summaries = []
    family_results = []
    for (model_id, method), group in results.groupby(["model_id", "method"]):
        per_family = (
            group.groupby(["family_id", "dataset_id"])[
                ["loss", "delta", "regret", "harmful", "switched"]
            ]
            .mean()
            .groupby(level="family_id")
            .mean()
        )
        rng = np.random.default_rng(config.selector_seed)
        deltas = per_family.delta.to_numpy()
        bootstrap = np.mean(rng.choice(deltas, size=(5000, len(deltas)), replace=True), axis=1)
        tail = (
            group.assign(positive_degradation=np.maximum(group.delta, 0))
            .groupby("family_id")
            .positive_degradation.apply(
                lambda values: float(
                    np.sort(values)[-max(1, int(np.ceil(0.1 * len(values)))) :].mean()
                )
            )
        )
        summaries.append(
            {
                "model_id": model_id,
                "method": method,
                "family_macro_mase": float(per_family.loss.mean()),
                "delta_vs_train_fixed": float(deltas.mean()),
                "exploratory_ci_low": float(np.quantile(bootstrap, 0.025)),
                "exploratory_ci_high": float(np.quantile(bootstrap, 0.975)),
                "mean_regret": float(per_family.regret.mean()),
                "harmful_fraction": float(per_family.harmful.mean()),
                "switch_fraction": float(per_family.switched.mean()),
                "tail_positive_degradation": float(tail.mean()),
                "family_count": len(per_family),
                "episode_count": int(group.episode_id.nunique()),
                "origin_count": int(group.origin_id.nunique()),
            }
        )
        family_results.append(per_family.reset_index().assign(model_id=model_id, method=method))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    results.to_csv(output / "episode_results.csv", index=False)
    pd.concat(family_results).to_csv(output / "family_results.csv", index=False)
    _write_json(output / "folds.json", fold_records)
    report = [
        "# R3 序列级填补选择：开发实验",
        "",
        "这些结果用于方法开发，未构成独立确认实验。每次留出一个完整数据家族，选择器拟合与阈值校准均排除该家族。相邻训练和验证分区的上下文与预测区间不重叠。",
        "",
        "MASE 采用历史训练前缀的逐变量尺度，按数据集、家族逐级等权。置信区间仅为开发家族上的探索性重采样结果，尚未作多重比较校正。",
        "",
    ]
    for model_id in model_ids:
        report.extend(
            [
                f"## {model_id}",
                "",
                "| 方法 | 家族平均 MASE | 相对训练固定基准的差值 | 有害比例 | 切换比例 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in (
            summary[summary.model_id == model_id].sort_values("family_macro_mase").itertuples()
        ):
            report.append(
                f"| {row.method} | {row.family_macro_mase:.6f} | {row.delta_vs_train_fixed:+.6f} | {row.harmful_fraction:.1%} | {row.switch_fraction:.1%} |"
            )
        report.append("")
    report.extend(
        [
            "clean 使用完整历史真值作为诊断；oracle 使用未来真值事后选择，二者均不可作为可部署方法。原生缺失输入仅在已验证支持的模型上启用。重建监督选择器仅覆盖有限值填补动作。",
            "",
            "预测均值和中位数集成使用与响应选择器相同的候选预测集合。缓存生成额外包含 clean 诊断调用，部署成本应排除该调用并加上实际选择器推理时间；本轮缓存总耗时不等于在线方法耗时。",
            "",
        ]
    )
    (output / "development_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "evidence_role": "development",
        "model_ids": list(model_ids),
        "source_sha256": _code_hash(),
        "summary_sha256": file_sha256(output / "summary.csv"),
        "episode_results_sha256": file_sha256(output / "episode_results.csv"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return manifest
