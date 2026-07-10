"""Opt-in implementations of the four experiment stages.

The CLI imports this module only for ``run --execute``. Merely validating a
configuration or preparing a stage never loads data, fits a model, or invokes
a forecaster.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import joblib
import numpy as np
from pandas.tseries.frequencies import to_offset

from tsfm_fais.config import AppConfig
from tsfm_fais.contracts import BudgetSpec, ForecastSpec, SeriesBatch, TimeSeriesItem
from tsfm_fais.data import (
    MaskingSpec,
    audit_dataset,
    build_episode,
    load_dataset,
    load_manifest,
    rolling_origins,
    stable_seed,
)
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry
from tsfm_fais.imputers import (
    DEFAULT_REGISTRY,
    CandidateRunner,
    load_dataset_imputer_artifacts,
)
from tsfm_fais.pipeline import BlockwiseFAIS
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
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
        yield spec, items


def _fit_region_end(length: int, context_length: int, horizon: int) -> int:
    if context_length < 2 or horizon < 1:
        raise ValueError("invalid context length or horizon")
    latest = length - context_length - 2 * horizon
    if latest < context_length:
        return context_length
    proposed = int(np.floor(0.2 * max(0, length - horizon)))
    return min(latest, max(context_length, proposed))


def _training_batch(
    items: Iterable[TimeSeriesItem],
    context_length: int,
    horizon: int,
) -> SeriesBatch:
    selected = [
        item
        for item in items
        if len(item.values) >= 2 * context_length + 2 * horizon
    ]
    if not selected:
        raise ValueError(
            "no item has enough history for fitting plus train/eval forecast origins"
        )
    dimensions = {item.values.shape[1] for item in selected}
    if len(dimensions) != 1:
        raise ValueError("training items must share the same variate dimension")
    windows: list[np.ndarray] = []
    window_ids: list[str] = []
    stride = max(1, int(horizon))
    for item in selected:
        fit_end = _fit_region_end(len(item.values), context_length, horizon)
        starts = list(range(0, fit_end - context_length + 1, stride))
        final_start = fit_end - context_length
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            windows.append(item.values[start : start + context_length])
            window_ids.append(f"{item.item_id}@{start}")
    values = np.stack(windows)
    return SeriesBatch(
        values,
        np.ones_like(values, dtype=bool),
        item_ids=tuple(window_ids),
    )


def _training_statistics(batch: SeriesBatch) -> tuple[np.ndarray, np.ndarray]:
    matrix = batch.values.reshape(-1, batch.shape[2])
    medians = np.median(matrix, axis=0)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
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


def _save_imputer_artifact(
    candidate_id: str,
    artifact: Any,
    directory: Path,
) -> tuple[str, str]:
    adapter = DEFAULT_REGISTRY.create(candidate_id)
    if hasattr(adapter, "save_artifact"):
        target = directory / candidate_id
        adapter.save_artifact(artifact, target)
        return "adapter", str(target.name)
    target = directory / f"{candidate_id}.joblib"
    joblib.dump(artifact, target)
    return "joblib", target.name


def execute_fit_imputers(
    preparation: StagePreparation,
    config: AppConfig,
    inputs: StageInputs,
) -> Mapping[str, Any]:
    if inputs.audit_artifact is None:
        raise ValueError("fit-imputers requires an audit artifact")
    output = preparation.store.root / "imputer_artifacts"
    output.mkdir(parents=True, exist_ok=False)
    runner = CandidateRunner(DEFAULT_REGISTRY)
    allowed_devices = _allowed_devices(config)
    manifest: dict[str, Any] = {"schema_version": 1, "datasets": {}}
    for dataset, items in _datasets(config, inputs.audit_artifact):
        batch = _training_batch(
            items,
            config.experiment.context_length,
            config.experiment.horizon,
        )
        dataset_dir = output / dataset.dataset_id
        dataset_dir.mkdir(parents=True)
        medians, correlation = _training_statistics(batch)
        np.savez(
            dataset_dir / "training_statistics.npz",
            medians=medians,
            correlation=correlation,
        )
        entries: dict[str, Any] = {}
        for spec in DEFAULT_REGISTRY.specs():
            if spec.fit_scope == "none":
                entries[spec.imputer_id] = {"status": "stateless"}
                continue
            if spec.device != "any" and spec.device not in allowed_devices:
                entries[spec.imputer_id] = {
                    "status": "unavailable",
                    "reason": f"device {spec.device!r} is excluded by runtime config",
                }
                continue
            availability = DEFAULT_REGISTRY.availability(spec.imputer_id)
            if not availability.available:
                entries[spec.imputer_id] = {
                    "status": "unavailable",
                    "missing_dependencies": list(availability.missing),
                }
                continue
            try:
                artifact = runner.fit(
                    spec.imputer_id,
                    batch,
                    {"period": dataset.period, "dataset_id": dataset.dataset_id},
                )
                serializer, relative_path = _save_imputer_artifact(
                    spec.imputer_id, artifact, dataset_dir
                )
            except Exception as error:
                entries[spec.imputer_id] = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
                if config.runtime.fail_fast:
                    raise
            else:
                entries[spec.imputer_id] = {
                    "status": "fitted",
                    "serializer": serializer,
                    "path": relative_path,
                }
        manifest["datasets"][dataset.dataset_id] = {
            "training_windows": batch.shape[0],
            "dimension": batch.shape[2],
            "context_length": batch.shape[1],
            "statistics": "training_statistics.npz",
            "candidates": entries,
        }
    if not manifest["datasets"]:
        raise ValueError("audit artifact contains no enabled dataset from the manifest")
    manifest_path = _write_json(output / "manifest.json", manifest)
    return {"imputer_artifacts": str(output), "manifest": str(manifest_path)}


def _load_imputer_artifacts(
    root: Path, dataset_id: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, str]]:
    return load_dataset_imputer_artifacts(root, dataset_id, DEFAULT_REGISTRY)


def _allowed_devices(config: AppConfig) -> tuple[str, ...]:
    if config.runtime.device == "cpu":
        return ("cpu",)
    if config.runtime.device == "gpu":
        return ("gpu",)
    return ("cpu", "gpu")


def _partition_origins(
    origins: tuple[int, ...],
    partition: Literal["all", "train", "eval"],
) -> tuple[int, ...]:
    if partition == "all":
        return origins
    if len(origins) < 2:
        return origins if partition == "train" else ()
    boundary = min(len(origins) - 1, max(1, int(np.ceil(0.7 * len(origins)))))
    return origins[:boundary] if partition == "train" else origins[boundary:]


def _episode_iter(
    config: AppConfig,
    dataset: Any,
    items: Iterable[TimeSeriesItem],
    partition: Literal["all", "train", "eval"] = "all",
):
    length = config.experiment.context_length
    for item in items:
        origins = rolling_origins(
            len(item.values),
            _fit_region_end(
                len(item.values), length, config.experiment.horizon
            )
            + length,
            config.experiment.horizon,
            config.experiment.horizon,
        )
        for origin in _partition_origins(origins, partition):
            for mechanism in config.experiment.missing_mechanisms:
                for rate in config.experiment.missing_rates:
                    for repetition, configured_seed in enumerate(config.experiment.seeds):
                        masking = MaskingSpec(mechanism, rate)
                        episode = build_episode(
                            item,
                            dataset.dataset_id,
                            origin,
                            length,
                            config.experiment.horizon,
                            masking,
                            repetition=int(configured_seed) + repetition,
                        )
                        episode_id = (
                            f"{dataset.dataset_id}__{item.item_id}__{origin}__"
                            f"{mechanism}__{rate:g}__{configured_seed}"
                        )
                        yield episode_id, episode


def _forecaster_artifacts(inputs: StageInputs) -> tuple[tuple[str, Path], ...]:
    if inputs.forecaster_id is None or inputs.forecaster_artifact is None:
        raise ValueError("forecaster ID and artifact are required")
    model_ids = parse_forecaster_ids(inputs.forecaster_id)
    source = inputs.forecaster_artifact.resolve()
    if len(model_ids) == 1:
        if not source.exists():
            raise ValueError(f"forecaster artifact does not exist: {source}")
        return ((model_ids[0], source),)

    resolved: dict[str, Path] = {}
    if source.is_dir():
        resolved = {model_id: source / model_id for model_id in model_ids}
    else:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "multiple forecasters require a checkpoint directory or JSON mapping"
            ) from error
        mapping = payload.get("artifacts") if isinstance(payload, dict) else None
        if mapping is None:
            mapping = payload
        if not isinstance(mapping, dict):
            raise ValueError("forecaster artifact JSON must map model IDs to paths")
        for model_id in model_ids:
            raw_path = mapping.get(model_id)
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(
                    f"forecaster artifact mapping has no path for {model_id!r}"
                )
            target = Path(raw_path)
            resolved[model_id] = (
                target if target.is_absolute() else (source.parent / target).resolve()
            )
    missing = [model_id for model_id, path in resolved.items() if not path.exists()]
    if missing:
        raise ValueError(
            "forecaster artifacts do not exist for: " + ", ".join(sorted(missing))
        )
    return tuple((model_id, resolved[model_id]) for model_id in model_ids)


def _preflight_forecaster(registry: Any, model_id: str, artifact: Path):
    adapter = registry.build(model_id, model_name=str(artifact))
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
    edge_list = [
        edge
        for edge in edges
        if eligible.get(edge.left) and eligible.get(edge.right)
    ]
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
        matching_left = [
            edge for edge in edge_list if candidate_id in eligible[edge.left]
        ]
        if matching_left:
            edge = matching_left[int(rng.integers(len(matching_left)))]
            right_values = eligible[edge.right]
            add(
                edge,
                candidate_id,
                right_values[int(rng.integers(len(right_values)))],
            )
        matching_right = [
            edge for edge in edge_list if candidate_id in eligible[edge.right]
        ]
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
    )


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
    assert inputs.audit_artifact is not None
    assert inputs.imputer_artifacts is not None
    assert inputs.forecaster_artifact is not None
    assert inputs.forecaster_id is not None
    registry = default_forecast_registry()
    selected_forecasters = _forecaster_artifacts(inputs)
    adapters = {
        model_id: _preflight_forecaster(registry, model_id, artifact)
        for model_id, artifact in selected_forecasters
    }
    forecast_runner = ForecastRunner(registry, adapters)
    candidate_runner = CandidateRunner(DEFAULT_REGISTRY)
    labels_path = preparation.store.root / "teacher_labels.jsonl"
    pairs_path = preparation.store.root / "pair_labels.jsonl"
    group_count = row_count = pair_count = 0
    with labels_path.open("w", encoding="utf-8") as labels_handle, pairs_path.open(
        "w", encoding="utf-8"
    ) as pairs_handle:
        for dataset, items in _datasets(config, inputs.audit_artifact):
            artifacts, _, correlation, _ = _load_imputer_artifacts(
                inputs.imputer_artifacts, dataset.dataset_id
            )
            for episode_id, episode in _episode_iter(
                config, dataset, items, partition="train"
            ):
                candidate_ids = tuple(
                    entry.imputer_id
                    for entry in DEFAULT_REGISTRY.specs()
                    if DEFAULT_REGISTRY.availability(entry.imputer_id).available
                    and (
                        entry.device == "any"
                        or entry.device in _allowed_devices(config)
                    )
                    and (entry.fit_scope == "none" or entry.imputer_id in artifacts)
                )
                budget = BudgetSpec(
                    max_candidates=len(candidate_ids),
                    allowed_devices=_allowed_devices(config),
                )
                candidates = candidate_runner.run_many(
                    candidate_ids,
                    episode.context,
                    artifacts,
                    seed=episode.seed,
                    budget=budget,
                )
                pipeline = BlockwiseFAIS(
                    imputer_artifacts=artifacts,
                    training_correlation=correlation,
                )
                pseudo = pipeline._pseudo_batch(
                    episode.context, episode.seed, max_blocks=8
                )
                pseudo_candidates = candidate_runner.run_many(
                    candidate_ids,
                    pseudo,
                    artifacts,
                    seed=episode.seed,
                    budget=budget,
                )
                proxy_mask = pseudo.observed_mask | ~episode.context.observed_mask
                graph = build_block_graph(episode.blocks, correlation)
                anchor = candidates["locf"].values
                by_block = {block.block_id: block for block in episode.blocks}
                eligible: dict[str, tuple[str, ...]] = {}
                for block in episode.blocks:
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
                                or DEFAULT_REGISTRY.get_spec(candidate_id).supports_tail
                            )
                            and candidates[candidate_id].native_valid_mask[selector].all()
                        )
                    )
                for model_id, _ in selected_forecasters:
                    spec = _forecast_spec(config, model_id, episode.context.shape[2])
                    teacher = TeacherBuilder(
                        forecast_runner.predict, seasonality=dataset.period
                    )
                    unary_labels = teacher.unary_labels(
                        episode_id,
                        episode.clean_context[None, ...],
                        episode.clean_future[None, ...],
                        anchor,
                        episode.blocks,
                        candidates,
                        spec,
                        candidate_filter=lambda block, candidate_id, _result: (
                            candidate_id in eligible[block.block_id]
                        ),
                    )
                    losses = {
                        (label.block_id, label.candidate_id): label
                        for label in unary_labels
                    }
                    clean_scales = teacher._scales(
                        episode.clean_context[None, ...], spec.target_indices or ()
                    )
                    clean_loss = teacher._loss(
                        episode.clean_context[None, ...],
                        episode.clean_future[None, ...],
                        spec,
                        clean_scales,
                    )
                    anchor_loss = teacher._loss(
                        anchor,
                        episode.clean_future[None, ...],
                        spec,
                        clean_scales,
                    )
                    for block in episode.blocks:
                        group_id = f"{model_id}::{episode_id}::{block.block_id}"
                        wrote_group = False
                        for candidate_id in candidate_ids:
                            label = losses.get((block.block_id, candidate_id))
                            if label is None:
                                continue
                            imputer_spec = DEFAULT_REGISTRY.get_spec(candidate_id)
                            prior = merge_features(
                                block_features(episode.context, block, dataset.period),
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
                        stable_seed(episode.seed, model_id, "pair_labels"),
                    )
                    for edge, left_candidate, right_candidate in requests:
                        left = by_block[edge.left]
                        right = by_block[edge.right]
                        interaction = teacher.pair_interaction(
                            episode.clean_future[None, ...],
                            anchor,
                            left,
                            right,
                            candidates[left_candidate],
                            candidates[right_candidate],
                            spec,
                            scale_context=episode.clean_context[None, ...],
                        )
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
    if row_count == 0:
        raise ValueError("no teacher label rows were generated")
    if pair_count == 0:
        raise ValueError("no pair label rows were generated")
    summary = {
        "teacher_labels": str(labels_path),
        "pair_labels": str(pairs_path),
        "forecasters": [model_id for model_id, _ in selected_forecasters],
        "imputer_artifacts": str(inputs.imputer_artifacts.resolve()),
        "origin_partition": "train",
        "ranking_groups": group_count,
        "unary_rows": row_count,
        "pair_rows": pair_count,
    }
    _write_json(preparation.store.root / "labels_manifest.json", summary)
    return summary


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
        sorted(
            set().union(
                *(set(row["unary_features"]) for row in ordered_rows)
            )
        )
    )
    prior = _matrix(ordered_rows, "prior_features", feature_names)
    unary = _matrix(ordered_rows, "unary_features", feature_names)
    labels = np.asarray([row["degradation"] for row in ordered_rows], dtype=float)
    groups = tuple(len(group) for group in grouped.values())

    pair_feature_names = tuple(
        sorted(set().union(*(set(row["features"]) for row in pair_rows)))
    )
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

    router_config = RouterConfig.model_validate(
        load_yaml(config.registries.router_config)
    )
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

    field = "family_id" if split == "leave_dataset_out" else "forecaster_id"
    held_out_values = tuple(sorted({str(row[field]) for row in rows}))
    if len(held_out_values) < 2:
        raise ValueError(
            f"{split} requires labels from at least two distinct {field} values"
        )
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
        None
        if item.timestamps is None
        else item.timestamps[start_index : episode.forecast_origin]
    )
    frequency = item.freq
    upper = frequency.upper()
    if upper.endswith("T") and upper[:-1].isdigit():
        frequency = f"{upper[:-1]}min"
    else:
        frequency = {"T": "min", "H": "h", "M": "ME"}.get(upper, frequency)
    context_start = (
        item.start + start_index * to_offset(frequency)
        if timestamps is None
        else timestamps[0]
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
    if split not in {"leave_dataset_out", "leave_model_out"}:
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
                raise ValueError("leave-dataset-out router has no held-out family metadata")
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
        held_out_field = (
            "family_id" if split == "leave_dataset_out" else "forecaster_id"
        )
        held_out = str(dataset.family_id) if split == "leave_dataset_out" else model_id
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
    output.mkdir(parents=True, exist_ok=False)
    assignments_path = preparation.store.root / "routing_assignments.jsonl"
    count = 0
    with assignments_path.open("w", encoding="utf-8") as handle:
        for dataset, items in _datasets(config, inputs.audit_artifact):
            router = router_for(dataset)
            if router is None:
                continue
            artifacts, medians, correlation, artifact_failures = _load_imputer_artifacts(
                inputs.imputer_artifacts, dataset.dataset_id
            )
            item_lookup = {item.item_id: item for item in items}
            pipeline = BlockwiseFAIS(
                config=config,
                router=router,
                imputer_artifacts=artifacts,
                artifact_load_failures=artifact_failures,
                training_medians=medians,
                training_correlation=correlation,
            )
            for episode_id, episode in _episode_iter(
                config, dataset, items, partition="eval"
            ):
                item = _context_item(item_lookup[episode.item_id], episode)
                spec = _forecast_spec(
                    config, model_id, episode.context.shape[2]
                )
                result = pipeline.impute(
                    item,
                    episode.context.observed_mask[0],
                    spec,
                    BudgetSpec(
                        max_candidates=pipeline.shortlist_size,
                        allowed_devices=_allowed_devices(config),
                    ),
                    seed=episode.seed,
                )
                relative = Path(dataset.dataset_id) / f"{count:08d}.npz"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    values=result.values,
                    observed_mask=result.observed_mask,
                    clean_context=episode.clean_context,
                )
                _append_jsonl(
                    handle,
                    {
                        "episode_id": episode_id,
                        "dataset_id": dataset.dataset_id,
                        "family_id": dataset.family_id,
                        "forecaster_id": model_id,
                        "item_id": episode.item_id,
                        "forecast_origin": episode.forecast_origin,
                        "file": str(relative),
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
                    },
                )
                count += 1
    if count == 0:
        raise ValueError("no imputation episode was generated")
    summary = {
        "imputations": str(output),
        "routing_assignments": str(assignments_path),
        "episode_count": count,
        "origin_partition": "eval",
        "forecaster_id": model_id,
    }
    _write_json(preparation.store.root / "imputation_manifest.json", summary)
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
