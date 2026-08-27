"""Numerical feature extraction for candidate ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tsfm_fais.contracts import (
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    ImputerSpec,
    MissingBlock,
    SeriesBatch,
)

from .graph import BlockGraph


@dataclass(frozen=True)
class RoutingFeatureTable:
    """Dense ``[block, candidate, feature]`` table with stable row keys."""

    block_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        expected = (len(self.block_ids), len(self.candidate_ids), len(self.feature_names))
        if values.shape != expected:
            raise ValueError(f"feature table must have shape {expected}, got {values.shape}")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block_ids must be unique")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("routing features must be finite")
        object.__setattr__(self, "values", values)

    def flatten(self) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
        keys = tuple(
            (block_id, candidate_id)
            for block_id in self.block_ids
            for candidate_id in self.candidate_ids
        )
        return self.values.reshape(len(keys), len(self.feature_names)), keys


class RoutingFeatureExtractor:
    """Extract deterministic block/candidate features without model imports."""

    feature_names = (
        "block_length",
        "block_length_ratio",
        "block_start_ratio",
        "block_tail_distance",
        "block_is_tail",
        "block_channel_ratio",
        "concurrent_missing",
        "target_missing_rate",
        "global_missing_rate",
        "local_missing_rate",
        "observed_mean",
        "observed_std",
        "boundary_gap",
        "graph_degree",
        "graph_weighted_degree",
        "candidate_runtime_seconds",
        "candidate_peak_memory_mb",
        "candidate_native_coverage",
        "candidate_success",
        "candidate_partial",
        "candidate_imputed_mean",
        "candidate_imputed_std",
        "candidate_boundary_jump",
        "candidate_mean_uncertainty",
        "candidate_cost_tier",
        "candidate_joint",
        "candidate_trainable",
        "candidate_supports_tail",
        "candidate_stochastic",
    )

    def __init__(self, *, period: int | None = None) -> None:
        if period is not None and period < 1:
            raise ValueError("period must be positive")
        self.period = period

    @staticmethod
    def _graph_features(graph: BlockGraph | None, block_id: str) -> tuple[float, float]:
        if graph is None:
            return 0.0, 0.0
        incident = [edge for edge in graph.edges if edge.left == block_id or edge.right == block_id]
        return float(len(incident)), float(sum(edge.weight for edge in incident))

    @staticmethod
    def _candidate_block_features(
        batch: SeriesBatch,
        block: MissingBlock,
        candidate: CandidateResult,
    ) -> tuple[float, ...]:
        if np.asarray(candidate.values).shape != batch.shape:
            raise ValueError(f"candidate {candidate.imputer_id} values do not match the batch")
        if np.asarray(candidate.native_valid_mask).shape != batch.shape:
            raise ValueError(
                f"candidate {candidate.imputer_id} native_valid_mask does not match the batch"
            )
        selector = (block.batch_index, slice(block.start, block.end), block.channel)
        filled = np.asarray(candidate.values, dtype=float)[selector]
        valid = np.asarray(candidate.native_valid_mask, dtype=bool)[selector]
        finite = filled[np.isfinite(filled)]
        mean = float(np.mean(finite)) if finite.size else 0.0
        std = float(np.std(finite)) if finite.size else 0.0
        jumps: list[float] = []
        if (
            block.start > 0
            and batch.observed_mask[block.batch_index, block.start - 1, block.channel]
        ):
            jumps.append(
                abs(
                    float(filled[0])
                    - float(batch.values[block.batch_index, block.start - 1, block.channel])
                )
            )
        if (
            block.end < batch.shape[1]
            and batch.observed_mask[block.batch_index, block.end, block.channel]
        ):
            jumps.append(
                abs(
                    float(filled[-1])
                    - float(batch.values[block.batch_index, block.end, block.channel])
                )
            )
        uncertainty = 0.0
        if candidate.uncertainty is not None:
            selected = np.asarray(candidate.uncertainty, dtype=float)[selector]
            selected = selected[np.isfinite(selected)]
            uncertainty = float(np.mean(selected)) if selected.size else 0.0
        return (
            max(0.0, float(candidate.runtime_seconds)),
            max(0.0, float(candidate.peak_memory_bytes)) / (1024**2),
            float(np.mean(valid)),
            float(candidate.status == CandidateStatus.SUCCESS),
            float(candidate.status == CandidateStatus.PARTIAL),
            mean,
            std,
            float(np.mean(jumps)) if jumps else 0.0,
            uncertainty,
        )

    def transform(
        self,
        batch: SeriesBatch,
        blocks: tuple[MissingBlock, ...] | list[MissingBlock],
        candidates: Mapping[str, CandidateResult],
        *,
        graph: BlockGraph | None = None,
        imputer_specs: Mapping[str, ImputerSpec] | None = None,
    ) -> RoutingFeatureTable:
        block_tuple = tuple(blocks)
        candidate_ids = tuple(candidates)
        if not block_tuple or not candidate_ids:
            raise ValueError("routing feature extraction requires blocks and candidates")
        rows = np.empty(
            (len(block_tuple), len(candidate_ids), len(self.feature_names)), dtype=float
        )
        for block_index, block in enumerate(block_tuple):
            base = block_features(batch, block, self.period)
            degree, weighted_degree = self._graph_features(graph, block.block_id)
            block_values = (
                base["length"],
                base["length_ratio"],
                base["start_ratio"],
                base["tail_distance"],
                base["is_tail"],
                base["channel_ratio"],
                base["concurrent_missing"],
                base["target_missing_rate"],
                base["global_missing_rate"],
                base["local_missing_rate"],
                base["observed_mean"],
                base["observed_std"],
                base["boundary_gap"],
                degree,
                weighted_degree,
            )
            for candidate_index, candidate_id in enumerate(candidate_ids):
                candidate = candidates[candidate_id]
                spec = (imputer_specs or {}).get(candidate_id)
                spec_values = (
                    float(spec.cost_tier) if spec is not None else 1.0,
                    float(spec.mode == "joint_multivariate") if spec is not None else 0.0,
                    float(spec.fit_scope != "none") if spec is not None else 0.0,
                    float(spec.supports_tail) if spec is not None else 1.0,
                    float(spec.stochastic) if spec is not None else 0.0,
                )
                rows[block_index, candidate_index] = (
                    *block_values,
                    *self._candidate_block_features(batch, block, candidate),
                    *spec_values,
                )
        rows = np.nan_to_num(rows, nan=0.0, posinf=1e12, neginf=-1e12)
        return RoutingFeatureTable(
            tuple(block.block_id for block in block_tuple),
            candidate_ids,
            tuple(self.feature_names),
            rows,
        )


def proxy_unary_scores(table: RoutingFeatureTable) -> dict[tuple[str, str], float]:
    """Deterministic score used before a learned unary model is available."""

    index = {name: position for position, name in enumerate(table.feature_names)}
    coverage = table.values[:, :, index["candidate_native_coverage"]]
    boundary = table.values[:, :, index["candidate_boundary_jump"]]
    uncertainty = table.values[:, :, index["candidate_mean_uncertainty"]]
    runtime = table.values[:, :, index["candidate_runtime_seconds"]]
    cost = table.values[:, :, index["candidate_cost_tier"]]
    score = (
        1000.0 * (1.0 - np.clip(coverage, 0.0, 1.0))
        + 0.01 * np.maximum(boundary, 0.0)
        + 0.01 * np.maximum(uncertainty, 0.0)
        + 0.001 * np.maximum(runtime, 0.0)
        + 0.001 * np.maximum(cost, 0.0)
    )
    return {
        (block_id, candidate_id): float(score[block_index, candidate_index])
        for block_index, block_id in enumerate(table.block_ids)
        for candidate_index, candidate_id in enumerate(table.candidate_ids)
    }


def proxy_pairwise_scores(
    graph: BlockGraph,
    candidates: Mapping[str, CandidateResult],
) -> dict[tuple[str, str, str, str], float]:
    """Estimate interaction risk from candidate disagreement on linked blocks."""

    blocks = {block.block_id: block for block in graph.blocks}
    scores: dict[tuple[str, str, str, str], float] = {}
    for edge in graph.edges:
        left = blocks[edge.left]
        right = blocks[edge.right]
        left_selector = (left.batch_index, slice(left.start, left.end), left.channel)
        right_selector = (right.batch_index, slice(right.start, right.end), right.channel)
        left_means = {
            candidate_id: float(np.mean(np.asarray(result.values)[left_selector]))
            for candidate_id, result in candidates.items()
        }
        right_means = {
            candidate_id: float(np.mean(np.asarray(result.values)[right_selector]))
            for candidate_id, result in candidates.items()
        }
        scale_values = np.asarray((*left_means.values(), *right_means.values()), dtype=float)
        scale = max(float(np.std(scale_values)), 1.0)
        for left_candidate in candidates:
            for right_candidate in candidates:
                disagreement = (
                    abs(left_means[left_candidate] - right_means[right_candidate]) / scale
                )
                scores[(edge.left, left_candidate, edge.right, right_candidate)] = float(
                    disagreement
                )
    return scores


def pair_features(
    batch: SeriesBatch,
    left_block: MissingBlock,
    right_block: MissingBlock,
    left_candidate: CandidateResult,
    right_candidate: CandidateResult,
    *,
    edge_weight: float = 1.0,
) -> dict[str, float]:
    """Numerical features for one linked block/candidate pair.

    The function is intentionally independent from the learned model so the
    same schema can be used when generating pair labels and during routing.
    """

    if left_block.batch_index != right_block.batch_index:
        raise ValueError("pair features require blocks from the same batch item")
    left_selector = (
        left_block.batch_index,
        slice(left_block.start, left_block.end),
        left_block.channel,
    )
    right_selector = (
        right_block.batch_index,
        slice(right_block.start, right_block.end),
        right_block.channel,
    )
    for candidate in (left_candidate, right_candidate):
        if np.asarray(candidate.values).shape != batch.shape:
            raise ValueError("candidate values do not match the batch")
    left_values = np.asarray(left_candidate.values, dtype=float)[left_selector]
    right_values = np.asarray(right_candidate.values, dtype=float)[right_selector]
    left_valid = np.asarray(left_candidate.native_valid_mask, dtype=bool)[left_selector]
    right_valid = np.asarray(right_candidate.native_valid_mask, dtype=bool)[right_selector]
    overlap = max(
        0,
        min(left_block.end, right_block.end) - max(left_block.start, right_block.start),
    )
    gap = max(
        0,
        max(left_block.start, right_block.start) - min(left_block.end, right_block.end),
    )
    scale = max(
        float(np.std(np.concatenate((left_values, right_values)))),
        1e-8,
    )
    features = {
        "pair_edge_weight": float(edge_weight),
        "pair_same_channel": float(left_block.channel == right_block.channel),
        "pair_temporal_overlap": float(overlap),
        "pair_temporal_gap": float(gap),
        "pair_length_ratio": float(left_block.length / max(1, right_block.length)),
        "pair_same_candidate": float(left_candidate.imputer_id == right_candidate.imputer_id),
        "pair_mean_disagreement": float(
            abs(float(np.mean(left_values)) - float(np.mean(right_values))) / scale
        ),
        "pair_std_disagreement": float(
            abs(float(np.std(left_values)) - float(np.std(right_values))) / scale
        ),
        "pair_native_coverage": float(
            0.5 * (float(np.mean(left_valid)) + float(np.mean(right_valid)))
        ),
        "pair_runtime_seconds": float(
            max(0.0, left_candidate.runtime_seconds) + max(0.0, right_candidate.runtime_seconds)
        ),
    }
    features[f"pair_left_candidate::{left_candidate.imputer_id}"] = 1.0
    features[f"pair_right_candidate::{right_candidate.imputer_id}"] = 1.0
    return features


def block_features(
    batch: SeriesBatch,
    block: MissingBlock,
    period: int | None = None,
) -> dict[str, float]:
    _, length, dimensions = batch.shape
    values = batch.values[block.batch_index, :, block.channel]
    observed = batch.observed_mask[block.batch_index, :, block.channel]
    finite = values[observed]
    left = values[block.start - 1] if block.start > 0 and observed[block.start - 1] else np.nan
    right = values[block.end] if block.end < length and observed[block.end] else np.nan
    overlap = float(np.mean(~batch.observed_mask[block.batch_index, block.start : block.end]))
    local_rate = float(
        batch.metadata.get(
            "local_missing_rate", float(np.mean(~batch.observed_mask[block.batch_index]))
        )
    )
    global_rate = float(batch.metadata.get("global_missing_rate", local_rate))
    target_rate = float(batch.metadata.get("target_missing_rate", global_rate))
    features = {
        "length": float(block.length),
        "length_ratio": block.length / length,
        "start_ratio": block.start / length,
        "tail_distance": float(length - block.end),
        "is_tail": float(block.end == length),
        "channel_ratio": block.channel / max(1, dimensions - 1),
        "concurrent_missing": overlap,
        "target_missing_rate": target_rate,
        "global_missing_rate": global_rate,
        "local_missing_rate": local_rate,
        "observed_mean": float(np.mean(finite)) if finite.size else 0.0,
        "observed_std": float(np.std(finite)) if finite.size else 0.0,
        "boundary_gap": float(abs(right - left))
        if np.isfinite(left) and np.isfinite(right)
        else 0.0,
        "period_ratio": block.length / max(1, period or length),
    }
    forecast_origin = batch.metadata.get("forecast_origin")
    try:
        numeric_origin = float(str(forecast_origin))
    except (TypeError, ValueError):
        numeric_origin = -1.0
    if np.isfinite(numeric_origin) and numeric_origin >= 0:
        features["forecast_origin_log1p"] = float(np.log1p(numeric_origin))
    for field in ("dataset_id", "family_id"):
        value = batch.metadata.get(field)
        if isinstance(value, str) and value:
            features[f"{field}::{value}"] = 1.0
    mechanism = batch.metadata.get("missing_mechanism")
    if isinstance(mechanism, str) and mechanism:
        features[f"missing_mechanism::{mechanism}"] = 1.0
    return features


def candidate_features(spec: ImputerSpec, forecast: ForecastSpec) -> dict[str, float]:
    features = {
        "candidate_cost": float(spec.cost_tier),
        "candidate_joint": float(spec.mode == "joint_multivariate"),
        "candidate_trainable": float(spec.fit_scope != "none"),
        "candidate_tail": float(spec.supports_tail),
        "candidate_periodic": float(spec.requires_period),
        "candidate_stochastic": float(spec.stochastic),
        "forecast_joint": float(forecast.mode == "joint_multivariate"),
        "forecast_horizon": float(forecast.horizon),
        "forecast_context": float(forecast.context_length or 0),
        "forecast_target_count": float(len(forecast.target_indices or ())),
    }
    features[f"candidate_id::{spec.imputer_id}"] = 1.0
    features[f"candidate_family::{spec.family}"] = 1.0
    features[f"forecast_model::{forecast.model_id}"] = 1.0
    return features


def forecast_block_features(
    block: MissingBlock,
    forecast: ForecastSpec,
) -> dict[str, float]:
    """Describe whether a missing block can directly enter the forecaster."""

    targets = set(forecast.target_indices or ())
    is_target = block.channel in targets
    is_visible = forecast.mode == "joint_multivariate" or is_target
    return {
        "block_is_forecast_target": float(is_target),
        "block_visible_to_forecaster": float(is_visible),
    }


def proxy_features(
    candidate: CandidateResult,
    pseudo_truth: np.ndarray,
    pseudo_mask: np.ndarray,
    source: np.ndarray | None = None,
    *,
    channel: int | None = None,
) -> dict[str, float]:
    feature_cap = 1e12
    hidden = ~np.asarray(pseudo_mask, dtype=bool)
    selected_hidden = hidden
    channel_available = 0.0
    if channel is not None:
        if not 0 <= int(channel) < hidden.shape[-1]:
            raise ValueError("proxy channel is outside the variate axis")
        channel_hidden = np.zeros_like(hidden, dtype=bool)
        channel_hidden[..., int(channel)] = hidden[..., int(channel)]
        if channel_hidden.any():
            selected_hidden = channel_hidden
            channel_available = 1.0
    predicted = np.asarray(candidate.values, dtype=float)
    truth = np.asarray(pseudo_truth, dtype=float)
    error = predicted[selected_hidden] - truth[selected_hidden]
    global_error = predicted[hidden] - truth[hidden]
    with np.errstate(over="ignore", invalid="ignore"):
        proxy_mae = float(np.mean(np.abs(error))) if error.size else 0.0
        proxy_rmse = float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
        global_mae = float(np.mean(np.abs(global_error))) if global_error.size else 0.0
        global_rmse = float(np.sqrt(np.mean(np.square(global_error)))) if global_error.size else 0.0
        proxy_bias = float(np.mean(error)) if error.size else 0.0
    result: dict[str, float] = {
        "proxy_mae": proxy_mae,
        "proxy_rmse": proxy_rmse,
        "proxy_global_mae": global_mae,
        "proxy_global_rmse": global_rmse,
        "proxy_bias": proxy_bias,
        "proxy_channel_available": channel_available,
        "proxy_channel_fraction": (
            float(np.mean(selected_hidden[hidden])) if hidden.any() else 0.0
        ),
        "runtime_seconds": float(candidate.runtime_seconds),
        "peak_memory_mb": candidate.peak_memory_bytes / (1024**2),
        "native_coverage": (
            float(np.mean(candidate.native_valid_mask[selected_hidden])) if error.size else 1.0
        ),
    }
    if source is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            completed_cov = np.cov(
                candidate.values.reshape(-1, candidate.values.shape[-1]),
                rowvar=False,
            )
            source_cov = np.cov(
                np.asarray(source).reshape(-1, candidate.values.shape[-1]),
                rowvar=False,
            )
            result["covariance_drift"] = float(np.linalg.norm(completed_cov - source_cov))
    else:
        result["covariance_drift"] = 0.0
    if candidate.uncertainty is not None:
        selected_uncertainty = np.asarray(candidate.uncertainty, dtype=float)[selected_hidden]
        if error.size and selected_uncertainty.size:
            # A stochastic candidate may return a finite point estimate while its
            # sample variance overflows.  Preserve that evidence as a large risk
            # instead of emitting NaN (or treating missing uncertainty as zero).
            bounded_uncertainty = np.nan_to_num(
                selected_uncertainty,
                nan=feature_cap,
                posinf=feature_cap,
                neginf=feature_cap,
            )
            bounded_uncertainty = np.clip(
                bounded_uncertainty,
                0.0,
                feature_cap,
            )
            result["mean_uncertainty"] = float(np.mean(bounded_uncertainty))
        else:
            result["mean_uncertainty"] = 0.0
    else:
        result["mean_uncertainty"] = 0.0
    return {
        name: float(
            np.clip(
                np.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=feature_cap,
                    neginf=-feature_cap,
                ),
                -feature_cap,
                feature_cap,
            )
        )
        for name, value in result.items()
    }


def merge_features(*parts: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"duplicate feature names: {sorted(overlap)}")
        merged.update(part)
    return merged
