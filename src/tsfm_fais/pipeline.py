"""End-to-end block-wise imputer selection and assembly."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_fais.config import AppConfig, load_config
from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    MissingBlock,
    RoutingResult,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers import (
    DEFAULT_REGISTRY,
    CandidateRunner,
    ImputerRegistry,
    load_dataset_imputer_artifacts,
)
from tsfm_fais.routing.blocks import build_block_graph, detect_missing_blocks
from tsfm_fais.routing.features import (
    block_features,
    candidate_features,
    forecast_block_features,
    merge_features,
    pair_features,
    proxy_features,
)
from tsfm_fais.routing.graph import BlockGraph
from tsfm_fais.routing.models import RouterBundle
from tsfm_fais.routing.solver import beam_search, greedy_shortlist


def _forecast_consensus_scores(
    candidate_values: Mapping[str, np.ndarray],
    forecast_spec: ForecastSpec,
    predictor: Callable[[np.ndarray, ForecastSpec], Any],
    mase_scale: np.ndarray,
) -> tuple[dict[str, float], dict[str, tuple[float, ...]]]:
    """Score candidate forecasts against their pointwise median."""

    candidate_ids = tuple(sorted(candidate_values))
    if not candidate_ids:
        raise ValueError("forecast medoid requires at least one candidate")
    contexts = np.concatenate(
        [np.asarray(candidate_values[candidate_id], dtype=float) for candidate_id in candidate_ids],
        axis=0,
    )
    forecast = predictor(contexts, forecast_spec)
    point = np.asarray(forecast.point, dtype=float)
    if point.shape[0] != len(candidate_ids):
        raise ValueError("forecast medoid predictor returned an incompatible batch")
    scale = np.asarray(mase_scale, dtype=float).reshape(-1)
    if scale.shape != (point.shape[2],) or not np.isfinite(scale).all():
        raise ValueError("forecast medoid scale does not match forecast targets")
    scale = np.maximum(np.abs(scale), 1e-8)
    normalized = point / scale[None, None, :]
    center = np.median(normalized, axis=0)
    raw_target_scores = np.mean(np.abs(normalized - center[None, ...]), axis=1)
    raw_scores = np.mean(raw_target_scores, axis=1)
    scores = {
        candidate_id: float(score)
        for candidate_id, score in zip(candidate_ids, raw_scores, strict=True)
    }
    target_scores = {
        candidate_id: tuple(map(float, candidate_scores))
        for candidate_id, candidate_scores in zip(
            candidate_ids,
            raw_target_scores,
            strict=True,
        )
    }
    return scores, target_scores


def _forecast_medoid_candidate(
    candidate_values: Mapping[str, np.ndarray],
    forecast_spec: ForecastSpec,
    predictor: Callable[[np.ndarray, ForecastSpec], Any],
    mase_scale: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Select the candidate closest to the median downstream forecast."""

    scores, _ = _forecast_consensus_scores(
        candidate_values,
        forecast_spec,
        predictor,
        mase_scale,
    )
    candidate_ids = tuple(sorted(scores))
    selected = min(candidate_ids, key=lambda candidate_id: (scores[candidate_id], candidate_id))
    return selected, scores


def _regularized_forecast_consensus_candidate(
    medoid_scores: Mapping[str, float],
    candidate_priors: Mapping[str, float],
    weight: float,
) -> tuple[str, dict[str, float]]:
    """Blend label-free forecast agreement with supported training risk priors."""

    candidate_ids = tuple(sorted(medoid_scores))
    if not candidate_ids:
        raise ValueError("regularized forecast consensus requires candidates")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("forecast consensus prior weight must lie in [0, 1]")
    medoid = np.asarray(
        [medoid_scores[candidate_id] for candidate_id in candidate_ids], dtype=float
    )
    if not np.isfinite(medoid).all():
        raise ValueError("forecast consensus medoid scores must be finite")
    supported = {
        candidate_id: float(candidate_priors[candidate_id])
        for candidate_id in candidate_ids
        if candidate_id in candidate_priors and np.isfinite(candidate_priors[candidate_id])
    }

    def normalize(values: np.ndarray) -> np.ndarray:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper <= lower:
            return np.zeros_like(values)
        return (values - lower) / (upper - lower)

    normalized_medoid = normalize(medoid)
    if weight == 0.0 or len(supported) < 2:
        combined = normalized_medoid
    else:
        unsupported_risk = max(supported.values())
        priors = np.asarray(
            [supported.get(candidate_id, unsupported_risk) for candidate_id in candidate_ids],
            dtype=float,
        )
        combined = (1.0 - weight) * normalized_medoid + weight * normalize(priors)
    scores = {
        candidate_id: float(score)
        for candidate_id, score in zip(candidate_ids, combined, strict=True)
    }
    selected = min(candidate_ids, key=lambda candidate_id: (scores[candidate_id], candidate_id))
    return selected, scores


def _top_k_consensus_weights(
    scores: Mapping[str, float],
    top_k: int,
    third_candidate_relative_gap: float | None = None,
) -> dict[str, float]:
    """Return deterministic uniform weights for the lowest-risk candidates."""

    if top_k < 2:
        raise ValueError("forecast consensus top-k must be at least two")
    if third_candidate_relative_gap is not None and (
        not np.isfinite(third_candidate_relative_gap)
        or third_candidate_relative_gap < 0.0
        or top_k != 2
    ):
        raise ValueError(
            "third-candidate relative gap must be finite and non-negative with top-k two"
        )
    ordered = sorted(
        (
            (float(score), str(candidate_id))
            for candidate_id, score in scores.items()
            if np.isfinite(float(score))
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ordered:
        raise ValueError("forecast consensus top-k requires finite candidate scores")
    selected_count = min(top_k, len(ordered))
    if third_candidate_relative_gap is not None and len(ordered) >= 3:
        second_score = ordered[1][0]
        relative_gap = (ordered[2][0] - second_score) / max(abs(second_score), 1e-8)
        if relative_gap <= third_candidate_relative_gap:
            selected_count = 3
    selected = ordered[:selected_count]
    weight = 1.0 / len(selected)
    return {candidate_id: weight for _, candidate_id in selected}


def _proxy_weighted_consensus_weights(
    candidate_weights: Mapping[str, float],
    proxy_scores: Mapping[str, float],
    *,
    power: float,
) -> dict[str, float]:
    """Reweight selected candidates by inverse pseudo-missing MAE.

    Missing or invalid proxy evidence falls back to the normalized selection
    weights, preserving the forecast-consensus decision.
    """

    if not np.isfinite(power) or power < 0.0:
        raise ValueError("proxy weight power must be finite and non-negative")
    weights = {
        str(candidate_id): float(weight) for candidate_id, weight in candidate_weights.items()
    }
    if not weights or any(not np.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("candidate weights must be finite and non-negative")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("candidate weights must have positive mass")
    normalized = {candidate_id: weight / total for candidate_id, weight in weights.items()}
    if power == 0.0 or len(normalized) == 1:
        return normalized
    errors = {
        candidate_id: float(proxy_scores.get(candidate_id, np.nan)) for candidate_id in normalized
    }
    if any(not np.isfinite(error) or error < 0.0 for error in errors.values()):
        return normalized
    reweighted = {
        candidate_id: weight * max(errors[candidate_id], 1e-8) ** (-power)
        for candidate_id, weight in normalized.items()
    }
    reweighted_total = float(sum(reweighted.values()))
    if not np.isfinite(reweighted_total) or reweighted_total <= 0.0:
        return normalized
    return {candidate_id: weight / reweighted_total for candidate_id, weight in reweighted.items()}


def _pseudo_calibrated_convex_weight(
    pseudo_truth: np.ndarray,
    pseudo_mask: np.ndarray,
    primary: CandidateResult,
    alternative: CandidateResult,
    *,
    channel: int | None,
    prior_weight: float,
    prior_strength: float,
    min_points: int,
) -> tuple[float, dict[str, Any]]:
    """Fit a deterministic convex weight on newly hidden historical values."""

    if not np.isfinite(prior_weight) or not 0.0 <= prior_weight <= 1.0:
        raise ValueError("pseudo calibration prior weight must lie in [0, 1]")
    if not np.isfinite(prior_strength) or prior_strength < 0.0:
        raise ValueError("pseudo calibration prior strength must be non-negative")
    if min_points < 1:
        raise ValueError("pseudo calibration minimum points must be positive")
    truth = np.asarray(pseudo_truth, dtype=float)
    mask = np.asarray(pseudo_mask, dtype=bool)
    primary_values = np.asarray(primary.values, dtype=float)
    alternative_values = np.asarray(alternative.values, dtype=float)
    primary_valid = np.asarray(primary.native_valid_mask, dtype=bool)
    alternative_valid = np.asarray(alternative.native_valid_mask, dtype=bool)
    if not (
        truth.shape
        == mask.shape
        == primary_values.shape
        == alternative_values.shape
        == primary_valid.shape
        == alternative_valid.shape
    ):
        raise ValueError("pseudo calibration tensors must have the same shape")
    if channel is not None and not 0 <= channel < truth.shape[-1]:
        raise ValueError("pseudo calibration channel is outside the variate axis")

    hidden = ~mask
    finite = np.isfinite(truth) & np.isfinite(primary_values) & np.isfinite(alternative_values)
    eligible = hidden & finite & primary_valid & alternative_valid

    def selected_values(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = alternative_values[selected] - primary_values[selected]
        residual = truth[selected] - primary_values[selected]
        return np.asarray(delta, dtype=float), np.asarray(residual, dtype=float)

    source = "channel"
    selected = eligible
    if channel is not None:
        selected = np.zeros_like(eligible, dtype=bool)
        selected[..., channel] = eligible[..., channel]
    if int(np.count_nonzero(selected)) < min_points:
        selected = eligible
        source = "global"
    sample_count = int(np.count_nonzero(selected))
    diagnostics: dict[str, Any] = {
        "protocol": "pseudo_convex_l2_v1",
        "source": source,
        "channel": channel,
        "sample_count": sample_count,
        "min_points": int(min_points),
        "prior_weight": float(prior_weight),
        "prior_strength": float(prior_strength),
        "applied": False,
        "reason": "insufficient_points",
    }
    if sample_count < min_points:
        diagnostics["source"] = "prior"
        diagnostics["weight"] = float(prior_weight)
        return float(prior_weight), diagnostics
    delta, residual = selected_values(selected)
    denominator = float(np.dot(delta, delta))
    numerator = float(np.dot(delta, residual))
    diagnostics.update(
        {
            "data_numerator": numerator,
            "data_denominator": denominator,
        }
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:
        diagnostics["reason"] = "indistinguishable_candidates"
        diagnostics["weight"] = float(prior_weight)
        return float(prior_weight), diagnostics
    ridge = float(prior_strength * denominator / sample_count)
    unconstrained = numerator / denominator
    calibrated = (numerator + ridge * prior_weight) / (denominator + ridge)
    weight = float(np.clip(calibrated, 0.0, 1.0))
    diagnostics.update(
        {
            "applied": True,
            "reason": "calibrated",
            "ridge": ridge,
            "unconstrained_weight": float(unconstrained),
            "regularized_weight": float(calibrated),
            "weight": weight,
        }
    )
    return weight, diagnostics


def _third_candidate_relative_gap(scores: Mapping[str, float]) -> float | None:
    """Return the raw consensus-score gap between ranks two and three."""

    ordered = sorted(float(score) for score in scores.values() if np.isfinite(float(score)))
    if len(ordered) < 3:
        return None
    return (ordered[2] - ordered[1]) / max(abs(ordered[1]), 1e-8)


def _safe_prior_consensus_override(
    selected: str,
    medoid_scores: Mapping[str, float],
    candidate_priors: Mapping[str, float],
    *,
    max_medoid_penalty: float | None,
    min_prior_margin: float,
) -> tuple[str, dict[str, Any]]:
    """Use the lowest-risk training candidate only when forecast agreement supports it."""

    if max_medoid_penalty is None:
        return selected, {"configured": False, "applied": False}
    if not 0.0 <= max_medoid_penalty <= 1.0:
        raise ValueError("prior override medoid penalty must lie in [0, 1]")
    if not np.isfinite(min_prior_margin) or min_prior_margin < 0.0:
        raise ValueError("prior override margin must be finite and non-negative")
    supported = {
        candidate_id: float(candidate_priors[candidate_id])
        for candidate_id in medoid_scores
        if candidate_id in candidate_priors
        and np.isfinite(float(candidate_priors[candidate_id]))
        and np.isfinite(float(medoid_scores[candidate_id]))
    }
    diagnostics: dict[str, Any] = {
        "configured": True,
        "applied": False,
        "original_candidate": selected,
        "max_medoid_penalty": float(max_medoid_penalty),
        "min_prior_margin": float(min_prior_margin),
    }
    if len(supported) < 2 or selected not in supported:
        diagnostics["reason"] = "insufficient_supported_priors"
        return selected, diagnostics
    prior_candidate = min(
        supported, key=lambda candidate_id: (supported[candidate_id], candidate_id)
    )
    medoid = np.asarray([float(medoid_scores[candidate_id]) for candidate_id in supported])
    lower = float(np.min(medoid))
    upper = float(np.max(medoid))
    penalty = (
        0.0 if upper <= lower else (float(medoid_scores[prior_candidate]) - lower) / (upper - lower)
    )
    margin = float(supported[selected] - supported[prior_candidate])
    applied = (
        prior_candidate != selected
        and margin + 1e-12 >= min_prior_margin
        and penalty <= max_medoid_penalty + 1e-12
    )
    diagnostics.update(
        {
            "applied": applied,
            "prior_candidate": prior_candidate,
            "prior_margin": margin,
            "medoid_penalty": penalty,
        }
    )
    return (prior_candidate if applied else selected), diagnostics


def _merge_forecast_consensus_config(
    artifact_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay an explicit inference configuration onto frozen router metadata."""

    merged = dict(artifact_config)
    merged.update(runtime_config)
    return merged


def _candidate_signal_scores(
    mode: str,
    candidate_ids: Iterable[str],
    visible_blocks: Sequence[MissingBlock],
    unary_risks: Mapping[tuple[str, str], float],
    proxy_scores: Mapping[str, float],
) -> dict[str, float]:
    """Aggregate inference-available candidate evidence for a soft global anchor."""

    scores: dict[str, float] = {}
    for candidate_id in candidate_ids:
        if mode == "router_risk":
            values = np.asarray(
                [
                    unary_risks[(block.block_id, candidate_id)]
                    for block in visible_blocks
                    if (block.block_id, candidate_id) in unary_risks
                ],
                dtype=float,
            )
            if values.shape != (len(visible_blocks),) or not np.isfinite(values).all():
                continue
            scores[candidate_id] = float(np.mean(values))
        elif mode == "proxy_min":
            value = float(proxy_scores.get(candidate_id, np.nan))
            if np.isfinite(value):
                scores[candidate_id] = value
        else:
            raise ValueError(f"unknown candidate signal mode {mode!r}")
    return scores


def _anchor_period_is_eligible(
    period: Any,
    context_length: int,
    max_period_ratio: float | None,
) -> tuple[bool, float | None]:
    """Gate an optional soft anchor using only declared historical periodicity."""

    if max_period_ratio is None:
        return True, None
    try:
        ratio = float(period) / float(context_length)
    except (TypeError, ValueError, ZeroDivisionError):
        return False, None
    if not np.isfinite(ratio) or ratio <= 0.0:
        return False, None
    return ratio <= max_period_ratio + 1e-12, ratio


def _forecast_consensus_inputs(
    candidate_values: Mapping[str, np.ndarray],
    forecast_spec: ForecastSpec,
    context_mode: str,
    *,
    correlation: np.ndarray | None = None,
    max_context_variates: int = 8,
) -> tuple[dict[str, np.ndarray], ForecastSpec, tuple[int, ...]]:
    """Project selector-only contexts without changing final forecast inputs."""

    values = {
        candidate_id: np.asarray(candidate_value, dtype=float)
        for candidate_id, candidate_value in candidate_values.items()
    }
    if not values:
        raise ValueError("forecast consensus inputs cannot be empty")
    first = next(iter(values.values()))
    if first.ndim != 3:
        raise ValueError("forecast consensus candidates must be rank-three tensors")
    variate_count = first.shape[2]
    if any(value.ndim != 3 or value.shape[2] != variate_count for value in values.values()):
        raise ValueError("forecast consensus candidates have inconsistent variates")
    native_indices = tuple(range(variate_count))
    if context_mode == "native":
        return values, forecast_spec, native_indices
    if context_mode not in {"targets_only", "targets_with_correlates"}:
        raise ValueError("unknown forecast consensus context mode")
    target_indices = tuple(forecast_spec.target_indices or ())
    if not target_indices:
        raise ValueError("target-only forecast consensus requires target indices")
    if max(target_indices) >= variate_count:
        raise ValueError("forecast consensus target is outside the candidate tensor")
    if context_mode == "targets_only":
        selected_indices = target_indices
    else:
        if max_context_variates < len(target_indices):
            raise ValueError("forecast consensus variate cap is smaller than the target count")
        if variate_count <= max_context_variates:
            return values, forecast_spec, native_indices
        matrix = np.asarray(correlation, dtype=float)
        if matrix.shape != (variate_count, variate_count) or not np.isfinite(matrix).all():
            raise ValueError("forecast consensus correlation has an invalid shape or value")
        target_set = set(target_indices)
        scores = np.max(np.abs(matrix[list(target_indices), :]), axis=0)
        extras = sorted(
            (index for index in native_indices if index not in target_set),
            key=lambda index: (-float(scores[index]), index),
        )[: max_context_variates - len(target_indices)]
        selected_indices = tuple(sorted((*target_indices, *extras)))
    projected: dict[str, np.ndarray] = {}
    for candidate_id, candidate_value in values.items():
        projected[candidate_id] = candidate_value[:, :, list(selected_indices)]
    projected_spec = replace(
        forecast_spec,
        target_indices=tuple(selected_indices.index(index) for index in target_indices),
    )
    return projected, projected_spec, selected_indices


def _historical_backtest_candidate(
    candidate_values: Mapping[str, np.ndarray],
    forecast_spec: ForecastSpec,
    predictor: Callable[[np.ndarray, ForecastSpec], Any],
    mase_scale: np.ndarray,
    clean_context: np.ndarray,
    observed_mask: np.ndarray,
    cutoff: int,
    min_observed_per_target: int,
) -> tuple[str, dict[str, float]]:
    """Select a candidate by forecasting a held-out observed context suffix."""

    candidate_ids = tuple(sorted(candidate_values))
    if not candidate_ids:
        raise ValueError("historical backtest requires at least one candidate")
    length = np.asarray(clean_context).shape[1]
    if not 2 <= cutoff < length:
        raise ValueError("historical backtest cutoff is outside the context")
    validation_length = length - cutoff
    contexts = np.concatenate(
        [
            np.asarray(candidate_values[candidate_id], dtype=float)[:, :cutoff]
            for candidate_id in candidate_ids
        ],
        axis=0,
    )
    validation_spec = replace(
        forecast_spec,
        horizon=validation_length,
        context_length=cutoff,
    )
    point = np.asarray(predictor(contexts, validation_spec).point, dtype=float)
    target_indices = tuple(forecast_spec.target_indices or ())
    expected = (len(candidate_ids), validation_length, len(target_indices))
    if point.shape != expected:
        raise ValueError("historical backtest predictor returned an incompatible batch")
    truth = np.asarray(clean_context, dtype=float)[0, cutoff:, :][:, list(target_indices)]
    validation_mask = np.asarray(observed_mask, dtype=bool)[0, cutoff:, :][:, list(target_indices)]
    scale = np.maximum(np.abs(np.asarray(mase_scale, dtype=float).reshape(-1)), 1e-8)
    if scale.shape != (len(target_indices),) or not np.isfinite(scale).all():
        raise ValueError("historical backtest scale does not match forecast targets")
    if any(
        int(validation_mask[:, target].sum()) < min_observed_per_target
        for target in range(len(target_indices))
    ):
        raise ValueError("historical backtest has insufficient observed targets")
    scores: dict[str, float] = {}
    for candidate_offset, candidate_id in enumerate(candidate_ids):
        target_scores = []
        for target in range(len(target_indices)):
            selected = validation_mask[:, target]
            target_scores.append(
                float(
                    np.mean(
                        np.abs(point[candidate_offset, selected, target] - truth[selected, target])
                    )
                    / scale[target]
                )
            )
        scores[candidate_id] = float(np.mean(target_scores))
    selected = min(
        candidate_ids,
        key=lambda candidate_id: (scores[candidate_id], candidate_id),
    )
    return selected, scores


@dataclass
class FAISResult:
    values: np.ndarray
    routing: RoutingResult
    candidates: dict[str, CandidateResult]
    observed_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutePlan:
    """Immutable routing preparation consumed by :meth:`finish_route`.

    Candidate execution is deliberately absent from the plan.  An external
    scheduler may run ``shortlist`` candidates in candidate-major order, then
    inject their actual and pseudo-missing results into ``finish_route``.
    Calling ``prepare_route`` again with a revised ``available_artifact_ids``
    rebuilds R0 and the shortlist after an artifact-load failure.
    """

    batch: SeriesBatch
    blocks: tuple[MissingBlock, ...]
    graph: BlockGraph
    forecast_spec: ForecastSpec
    budget: BudgetSpec
    seed: int
    period: int | None
    correlation_source: str
    candidate_ids: tuple[str, ...]
    shortlist: tuple[str, ...]
    costs: Mapping[str, float]
    prior_unary: Mapping[tuple[str, str], float]
    pseudo_batch: SeriesBatch | None
    training_medians: np.ndarray | None
    artifact_load_failures: Mapping[str, str]
    available_artifact_ids: frozenset[str]
    allow_fallback_execution: bool
    backtest_batch: SeriesBatch | None = None
    backtest_cutoff: int | None = None

    @property
    def is_noop(self) -> bool:
        return not self.blocks


def _correlation(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    completed = values.copy()
    for channel in range(completed.shape[1]):
        observed = mask[:, channel]
        median = float(np.median(completed[observed, channel])) if observed.any() else 0.0
        completed[~observed, channel] = median
    centered = completed - np.mean(completed, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(
        centered,
        norms[None, :],
        out=np.zeros_like(centered),
        where=norms[None, :] > 0,
    )
    correlation = normalized.T @ normalized
    diagonal = np.flatnonzero(norms > 0)
    correlation[diagonal, diagonal] = 1.0
    return np.clip(correlation, -1.0, 1.0)


def _feature_matrix(rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=float)


def _nonnegative_mapping_value(values: object, key: str) -> float:
    if not isinstance(values, Mapping):
        return 0.0
    try:
        value = float(values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) and value >= 0.0 else 0.0


def _ranker_risks(
    keys: list[tuple[str, str]],
    predictions: np.ndarray,
    scale: float,
) -> dict[tuple[str, str], float]:
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("router ranker returned invalid predictions")
    grouped: dict[str, list[int]] = {}
    for index, (block_id, _) in enumerate(keys):
        grouped.setdefault(block_id, []).append(index)
    risks = np.zeros_like(values)
    for indices in grouped.values():
        group = values[indices]
        span = float(np.max(group) - np.min(group))
        if span > 1e-12:
            risks[indices] = (np.max(group) - group) / span * max(float(scale), 1e-6)
    return {key: float(risks[index]) for index, key in enumerate(keys)}


def _blend_candidate_global_priors(
    risks: Mapping[tuple[str, str], float],
    priors: Mapping[str, float],
    *,
    weight: float,
    scale: float,
) -> dict[tuple[str, str], float]:
    """Blend block-level ranker risks with training-only candidate evidence."""

    if not priors or weight <= 0:
        return dict(risks)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("candidate global prior weight must lie in [0, 1]")
    finite_priors = {
        candidate_id: float(value)
        for candidate_id, value in priors.items()
        if np.isfinite(float(value))
    }
    if not finite_priors:
        return dict(risks)
    risk_scale = max(float(scale), 1e-6)
    low = min(finite_priors.values())
    high = max(finite_priors.values())
    span = high - low
    if span > 1e-12:
        normalized = {
            candidate_id: (value - low) / span * risk_scale
            for candidate_id, value in finite_priors.items()
        }
        neutral = 0.5 * risk_scale
    else:
        normalized = {candidate_id: 0.0 for candidate_id in finite_priors}
        neutral = 0.0
    return {
        key: float((1.0 - weight) * value + weight * normalized.get(key[1], neutral))
        for key, value in risks.items()
    }


def _normalize_block_evidence(
    values: Mapping[tuple[str, str], float],
    keys: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """Normalize lower-is-better evidence independently inside each block."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(key)
    normalized: dict[tuple[str, str], float] = {}
    for group_keys in grouped.values():
        finite = {
            key: float(values[key])
            for key in group_keys
            if key in values and np.isfinite(float(values[key]))
        }
        if not finite:
            normalized.update({key: 0.0 for key in group_keys})
            continue
        low = min(finite.values())
        high = max(finite.values())
        span = high - low
        if span <= 1e-12:
            normalized.update({key: 0.0 for key in group_keys})
            continue
        normalized.update(
            {key: ((finite[key] - low) / span if key in finite else 0.5) for key in group_keys}
        )
    return normalized


def _blend_routing_evidence(
    r0_risks: Mapping[tuple[str, str], float],
    r1_risks: Mapping[tuple[str, str], float],
    proxy_risks: Mapping[tuple[str, str], float],
    candidate_priors: Mapping[str, float],
    *,
    weights: Mapping[str, float],
    scale: float,
) -> dict[tuple[str, str], float]:
    """Combine held-out-family calibrated routing evidence."""

    names = ("r0", "r1", "proxy", "global_prior")
    resolved = {name: float(weights.get(name, 0.0)) for name in names}
    if any(not np.isfinite(value) or value < 0 for value in resolved.values()):
        raise ValueError("routing evidence weights must be finite and non-negative")
    if not np.isclose(sum(resolved.values()), 1.0, atol=1e-9):
        raise ValueError("routing evidence weights must sum to one")
    keys = tuple(r1_risks)
    if set(keys) != set(r0_risks) or set(keys) != set(proxy_risks):
        raise ValueError("routing evidence must cover identical block-candidate keys")
    prior_risks = {
        key: float(candidate_priors[key[1]]) for key in keys if key[1] in candidate_priors
    }
    components = {
        "r0": _normalize_block_evidence(r0_risks, keys),
        "r1": _normalize_block_evidence(r1_risks, keys),
        "proxy": _normalize_block_evidence(proxy_risks, keys),
        "global_prior": _normalize_block_evidence(prior_risks, keys),
    }
    risk_scale = max(float(scale), 1e-6)
    return {
        key: float(risk_scale * sum(resolved[name] * components[name][key] for name in names))
        for key in keys
    }


def _candidate_switch_penalties(
    graph: BlockGraph,
    candidates: Iterable[str],
    forecast_spec: ForecastSpec,
    *,
    weight: float,
) -> dict[tuple[str, str, str, str], float]:
    """Penalize unsupported method switches across forecast-relevant edges."""

    if not np.isfinite(weight) or weight < 0:
        raise ValueError("candidate switch penalty must be finite and non-negative")
    if weight == 0:
        return {}
    candidate_ids = tuple(dict.fromkeys(candidates))
    by_id = {block.block_id: block for block in graph.blocks}
    penalties: dict[tuple[str, str, str, str], float] = {}
    for edge in graph.edges:
        left = by_id[edge.left]
        right = by_id[edge.right]
        if not (
            _block_visible_to_forecaster(left, forecast_spec)
            and _block_visible_to_forecaster(right, forecast_spec)
        ):
            continue
        for left_candidate in candidate_ids:
            for right_candidate in candidate_ids:
                if left_candidate != right_candidate:
                    penalties[(edge.left, edge.right, left_candidate, right_candidate)] = float(
                        weight
                    )
    return penalties


def _proxy_outlier_candidates(
    scores: Mapping[str, float],
    *,
    multiplier: float = 5.0,
) -> tuple[frozenset[str], float | None]:
    """Reject only extreme pseudo-missing errors relative to the shortlist."""

    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("proxy outlier multiplier must be finite and at least one")
    finite = {
        candidate_id: float(value)
        for candidate_id, value in scores.items()
        if np.isfinite(float(value)) and float(value) >= 0
    }
    if len(finite) < 3:
        return frozenset(), None
    values = np.asarray(tuple(finite.values()), dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = 1.4826 * mad
    threshold = max(
        multiplier * max(median, 1e-8),
        median + multiplier * robust_scale,
    )
    rejected = frozenset(
        candidate_id for candidate_id, value in finite.items() if value > threshold
    )
    return rejected, float(threshold)


def _select_extrapolation_anchor(
    calibrations: Mapping[str, Any],
    forecast_spec: ForecastSpec,
    batch_metadata: Mapping[str, Any],
    proxy_scores: Mapping[str, float],
    available_candidates: Iterable[str],
) -> tuple[str | None, dict[str, Any]]:
    """Select a calibrated anchor only beyond the labelled origin range."""

    diagnostics: dict[str, Any] = {"active": False}
    model_calibrations = calibrations.get(forecast_spec.model_id, {})
    dataset_id = str(batch_metadata.get("dataset_id", ""))
    if not isinstance(model_calibrations, Mapping) or not dataset_id:
        diagnostics["reason"] = "calibration_unavailable"
        return None, diagnostics
    calibration = model_calibrations.get(dataset_id)
    calibration_scope = "dataset"
    if not isinstance(calibration, Mapping):
        calibration = model_calibrations.get("__all__")
        calibration_scope = "cross_dataset"
    if not isinstance(calibration, Mapping):
        diagnostics["reason"] = "calibration_unavailable"
        return None, diagnostics
    try:
        forecast_origin = int(batch_metadata["forecast_origin"])
        threshold = float(calibration["proxy_log_ratio_threshold"])
    except (KeyError, TypeError, ValueError):
        diagnostics["reason"] = "invalid_calibration"
        return None, diagnostics
    max_training_origin: int | None = None
    if calibration_scope == "dataset":
        try:
            max_training_origin = int(calibration["max_training_origin"])
        except (KeyError, TypeError, ValueError):
            diagnostics["reason"] = "invalid_calibration"
            return None, diagnostics
    candidates = calibration.get("candidates", ())
    if (
        not isinstance(candidates, (list, tuple))
        or len(candidates) != 2
        or len(set(candidates)) != 2
        or not all(isinstance(candidate_id, str) for candidate_id in candidates)
        or not np.isfinite(threshold)
    ):
        diagnostics["reason"] = "invalid_calibration"
        return None, diagnostics
    diagnostics.update(
        {
            "dataset_id": dataset_id,
            "scope": calibration_scope,
            "forecast_origin": forecast_origin,
            "max_training_origin": max_training_origin,
            "candidates": list(candidates),
            "proxy_log_ratio_threshold": threshold,
        }
    )
    if max_training_origin is not None and forecast_origin <= max_training_origin:
        diagnostics["reason"] = "within_training_origin_range"
        return None, diagnostics

    available = set(available_candidates)
    usable: list[str] = []
    for candidate_id in candidates:
        try:
            score = float(proxy_scores[candidate_id])
        except (KeyError, TypeError, ValueError):
            continue
        if candidate_id in available and np.isfinite(score) and score >= 0:
            usable.append(candidate_id)
    if not usable:
        diagnostics["reason"] = "anchor_candidates_unavailable"
        return None, diagnostics
    if len(usable) == 1:
        selected = usable[0]
        diagnostics.update(
            {
                "active": True,
                "reason": "single_available_anchor",
                "selected_candidate": selected,
            }
        )
        return selected, diagnostics

    first_candidate, second_candidate = candidates
    proxy_log_ratio = float(
        np.log1p(float(proxy_scores[first_candidate]))
        - np.log1p(float(proxy_scores[second_candidate]))
    )
    selected = first_candidate if proxy_log_ratio <= threshold else second_candidate
    diagnostics.update(
        {
            "active": True,
            "reason": "calibrated_proxy_threshold",
            "proxy_log_ratio": proxy_log_ratio,
            "selected_candidate": selected,
        }
    )
    return selected, diagnostics


def _pairwise_free_search(
    blocks: tuple[Any, ...],
    candidates: tuple[str, ...],
    unary: Mapping[tuple[str, str], float],
    costs: Mapping[str, float],
    budget: BudgetSpec,
    cost_weight: float,
    invalid: set[tuple[str, str]],
) -> RoutingResult:
    """Solve the pairwise-free objective by enumerating active candidates.

    Candidate shortlists contain at most a handful of methods, so enumerating
    their active subsets is cheap and makes the work linear in the number of
    blocks. This avoids the quadratic assignment copying and repeated edge
    scans of beam search when pairwise terms are deliberately disabled for a
    large block set.
    """

    if not candidates:
        raise RuntimeError("no candidate is available for pairwise-free routing")
    max_active = budget.max_active_candidates or len(candidates)
    max_active = min(max_active, len(candidates))
    best: (
        tuple[
            tuple[float, tuple[tuple[str, str], ...]],
            dict[str, str],
            float,
            float,
        ]
        | None
    ) = None
    for count in range(1, max_active + 1):
        for enabled in itertools.combinations(candidates, count):
            assignments: dict[str, str] = {}
            unary_energy = 0.0
            feasible = True
            for block in blocks:
                choices = [
                    (float(unary.get((block.block_id, candidate), float("inf"))), candidate)
                    for candidate in enabled
                    if (block.block_id, candidate) not in invalid
                ]
                choices = [choice for choice in choices if np.isfinite(choice[0])]
                if not choices:
                    feasible = False
                    break
                risk, candidate = min(choices)
                assignments[block.block_id] = candidate
                unary_energy += risk
            if not feasible:
                continue
            active = set(assignments.values())
            activated_cost = float(sum(float(costs.get(candidate, 1.0)) for candidate in active))
            total_energy = float(unary_energy + cost_weight * activated_cost)
            key = (total_energy, tuple(sorted(assignments.items())))
            entry = (key, assignments, float(unary_energy), activated_cost)
            if best is None or entry[0] < best[0]:
                best = entry
    if best is None:
        raise RuntimeError("no feasible pairwise-free routing assignment")
    key, assignments, unary_energy, activated_cost = best
    active_candidates = tuple(sorted(set(assignments.values())))
    cost_energy = float(cost_weight * activated_cost)
    return RoutingResult(
        assignments=assignments,
        shortlist=candidates,
        total_energy=float(key[0]),
        predicted_unary=dict(unary),
        predicted_pairwise={},
        activated_candidates=active_candidates,
        candidate_costs={candidate: float(costs.get(candidate, 1.0)) for candidate in candidates},
        activated_cost=activated_cost,
        risk_energy=unary_energy,
        cost_energy=cost_energy,
        metadata={"solver": "pairwise_free_subset"},
    )


def _native_block_is_valid(result: CandidateResult, block) -> bool:
    native = result.native_valid_mask[block.batch_index, block.start : block.end, block.channel]
    return result.status not in {CandidateStatus.FAILED, CandidateStatus.UNAVAILABLE} and bool(
        native.all()
    )


def _block_visible_to_forecaster(
    block: MissingBlock,
    forecast_spec: ForecastSpec,
) -> bool:
    return forecast_spec.mode == "joint_multivariate" or block.channel in set(
        forecast_spec.target_indices or ()
    )


def _median_fallback(
    batch: SeriesBatch,
    block,
    training_medians: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if training_medians is not None:
        medians = np.asarray(training_medians, dtype=float).reshape(-1)
        if block.channel < len(medians) and np.isfinite(medians[block.channel]):
            return np.full(block.length, medians[block.channel], dtype=float), "train_median"
    observed = batch.observed_mask[:, :, block.channel]
    channel_values = batch.values[:, :, block.channel][observed]
    if channel_values.size:
        value = float(np.median(channel_values))
    else:
        all_observed = batch.values[batch.observed_mask]
        value = float(np.median(all_observed)) if all_observed.size else 0.0
    return np.full(block.length, value, dtype=float), "context_median"


class BlockwiseFAIS:
    """Facade for heterogeneous block-level imputation.

    Candidate models always receive the same corrupted context. Block assembly
    happens only after every candidate has completed, avoiding order effects.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        router: RouterBundle | None = None,
        imputer_registry: ImputerRegistry | None = None,
        imputer_artifacts: Mapping[str, Any] | None = None,
        imputer_artifact_root: str | Path | None = None,
        artifact_load_failures: Mapping[str, str] | None = None,
        candidate_params: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        beam_width: int | None = None,
        beta: float | None = None,
        cost_weight: float | None = None,
        training_medians: np.ndarray | None = None,
        training_correlation: np.ndarray | None = None,
        fallback_internal: tuple[str, ...] | None = None,
        fallback_tail: tuple[str, ...] | None = None,
        forecast_predictor: Callable[[np.ndarray, ForecastSpec], Any] | None = None,
        max_pairwise_blocks: int = 512,
    ) -> None:
        self.config = config
        self.router = router
        self.imputer_registry = imputer_registry or DEFAULT_REGISTRY
        self.candidate_runner = CandidateRunner(self.imputer_registry)
        self._artifacts_supplied = imputer_artifacts is not None
        self.imputer_artifacts = dict(imputer_artifacts or {})
        self.imputer_artifact_root = (
            None if imputer_artifact_root is None else Path(imputer_artifact_root).resolve()
        )
        self._loaded_artifact_dataset: str | None = None
        self.artifact_load_failures = dict(artifact_load_failures or {})
        if candidate_params is None and config is not None:
            candidate_params = {
                spec.imputer_id: {
                    "num_samples": config.experiment.csdi_num_samples,
                }
                for spec in self.imputer_registry.specs()
                if str(spec.factory).startswith("tsfm_fais.imputers.pypots:")
            }
        self.candidate_params = {
            candidate_id: dict(params) for candidate_id, params in (candidate_params or {}).items()
        }
        self.training_medians = (
            None
            if training_medians is None
            else np.asarray(training_medians, dtype=float).reshape(-1)
        )
        self.training_correlation = (
            None if training_correlation is None else np.asarray(training_correlation, dtype=float)
        )
        configured_internal: tuple[str, ...] = (
            "linear_interp",
            "locf",
            "train_median",
        )
        configured_tail: tuple[str, ...] = ("locf", "train_median")
        configured_forced: tuple[str, ...] = ("locf", "linear_interp")
        configured_shortlist_size = 6
        configured_pseudo_blocks = 8
        configured_beam_width = 32
        configured_beta = 1.0
        configured_cost_weight = 0.0
        configured_forecast_consensus: Mapping[str, Any] = {
            "mode": "disabled",
            "candidates": (),
            "ensemble_top_k": 2,
            "ensemble_third_relative_gap": None,
            "ensemble_proxy_weight_power": 0.0,
            "pseudo_weight_calibration": "disabled",
            "pseudo_weight_prior_strength": 8.0,
            "pseudo_weight_min_points": 4,
            "context_mode": "native",
            "max_context_variates": 8,
            "prior_weight": 0.0,
            "prior_weight_by_model": {},
            "prior_override_max_medoid_penalty": None,
            "prior_override_max_medoid_penalty_by_model": {},
            "prior_override_min_margin": 0.0,
            "prior_override_min_margin_by_model": {},
            "anchor_weight": 1.0,
            "anchor_weight_by_model": {},
            "proxy_blend_weight": 0.0,
            "proxy_blend_weight_by_model": {},
            "proxy_blend_min_relative_margin": 0.0,
            "proxy_blend_min_relative_margin_by_model": {},
            "candidate_shrinkage_id": None,
            "candidate_shrinkage_weight": 0.0,
            "candidate_shrinkage_fallback_id": None,
            "candidate_shrinkage_fallback_weight": 0.0,
            "anchor_max_period_ratio": None,
            "anchor_max_period_ratio_by_model": {},
            "anchor_period_exceeded_mode": "disabled",
            "selection_granularity": "episode",
            "selection_granularity_by_model": {},
            "validation_length": 24,
            "min_observed_per_target": 4,
        }
        runtime_forecast_consensus: Mapping[str, Any] | None = None
        if config is not None:
            from tsfm_fais.config import load_yaml
            from tsfm_fais.registry_configs import RouterConfig

            router_config = RouterConfig.model_validate(load_yaml(config.registries.router_config))
            configured_internal = router_config.fallback.internal
            configured_tail = router_config.fallback.tail
            configured_forced = router_config.forced_candidates
            configured_shortlist_size = router_config.shortlist_size
            configured_pseudo_blocks = router_config.pseudo_blocks
            configured_beam_width = router_config.beam_width
            configured_beta = router_config.beta
            configured_cost_weight = router_config.cost_weight
            runtime_forecast_consensus = router_config.forecast_consensus.model_dump()
            configured_forecast_consensus = runtime_forecast_consensus
        if router is not None:
            configured_beta = float(router.metadata.get("beta", configured_beta))
            configured_cost_weight = float(
                router.metadata.get("cost_weight", configured_cost_weight)
            )
            artifact_forecast_consensus = router.metadata.get(
                "forecast_consensus", configured_forecast_consensus
            )
            if runtime_forecast_consensus is None:
                configured_forecast_consensus = artifact_forecast_consensus
            elif isinstance(artifact_forecast_consensus, Mapping):
                configured_forecast_consensus = _merge_forecast_consensus_config(
                    artifact_forecast_consensus,
                    runtime_forecast_consensus,
                )
            else:
                configured_forecast_consensus = artifact_forecast_consensus
        configured_prior_weight = (
            float(router.metadata.get("candidate_global_prior_weight", 0.0))
            if router is not None
            else 0.0
        )
        configured_prior_min_support = (
            int(router.metadata.get("candidate_global_prior_min_support", 2))
            if router is not None
            else 2
        )
        configured_prior_forced_count = (
            int(router.metadata.get("candidate_global_prior_forced_count", 0))
            if router is not None
            else 0
        )
        configured_shortlist_anchors = (
            tuple(router.metadata.get("shortlist_anchor_candidates", ()))
            if router is not None
            else ()
        )
        configured_switch_penalty = (
            float(router.metadata.get("candidate_switch_penalty", 0.0))
            if router is not None
            else 0.0
        )
        configured_proxy_outlier_multiplier = (
            float(router.metadata.get("proxy_outlier_multiplier", 5.0))
            if router is not None
            else 5.0
        )
        configured_anchor_calibrations = (
            router.metadata.get("candidate_anchor_calibrations", {}) if router is not None else {}
        )
        configured_evidence_blend = (
            router.metadata.get("evidence_blend", {}) if router is not None else {}
        )
        self.beta = float(configured_beta if beta is None else beta)
        self.cost_weight = float(configured_cost_weight if cost_weight is None else cost_weight)
        if (
            not np.isfinite(self.beta)
            or not np.isfinite(self.cost_weight)
            or self.beta < 0
            or self.cost_weight < 0
        ):
            raise ValueError("beta and cost_weight must be finite and non-negative")
        self.fallback_internal = tuple(fallback_internal or configured_internal)
        self.fallback_tail = tuple(fallback_tail or configured_tail)
        self.forced_candidates = tuple(configured_forced)
        if not 0 <= configured_prior_weight <= 1:
            raise ValueError("candidate global prior weight must lie in [0, 1]")
        if configured_prior_min_support < 1:
            raise ValueError("candidate global prior minimum support must be positive")
        if configured_prior_forced_count < 0:
            raise ValueError("candidate global prior forced count cannot be negative")
        if len(set(configured_shortlist_anchors)) != len(configured_shortlist_anchors) or any(
            not isinstance(candidate_id, str) or not candidate_id
            for candidate_id in configured_shortlist_anchors
        ):
            raise ValueError("shortlist anchor candidates must be unique non-empty IDs")
        if not np.isfinite(configured_switch_penalty) or configured_switch_penalty < 0:
            raise ValueError("candidate switch penalty must be finite and non-negative")
        if (
            not np.isfinite(configured_proxy_outlier_multiplier)
            or configured_proxy_outlier_multiplier < 1
        ):
            raise ValueError("proxy outlier multiplier must be finite and at least one")
        if not isinstance(configured_anchor_calibrations, Mapping):
            raise ValueError("candidate anchor calibrations must be a mapping")
        if not isinstance(configured_evidence_blend, Mapping):
            raise ValueError("routing evidence blend must be a mapping")
        if not isinstance(configured_forecast_consensus, Mapping):
            raise ValueError("forecast consensus configuration must be a mapping")
        forecast_consensus_mode = str(configured_forecast_consensus.get("mode", "disabled"))
        forecast_consensus_candidates = tuple(configured_forecast_consensus.get("candidates", ()))
        forecast_consensus_ensemble_top_k = int(
            configured_forecast_consensus.get("ensemble_top_k", 2)
        )
        raw_ensemble_third_relative_gap = configured_forecast_consensus.get(
            "ensemble_third_relative_gap"
        )
        forecast_consensus_ensemble_third_relative_gap = (
            None
            if raw_ensemble_third_relative_gap is None
            else float(raw_ensemble_third_relative_gap)
        )
        forecast_consensus_ensemble_proxy_weight_power = float(
            configured_forecast_consensus.get("ensemble_proxy_weight_power", 0.0)
        )
        forecast_consensus_pseudo_weight_calibration = str(
            configured_forecast_consensus.get("pseudo_weight_calibration", "disabled")
        )
        forecast_consensus_pseudo_weight_prior_strength = float(
            configured_forecast_consensus.get("pseudo_weight_prior_strength", 8.0)
        )
        forecast_consensus_pseudo_weight_min_points = int(
            configured_forecast_consensus.get("pseudo_weight_min_points", 4)
        )
        forecast_consensus_dataset_prior_candidates = int(
            configured_forecast_consensus.get("dataset_prior_candidates", 0)
        )
        raw_model_prior_counts = configured_forecast_consensus.get(
            "dataset_prior_candidates_by_model", {}
        )
        if not isinstance(raw_model_prior_counts, Mapping):
            raise ValueError("forecast consensus model prior counts must be a mapping")
        forecast_consensus_model_prior_candidates = {
            str(model_id): int(count) for model_id, count in raw_model_prior_counts.items()
        }
        forecast_consensus_prior_weight = float(
            configured_forecast_consensus.get("prior_weight", 0.0)
        )
        raw_model_prior_weights = configured_forecast_consensus.get("prior_weight_by_model", {})
        if not isinstance(raw_model_prior_weights, Mapping):
            raise ValueError("forecast consensus model prior weights must be a mapping")
        forecast_consensus_model_prior_weights = {
            str(model_id): float(weight) for model_id, weight in raw_model_prior_weights.items()
        }
        raw_prior_override_penalty = configured_forecast_consensus.get(
            "prior_override_max_medoid_penalty"
        )
        forecast_consensus_prior_override_penalty = (
            None if raw_prior_override_penalty is None else float(raw_prior_override_penalty)
        )
        raw_model_prior_override_penalties = configured_forecast_consensus.get(
            "prior_override_max_medoid_penalty_by_model", {}
        )
        if not isinstance(raw_model_prior_override_penalties, Mapping):
            raise ValueError("forecast consensus model override penalties must be a mapping")
        forecast_consensus_model_prior_override_penalties = {
            str(model_id): float(penalty)
            for model_id, penalty in raw_model_prior_override_penalties.items()
        }
        forecast_consensus_prior_override_margin = float(
            configured_forecast_consensus.get("prior_override_min_margin", 0.0)
        )
        raw_model_prior_override_margins = configured_forecast_consensus.get(
            "prior_override_min_margin_by_model", {}
        )
        if not isinstance(raw_model_prior_override_margins, Mapping):
            raise ValueError("forecast consensus model override margins must be a mapping")
        forecast_consensus_model_prior_override_margins = {
            str(model_id): float(margin)
            for model_id, margin in raw_model_prior_override_margins.items()
        }
        forecast_consensus_anchor_weight = float(
            configured_forecast_consensus.get("anchor_weight", 1.0)
        )
        raw_model_anchor_weights = configured_forecast_consensus.get("anchor_weight_by_model", {})
        if not isinstance(raw_model_anchor_weights, Mapping):
            raise ValueError("forecast consensus model anchor weights must be a mapping")
        forecast_consensus_model_anchor_weights = {
            str(model_id): float(weight) for model_id, weight in raw_model_anchor_weights.items()
        }
        forecast_consensus_proxy_blend_weight = float(
            configured_forecast_consensus.get("proxy_blend_weight", 0.0)
        )
        raw_model_proxy_blend_weights = configured_forecast_consensus.get(
            "proxy_blend_weight_by_model", {}
        )
        if not isinstance(raw_model_proxy_blend_weights, Mapping):
            raise ValueError("forecast consensus model proxy blend weights must be a mapping")
        forecast_consensus_model_proxy_blend_weights = {
            str(model_id): float(weight)
            for model_id, weight in raw_model_proxy_blend_weights.items()
        }
        forecast_consensus_proxy_blend_min_relative_margin = float(
            configured_forecast_consensus.get(
                "proxy_blend_min_relative_margin",
                0.0,
            )
        )
        raw_model_proxy_blend_min_relative_margins = configured_forecast_consensus.get(
            "proxy_blend_min_relative_margin_by_model",
            {},
        )
        if not isinstance(raw_model_proxy_blend_min_relative_margins, Mapping):
            raise ValueError("forecast consensus model proxy margins must be a mapping")
        forecast_consensus_model_proxy_blend_min_relative_margins = {
            str(model_id): float(margin)
            for model_id, margin in raw_model_proxy_blend_min_relative_margins.items()
        }
        raw_candidate_shrinkage_id = configured_forecast_consensus.get("candidate_shrinkage_id")
        forecast_consensus_candidate_shrinkage_id = (
            None if raw_candidate_shrinkage_id is None else str(raw_candidate_shrinkage_id)
        )
        forecast_consensus_candidate_shrinkage_weight = float(
            configured_forecast_consensus.get("candidate_shrinkage_weight", 0.0)
        )
        raw_candidate_shrinkage_fallback_id = configured_forecast_consensus.get(
            "candidate_shrinkage_fallback_id"
        )
        forecast_consensus_candidate_shrinkage_fallback_id = (
            None
            if raw_candidate_shrinkage_fallback_id is None
            else str(raw_candidate_shrinkage_fallback_id)
        )
        forecast_consensus_candidate_shrinkage_fallback_weight = float(
            configured_forecast_consensus.get(
                "candidate_shrinkage_fallback_weight",
                0.0,
            )
        )
        raw_anchor_period_ratio = configured_forecast_consensus.get("anchor_max_period_ratio")
        forecast_consensus_anchor_period_ratio = (
            None if raw_anchor_period_ratio is None else float(raw_anchor_period_ratio)
        )
        raw_model_anchor_period_ratios = configured_forecast_consensus.get(
            "anchor_max_period_ratio_by_model", {}
        )
        if not isinstance(raw_model_anchor_period_ratios, Mapping):
            raise ValueError("forecast consensus model period ratios must be a mapping")
        forecast_consensus_model_anchor_period_ratios = {
            str(model_id): float(ratio)
            for model_id, ratio in raw_model_anchor_period_ratios.items()
        }
        forecast_consensus_anchor_period_exceeded_mode = str(
            configured_forecast_consensus.get("anchor_period_exceeded_mode", "disabled")
        )
        forecast_consensus_selection_granularity = str(
            configured_forecast_consensus.get("selection_granularity", "episode")
        )
        raw_model_granularities = configured_forecast_consensus.get(
            "selection_granularity_by_model", {}
        )
        if not isinstance(raw_model_granularities, Mapping):
            raise ValueError("forecast consensus model granularities must be a mapping")
        forecast_consensus_model_granularities = {
            str(model_id): str(granularity)
            for model_id, granularity in raw_model_granularities.items()
        }
        forecast_consensus_context_mode = str(
            configured_forecast_consensus.get("context_mode", "native")
        )
        forecast_consensus_max_context_variates = int(
            configured_forecast_consensus.get("max_context_variates", 8)
        )
        forecast_consensus_validation_length = int(
            configured_forecast_consensus.get("validation_length", 24)
        )
        forecast_consensus_min_observed = int(
            configured_forecast_consensus.get("min_observed_per_target", 4)
        )
        if forecast_consensus_mode not in {
            "disabled",
            "medoid",
            "historical_backtest",
            "value_median",
            "value_topk_mean",
            "router_risk",
            "proxy_min",
        }:
            raise ValueError("unknown forecast consensus mode")
        if len(set(forecast_consensus_candidates)) != len(forecast_consensus_candidates) or any(
            not isinstance(candidate_id, str) or candidate_id not in self.imputer_registry
            for candidate_id in forecast_consensus_candidates
        ):
            raise ValueError("forecast consensus candidates are invalid")
        if (
            not np.isfinite(forecast_consensus_candidate_shrinkage_weight)
            or not 0.0 <= forecast_consensus_candidate_shrinkage_weight <= 1.0
            or (forecast_consensus_candidate_shrinkage_id is None)
            != (forecast_consensus_candidate_shrinkage_weight == 0.0)
            or (
                forecast_consensus_candidate_shrinkage_id is not None
                and forecast_consensus_candidate_shrinkage_id not in forecast_consensus_candidates
            )
        ):
            raise ValueError(
                "candidate shrinkage requires a configured consensus candidate and weight in (0, 1]"
            )
        if (
            not np.isfinite(forecast_consensus_candidate_shrinkage_fallback_weight)
            or not 0.0 <= forecast_consensus_candidate_shrinkage_fallback_weight <= 1.0
            or (forecast_consensus_candidate_shrinkage_fallback_id is None)
            != (forecast_consensus_candidate_shrinkage_fallback_weight == 0.0)
            or (
                forecast_consensus_candidate_shrinkage_fallback_id is not None
                and (
                    forecast_consensus_candidate_shrinkage_id is None
                    or forecast_consensus_candidate_shrinkage_fallback_id
                    == forecast_consensus_candidate_shrinkage_id
                    or forecast_consensus_candidate_shrinkage_fallback_id
                    not in forecast_consensus_candidates
                )
            )
        ):
            raise ValueError(
                "candidate shrinkage fallback requires a distinct configured consensus "
                "candidate and weight in (0, 1]"
            )
        if forecast_consensus_ensemble_top_k < 2:
            raise ValueError("forecast consensus top-k must be at least two")
        if forecast_consensus_ensemble_third_relative_gap is not None and (
            not np.isfinite(forecast_consensus_ensemble_third_relative_gap)
            or forecast_consensus_ensemble_third_relative_gap < 0.0
            or forecast_consensus_mode != "value_topk_mean"
            or forecast_consensus_ensemble_top_k != 2
        ):
            raise ValueError(
                "third-candidate gap gating requires a finite non-negative threshold "
                "with value_topk_mean and top-k two"
            )
        if (
            not np.isfinite(forecast_consensus_ensemble_proxy_weight_power)
            or forecast_consensus_ensemble_proxy_weight_power < 0.0
            or (
                forecast_consensus_ensemble_proxy_weight_power > 0.0
                and forecast_consensus_mode != "value_topk_mean"
            )
        ):
            raise ValueError(
                "proxy-weighted candidate aggregation requires a finite non-negative "
                "power with value_topk_mean"
            )
        if forecast_consensus_pseudo_weight_calibration not in {
            "disabled",
            "convex_l2",
        }:
            raise ValueError("unknown pseudo-weight calibration mode")
        if (
            not np.isfinite(forecast_consensus_pseudo_weight_prior_strength)
            or forecast_consensus_pseudo_weight_prior_strength < 0.0
            or forecast_consensus_pseudo_weight_min_points < 1
        ):
            raise ValueError("pseudo-weight calibration parameters are invalid")
        if forecast_consensus_dataset_prior_candidates < 0:
            raise ValueError("forecast consensus dataset prior count cannot be negative")
        if any(
            not model_id or count < 0
            for model_id, count in forecast_consensus_model_prior_candidates.items()
        ):
            raise ValueError("forecast consensus model prior counts are invalid")
        if not 0.0 <= forecast_consensus_prior_weight <= 1.0 or any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in forecast_consensus_model_prior_weights.items()
        ):
            raise ValueError("forecast consensus prior weights are invalid")
        if (
            forecast_consensus_prior_override_penalty is not None
            and (
                not np.isfinite(forecast_consensus_prior_override_penalty)
                or not 0.0 <= forecast_consensus_prior_override_penalty <= 1.0
            )
        ) or any(
            not model_id or not np.isfinite(penalty) or not 0.0 <= penalty <= 1.0
            for model_id, penalty in (forecast_consensus_model_prior_override_penalties.items())
        ):
            raise ValueError("forecast consensus prior override penalties are invalid")
        if (
            not np.isfinite(forecast_consensus_prior_override_margin)
            or forecast_consensus_prior_override_margin < 0.0
            or any(
                not model_id or not np.isfinite(margin) or margin < 0.0
                for model_id, margin in (forecast_consensus_model_prior_override_margins.items())
            )
        ):
            raise ValueError("forecast consensus prior override margins are invalid")
        if not 0.0 <= forecast_consensus_anchor_weight <= 1.0 or any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in forecast_consensus_model_anchor_weights.items()
        ):
            raise ValueError("forecast consensus anchor weights are invalid")
        if not 0.0 <= forecast_consensus_proxy_blend_weight <= 1.0 or any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in forecast_consensus_model_proxy_blend_weights.items()
        ):
            raise ValueError("forecast consensus proxy blend weights are invalid")
        proxy_blend_configured = bool(
            forecast_consensus_proxy_blend_weight
            or any(forecast_consensus_model_proxy_blend_weights.values())
        )
        calibrated_top_two = bool(
            forecast_consensus_mode == "value_topk_mean"
            and forecast_consensus_ensemble_top_k == 2
            and forecast_consensus_ensemble_third_relative_gap is None
        )
        if forecast_consensus_pseudo_weight_calibration != "disabled" and not (
            proxy_blend_configured or calibrated_top_two
        ):
            raise ValueError(
                "pseudo-weight calibration requires a proxy blend or fixed top-two mean"
            )
        if (
            forecast_consensus_pseudo_weight_calibration != "disabled"
            and forecast_consensus_ensemble_proxy_weight_power > 0.0
        ):
            raise ValueError(
                "pseudo-weight calibration and inverse-error weighting are mutually exclusive"
            )
        if (
            not np.isfinite(forecast_consensus_proxy_blend_min_relative_margin)
            or forecast_consensus_proxy_blend_min_relative_margin < 0.0
            or any(
                not model_id or not np.isfinite(margin) or margin < 0.0
                for model_id, margin in (
                    forecast_consensus_model_proxy_blend_min_relative_margins.items()
                )
            )
        ):
            raise ValueError("forecast consensus proxy blend margins are invalid")
        if (
            forecast_consensus_anchor_period_ratio is not None
            and (
                not np.isfinite(forecast_consensus_anchor_period_ratio)
                or forecast_consensus_anchor_period_ratio <= 0.0
            )
        ) or any(
            not model_id or not np.isfinite(ratio) or ratio <= 0.0
            for model_id, ratio in forecast_consensus_model_anchor_period_ratios.items()
        ):
            raise ValueError("forecast consensus anchor period ratios are invalid")
        if forecast_consensus_anchor_period_exceeded_mode not in {"disabled", "medoid"}:
            raise ValueError("unknown forecast consensus period-exceeded mode")
        if forecast_consensus_selection_granularity not in {"episode", "target"} or any(
            not model_id or granularity not in {"episode", "target"}
            for model_id, granularity in forecast_consensus_model_granularities.items()
        ):
            raise ValueError("forecast consensus granularities are invalid")
        configured_granularities = {
            forecast_consensus_selection_granularity,
            *forecast_consensus_model_granularities.values(),
        }
        if "target" in configured_granularities and forecast_consensus_mode not in {
            "medoid",
            "value_topk_mean",
        }:
            raise ValueError(
                "target-level forecast consensus requires medoid or value_topk_mean mode"
            )
        if forecast_consensus_context_mode not in {
            "native",
            "targets_only",
            "targets_with_correlates",
        }:
            raise ValueError("unknown forecast consensus context mode")
        if (
            forecast_consensus_mode == "historical_backtest"
            and forecast_consensus_context_mode != "native"
        ):
            raise ValueError("historical forecast consensus requires native context")
        if forecast_consensus_max_context_variates < 2:
            raise ValueError("forecast consensus variate cap must be at least two")
        if forecast_consensus_mode != "disabled" and (
            len(forecast_consensus_candidates) + forecast_consensus_dataset_prior_candidates < 2
        ):
            raise ValueError("forecast consensus selection requires two candidates")
        if forecast_consensus_validation_length < 1 or forecast_consensus_min_observed < 1:
            raise ValueError("forecast consensus validation limits must be positive")
        evidence_blend: dict[str, dict[str, float]] = {}
        for model_id, raw_weights in configured_evidence_blend.items():
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("routing evidence blend model IDs must be non-empty")
            if not isinstance(raw_weights, Mapping):
                raise ValueError("routing evidence weights must be mappings")
            weights = {
                name: float(raw_weights.get(name, 0.0))
                for name in ("r0", "r1", "proxy", "global_prior")
            }
            if any(
                not np.isfinite(value) or value < 0 for value in weights.values()
            ) or not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
                raise ValueError("routing evidence weights must be non-negative and sum to one")
            evidence_blend[model_id] = weights
        self.candidate_global_prior_weight = configured_prior_weight
        self.candidate_global_prior_min_support = configured_prior_min_support
        self.candidate_global_prior_forced_count = configured_prior_forced_count
        self.shortlist_anchor_candidates = configured_shortlist_anchors
        self.candidate_switch_penalty = configured_switch_penalty
        self.proxy_outlier_multiplier = configured_proxy_outlier_multiplier
        self.candidate_anchor_calibrations = configured_anchor_calibrations
        self.evidence_blend = evidence_blend
        self.forecast_consensus_mode = forecast_consensus_mode
        self.forecast_consensus_candidates = forecast_consensus_candidates
        self.forecast_consensus_ensemble_top_k = forecast_consensus_ensemble_top_k
        self.forecast_consensus_ensemble_third_relative_gap = (
            forecast_consensus_ensemble_third_relative_gap
        )
        self.forecast_consensus_ensemble_proxy_weight_power = (
            forecast_consensus_ensemble_proxy_weight_power
        )
        self.forecast_consensus_pseudo_weight_calibration = (
            forecast_consensus_pseudo_weight_calibration
        )
        self.forecast_consensus_pseudo_weight_prior_strength = (
            forecast_consensus_pseudo_weight_prior_strength
        )
        self.forecast_consensus_pseudo_weight_min_points = (
            forecast_consensus_pseudo_weight_min_points
        )
        self.forecast_consensus_dataset_prior_candidates = (
            forecast_consensus_dataset_prior_candidates
        )
        self.forecast_consensus_model_prior_candidates = forecast_consensus_model_prior_candidates
        self.forecast_consensus_prior_weight = forecast_consensus_prior_weight
        self.forecast_consensus_model_prior_weights = forecast_consensus_model_prior_weights
        self.forecast_consensus_prior_override_penalty = forecast_consensus_prior_override_penalty
        self.forecast_consensus_model_prior_override_penalties = (
            forecast_consensus_model_prior_override_penalties
        )
        self.forecast_consensus_prior_override_margin = forecast_consensus_prior_override_margin
        self.forecast_consensus_model_prior_override_margins = (
            forecast_consensus_model_prior_override_margins
        )
        self.forecast_consensus_anchor_weight = forecast_consensus_anchor_weight
        self.forecast_consensus_model_anchor_weights = forecast_consensus_model_anchor_weights
        self.forecast_consensus_proxy_blend_weight = forecast_consensus_proxy_blend_weight
        self.forecast_consensus_model_proxy_blend_weights = (
            forecast_consensus_model_proxy_blend_weights
        )
        self.forecast_consensus_proxy_blend_min_relative_margin = (
            forecast_consensus_proxy_blend_min_relative_margin
        )
        self.forecast_consensus_model_proxy_blend_min_relative_margins = (
            forecast_consensus_model_proxy_blend_min_relative_margins
        )
        self.forecast_consensus_candidate_shrinkage_id = forecast_consensus_candidate_shrinkage_id
        self.forecast_consensus_candidate_shrinkage_weight = (
            forecast_consensus_candidate_shrinkage_weight
        )
        self.forecast_consensus_candidate_shrinkage_fallback_id = (
            forecast_consensus_candidate_shrinkage_fallback_id
        )
        self.forecast_consensus_candidate_shrinkage_fallback_weight = (
            forecast_consensus_candidate_shrinkage_fallback_weight
        )
        self.forecast_consensus_anchor_period_ratio = forecast_consensus_anchor_period_ratio
        self.forecast_consensus_model_anchor_period_ratios = (
            forecast_consensus_model_anchor_period_ratios
        )
        self.forecast_consensus_anchor_period_exceeded_mode = (
            forecast_consensus_anchor_period_exceeded_mode
        )
        self.forecast_consensus_selection_granularity = forecast_consensus_selection_granularity
        self.forecast_consensus_model_granularities = forecast_consensus_model_granularities
        self.forecast_consensus_context_mode = forecast_consensus_context_mode
        self.forecast_consensus_max_context_variates = forecast_consensus_max_context_variates
        self.forecast_consensus_validation_length = forecast_consensus_validation_length
        self.forecast_consensus_min_observed = forecast_consensus_min_observed
        self.forecast_predictor = forecast_predictor
        self.shortlist_size = int(configured_shortlist_size)
        self.pseudo_blocks = int(configured_pseudo_blocks)
        self.beam_width = int(configured_beam_width if beam_width is None else beam_width)
        if max_pairwise_blocks < 1:
            raise ValueError("max_pairwise_blocks must be positive")
        self.max_pairwise_blocks = int(max_pairwise_blocks)

    def _evidence_weights(self, forecast_spec: ForecastSpec) -> dict[str, float]:
        return dict(
            self.evidence_blend.get(
                forecast_spec.model_id,
                {"r0": 0.0, "r1": 1.0, "proxy": 0.0, "global_prior": 0.0},
            )
        )

    def _forecast_consensus_prior_count(self, forecast_spec: ForecastSpec) -> int:
        return int(
            self.forecast_consensus_model_prior_candidates.get(
                forecast_spec.model_id,
                self.forecast_consensus_dataset_prior_candidates,
            )
        )

    def _forecast_consensus_prior_weight(self, forecast_spec: ForecastSpec) -> float:
        return float(
            self.forecast_consensus_model_prior_weights.get(
                forecast_spec.model_id,
                self.forecast_consensus_prior_weight,
            )
        )

    def _forecast_consensus_anchor_weight(self, forecast_spec: ForecastSpec) -> float:
        return float(
            self.forecast_consensus_model_anchor_weights.get(
                forecast_spec.model_id,
                self.forecast_consensus_anchor_weight,
            )
        )

    def _forecast_consensus_proxy_blend_weight(
        self,
        forecast_spec: ForecastSpec,
    ) -> float:
        return float(
            self.forecast_consensus_model_proxy_blend_weights.get(
                forecast_spec.model_id,
                self.forecast_consensus_proxy_blend_weight,
            )
        )

    def _forecast_consensus_proxy_blend_margin(
        self,
        forecast_spec: ForecastSpec,
    ) -> float:
        return float(
            self.forecast_consensus_model_proxy_blend_min_relative_margins.get(
                forecast_spec.model_id,
                self.forecast_consensus_proxy_blend_min_relative_margin,
            )
        )

    def _forecast_consensus_anchor_period_ratio(
        self,
        forecast_spec: ForecastSpec,
    ) -> float | None:
        return self.forecast_consensus_model_anchor_period_ratios.get(
            forecast_spec.model_id,
            self.forecast_consensus_anchor_period_ratio,
        )

    def _forecast_consensus_prior_override(
        self,
        forecast_spec: ForecastSpec,
    ) -> tuple[float | None, float]:
        return (
            self.forecast_consensus_model_prior_override_penalties.get(
                forecast_spec.model_id,
                self.forecast_consensus_prior_override_penalty,
            ),
            float(
                self.forecast_consensus_model_prior_override_margins.get(
                    forecast_spec.model_id,
                    self.forecast_consensus_prior_override_margin,
                )
            ),
        )

    def _forecast_consensus_granularity(self, forecast_spec: ForecastSpec) -> str:
        return str(
            self.forecast_consensus_model_granularities.get(
                forecast_spec.model_id,
                self.forecast_consensus_selection_granularity,
            )
        )

    def _reliable_candidate_global_priors(
        self,
        forecast_spec: ForecastSpec,
        candidates: Iterable[str],
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        if self.router is None:
            return {}
        all_priors = self.router.metadata.get("candidate_global_priors", {})
        all_support = self.router.metadata.get("candidate_global_support", {})
        if not isinstance(all_priors, Mapping) or not isinstance(all_support, Mapping):
            return {}
        model_priors = all_priors.get(forecast_spec.model_id, {})
        model_support = all_support.get(forecast_spec.model_id, {})
        if not isinstance(model_priors, Mapping) or not isinstance(model_support, Mapping):
            return {}
        dataset_id = str((batch_metadata or {}).get("dataset_id", ""))
        all_dataset_priors = self.router.metadata.get("candidate_dataset_priors", {})
        all_dataset_support = self.router.metadata.get("candidate_dataset_support", {})
        model_dataset_priors = (
            all_dataset_priors.get(forecast_spec.model_id, {})
            if isinstance(all_dataset_priors, Mapping)
            else {}
        )
        model_dataset_support = (
            all_dataset_support.get(forecast_spec.model_id, {})
            if isinstance(all_dataset_support, Mapping)
            else {}
        )
        dataset_priors = (
            model_dataset_priors.get(dataset_id, {})
            if isinstance(model_dataset_priors, Mapping)
            else {}
        )
        dataset_support = (
            model_dataset_support.get(dataset_id, {})
            if isinstance(model_dataset_support, Mapping)
            else {}
        )
        if not isinstance(dataset_priors, Mapping):
            dataset_priors = {}
        if not isinstance(dataset_support, Mapping):
            dataset_support = {}
        available = set(candidates)
        reliable: dict[str, float] = {}
        for candidate_id in set(model_priors) | set(dataset_priors):
            if candidate_id not in available:
                continue
            try:
                local_support = int(dataset_support.get(candidate_id, 0))
                if local_support >= self.candidate_global_prior_min_support:
                    prior = float(dataset_priors[candidate_id])
                    support = local_support
                else:
                    prior = float(model_priors[candidate_id])
                    support = int(model_support.get(candidate_id, 0))
            except (KeyError, TypeError, ValueError):
                continue
            if support >= self.candidate_global_prior_min_support and np.isfinite(prior):
                reliable[str(candidate_id)] = prior
        return reliable

    def _blend_with_candidate_global_priors(
        self,
        risks: Mapping[tuple[str, str], float],
        forecast_spec: ForecastSpec,
        candidates: Iterable[str],
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> dict[tuple[str, str], float]:
        scale = (
            float(self.router.metadata.get("unary_risk_scale", 1.0))
            if self.router is not None
            else 1.0
        )
        return _blend_candidate_global_priors(
            risks,
            self._reliable_candidate_global_priors(
                forecast_spec,
                candidates,
                batch_metadata,
            ),
            weight=self.candidate_global_prior_weight,
            scale=scale,
        )

    @classmethod
    def load(
        cls,
        config: str | Path | AppConfig,
        router_artifact: str | Path | None = None,
        **kwargs: Any,
    ) -> BlockwiseFAIS:
        resolved = load_config(config) if isinstance(config, (str, Path)) else config
        from tsfm_fais.registry_configs import validate_project_configuration

        validate_project_configuration(resolved)
        router = RouterBundle.load(router_artifact) if router_artifact is not None else None
        artifact_root = kwargs.pop("imputer_artifact_root", None)
        if artifact_root is None and router is not None:
            artifact_root = router.metadata.get("imputer_artifacts")
        return cls(
            config=resolved,
            router=router,
            imputer_artifact_root=artifact_root,
            **kwargs,
        )

    def _ensure_item_artifacts(self, item: TimeSeriesItem) -> None:
        if self.imputer_artifact_root is not None:
            dataset_id = item.metadata.get("dataset_id")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError("item.metadata['dataset_id'] is required to load fitted imputers")
            if self._loaded_artifact_dataset != dataset_id:
                artifacts, medians, correlation, failures = load_dataset_imputer_artifacts(
                    self.imputer_artifact_root,
                    dataset_id,
                    self.imputer_registry,
                )
                self.imputer_artifacts = artifacts
                self.training_medians = medians
                self.training_correlation = correlation
                self.artifact_load_failures = failures
                self._loaded_artifact_dataset = dataset_id

        if self.router is None:
            return
        missing = [
            candidate_id
            for candidate_id in self.router.candidate_ids
            if (
                candidate_id in self.imputer_registry
                and self.imputer_registry.get_spec(candidate_id).fit_scope != "none"
                and candidate_id not in self.imputer_artifacts
            )
        ]
        if missing and self.imputer_artifact_root is None and not self._artifacts_supplied:
            raise RuntimeError(
                "router requires fitted imputer artifacts that were not loaded: "
                + ", ".join(sorted(missing))
            )

    def _heuristic_unary(
        self,
        batch: SeriesBatch,
        blocks,
        candidates: tuple[str, ...],
        forecast_spec: ForecastSpec,
        period: int | None,
    ) -> dict[tuple[str, str], float]:
        unary: dict[tuple[str, str], float] = {}
        for block in blocks:
            for candidate_id in candidates:
                spec = self.imputer_registry.get_spec(candidate_id)
                score = 0.05 * spec.cost_tier
                if block.end == batch.shape[1] and not spec.supports_tail:
                    score += 1000.0
                if spec.requires_period and not period:
                    score += 1000.0
                concurrent = np.mean(
                    ~batch.observed_mask[block.batch_index, block.start : block.end, :]
                )
                if spec.mode == "joint_multivariate":
                    score -= 0.1 * float(concurrent)
                if candidate_id == "linear_interp":
                    score += 0.2 * block.length / batch.shape[1]
                if candidate_id == "seasonal_lag" and period:
                    score += abs(block.length - period) / max(period, 1) * 0.05
                unary[(block.block_id, candidate_id)] = score
        if self.router is None:
            return unary
        rows: list[dict[str, float]] = []
        keys: list[tuple[str, str]] = []
        for block in blocks:
            for candidate_id in candidates:
                spec = self.imputer_registry.get_spec(candidate_id)
                rows.append(
                    merge_features(
                        block_features(batch, block, period),
                        forecast_block_features(block, forecast_spec),
                        candidate_features(spec, forecast_spec),
                    )
                )
                keys.append((block.block_id, candidate_id))
        predictions = self.router.prior.predict(_feature_matrix(rows, self.router.feature_names))
        risks = _ranker_risks(
            keys,
            predictions,
            float(self.router.metadata.get("unary_risk_scale", 1.0)),
        )
        return self._blend_with_candidate_global_priors(
            risks,
            forecast_spec,
            candidates,
            batch.metadata,
        )

    def _pseudo_batch(
        self,
        batch: SeriesBatch,
        seed: int,
        max_blocks: int = 8,
        *,
        target_blocks: Sequence[MissingBlock] | None = None,
        priority_channels: Sequence[int] = (),
    ) -> SeriesBatch:
        """Hide observed analogues matched to routed block channels and lengths."""

        rng = np.random.default_rng(seed)
        mask = batch.observed_mask.copy()
        length = batch.shape[1]
        if not np.any(mask[0]) or max_blocks < 1:
            return batch
        default_length = max(1, min(length // 20, 8))
        maximum_length = max(default_length, min(max(1, length // 8), 12))
        templates = list(target_blocks or ())
        if templates:
            priority = tuple(dict.fromkeys(int(value) for value in priority_channels))
            ordered: list[MissingBlock] = []
            seen_channels: set[int] = set()
            for channel in priority:
                match = next(
                    (block for block in templates if block.channel == channel),
                    None,
                )
                if match is not None:
                    ordered.append(match)
                    seen_channels.add(channel)
            for block in templates:
                if block.channel not in seen_channels:
                    ordered.append(block)
                    seen_channels.add(block.channel)
            ordered.extend(block for block in templates if block not in ordered)
            templates = ordered

        occupied_time = np.zeros(length, dtype=bool)
        placed = 0
        attempts = 0
        maximum_attempts = max(32, max_blocks * 8)
        while placed < max_blocks and attempts < maximum_attempts:
            attempts += 1
            template = templates[placed] if placed < len(templates) else None
            if template is None:
                channel = int(rng.integers(0, batch.shape[2]))
                desired_start = int(rng.integers(0, max(1, length - default_length + 1)))
                desired_length = default_length
            else:
                channel = int(template.channel)
                desired_start = int(template.start)
                desired_length = min(maximum_length, max(1, int(template.length)))
            candidate_lengths = tuple(dict.fromkeys((desired_length, default_length, 1)))
            selected: tuple[int, int] | None = None
            for block_length in candidate_lengths:
                starts = [
                    start
                    for start in range(0, length - block_length + 1)
                    if mask[0, start : start + block_length, channel].all()
                    and not occupied_time[start : start + block_length].any()
                ]
                if not starts:
                    continue
                distances = np.abs(np.asarray(starts, dtype=int) - desired_start)
                nearest = np.flatnonzero(distances == np.min(distances))
                chosen = int(starts[int(rng.choice(nearest))])
                selected = (chosen, block_length)
                break
            if selected is None:
                if template is not None:
                    templates.pop(placed)
                continue
            start, block_length = selected
            mask[0, start : start + block_length, channel] = False
            occupied_time[start : start + block_length] = True
            placed += 1
        return SeriesBatch(
            values=batch.values.copy(),
            observed_mask=mask,
            item_ids=batch.item_ids,
            metadata=batch.metadata,
        )

    def _refined_unary(
        self,
        batch: SeriesBatch,
        pseudo_batch: SeriesBatch,
        blocks,
        shortlist: tuple[str, ...],
        candidates: Mapping[str, CandidateResult],
        pseudo_candidates: Mapping[str, CandidateResult],
        forecast_spec: ForecastSpec,
        period: int | None,
        fallback: Mapping[tuple[str, str], float],
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        if self.router is None:
            return (
                {key: value for key, value in fallback.items() if key[1] in shortlist},
                {},
            )
        rows: list[dict[str, float]] = []
        keys: list[tuple[str, str]] = []
        proxy_risks: dict[tuple[str, str], float] = {}
        # Original missing positions have no ground truth. Mark them as
        # excluded so proxy errors are computed only on newly hidden values.
        proxy_mask = pseudo_batch.observed_mask | ~batch.observed_mask
        runtime_by_candidate = self.router.metadata.get(
            "candidate_runtime_seconds",
            {},
        )
        memory_by_candidate = self.router.metadata.get(
            "candidate_peak_memory_mb",
            {},
        )
        for block in blocks:
            for candidate_id in shortlist:
                spec = self.imputer_registry.get_spec(candidate_id)
                proxy = proxy_features(
                    pseudo_candidates[candidate_id],
                    batch.values,
                    proxy_mask,
                    channel=block.channel,
                )
                # Wall-clock time and RSS deltas depend on process scheduling and
                # must not change a routing decision.  Router artifacts persist
                # training medians for these operational features; legacy
                # artifacts without a memory statistic use the training median
                # default of zero instead of the current invocation's measurement.
                proxy["runtime_seconds"] = _nonnegative_mapping_value(
                    runtime_by_candidate,
                    candidate_id,
                )
                proxy["peak_memory_mb"] = _nonnegative_mapping_value(
                    memory_by_candidate,
                    candidate_id,
                )
                proxy_risks[(block.block_id, candidate_id)] = float(proxy["proxy_mae"])
                rows.append(
                    merge_features(
                        block_features(batch, block, period),
                        forecast_block_features(block, forecast_spec),
                        candidate_features(spec, forecast_spec),
                        proxy,
                    )
                )
                keys.append((block.block_id, candidate_id))
        predictions = self.router.unary.predict(_feature_matrix(rows, self.router.feature_names))
        r1_risks = _ranker_risks(
            keys,
            predictions,
            float(self.router.metadata.get("unary_risk_scale", 1.0)),
        )
        r0_risks = {key: float(fallback[key]) for key in keys}
        return (
            _blend_routing_evidence(
                r0_risks,
                r1_risks,
                proxy_risks,
                self._reliable_candidate_global_priors(
                    forecast_spec,
                    shortlist,
                    batch.metadata,
                ),
                weights=self._evidence_weights(forecast_spec),
                scale=float(self.router.metadata.get("unary_risk_scale", 1.0)),
            ),
            proxy_risks,
        )

    def _pairwise_risk(
        self,
        batch: SeriesBatch,
        graph,
        shortlist: tuple[str, ...],
        candidates: Mapping[str, CandidateResult],
        forecast_spec: ForecastSpec,
    ) -> dict[tuple[str, str, str, str], float]:
        feature_names = tuple(getattr(self.router, "pair_feature_names", ())) if self.router else ()
        pairwise: dict[tuple[str, str, str, str], float] = {}
        if feature_names and self.router is not None and self.router.pairwise.model is not None:
            by_id = {block.block_id: block for block in graph.blocks}
            rows: list[dict[str, float]] = []
            keys: list[tuple[str, str, str, str]] = []
            for edge in graph.edges:
                left_id, right_id = edge.left, edge.right
                left = by_id[left_id]
                right = by_id[right_id]
                for left_candidate in shortlist:
                    for right_candidate in shortlist:
                        rows.append(
                            pair_features(
                                batch,
                                left,
                                right,
                                candidates[left_candidate],
                                candidates[right_candidate],
                                edge_weight=edge.weight,
                            )
                        )
                        keys.append((left_id, right_id, left_candidate, right_candidate))
            if rows:
                predictions = self.router.pairwise.predict(_feature_matrix(rows, feature_names))
                pair_scale = max(
                    float(self.router.metadata.get("pair_risk_scale", 1.0)),
                    1e-6,
                )
                predictions = np.clip(
                    predictions,
                    -10.0 * pair_scale,
                    10.0 * pair_scale,
                )
                pairwise.update(
                    {key: float(value) for key, value in zip(keys, predictions, strict=True)}
                )
        for key, penalty in _candidate_switch_penalties(
            graph,
            shortlist,
            forecast_spec,
            weight=self.candidate_switch_penalty,
        ).items():
            pairwise[key] = pairwise.get(key, 0.0) + penalty
        return pairwise

    def _forecast_consensus_anchor(
        self,
        batch: SeriesBatch,
        blocks: Sequence[MissingBlock],
        shortlist: tuple[str, ...],
        candidates: Mapping[str, CandidateResult],
        forecast_spec: ForecastSpec,
        invalid: set[tuple[str, str]],
        unary_risks: Mapping[tuple[str, str], float] | None = None,
        proxy_scores: Mapping[str, float] | None = None,
        backtest_batch: SeriesBatch | None = None,
        backtest_candidates: Mapping[str, CandidateResult] | None = None,
        backtest_cutoff: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        if self.forecast_consensus_mode == "disabled":
            return None, {"active": False, "reason": "disabled"}
        if (
            self.forecast_consensus_mode
            not in {
                "value_median",
                "router_risk",
                "proxy_min",
            }
            and self.forecast_predictor is None
        ):
            raise ValueError("forecast consensus requires --forecaster-artifact during imputation")
        if (
            self.forecast_consensus_anchor_period_exceeded_mode == "medoid"
            and self.forecast_predictor is None
        ):
            raise ValueError("medoid period fallback requires --forecaster-artifact")
        visible_blocks = tuple(
            block for block in blocks if _block_visible_to_forecaster(block, forecast_spec)
        )
        reliable_priors = self._reliable_candidate_global_priors(
            forecast_spec,
            shortlist,
            batch.metadata,
        )
        dynamic_ids = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                (
                    (candidate_id, prior)
                    for candidate_id, prior in reliable_priors.items()
                    if candidate_id not in self.forecast_consensus_candidates
                ),
                key=lambda entry: (entry[1], entry[0]),
            )[: self._forecast_consensus_prior_count(forecast_spec)]
        )
        consensus_pool = tuple(dict.fromkeys((*self.forecast_consensus_candidates, *dynamic_ids)))
        eligible_ids = tuple(
            candidate_id
            for candidate_id in consensus_pool
            if candidate_id in shortlist
            and candidate_id in candidates
            and all((block.block_id, candidate_id) not in invalid for block in visible_blocks)
        )
        if not eligible_ids:
            return None, {"active": False, "reason": "no_eligible_candidate"}
        if not visible_blocks:
            return None, {"active": False, "reason": "no_forecast_visible_block"}
        selection_mode = self.forecast_consensus_mode
        period_fallback = False
        max_period_ratio = self._forecast_consensus_anchor_period_ratio(forecast_spec)
        period_eligible, period_ratio = _anchor_period_is_eligible(
            batch.metadata.get("period"),
            batch.shape[1],
            max_period_ratio,
        )
        if not period_eligible:
            if self.forecast_consensus_anchor_period_exceeded_mode == "medoid":
                selection_mode = "medoid"
                period_fallback = True
            else:
                return None, {
                    "active": False,
                    "reason": "anchor_period_ratio_exceeded",
                    "period_ratio": period_ratio,
                    "max_period_ratio": max_period_ratio,
                }
        if selection_mode == "value_median":
            return None, {
                "active": True,
                "mode": selection_mode,
                "aggregation": "elementwise_median",
                "ensemble_candidates": list(eligible_ids),
                "dataset_prior_candidates": list(dynamic_ids),
            }
        configured_prior_weight = self._forecast_consensus_prior_weight(forecast_spec)
        candidate_priors = {
            candidate_id: reliable_priors[candidate_id]
            for candidate_id in eligible_ids
            if candidate_id in reliable_priors
        }
        if selection_mode in {"router_risk", "proxy_min"}:
            scores = _candidate_signal_scores(
                selection_mode,
                eligible_ids,
                visible_blocks,
                unary_risks or {},
                proxy_scores or {},
            )
            if not scores:
                return None, {"active": False, "reason": "no_finite_candidate_signal"}
            selected, selection_scores = _regularized_forecast_consensus_candidate(
                scores,
                candidate_priors,
                configured_prior_weight,
            )
            return selected, {
                "active": True,
                "mode": selection_mode,
                "configured_mode": self.forecast_consensus_mode,
                "period_fallback": False,
                "period_ratio": period_ratio,
                "selected_candidate": selected,
                "eligible_candidates": list(scores),
                "dataset_prior_candidates": list(dynamic_ids),
                "selection_granularity": "episode",
                "selected_candidates_by_target": {},
                "scores": scores,
                "selection_scores": selection_scores,
                "candidate_priors": candidate_priors,
                "prior_weight": configured_prior_weight,
            }
        raw_scale = batch.metadata.get("mase_scale")
        if raw_scale is None:
            raise ValueError("forecast consensus requires a frozen MASE scale")
        full_scale = np.asarray(raw_scale, dtype=float).reshape(-1)
        target_indices = tuple(forecast_spec.target_indices or ())
        if full_scale.shape == (batch.shape[2],):
            scale = full_scale[list(target_indices)]
        elif full_scale.shape == (len(target_indices),):
            scale = full_scale
        else:
            raise ValueError("forecast consensus MASE scale has an invalid shape")
        target_scores: dict[str, tuple[float, ...]] = {}
        selected_by_target: dict[str, str] = {}
        target_selection_scores: dict[str, dict[str, float]] = {}
        prior_override_diagnostics: dict[str, Any] = {
            "configured": False,
            "applied": False,
        }
        target_prior_override_diagnostics: dict[str, dict[str, Any]] = {}
        selection_granularity = "episode"
        if selection_mode == "historical_backtest":
            consensus_indices = tuple(range(batch.shape[2]))
            if backtest_batch is None or backtest_candidates is None or backtest_cutoff is None:
                return None, {
                    "active": False,
                    "reason": "historical_backtest_unavailable",
                }
            backtest_ids = tuple(
                candidate_id
                for candidate_id in eligible_ids
                if candidate_id in backtest_candidates
                and np.asarray(
                    backtest_candidates[candidate_id].native_valid_mask,
                    dtype=bool,
                )[:, :backtest_cutoff][~backtest_batch.observed_mask[:, :backtest_cutoff]].all()
            )
            if not backtest_ids:
                return None, {
                    "active": False,
                    "reason": "no_valid_historical_backtest_candidate",
                }
            try:
                selected, scores = _historical_backtest_candidate(
                    {
                        candidate_id: backtest_candidates[candidate_id].values
                        for candidate_id in backtest_ids
                    },
                    forecast_spec,
                    self.forecast_predictor,
                    scale,
                    batch.values,
                    batch.observed_mask,
                    backtest_cutoff,
                    self.forecast_consensus_min_observed,
                )
            except ValueError as error:
                return None, {
                    "active": False,
                    "reason": "historical_backtest_invalid",
                    "detail": str(error),
                }
            eligible_ids = backtest_ids
            selection_scores = dict(scores)
            applied_prior_weight = 0.0
        else:
            raw_correlation = batch.metadata.get("training_correlation")
            consensus_correlation = (
                _correlation(batch.values[0], batch.observed_mask[0])
                if raw_correlation is None
                else np.asarray(raw_correlation, dtype=float)
            )
            consensus_values, consensus_spec, consensus_indices = _forecast_consensus_inputs(
                {candidate_id: candidates[candidate_id].values for candidate_id in eligible_ids},
                forecast_spec,
                self.forecast_consensus_context_mode,
                correlation=consensus_correlation,
                max_context_variates=self.forecast_consensus_max_context_variates,
            )
            scores, target_scores = _forecast_consensus_scores(
                consensus_values,
                consensus_spec,
                self.forecast_predictor,
                scale,
            )
            selected, selection_scores = _regularized_forecast_consensus_candidate(
                scores,
                candidate_priors,
                configured_prior_weight,
            )
            applied_prior_weight = configured_prior_weight
            selection_granularity = self._forecast_consensus_granularity(forecast_spec)
            override_penalty, override_margin = self._forecast_consensus_prior_override(
                forecast_spec
            )
            if selection_granularity == "episode":
                selected, prior_override_diagnostics = _safe_prior_consensus_override(
                    selected,
                    scores,
                    candidate_priors,
                    max_medoid_penalty=override_penalty,
                    min_prior_margin=override_margin,
                )
            else:
                for target_offset, target_index in enumerate(target_indices):
                    per_target_scores = {
                        candidate_id: candidate_scores[target_offset]
                        for candidate_id, candidate_scores in target_scores.items()
                    }
                    target_selected, combined_scores = _regularized_forecast_consensus_candidate(
                        per_target_scores,
                        candidate_priors,
                        configured_prior_weight,
                    )
                    target_selected, target_override = _safe_prior_consensus_override(
                        target_selected,
                        per_target_scores,
                        candidate_priors,
                        max_medoid_penalty=override_penalty,
                        min_prior_margin=override_margin,
                    )
                    selected_by_target[str(target_index)] = target_selected
                    target_selection_scores[str(target_index)] = combined_scores
                    target_prior_override_diagnostics[str(target_index)] = target_override
        diagnostics = {
            "active": True,
            "mode": selection_mode,
            "configured_mode": self.forecast_consensus_mode,
            "period_fallback": period_fallback,
            "period_ratio": period_ratio,
            "anchor_weight_override": 1.0 if period_fallback else None,
            "selected_candidate": selected,
            "eligible_candidates": list(eligible_ids),
            "dataset_prior_candidates": list(dynamic_ids),
            "context_mode": self.forecast_consensus_context_mode,
            "consensus_input_variates": len(consensus_indices),
            "consensus_variate_indices": list(consensus_indices),
            "selection_granularity": selection_granularity,
            "selected_candidates_by_target": selected_by_target,
            "scores": scores,
            "scores_by_target": {
                candidate_id: list(candidate_scores)
                for candidate_id, candidate_scores in target_scores.items()
            },
            "selection_scores": selection_scores,
            "target_selection_scores": target_selection_scores,
            "candidate_priors": candidate_priors,
            "prior_weight": applied_prior_weight,
            "prior_override": prior_override_diagnostics,
            "prior_override_by_target": target_prior_override_diagnostics,
            "scale": list(map(float, scale)),
            "backtest_cutoff": backtest_cutoff,
        }
        if selection_mode == "value_topk_mean":
            ensemble_weights_by_target: dict[str, dict[str, float]] = {}
            third_relative_gap_by_target: dict[str, float | None] = {}
            if selection_granularity == "target":
                ensemble_weights: dict[str, float] = {}
                selected_union: dict[str, None] = {}
                for target_offset, target_index in enumerate(target_indices):
                    target_key = str(target_index)
                    per_target_raw_scores = {
                        candidate_id: candidate_scores[target_offset]
                        for candidate_id, candidate_scores in target_scores.items()
                    }
                    third_relative_gap = _third_candidate_relative_gap(per_target_raw_scores)
                    selected_top_k = self.forecast_consensus_ensemble_top_k
                    if (
                        self.forecast_consensus_ensemble_third_relative_gap is not None
                        and third_relative_gap is not None
                        and third_relative_gap
                        <= self.forecast_consensus_ensemble_third_relative_gap
                    ):
                        selected_top_k = 3
                    target_weights = _top_k_consensus_weights(
                        target_selection_scores[target_key],
                        selected_top_k,
                    )
                    target_weights = _proxy_weighted_consensus_weights(
                        target_weights,
                        proxy_scores,
                        power=self.forecast_consensus_ensemble_proxy_weight_power,
                    )
                    ensemble_weights_by_target[target_key] = target_weights
                    third_relative_gap_by_target[target_key] = third_relative_gap
                    selected_union.update(dict.fromkeys(target_weights))
                ensemble_candidates = list(selected_union)
                third_relative_gap = None
                ensemble_selected_k = None
            else:
                third_relative_gap = _third_candidate_relative_gap(scores)
                selected_top_k = self.forecast_consensus_ensemble_top_k
                if (
                    self.forecast_consensus_ensemble_third_relative_gap is not None
                    and third_relative_gap is not None
                    and third_relative_gap <= self.forecast_consensus_ensemble_third_relative_gap
                ):
                    selected_top_k = 3
                ensemble_weights = _top_k_consensus_weights(
                    selection_scores,
                    selected_top_k,
                )
                ensemble_weights = _proxy_weighted_consensus_weights(
                    ensemble_weights,
                    proxy_scores,
                    power=self.forecast_consensus_ensemble_proxy_weight_power,
                )
                ensemble_candidates = list(ensemble_weights)
                ensemble_selected_k = len(ensemble_weights)
            diagnostics.update(
                {
                    "aggregation": "weighted_mean",
                    "ensemble_candidates": ensemble_candidates,
                    "ensemble_weights": ensemble_weights,
                    "ensemble_weights_by_target": ensemble_weights_by_target,
                    "ensemble_weight_source": (
                        "proxy_global_mae_inverse_power_v1"
                        if self.forecast_consensus_ensemble_proxy_weight_power > 0.0
                        else "uniform_v1"
                    ),
                    "ensemble_proxy_weight_power": (
                        self.forecast_consensus_ensemble_proxy_weight_power
                    ),
                    "ensemble_proxy_scores": {
                        candidate_id: float(proxy_scores[candidate_id])
                        for candidate_id in ensemble_weights
                        if candidate_id in proxy_scores
                        and np.isfinite(float(proxy_scores[candidate_id]))
                    },
                    "ensemble_top_k": self.forecast_consensus_ensemble_top_k,
                    "ensemble_selected_k": ensemble_selected_k,
                    "ensemble_selected_k_by_target": {
                        target_key: len(target_weights)
                        for target_key, target_weights in (ensemble_weights_by_target.items())
                    },
                    "ensemble_third_relative_gap": third_relative_gap,
                    "ensemble_third_relative_gap_by_target": (third_relative_gap_by_target),
                    "ensemble_third_relative_gap_source": ("raw_forecast_consensus_score_v1"),
                    "ensemble_third_relative_gap_threshold": (
                        self.forecast_consensus_ensemble_third_relative_gap
                    ),
                }
            )
            return None, diagnostics
        return selected, diagnostics

    def _configured_fallback(
        self,
        batch: SeriesBatch,
        block,
        candidates: dict[str, CandidateResult],
        training_medians: np.ndarray | None,
        seed: int,
        precomputed: Mapping[str, CandidateResult] | None = None,
        allow_execution: bool = True,
    ) -> tuple[np.ndarray, str, tuple[str, ...]]:
        precomputed = precomputed or {}
        sequence = self.fallback_tail if block.end == batch.shape[1] else self.fallback_internal
        attempts: list[str] = []
        for fallback_id in sequence:
            attempts.append(fallback_id)
            if fallback_id == "train_median":
                values, name = _median_fallback(batch, block, training_medians)
                return values, name, tuple(attempts)
            if fallback_id not in self.imputer_registry:
                continue
            spec = self.imputer_registry.get_spec(fallback_id)
            if block.end == batch.shape[1] and not spec.supports_tail:
                continue
            if spec.requires_period and not batch.metadata.get("period"):
                continue
            result = candidates.get(fallback_id)
            if result is None:
                result = precomputed.get(fallback_id)
                if result is None:
                    if not allow_execution:
                        continue
                    if spec.fit_scope != "none" and fallback_id not in self.imputer_artifacts:
                        continue
                    result = self.candidate_runner.run(
                        fallback_id,
                        batch,
                        self.imputer_artifacts.get(fallback_id),
                        seed=seed,
                    )
                candidates[fallback_id] = result
            if _native_block_is_valid(result, block):
                selector = (
                    block.batch_index,
                    slice(block.start, block.end),
                    block.channel,
                )
                return (
                    np.asarray(result.values[selector], dtype=float),
                    fallback_id,
                    tuple(attempts),
                )
        values, name = _median_fallback(batch, block, training_medians)
        attempts.append(name)
        return values, name, tuple(attempts)

    def prepare_route(
        self,
        item: TimeSeriesItem,
        observed_mask: np.ndarray,
        forecast_spec: ForecastSpec,
        budget: BudgetSpec | None = None,
        *,
        seed: int = 20260710,
        available_artifact_ids: Iterable[str] | None = None,
        artifact_load_failures: Mapping[str, str] | None = None,
    ) -> RoutePlan:
        """Build R0 state without executing any imputation candidate.

        ``available_artifact_ids`` is the explicit hook for a candidate-major
        scheduler.  When omitted, the legacy eager artifact path is used.  A
        caller that encounters an artifact-load failure should call this method
        again with the successfully loaded fitted IDs and the updated failure
        mapping; R0 and the shortlist are then rebuilt deterministically.  In
        explicit mode, training medians and correlation must already have been
        supplied on the pipeline or item because no artifact loader is invoked.
        """

        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != item.values.shape:
            raise ValueError("observed_mask must match item.values")
        batch = SeriesBatch(
            values=item.values[None, ...],
            observed_mask=mask[None, ...],
            item_ids=(item.item_id,),
            metadata=item.metadata,
        )
        blocks = detect_missing_blocks(batch.observed_mask)
        resolved_budget = budget or BudgetSpec(max_candidates=self.shortlist_size)
        period = item.metadata.get("period")
        if not blocks:
            return RoutePlan(
                batch=batch,
                blocks=(),
                graph=BlockGraph((), ()),
                forecast_spec=forecast_spec,
                budget=resolved_budget,
                seed=seed,
                period=period,
                correlation_source="none",
                candidate_ids=(),
                shortlist=(),
                costs={},
                prior_unary={},
                pseudo_batch=None,
                training_medians=item.metadata.get("training_medians", self.training_medians),
                artifact_load_failures={},
                available_artifact_ids=frozenset(),
                allow_fallback_execution=False,
            )

        if available_artifact_ids is None:
            self._ensure_item_artifacts(item)
            available = frozenset(self.imputer_artifacts)
            failures = dict(
                self.artifact_load_failures
                if artifact_load_failures is None
                else artifact_load_failures
            )
        else:
            available = frozenset(str(value) for value in available_artifact_ids)
            unknown = available.difference(self.imputer_registry.ids)
            if unknown:
                raise ValueError(
                    "available_artifact_ids contains unknown candidates: "
                    + ", ".join(sorted(unknown))
                )
            failures = dict(
                self.artifact_load_failures
                if artifact_load_failures is None
                else artifact_load_failures
            )
        training_medians = item.metadata.get("training_medians", self.training_medians)

        supplied_correlation = item.metadata.get("training_correlation", self.training_correlation)
        if supplied_correlation is None:
            correlation = _correlation(item.values, mask)
            correlation_source = "context"
        else:
            correlation = np.asarray(supplied_correlation, dtype=float)
            expected = (item.values.shape[1], item.values.shape[1])
            if correlation.shape != expected or not np.isfinite(correlation).all():
                raise ValueError(
                    f"training_correlation must be a finite matrix with shape {expected}"
                )
            correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
            correlation_source = "training"
        graph = build_block_graph(blocks, correlation)
        forced = set(self.forced_candidates)
        trained_candidates = (
            set(self.router.candidate_ids)
            if self.router is not None and self.router.candidate_ids
            else None
        )
        candidate_ids = tuple(
            spec.imputer_id
            for spec in self.imputer_registry.specs()
            if (
                (trained_candidates is None or spec.imputer_id in trained_candidates)
                and (
                    spec.imputer_id in forced
                    or (
                        self.imputer_registry.availability(spec.imputer_id).available
                        and (spec.device == "any" or spec.device in resolved_budget.allowed_devices)
                        and (not spec.requires_period or period)
                        and (spec.fit_scope == "none" or spec.imputer_id in available)
                    )
                )
            )
        )
        if forecast_spec.mode == "independent_univariate" and not any(
            _block_visible_to_forecaster(block, forecast_spec) for block in blocks
        ):
            candidate_ids = tuple(
                candidate_id
                for candidate_id in self.forced_candidates
                if candidate_id in candidate_ids
            )
        if not candidate_ids:
            raise RuntimeError("no imputation candidate is available under the budget")
        learned_costs = (
            self.router.metadata.get("candidate_shortlist_costs", {})
            if self.router is not None
            else {}
        )
        costs = {
            spec.imputer_id: float(
                learned_costs.get(spec.imputer_id, spec.cost_tier)
                if isinstance(learned_costs, Mapping)
                else spec.cost_tier
            )
            for spec in self.imputer_registry.specs()
        }
        for spec in self.imputer_registry.specs():
            if not np.isfinite(costs[spec.imputer_id]) or costs[spec.imputer_id] < 0:
                costs[spec.imputer_id] = float(spec.cost_tier)
        prior_unary = self._heuristic_unary(batch, blocks, candidate_ids, forecast_spec, period)
        reliable_priors = self._reliable_candidate_global_priors(
            forecast_spec,
            candidate_ids,
            batch.metadata,
        )
        prior_forced = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                reliable_priors.items(),
                key=lambda entry: (entry[1], entry[0]),
            )[: self.candidate_global_prior_forced_count]
        )
        consensus_prior_forced = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                (
                    (candidate_id, prior)
                    for candidate_id, prior in reliable_priors.items()
                    if candidate_id not in self.forecast_consensus_candidates
                ),
                key=lambda entry: (entry[1], entry[0]),
            )[: self._forecast_consensus_prior_count(forecast_spec)]
        )
        shortlist_forced = tuple(
            dict.fromkeys(
                (
                    *self.forced_candidates,
                    *(
                        candidate_id
                        for candidate_id in self.shortlist_anchor_candidates
                        if candidate_id in candidate_ids
                    ),
                    *prior_forced,
                    *consensus_prior_forced,
                )
            )
        )
        shortlist = greedy_shortlist(
            blocks,
            candidate_ids,
            prior_unary,
            costs,
            max_candidates=min(resolved_budget.max_candidates, self.shortlist_size),
            forced=shortlist_forced,
        )
        pseudo_batch = (
            None
            if self.router is None
            else self._pseudo_batch(
                batch,
                seed,
                max_blocks=self.pseudo_blocks,
                target_blocks=blocks,
                priority_channels=forecast_spec.target_indices or (),
            )
        )
        backtest_batch: SeriesBatch | None = None
        backtest_cutoff: int | None = None
        if self.forecast_consensus_mode == "historical_backtest":
            validation_length = min(
                self.forecast_consensus_validation_length,
                batch.shape[1] - 2,
            )
            cutoff = batch.shape[1] - validation_length
            targets = tuple(forecast_spec.target_indices or ())
            has_validation = validation_length > 0 and all(
                int(batch.observed_mask[0, cutoff:, target].sum())
                >= self.forecast_consensus_min_observed
                for target in targets
            )
            if has_validation:
                source = pseudo_batch or batch
                backtest_mask = np.asarray(source.observed_mask, dtype=bool).copy()
                backtest_mask[:, cutoff:, :] = False
                backtest_batch = SeriesBatch(
                    values=source.values.copy(),
                    observed_mask=backtest_mask,
                    item_ids=source.item_ids,
                    metadata=source.metadata,
                )
                backtest_cutoff = cutoff
        return RoutePlan(
            batch=batch,
            blocks=blocks,
            graph=graph,
            forecast_spec=forecast_spec,
            budget=resolved_budget,
            seed=seed,
            period=period,
            correlation_source=correlation_source,
            candidate_ids=candidate_ids,
            shortlist=shortlist,
            costs=costs,
            prior_unary=prior_unary,
            pseudo_batch=pseudo_batch,
            training_medians=training_medians,
            artifact_load_failures=failures,
            available_artifact_ids=available,
            allow_fallback_execution=available_artifact_ids is None,
            backtest_batch=backtest_batch,
            backtest_cutoff=backtest_cutoff,
        )

    @staticmethod
    def _injected_results(
        required_ids: tuple[str, ...],
        supplied: Mapping[str, CandidateResult],
        batch: SeriesBatch,
        label: str,
    ) -> dict[str, CandidateResult]:
        missing = [candidate_id for candidate_id in required_ids if candidate_id not in supplied]
        if missing:
            raise ValueError(f"missing {label} candidate results: {', '.join(missing)}")
        results: dict[str, CandidateResult] = {}
        for candidate_id in required_ids:
            result = supplied[candidate_id]
            if result.imputer_id != candidate_id:
                raise ValueError(
                    f"{label} result key {candidate_id!r} contains {result.imputer_id!r}"
                )
            result.validate_against(batch)
            results[candidate_id] = result
        return results

    def finish_route(
        self,
        plan: RoutePlan,
        candidates: Mapping[str, CandidateResult],
        pseudo_candidates: Mapping[str, CandidateResult] | None = None,
        *,
        backtest_candidates: Mapping[str, CandidateResult] | None = None,
        fallback_candidates: Mapping[str, CandidateResult] | None = None,
    ) -> FAISResult:
        """Finish R1, structured search, fallback, and block assembly.

        Supplied mappings may contain evaluation-only candidates; only
        ``plan.shortlist`` participates in routing.  Precomputed fallback
        results are consulted only if the configured fallback sequence needs
        them, so they cannot change shortlist risks or candidate costs.  Plans
        built with explicit artifact IDs never execute an implicit fallback;
        a candidate-major scheduler must inject any such result through
        ``fallback_candidates``.
        """

        batch = plan.batch
        mask = np.asarray(batch.observed_mask[0], dtype=bool)
        if plan.is_noop:
            return FAISResult(
                values=batch.values[0].copy(),
                routing=RoutingResult({}, (), 0.0),
                candidates={},
                observed_mask=mask,
            )
        route_candidates = self._injected_results(plan.shortlist, candidates, batch, "actual")
        routed_backtest: dict[str, CandidateResult] = {}
        if plan.backtest_batch is not None:
            routed_backtest = self._injected_results(
                plan.shortlist,
                backtest_candidates or {},
                plan.backtest_batch,
                "historical backtest",
            )
        proxy_error_scores: dict[str, float] = {}
        proxy_rejected_candidates: frozenset[str] = frozenset()
        proxy_outlier_threshold: float | None = None
        routed_pseudo: dict[str, CandidateResult] = {}
        proxy_mask: np.ndarray | None = None
        precomputed_fallbacks = dict(fallback_candidates or {})
        for candidate_id, result in precomputed_fallbacks.items():
            if candidate_id not in self.imputer_registry:
                raise ValueError(f"unknown fallback candidate result: {candidate_id!r}")
            if result.imputer_id != candidate_id:
                raise ValueError(
                    f"fallback result key {candidate_id!r} contains {result.imputer_id!r}"
                )
            result.validate_against(batch)

        if self.router is None:
            unary = {
                key: value for key, value in plan.prior_unary.items() if key[1] in plan.shortlist
            }
            block_proxy_risks: dict[tuple[str, str], float] = {}
        else:
            if plan.pseudo_batch is None:
                raise ValueError("router route plan is missing its pseudo batch")
            routed_pseudo = self._injected_results(
                plan.shortlist,
                pseudo_candidates or {},
                plan.pseudo_batch,
                "pseudo",
            )
            proxy_mask = plan.pseudo_batch.observed_mask | ~batch.observed_mask
            proxy_error_scores = {
                candidate_id: float(
                    proxy_features(
                        result,
                        batch.values,
                        proxy_mask,
                    )["proxy_global_mae"]
                )
                for candidate_id, result in routed_pseudo.items()
            }
            (
                proxy_rejected_candidates,
                proxy_outlier_threshold,
            ) = _proxy_outlier_candidates(
                proxy_error_scores,
                multiplier=self.proxy_outlier_multiplier,
            )
            unary, block_proxy_risks = self._refined_unary(
                batch,
                plan.pseudo_batch,
                plan.blocks,
                plan.shortlist,
                route_candidates,
                routed_pseudo,
                plan.forecast_spec,
                plan.period,
                plan.prior_unary,
            )
        pairwise_skipped_for_scale = len(plan.blocks) > self.max_pairwise_blocks
        pairwise = (
            {}
            if pairwise_skipped_for_scale
            else self._pairwise_risk(
                batch,
                plan.graph,
                plan.shortlist,
                route_candidates,
                plan.forecast_spec,
            )
        )
        invalid: set[tuple[str, str]] = set()
        for block in plan.blocks:
            for candidate_id, result in route_candidates.items():
                candidate_spec = self.imputer_registry.get_spec(candidate_id)
                capability_mismatch = (
                    block.end == batch.shape[1] and not candidate_spec.supports_tail
                ) or (candidate_spec.requires_period and not plan.period)
                if capability_mismatch or not _native_block_is_valid(result, block):
                    invalid.add((block.block_id, candidate_id))
                elif candidate_id in proxy_rejected_candidates:
                    invalid.add((block.block_id, candidate_id))

        calibrated_anchor, calibrated_diagnostics = _select_extrapolation_anchor(
            self.candidate_anchor_calibrations,
            plan.forecast_spec,
            batch.metadata,
            proxy_error_scores,
            (
                candidate_id
                for candidate_id in plan.shortlist
                if candidate_id not in proxy_rejected_candidates
            ),
        )
        consensus_anchor, consensus_diagnostics = self._forecast_consensus_anchor(
            batch,
            plan.blocks,
            plan.shortlist,
            route_candidates,
            plan.forecast_spec,
            invalid,
            unary,
            proxy_error_scores,
            plan.backtest_batch,
            routed_backtest,
            plan.backtest_cutoff,
        )
        selected_anchor = consensus_anchor or calibrated_anchor
        anchor_diagnostics = (
            {
                "active": True,
                "source": "forecast_consensus",
                **consensus_diagnostics,
            }
            if consensus_anchor is not None
            else {
                "source": "proxy_calibration",
                **calibrated_diagnostics,
            }
        )
        anchor_block_assignments: dict[str, str] = {}
        if selected_anchor is not None:
            anchor_weight_override = consensus_diagnostics.get("anchor_weight_override")
            anchor_weight = (
                float(
                    self._forecast_consensus_anchor_weight(plan.forecast_spec)
                    if anchor_weight_override is None
                    else anchor_weight_override
                )
                if consensus_anchor is not None
                else 1.0
            )
            anchor_diagnostics["anchor_weight"] = anchor_weight
            anchor_risk_scale = max(
                float(
                    self.router.metadata.get("unary_risk_scale", 1.0)
                    if self.router is not None
                    else 1.0
                ),
                1e-6,
            )
            calibrated_candidates = tuple(anchor_diagnostics.get("candidates", ()))
            raw_target_anchors = anchor_diagnostics.get("selected_candidates_by_target", {})
            target_anchor_candidates = (
                {
                    int(target_index): str(candidate_id)
                    for target_index, candidate_id in raw_target_anchors.items()
                }
                if isinstance(raw_target_anchors, Mapping)
                else {}
            )
            for block in plan.blocks:
                if not _block_visible_to_forecaster(block, plan.forecast_spec):
                    continue
                block_selected_anchor = target_anchor_candidates.get(
                    block.channel,
                    selected_anchor,
                )
                preferred = tuple(dict.fromkeys((block_selected_anchor, *calibrated_candidates)))
                block_anchor = next(
                    (
                        candidate_id
                        for candidate_id in preferred
                        if candidate_id in plan.shortlist
                        and (block.block_id, candidate_id) not in invalid
                    ),
                    None,
                )
                if block_anchor is None:
                    continue
                anchor_block_assignments[block.block_id] = block_anchor
                for candidate_id in plan.shortlist:
                    key = (block.block_id, candidate_id)
                    if anchor_weight >= 1.0 - 1e-12:
                        unary[key] = 0.0 if candidate_id == block_anchor else 1e6
                    elif anchor_weight > 0.0:
                        preference_risk = 0.0 if candidate_id == block_anchor else anchor_risk_scale
                        unary[key] = float(
                            (1.0 - anchor_weight) * unary[key] + anchor_weight * preference_risk
                        )

        forecast_irrelevant_assignments: dict[str, str] = {}
        if plan.forecast_spec.mode == "independent_univariate":
            for block in plan.blocks:
                if _block_visible_to_forecaster(block, plan.forecast_spec):
                    continue
                preferred = (
                    ("locf", "linear_interp")
                    if block.end == batch.shape[1]
                    else ("linear_interp", "locf")
                )
                ordered_candidates = tuple(
                    dict.fromkeys(
                        (
                            *preferred,
                            *sorted(
                                plan.shortlist,
                                key=lambda candidate_id: (
                                    float(plan.costs.get(candidate_id, 1.0)),
                                    candidate_id,
                                ),
                            ),
                        )
                    )
                )
                selected = next(
                    (
                        candidate_id
                        for candidate_id in ordered_candidates
                        if candidate_id in plan.shortlist
                        and (block.block_id, candidate_id) not in invalid
                    ),
                    None,
                )
                if selected is None:
                    continue
                forecast_irrelevant_assignments[block.block_id] = selected
                for candidate_id in plan.shortlist:
                    unary[(block.block_id, candidate_id)] = 0.0 if candidate_id == selected else 1e6

        # Keep safe completion outside candidate scoring. A synthetic fallback
        # state is available only when every shortlisted candidate failed for
        # a block; the router therefore never interprets fallback values as a
        # successful candidate result.
        fallback_id = "__fallback__"
        solver_candidates = list(plan.shortlist)
        no_native_candidate = {
            block.block_id
            for block in plan.blocks
            if all((block.block_id, candidate_id) in invalid for candidate_id in plan.shortlist)
        }
        costs = dict(plan.costs)
        if no_native_candidate:
            solver_candidates.append(fallback_id)
            costs[fallback_id] = 0.0
            for block in plan.blocks:
                unary[(block.block_id, fallback_id)] = (
                    1e6 if block.block_id in no_native_candidate else float("inf")
                )
        solver_budget = plan.budget
        if no_native_candidate and plan.budget.max_active_candidates is not None:
            solver_budget = BudgetSpec(
                max_candidates=max(
                    plan.budget.max_candidates + 1,
                    plan.budget.max_active_candidates + 1,
                ),
                max_active_candidates=plan.budget.max_active_candidates + 1,
                max_runtime_seconds=plan.budget.max_runtime_seconds,
                max_memory_bytes=plan.budget.max_memory_bytes,
                allowed_devices=plan.budget.allowed_devices,
            )
        if pairwise_skipped_for_scale:
            routing = _pairwise_free_search(
                plan.blocks,
                tuple(solver_candidates),
                unary,
                costs,
                solver_budget,
                self.cost_weight,
                invalid,
            )
        else:
            routing = beam_search(
                plan.graph,
                tuple(solver_candidates),
                unary,
                pairwise=pairwise,
                costs=costs,
                budget=solver_budget,
                beam_width=self.beam_width,
                beta=self.beta,
                cost_weight=self.cost_weight,
                invalid=invalid,
            )
        completed = batch.values.copy()
        fallback_blocks: list[str] = []
        fallback_records: dict[str, dict[str, Any]] = {}
        ensemble_candidates = tuple(
            str(candidate_id)
            for candidate_id in consensus_diagnostics.get("ensemble_candidates", ())
            if candidate_id in route_candidates
        )
        ensemble_block_assignments: dict[str, tuple[str, ...]] = {}
        ensemble_block_weights: dict[str, dict[str, float]] = {}
        ensemble_aggregation = str(consensus_diagnostics.get("aggregation", ""))
        raw_ensemble_weights = consensus_diagnostics.get("ensemble_weights", {})
        configured_ensemble_weights = (
            {
                str(candidate_id): float(weight)
                for candidate_id, weight in raw_ensemble_weights.items()
                if np.isfinite(float(weight)) and float(weight) >= 0.0
            }
            if isinstance(raw_ensemble_weights, Mapping)
            else {}
        )
        raw_target_ensemble_weights = consensus_diagnostics.get(
            "ensemble_weights_by_target",
            {},
        )
        ensemble_weights_by_target = (
            {
                str(target_index): {
                    str(candidate_id): float(weight)
                    for candidate_id, weight in target_weights.items()
                    if np.isfinite(float(weight)) and float(weight) >= 0.0
                }
                for target_index, target_weights in raw_target_ensemble_weights.items()
                if isinstance(target_weights, Mapping)
            }
            if isinstance(raw_target_ensemble_weights, Mapping)
            else {}
        )
        ensemble_channel_weights: dict[str, dict[str, float]] = {}
        ensemble_weight_calibration: dict[str, dict[str, Any]] = {}
        if (
            self.forecast_consensus_pseudo_weight_calibration == "convex_l2"
            and len(ensemble_candidates) == 2
            and proxy_mask is not None
            and all(candidate_id in routed_pseudo for candidate_id in ensemble_candidates)
        ):
            first_candidate, second_candidate = ensemble_candidates
            prior = np.asarray(
                [
                    configured_ensemble_weights.get(first_candidate, 0.0),
                    configured_ensemble_weights.get(second_candidate, 0.0),
                ],
                dtype=float,
            )
            if not np.isfinite(prior).all() or float(prior.sum()) <= 0.0:
                prior = np.ones(2, dtype=float)
            prior /= prior.sum()
            visible_channels = sorted(
                {
                    block.channel
                    for block in plan.blocks
                    if _block_visible_to_forecaster(block, plan.forecast_spec)
                }
            )
            for channel in visible_channels:
                second_weight, calibration = _pseudo_calibrated_convex_weight(
                    batch.values,
                    proxy_mask,
                    routed_pseudo[first_candidate],
                    routed_pseudo[second_candidate],
                    channel=channel,
                    prior_weight=float(prior[1]),
                    prior_strength=(self.forecast_consensus_pseudo_weight_prior_strength),
                    min_points=self.forecast_consensus_pseudo_weight_min_points,
                )
                ensemble_channel_weights[str(channel)] = {
                    first_candidate: 1.0 - second_weight,
                    second_candidate: second_weight,
                }
                ensemble_weight_calibration[str(channel)] = calibration
        consensus_diagnostics["ensemble_channel_weights"] = ensemble_channel_weights
        consensus_diagnostics["ensemble_weight_calibration"] = ensemble_weight_calibration
        consensus_diagnostics["pseudo_weight_calibration"] = (
            self.forecast_consensus_pseudo_weight_calibration
        )
        for block in plan.blocks:
            if ensemble_candidates and _block_visible_to_forecaster(block, plan.forecast_spec):
                target_weights = ensemble_weights_by_target.get(str(block.channel))
                block_ensemble_candidates = (
                    tuple(target_weights) if target_weights else ensemble_candidates
                )
                valid_candidates = tuple(
                    candidate_id
                    for candidate_id in block_ensemble_candidates
                    if (block.block_id, candidate_id) not in invalid
                )
                if valid_candidates:
                    selector = (
                        block.batch_index,
                        slice(block.start, block.end),
                        block.channel,
                    )
                    stacked = np.stack(
                        [
                            np.asarray(route_candidates[candidate_id].values[selector])
                            for candidate_id in valid_candidates
                        ],
                        axis=0,
                    )
                    if ensemble_aggregation == "weighted_mean":
                        block_configured_weights = ensemble_channel_weights.get(
                            str(block.channel),
                            target_weights or configured_ensemble_weights,
                        )
                        weights = np.asarray(
                            [
                                block_configured_weights.get(candidate_id, 0.0)
                                for candidate_id in valid_candidates
                            ],
                            dtype=float,
                        )
                        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                            weights = np.ones(len(valid_candidates), dtype=float)
                        weights /= weights.sum()
                        completed[selector] = np.tensordot(weights, stacked, axes=(0, 0))
                        block_weights = {
                            candidate_id: float(weight)
                            for candidate_id, weight in zip(
                                valid_candidates,
                                weights,
                                strict=True,
                            )
                        }
                        assignment_prefix = "value_topk_mean"
                    else:
                        completed[selector] = np.median(stacked, axis=0)
                        block_weights = {
                            candidate_id: 1.0 / len(valid_candidates)
                            for candidate_id in valid_candidates
                        }
                        assignment_prefix = "value_median"
                    ensemble_block_assignments[block.block_id] = valid_candidates
                    ensemble_block_weights[block.block_id] = block_weights
                    routing.assignments[block.block_id] = (
                        assignment_prefix + "[" + ",".join(valid_candidates) + "]"
                    )
                    continue
            candidate_id = routing.assignments[block.block_id]
            if candidate_id == fallback_id:
                fallback_values, fallback_name, attempts = self._configured_fallback(
                    batch,
                    block,
                    route_candidates,
                    plan.training_medians,
                    plan.seed,
                    precomputed_fallbacks,
                    plan.allow_fallback_execution,
                )
                completed[block.batch_index, block.start : block.end, block.channel] = (
                    fallback_values
                )
                routing.assignments[block.block_id] = fallback_name
                fallback_blocks.append(block.block_id)
                fallback_records[block.block_id] = {
                    "selected": fallback_name,
                    "attempts": attempts,
                    "kind": "tail" if block.end == batch.shape[1] else "internal",
                }
                continue
            result = route_candidates[candidate_id]
            completed[block.batch_index, block.start : block.end, block.channel] = result.values[
                block.batch_index, block.start : block.end, block.channel
            ]
        proxy_blend_weight = self._forecast_consensus_proxy_blend_weight(plan.forecast_spec)
        proxy_blend_min_relative_margin = self._forecast_consensus_proxy_blend_margin(
            plan.forecast_spec
        )
        proxy_blend_block_assignments: dict[str, dict[str, Any]] = {}
        proxy_blend_primary_candidates: set[str] = set()
        proxy_weight_calibration_cache: dict[
            tuple[int, str, str], tuple[float, dict[str, Any]]
        ] = {}
        if proxy_blend_weight > 0.0 and block_proxy_risks:
            for block in plan.blocks:
                if not _block_visible_to_forecaster(block, plan.forecast_spec):
                    continue
                eligible_proxy = tuple(
                    candidate_id
                    for candidate_id in plan.shortlist
                    if (block.block_id, candidate_id) not in invalid
                    and np.isfinite(
                        float(block_proxy_risks.get((block.block_id, candidate_id), np.nan))
                    )
                )
                if not eligible_proxy:
                    continue
                proxy_candidate = min(
                    eligible_proxy,
                    key=lambda candidate_id: (
                        float(block_proxy_risks[(block.block_id, candidate_id)]),
                        candidate_id,
                    ),
                )
                selector = (
                    block.batch_index,
                    slice(block.start, block.end),
                    block.channel,
                )
                primary_assignment = routing.assignments[block.block_id]
                if primary_assignment in self.imputer_registry:
                    proxy_blend_primary_candidates.add(primary_assignment)
                primary_proxy_risk = block_proxy_risks.get((block.block_id, primary_assignment))
                if primary_proxy_risk is not None and not np.isfinite(float(primary_proxy_risk)):
                    primary_proxy_risk = None
                proxy_risk = float(block_proxy_risks[(block.block_id, proxy_candidate)])
                proxy_risk_margin = (
                    None if primary_proxy_risk is None else float(primary_proxy_risk) - proxy_risk
                )
                proxy_risk_relative_margin = (
                    None
                    if proxy_risk_margin is None
                    else proxy_risk_margin / max(abs(float(primary_proxy_risk)), 1e-8)
                )
                proxy_gate_applied = bool(
                    proxy_risk_relative_margin is not None
                    and proxy_risk_relative_margin + 1e-12 >= proxy_blend_min_relative_margin
                )
                applied_proxy_weight = proxy_blend_weight if proxy_gate_applied else 0.0
                weight_calibration: dict[str, Any] | None = None
                calibration_key = (
                    block.channel,
                    primary_assignment,
                    proxy_candidate,
                )
                if (
                    proxy_gate_applied
                    and self.forecast_consensus_pseudo_weight_calibration == "convex_l2"
                    and proxy_mask is not None
                    and primary_assignment in routed_pseudo
                    and proxy_candidate in routed_pseudo
                ):
                    if calibration_key not in proxy_weight_calibration_cache:
                        proxy_weight_calibration_cache[calibration_key] = (
                            _pseudo_calibrated_convex_weight(
                                batch.values,
                                proxy_mask,
                                routed_pseudo[primary_assignment],
                                routed_pseudo[proxy_candidate],
                                channel=block.channel,
                                prior_weight=proxy_blend_weight,
                                prior_strength=(
                                    self.forecast_consensus_pseudo_weight_prior_strength
                                ),
                                min_points=(self.forecast_consensus_pseudo_weight_min_points),
                            )
                        )
                    applied_proxy_weight, weight_calibration = proxy_weight_calibration_cache[
                        calibration_key
                    ]
                if applied_proxy_weight > 0.0:
                    completed[selector] = (1.0 - applied_proxy_weight) * completed[
                        selector
                    ] + applied_proxy_weight * route_candidates[proxy_candidate].values[selector]
                proxy_blend_block_assignments[block.block_id] = {
                    "primary_assignment": primary_assignment,
                    "proxy_candidate": proxy_candidate,
                    "primary_proxy_risk": (
                        None if primary_proxy_risk is None else float(primary_proxy_risk)
                    ),
                    "proxy_risk": proxy_risk,
                    "proxy_risk_margin": proxy_risk_margin,
                    "proxy_risk_relative_margin": proxy_risk_relative_margin,
                    "proxy_gate_threshold": proxy_blend_min_relative_margin,
                    "proxy_gate_applied": proxy_gate_applied,
                    "configured_proxy_weight": proxy_blend_weight,
                    "proxy_weight": applied_proxy_weight,
                    "proxy_weight_calibration": weight_calibration,
                }
                if applied_proxy_weight > 0.0:
                    routing.assignments[block.block_id] = (
                        "selector_blend[" + primary_assignment + "," + proxy_candidate + "]"
                    )
        candidate_shrinkage_block_assignments: dict[str, dict[str, Any]] = {}
        candidate_shrinkage_activated: set[str] = set()
        candidate_shrinkage_primary_candidates: set[str] = set()
        shrinkage_candidate = self.forecast_consensus_candidate_shrinkage_id
        shrinkage_weight = self.forecast_consensus_candidate_shrinkage_weight
        shrinkage_fallback_candidate = self.forecast_consensus_candidate_shrinkage_fallback_id
        shrinkage_fallback_weight = self.forecast_consensus_candidate_shrinkage_fallback_weight
        if shrinkage_candidate is not None and shrinkage_weight > 0.0:
            for block in plan.blocks:
                if not _block_visible_to_forecaster(block, plan.forecast_spec):
                    continue
                primary_valid = bool(
                    shrinkage_candidate in route_candidates
                    and (block.block_id, shrinkage_candidate) not in invalid
                )
                fallback_valid = bool(
                    not primary_valid
                    and shrinkage_fallback_candidate is not None
                    and shrinkage_fallback_candidate in route_candidates
                    and (block.block_id, shrinkage_fallback_candidate) not in invalid
                )
                applied_candidate = (
                    shrinkage_candidate
                    if primary_valid
                    else shrinkage_fallback_candidate
                    if fallback_valid
                    else None
                )
                applied_weight = (
                    shrinkage_weight
                    if primary_valid
                    else shrinkage_fallback_weight
                    if fallback_valid
                    else 0.0
                )
                primary_assignment = routing.assignments[block.block_id]
                if applied_candidate is not None:
                    if primary_assignment in self.imputer_registry:
                        candidate_shrinkage_primary_candidates.add(primary_assignment)
                    selector = (
                        block.batch_index,
                        slice(block.start, block.end),
                        block.channel,
                    )
                    completed[selector] = (1.0 - applied_weight) * completed[
                        selector
                    ] + applied_weight * route_candidates[applied_candidate].values[selector]
                    routing.assignments[block.block_id] = (
                        "candidate_shrink[" + primary_assignment + "," + applied_candidate + "]"
                    )
                    candidate_shrinkage_activated.add(applied_candidate)
                candidate_shrinkage_block_assignments[block.block_id] = {
                    "primary_assignment": primary_assignment,
                    "primary_candidate_id": shrinkage_candidate,
                    "fallback_candidate_id": shrinkage_fallback_candidate,
                    "candidate_id": applied_candidate or shrinkage_candidate,
                    "weight": applied_weight,
                    "applied": applied_candidate is not None,
                    "used_fallback": fallback_valid,
                    "reason": (
                        None if applied_candidate is not None else "candidate_not_native_valid"
                    ),
                }
        completed[batch.observed_mask] = batch.values[batch.observed_mask]
        if not np.isfinite(completed).all():
            raise RuntimeError("assembled imputation contains non-finite values")
        routing.fallback_blocks = tuple(fallback_blocks)
        routing.fallback_records = fallback_records
        routing.metadata["fallback_records"] = fallback_records
        routing.metadata["correlation_source"] = plan.correlation_source
        routing.metadata["pairwise_skipped_for_scale"] = pairwise_skipped_for_scale
        routing.metadata["block_graph_edges"] = len(plan.graph.edges)
        routing.metadata["forecast_irrelevant_assignments"] = dict(forecast_irrelevant_assignments)
        routing.metadata["forecast_irrelevant_block_count"] = len(forecast_irrelevant_assignments)
        routing.metadata["proxy_global_mae"] = dict(proxy_error_scores)
        routing.metadata["proxy_outlier_threshold"] = proxy_outlier_threshold
        routing.metadata["proxy_rejected_candidates"] = sorted(proxy_rejected_candidates)
        routing.metadata["proxy_blend_weight"] = proxy_blend_weight
        routing.metadata["proxy_blend_min_relative_margin"] = proxy_blend_min_relative_margin
        routing.metadata["pseudo_weight_calibration"] = (
            self.forecast_consensus_pseudo_weight_calibration
        )
        routing.metadata["proxy_blend_block_assignments"] = proxy_blend_block_assignments
        routing.metadata["candidate_shrinkage_id"] = shrinkage_candidate
        routing.metadata["candidate_shrinkage_weight"] = shrinkage_weight
        routing.metadata["candidate_shrinkage_fallback_id"] = shrinkage_fallback_candidate
        routing.metadata["candidate_shrinkage_fallback_weight"] = shrinkage_fallback_weight
        routing.metadata["candidate_shrinkage_block_assignments"] = (
            candidate_shrinkage_block_assignments
        )
        routing.metadata["evidence_blend"] = self._evidence_weights(plan.forecast_spec)
        routing.metadata["candidate_anchor_calibration"] = anchor_diagnostics
        routing.metadata["forecast_consensus"] = consensus_diagnostics
        routing.metadata["ensemble_block_assignments"] = {
            block_id: list(candidate_ids)
            for block_id, candidate_ids in ensemble_block_assignments.items()
        }
        routing.metadata["ensemble_block_weights"] = ensemble_block_weights
        routing.metadata["candidate_anchor_block_assignments"] = dict(anchor_block_assignments)
        routing.metadata["artifact_load_failures"] = dict(plan.artifact_load_failures)
        routing.shortlist = plan.shortlist
        routing.activated_candidates = tuple(
            sorted(
                {
                    candidate
                    for candidate in routing.assignments.values()
                    if candidate in self.imputer_registry
                }
                | {
                    candidate
                    for candidates_for_block in ensemble_block_assignments.values()
                    for candidate in candidates_for_block
                }
                | {
                    str(record["proxy_candidate"])
                    for record in proxy_blend_block_assignments.values()
                    if float(record["proxy_weight"]) > 0.0
                }
                | proxy_blend_primary_candidates
                | candidate_shrinkage_activated
                | candidate_shrinkage_primary_candidates
            )
        )
        routing.candidate_costs = {
            candidate_id: float(costs[candidate_id])
            for candidate_id in route_candidates
            if candidate_id in costs
        }
        routing.activated_cost = float(
            sum(
                routing.candidate_costs.get(candidate_id, 0.0)
                for candidate_id in routing.activated_candidates
            )
        )
        routing.cost_energy = float(self.cost_weight * routing.activated_cost)
        routing.total_energy = float(routing.risk_energy + routing.cost_energy)
        return FAISResult(
            values=completed[0],
            routing=routing,
            candidates=route_candidates,
            observed_mask=mask,
            metadata={"period": plan.period, "block_count": len(plan.blocks)},
        )

    def impute(
        self,
        item: TimeSeriesItem,
        observed_mask: np.ndarray,
        forecast_spec: ForecastSpec,
        budget: BudgetSpec | None = None,
        *,
        seed: int = 20260710,
    ) -> FAISResult:
        """Execute the legacy episode-major path through reusable route phases."""

        plan = self.prepare_route(
            item,
            observed_mask,
            forecast_spec,
            budget,
            seed=seed,
        )
        if plan.is_noop:
            return self.finish_route(plan, {}, {})
        candidates = self.candidate_runner.run_many(
            plan.shortlist,
            plan.batch,
            self.imputer_artifacts,
            seed=seed,
            params=self.candidate_params,
            budget=plan.budget,
        )
        pseudo_candidates: dict[str, CandidateResult] = {}
        if self.router is not None:
            if plan.pseudo_batch is None:  # pragma: no cover - prepare invariant
                raise RuntimeError("router route plan has no pseudo batch")
            pseudo_candidates = self.candidate_runner.run_many(
                plan.shortlist,
                plan.pseudo_batch,
                self.imputer_artifacts,
                seed=seed,
                params=self.candidate_params,
                budget=plan.budget,
                runtime_already_spent=sum(result.runtime_seconds for result in candidates.values()),
            )
        backtest_candidates: dict[str, CandidateResult] = {}
        if plan.backtest_batch is not None:
            backtest_candidates = self.candidate_runner.run_many(
                plan.shortlist,
                plan.backtest_batch,
                self.imputer_artifacts,
                seed=seed,
                params=self.candidate_params,
                budget=plan.budget,
                runtime_already_spent=(
                    sum(result.runtime_seconds for result in candidates.values())
                    + sum(result.runtime_seconds for result in pseudo_candidates.values())
                ),
            )
        return self.finish_route(
            plan,
            candidates,
            pseudo_candidates,
            backtest_candidates=backtest_candidates,
        )
