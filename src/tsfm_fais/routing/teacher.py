"""Counterfactual TSFM labels for block-level candidate ranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from tsfm_fais.contracts import (
    CandidateResult,
    ForecastResult,
    ForecastSpec,
    MissingBlock,
    SeriesBatch,
)

from .graph import BlockGraph
from .metrics import block_candidate_losses


@dataclass(frozen=True)
class TeacherWeights:
    """Weights for reconstruction, forecast, resource, and validity evidence."""

    reconstruction: float = 1.0
    forecast: float = 1.0
    runtime: float = 0.01
    memory: float = 0.001
    invalid_native: float = 1000.0

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("teacher weights must be finite and non-negative")


@dataclass(frozen=True)
class TeacherTargets:
    unary: Mapping[tuple[str, str], float]
    rankings: Mapping[str, tuple[str, ...]]
    pairwise: Mapping[tuple[str, str, str, str], float] = field(default_factory=dict)
    reconstruction: Mapping[tuple[str, str], float] = field(default_factory=dict)


def _lookup_forecast_loss(
    forecast_losses: Mapping[object, object] | None,
    block_id: str,
    candidate_id: str,
) -> float:
    if not forecast_losses:
        return 0.0
    pair_key = (block_id, candidate_id)
    if pair_key in forecast_losses:
        return float(cast(Any, forecast_losses[pair_key]))
    nested = forecast_losses.get(block_id)
    if isinstance(nested, Mapping) and candidate_id in nested:
        return float(cast(Any, nested[candidate_id]))
    return 0.0


def _teacher_pairwise(
    graph: BlockGraph | None,
    clean: np.ndarray,
    candidates: Mapping[str, CandidateResult],
) -> dict[tuple[str, str, str, str], float]:
    if graph is None:
        return {}
    blocks = {block.block_id: block for block in graph.blocks}
    pairwise: dict[tuple[str, str, str, str], float] = {}
    for edge in graph.edges:
        left = blocks[edge.left]
        right = blocks[edge.right]
        left_selector = (left.batch_index, slice(left.start, left.end), left.channel)
        right_selector = (right.batch_index, slice(right.start, right.end), right.channel)
        clean_relation = float(np.mean(clean[left_selector]) - np.mean(clean[right_selector]))
        for left_id, left_candidate in candidates.items():
            left_mean = float(np.mean(left_candidate.values[left_selector]))
            for right_id, right_candidate in candidates.items():
                right_mean = float(np.mean(right_candidate.values[right_selector]))
                relation = left_mean - right_mean
                pairwise[(edge.left, left_id, edge.right, right_id)] = float(
                    abs(relation - clean_relation)
                )
    return pairwise


class RoutingTeacher:
    """Build finite routing targets from pseudo-missing reconstruction evidence."""

    def __init__(
        self,
        weights: TeacherWeights | None = None,
        reconstruction_metric: str = "mae",
    ) -> None:
        if reconstruction_metric not in {"mae", "mse", "rmse"}:
            raise ValueError("reconstruction_metric must be mae, mse, or rmse")
        self.weights = weights or TeacherWeights()
        self.reconstruction_metric = reconstruction_metric

    def build_targets(
        self,
        batch: SeriesBatch,
        clean_values: np.ndarray,
        blocks: Sequence[MissingBlock],
        candidates: Mapping[str, CandidateResult],
        *,
        forecast_losses: Mapping[object, object] | None = None,
        graph: BlockGraph | None = None,
    ) -> TeacherTargets:
        clean = np.asarray(clean_values, dtype=float)
        if clean.shape != batch.shape or not np.all(np.isfinite(clean)):
            raise ValueError("clean_values must be finite and match the batch")
        if not blocks or not candidates:
            raise ValueError("teacher target construction requires blocks and candidates")
        reconstruction = block_candidate_losses(
            clean,
            blocks,
            candidates,
            metric=self.reconstruction_metric,
        )
        unary: dict[tuple[str, str], float] = {}
        rankings: dict[str, tuple[str, ...]] = {}
        for block in blocks:
            selector = (block.batch_index, slice(block.start, block.end), block.channel)
            for candidate_id, candidate in candidates.items():
                native_coverage = float(np.mean(candidate.native_valid_mask[selector]))
                unary[(block.block_id, candidate_id)] = float(
                    self.weights.reconstruction
                    * reconstruction[(block.block_id, candidate_id)]
                    + self.weights.forecast
                    * _lookup_forecast_loss(
                        forecast_losses, block.block_id, candidate_id
                    )
                    + self.weights.runtime
                    * np.log1p(max(0.0, float(candidate.runtime_seconds)))
                    + self.weights.memory
                    * np.log1p(max(0.0, float(candidate.peak_memory_bytes)) / (1024**2))
                    + self.weights.invalid_native * (1.0 - native_coverage)
                )
            rankings[block.block_id] = tuple(
                candidate_id
                for _, candidate_id in sorted(
                    (unary[(block.block_id, candidate_id)], candidate_id)
                    for candidate_id in candidates
                )
            )
        return TeacherTargets(
            unary=unary,
            rankings=rankings,
            pairwise=_teacher_pairwise(graph, clean, candidates),
            reconstruction=reconstruction,
        )

    @staticmethod
    def fit_arrays(
        feature_keys: Sequence[tuple[str, str]],
        targets: TeacherTargets,
    ) -> np.ndarray:
        try:
            return np.asarray([targets.unary[key] for key in feature_keys], dtype=float)
        except KeyError as exc:
            raise ValueError(f"missing teacher target for {exc.args[0]}") from exc


@dataclass(frozen=True)
class TeacherLabel:
    episode_id: str
    block_id: str
    candidate_id: str
    forecast_loss: float
    clean_loss: float
    degradation: float


@dataclass(frozen=True)
class CoherenceAdjustedTarget:
    """Auditable block risk after allocating a full-candidate forecast effect."""

    local_marginal: float
    full_candidate_loss: float | None
    global_marginal_per_block: float
    coherence_adjustment: float
    routing_target: float


def coherence_adjusted_targets(
    labels: Sequence[TeacherLabel],
    full_candidate_losses: Mapping[str, float],
    *,
    anchor_loss: float,
    visible_block_count: int,
) -> dict[tuple[str, str], CoherenceAdjustedTarget]:
    """Combine local counterfactuals with each candidate's full-context loss.

    The candidate-wide residual is centered by the mean observed local
    marginal.  Consequently, when a candidate has labels for every visible
    block, its adjusted block risks sum exactly to its full-context loss delta
    relative to the anchor.
    """

    if visible_block_count < 1:
        raise ValueError("visible_block_count must be positive")
    if not np.isfinite(anchor_loss):
        raise ValueError("anchor_loss must be finite")
    local_by_candidate: dict[str, list[float]] = {}
    for label in labels:
        local = float(label.forecast_loss - anchor_loss)
        if not np.isfinite(local):
            raise ValueError("teacher label marginal must be finite")
        local_by_candidate.setdefault(label.candidate_id, []).append(local)

    adjustments: dict[str, tuple[float | None, float, float]] = {}
    for candidate_id, local_values in local_by_candidate.items():
        mean_local = float(np.mean(local_values))
        full_loss = full_candidate_losses.get(candidate_id)
        if full_loss is None:
            adjustments[candidate_id] = (None, mean_local, 0.0)
            continue
        full_loss = float(full_loss)
        if not np.isfinite(full_loss):
            raise ValueError("full candidate loss must be finite")
        global_per_block = (full_loss - anchor_loss) / visible_block_count
        adjustments[candidate_id] = (
            full_loss,
            float(global_per_block),
            float(global_per_block - mean_local),
        )

    targets: dict[tuple[str, str], CoherenceAdjustedTarget] = {}
    for label in labels:
        local = float(label.forecast_loss - anchor_loss)
        full_loss, global_per_block, adjustment = adjustments[label.candidate_id]
        targets[(label.block_id, label.candidate_id)] = CoherenceAdjustedTarget(
            local_marginal=local,
            full_candidate_loss=full_loss,
            global_marginal_per_block=global_per_block,
            coherence_adjustment=adjustment,
            routing_target=float(local + adjustment),
        )
    return targets


ForecastCallable = Callable[[np.ndarray, ForecastSpec], ForecastResult]


def replace_block(base: np.ndarray, candidate: np.ndarray, block: MissingBlock) -> np.ndarray:
    result = np.asarray(base, dtype=float).copy()
    result[
        block.batch_index,
        block.start : block.end,
        block.channel,
    ] = candidate[
        block.batch_index,
        block.start : block.end,
        block.channel,
    ]
    return result


class TeacherBuilder:
    def __init__(
        self,
        forecast: ForecastCallable,
        seasonality: int = 1,
        *,
        mase_scales: Mapping[int, float] | None = None,
    ):
        self.forecast = forecast
        self.seasonality = seasonality
        self.mase_scales = None if mase_scales is None else dict(mase_scales)
        if self.mase_scales is not None and (
            not self.mase_scales
            or any(not np.isfinite(value) or value <= 0 for value in self.mase_scales.values())
        ):
            raise ValueError("teacher MASE scales must be nonempty, positive, and finite")

    def _scales(
        self,
        context: np.ndarray,
        targets: Sequence[int],
    ) -> dict[int, float]:
        if self.mase_scales is not None:
            if set(targets).difference(self.mase_scales):
                raise ValueError("teacher MASE scales must cover all forecast targets")
            return {target: self.mase_scales[target] for target in targets}
        # R2 replay retains its original context-based objective. New experiments
        # supply explicit training-prefix scales for every teacher request.
        values = np.asarray(context, dtype=float)
        scales: dict[int, float] = {}
        for target in targets:
            history = values[:, :, target]
            requested_lag = max(1, int(self.seasonality))
            lag = requested_lag if history.shape[-1] > requested_lag else 1
            differences = np.abs(history[..., lag:] - history[..., :-lag])
            scales[target] = max(
                float(np.mean(differences)) if differences.size else 0.0,
                1e-8,
            )
        return scales

    def _loss(
        self,
        context: np.ndarray,
        future: np.ndarray,
        spec: ForecastSpec,
        scales: Mapping[int, float] | None = None,
    ) -> float:
        return float(np.mean(self._losses(context, future, spec, scales)))

    def _losses(
        self,
        context: np.ndarray,
        future: np.ndarray,
        spec: ForecastSpec,
        scales: Mapping[int, float] | None = None,
    ) -> np.ndarray:
        """Return one macro-MASE value per context row.

        Keeping the row dimension makes counterfactual contexts batchable while
        preserving the exact scalar objective used by ``_loss``.
        """

        values = np.asarray(context, dtype=float)
        truth_values = np.asarray(future, dtype=float)
        if truth_values.shape[0] == 1 and values.shape[0] != 1:
            truth_values = np.repeat(truth_values, values.shape[0], axis=0)
        if truth_values.shape[0] != values.shape[0]:
            raise ValueError("future batch size must match forecast contexts")
        forecast = self.forecast(values, spec)
        targets = forecast.target_indices
        truth = truth_values[:, :, targets]
        fixed_scales = dict(scales or self._scales(values, targets))
        errors = np.mean(np.abs(truth - forecast.point), axis=1)
        scale_array = np.asarray([fixed_scales[target] for target in targets])
        return np.mean(errors / scale_array[None, :], axis=1)

    def unary_labels(
        self,
        episode_id: str,
        clean_context: np.ndarray,
        clean_future: np.ndarray,
        anchor: np.ndarray,
        blocks: Sequence[MissingBlock],
        candidates: Mapping[str, CandidateResult],
        spec: ForecastSpec,
        candidate_filter: Callable[
            [MissingBlock, str, CandidateResult], bool
        ]
        | None = None,
    ) -> list[TeacherLabel]:
        targets = spec.target_indices or tuple(range(clean_context.shape[2]))
        scales = self._scales(clean_context, targets)
        clean_loss = self._loss(clean_context, clean_future, spec, scales)
        labels: list[TeacherLabel] = []
        for block in blocks:
            for candidate_id, candidate in candidates.items():
                if candidate_filter is not None and not candidate_filter(
                    block, candidate_id, candidate
                ):
                    continue
                if not candidate.native_valid_mask[
                    block.batch_index, block.start : block.end, block.channel
                ].all():
                    continue
                hybrid = replace_block(anchor, candidate.values, block)
                loss = self._loss(hybrid, clean_future, spec, scales)
                labels.append(
                    TeacherLabel(
                        episode_id=episode_id,
                        block_id=block.block_id,
                        candidate_id=candidate_id,
                        forecast_loss=loss,
                        clean_loss=clean_loss,
                        degradation=loss - clean_loss,
                    )
                )
        return labels

    def unary_labels_batched(
        self,
        episode_id: str,
        clean_context: np.ndarray,
        clean_future: np.ndarray,
        anchor: np.ndarray,
        blocks: Sequence[MissingBlock],
        candidates: Mapping[str, CandidateResult],
        spec: ForecastSpec,
        candidate_filter: Callable[
            [MissingBlock, str, CandidateResult], bool
        ]
        | None = None,
    ) -> tuple[list[TeacherLabel], float, float]:
        """Evaluate clean, anchor, and all unary counterfactuals in one call."""

        clean = np.asarray(clean_context, dtype=float)
        anchor_values = np.asarray(anchor, dtype=float)
        if clean.shape != anchor_values.shape:
            raise ValueError("clean context and anchor must have the same shape")
        targets = spec.target_indices or tuple(range(clean.shape[2]))
        scales = self._scales(clean, targets)
        contexts = [clean, anchor_values]
        keys: list[tuple[MissingBlock, str]] = []
        for block in blocks:
            for candidate_id, candidate in candidates.items():
                if candidate_filter is not None and not candidate_filter(
                    block, candidate_id, candidate
                ):
                    continue
                if not candidate.native_valid_mask[
                    block.batch_index, block.start : block.end, block.channel
                ].all():
                    continue
                contexts.append(replace_block(anchor_values, candidate.values, block))
                keys.append((block, candidate_id))
        batch_size = clean.shape[0]
        repeated_future = np.concatenate([clean_future] * len(contexts), axis=0)
        losses = self._losses(
            np.concatenate(contexts, axis=0),
            repeated_future,
            spec,
            scales,
        ).reshape(len(contexts), batch_size).mean(axis=1)
        clean_loss = float(losses[0])
        anchor_loss = float(losses[1])
        labels = [
            TeacherLabel(
                episode_id=episode_id,
                block_id=block.block_id,
                candidate_id=candidate_id,
                forecast_loss=float(loss),
                clean_loss=clean_loss,
                degradation=float(loss - clean_loss),
            )
            for (block, candidate_id), loss in zip(
                keys,
                losses[2:],
                strict=True,
            )
        ]
        return labels, clean_loss, anchor_loss

    def candidate_losses_batched(
        self,
        clean_context: np.ndarray,
        clean_future: np.ndarray,
        candidates: Mapping[str, CandidateResult],
        spec: ForecastSpec,
    ) -> dict[str, float]:
        """Evaluate each complete candidate context in one forecast call."""

        if not candidates:
            return {}
        clean = np.asarray(clean_context, dtype=float)
        targets = spec.target_indices or tuple(range(clean.shape[2]))
        scales = self._scales(clean, targets)
        candidate_ids = tuple(candidates)
        contexts: list[np.ndarray] = []
        for candidate_id in candidate_ids:
            values = np.asarray(candidates[candidate_id].values, dtype=float)
            if values.shape != clean.shape or not np.isfinite(values).all():
                raise ValueError(
                    "complete candidate values must be finite and match clean_context"
                )
            contexts.append(values)
        batch_size = clean.shape[0]
        repeated_future = np.concatenate([clean_future] * len(contexts), axis=0)
        losses = self._losses(
            np.concatenate(contexts, axis=0),
            repeated_future,
            spec,
            scales,
        ).reshape(len(contexts), batch_size).mean(axis=1)
        return {
            candidate_id: float(loss)
            for candidate_id, loss in zip(candidate_ids, losses, strict=True)
        }

    def pair_interaction(
        self,
        clean_future: np.ndarray,
        anchor: np.ndarray,
        left_block: MissingBlock,
        right_block: MissingBlock,
        left_candidate: CandidateResult,
        right_candidate: CandidateResult,
        spec: ForecastSpec,
        scale_context: np.ndarray | None = None,
    ) -> float:
        targets = spec.target_indices or tuple(range(anchor.shape[2]))
        scales = self._scales(
            anchor if scale_context is None else scale_context,
            targets,
        )
        base_loss = self._loss(anchor, clean_future, spec, scales)
        left_values = replace_block(anchor, left_candidate.values, left_block)
        right_values = replace_block(anchor, right_candidate.values, right_block)
        both = replace_block(left_values, right_candidate.values, right_block)
        return (
            self._loss(both, clean_future, spec, scales)
            - self._loss(left_values, clean_future, spec, scales)
            - self._loss(right_values, clean_future, spec, scales)
            + base_loss
        )

    def pair_interactions_batched(
        self,
        clean_future: np.ndarray,
        anchor: np.ndarray,
        requests: Sequence[
            tuple[MissingBlock, MissingBlock, CandidateResult, CandidateResult]
        ],
        spec: ForecastSpec,
        *,
        anchor_loss: float,
        scale_context: np.ndarray | None = None,
    ) -> tuple[float, ...]:
        """Evaluate all left/right/both pair counterfactuals in one call."""

        if not requests:
            return ()
        anchor_values = np.asarray(anchor, dtype=float)
        targets = spec.target_indices or tuple(range(anchor_values.shape[2]))
        scales = self._scales(
            anchor_values if scale_context is None else scale_context,
            targets,
        )
        contexts: list[np.ndarray] = []
        for left, right, left_candidate, right_candidate in requests:
            left_values = replace_block(anchor_values, left_candidate.values, left)
            right_values = replace_block(anchor_values, right_candidate.values, right)
            both = replace_block(left_values, right_candidate.values, right)
            contexts.extend((left_values, right_values, both))
        batch_size = anchor_values.shape[0]
        repeated_future = np.concatenate([clean_future] * len(contexts), axis=0)
        losses = self._losses(
            np.concatenate(contexts, axis=0),
            repeated_future,
            spec,
            scales,
        ).reshape(len(contexts), batch_size).mean(axis=1)
        interactions = []
        for index in range(len(requests)):
            left_loss, right_loss, both_loss = losses[3 * index : 3 * index + 3]
            interactions.append(
                float(both_loss - left_loss - right_loss + anchor_loss)
            )
        return tuple(interactions)
