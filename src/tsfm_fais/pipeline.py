"""End-to-end block-wise imputer selection and assembly."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
    merge_features,
    pair_features,
    proxy_features,
)
from tsfm_fais.routing.graph import BlockGraph
from tsfm_fais.routing.models import RouterBundle
from tsfm_fais.routing.solver import beam_search, greedy_shortlist


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
    best: tuple[
        tuple[float, tuple[tuple[str, str], ...]],
        dict[str, str],
        float,
        float,
    ] | None = None
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
        candidate_costs={
            candidate: float(costs.get(candidate, 1.0)) for candidate in candidates
        },
        activated_cost=activated_cost,
        risk_energy=unary_energy,
        cost_energy=cost_energy,
        metadata={"solver": "pairwise_free_subset"},
    )


def _native_block_is_valid(result: CandidateResult, block) -> bool:
    native = result.native_valid_mask[
        block.batch_index, block.start : block.end, block.channel
    ]
    return (
        result.status not in {CandidateStatus.FAILED, CandidateStatus.UNAVAILABLE}
        and bool(native.all())
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
        max_pairwise_blocks: int = 512,
    ) -> None:
        self.config = config
        self.router = router
        self.imputer_registry = imputer_registry or DEFAULT_REGISTRY
        self.candidate_runner = CandidateRunner(self.imputer_registry)
        self._artifacts_supplied = imputer_artifacts is not None
        self.imputer_artifacts = dict(imputer_artifacts or {})
        self.imputer_artifact_root = (
            None
            if imputer_artifact_root is None
            else Path(imputer_artifact_root).resolve()
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
            candidate_id: dict(params)
            for candidate_id, params in (candidate_params or {}).items()
        }
        self.training_medians = (
            None
            if training_medians is None
            else np.asarray(training_medians, dtype=float).reshape(-1)
        )
        self.training_correlation = (
            None
            if training_correlation is None
            else np.asarray(training_correlation, dtype=float)
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
        configured_cost_weight = 0.05
        if config is not None:
            from tsfm_fais.config import load_yaml
            from tsfm_fais.registry_configs import RouterConfig

            router_config = RouterConfig.model_validate(
                load_yaml(config.registries.router_config)
            )
            configured_internal = router_config.fallback.internal
            configured_tail = router_config.fallback.tail
            configured_forced = router_config.forced_candidates
            configured_shortlist_size = router_config.shortlist_size
            configured_pseudo_blocks = router_config.pseudo_blocks
            configured_beam_width = router_config.beam_width
            configured_beta = router_config.beta
            configured_cost_weight = router_config.cost_weight
        if router is not None:
            configured_beta = float(router.metadata.get("beta", configured_beta))
            configured_cost_weight = float(
                router.metadata.get("cost_weight", configured_cost_weight)
            )
        self.beta = float(configured_beta if beta is None else beta)
        self.cost_weight = float(
            configured_cost_weight if cost_weight is None else cost_weight
        )
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
        self.shortlist_size = int(configured_shortlist_size)
        self.pseudo_blocks = int(configured_pseudo_blocks)
        self.beam_width = int(
            configured_beam_width if beam_width is None else beam_width
        )
        if max_pairwise_blocks < 1:
            raise ValueError("max_pairwise_blocks must be positive")
        self.max_pairwise_blocks = int(max_pairwise_blocks)

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
                raise ValueError(
                    "item.metadata['dataset_id'] is required to load fitted imputers"
                )
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
        if (
            missing
            and self.imputer_artifact_root is None
            and not self._artifacts_supplied
        ):
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
                    ~batch.observed_mask[
                        block.batch_index, block.start : block.end, :
                    ]
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
                        candidate_features(spec, forecast_spec),
                    )
                )
                keys.append((block.block_id, candidate_id))
        predictions = self.router.prior.predict(_feature_matrix(rows, self.router.feature_names))
        return _ranker_risks(
            keys,
            predictions,
            float(self.router.metadata.get("unary_risk_scale", 1.0)),
        )

    def _pseudo_batch(self, batch: SeriesBatch, seed: int, max_blocks: int = 8) -> SeriesBatch:
        rng = np.random.default_rng(seed)
        mask = batch.observed_mask.copy()
        length = batch.shape[1]
        available = np.argwhere(mask[0])
        if len(available) == 0:
            return batch
        block_length = max(1, min(length // 20, 8))
        occupied_time = np.zeros(length, dtype=bool)
        for _ in range(max_blocks):
            channel = int(rng.integers(0, batch.shape[2]))
            starts = [
                start
                for start in range(0, length - block_length + 1)
                if mask[0, start : start + block_length, channel].all()
                and not occupied_time[start : start + block_length].any()
            ]
            if not starts:
                continue
            start = int(rng.choice(starts))
            mask[0, start : start + block_length, channel] = False
            occupied_time[start : start + block_length] = True
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
    ) -> dict[tuple[str, str], float]:
        if self.router is None:
            return {key: value for key, value in fallback.items() if key[1] in shortlist}
        rows: list[dict[str, float]] = []
        keys: list[tuple[str, str]] = []
        # Original missing positions have no ground truth. Mark them as
        # excluded so proxy errors are computed only on newly hidden values.
        proxy_mask = pseudo_batch.observed_mask | ~batch.observed_mask
        for block in blocks:
            for candidate_id in shortlist:
                spec = self.imputer_registry.get_spec(candidate_id)
                proxy = proxy_features(
                    pseudo_candidates[candidate_id],
                    batch.values,
                    proxy_mask,
                )
                rows.append(
                    merge_features(
                        block_features(batch, block, period),
                        candidate_features(spec, forecast_spec),
                        proxy,
                    )
                )
                keys.append((block.block_id, candidate_id))
        predictions = self.router.unary.predict(_feature_matrix(rows, self.router.feature_names))
        return _ranker_risks(
            keys,
            predictions,
            float(self.router.metadata.get("unary_risk_scale", 1.0)),
        )

    def _pairwise_risk(
        self,
        batch: SeriesBatch,
        graph,
        shortlist: tuple[str, ...],
        candidates: Mapping[str, CandidateResult],
    ) -> dict[tuple[str, str, str, str], float]:
        feature_names = tuple(getattr(self.router, "pair_feature_names", ())) if self.router else ()
        if not feature_names or self.router is None or self.router.pairwise.model is None:
            return {}
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
                    keys.append(
                        (left_id, right_id, left_candidate, right_candidate)
                    )
        if not rows:
            return {}
        predictions = self.router.pairwise.predict(
            _feature_matrix(rows, feature_names)
        )
        pair_scale = max(
            float(self.router.metadata.get("pair_risk_scale", 1.0)),
            1e-6,
        )
        predictions = np.clip(predictions, -10.0 * pair_scale, 10.0 * pair_scale)
        return {
            key: float(value)
            for key, value in zip(keys, predictions, strict=True)
        }

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
                return np.asarray(result.values[selector], dtype=float), fallback_id, tuple(attempts)
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
                training_medians=item.metadata.get(
                    "training_medians", self.training_medians
                ),
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
        training_medians = item.metadata.get(
            "training_medians", self.training_medians
        )

        supplied_correlation = item.metadata.get(
            "training_correlation", self.training_correlation
        )
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
                        and (
                            spec.device == "any"
                            or spec.device in resolved_budget.allowed_devices
                        )
                        and (not spec.requires_period or period)
                        and (spec.fit_scope == "none" or spec.imputer_id in available)
                    )
                )
            )
        )
        if not candidate_ids:
            raise RuntimeError("no imputation candidate is available under the budget")
        costs = {
            spec.imputer_id: float(spec.cost_tier)
            for spec in self.imputer_registry.specs()
        }
        prior_unary = self._heuristic_unary(
            batch, blocks, candidate_ids, forecast_spec, period
        )
        shortlist = greedy_shortlist(
            blocks,
            candidate_ids,
            prior_unary,
            costs,
            max_candidates=min(resolved_budget.max_candidates, self.shortlist_size),
            forced=self.forced_candidates,
        )
        pseudo_batch = (
            None
            if self.router is None
            else self._pseudo_batch(batch, seed, max_blocks=self.pseudo_blocks)
        )
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
        route_candidates = self._injected_results(
            plan.shortlist, candidates, batch, "actual"
        )
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
                key: value
                for key, value in plan.prior_unary.items()
                if key[1] in plan.shortlist
            }
        else:
            if plan.pseudo_batch is None:
                raise ValueError("router route plan is missing its pseudo batch")
            routed_pseudo = self._injected_results(
                plan.shortlist,
                pseudo_candidates or {},
                plan.pseudo_batch,
                "pseudo",
            )
            unary = self._refined_unary(
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
                batch, plan.graph, plan.shortlist, route_candidates
            )
        )
        invalid: set[tuple[str, str]] = set()
        for block in plan.blocks:
            for candidate_id, result in route_candidates.items():
                candidate_spec = self.imputer_registry.get_spec(candidate_id)
                capability_mismatch = (
                    (block.end == batch.shape[1] and not candidate_spec.supports_tail)
                    or (candidate_spec.requires_period and not plan.period)
                )
                if capability_mismatch or not _native_block_is_valid(result, block):
                    invalid.add((block.block_id, candidate_id))

        # Keep safe completion outside candidate scoring. A synthetic fallback
        # state is available only when every shortlisted candidate failed for
        # a block; the router therefore never interprets fallback values as a
        # successful candidate result.
        fallback_id = "__fallback__"
        solver_candidates = list(plan.shortlist)
        no_native_candidate = {
            block.block_id
            for block in plan.blocks
            if all(
                (block.block_id, candidate_id) in invalid
                for candidate_id in plan.shortlist
            )
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
        for block in plan.blocks:
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
                completed[
                    block.batch_index, block.start : block.end, block.channel
                ] = fallback_values
                routing.assignments[block.block_id] = fallback_name
                fallback_blocks.append(block.block_id)
                fallback_records[block.block_id] = {
                    "selected": fallback_name,
                    "attempts": attempts,
                    "kind": "tail" if block.end == batch.shape[1] else "internal",
                }
                continue
            result = route_candidates[candidate_id]
            completed[
                block.batch_index, block.start : block.end, block.channel
            ] = result.values[
                block.batch_index, block.start : block.end, block.channel
            ]
        completed[batch.observed_mask] = batch.values[batch.observed_mask]
        if not np.isfinite(completed).all():
            raise RuntimeError("assembled imputation contains non-finite values")
        routing.fallback_blocks = tuple(fallback_blocks)
        routing.fallback_records = fallback_records
        routing.metadata["fallback_records"] = fallback_records
        routing.metadata["correlation_source"] = plan.correlation_source
        routing.metadata["pairwise_skipped_for_scale"] = pairwise_skipped_for_scale
        routing.metadata["block_graph_edges"] = len(plan.graph.edges)
        routing.metadata["artifact_load_failures"] = dict(
            plan.artifact_load_failures
        )
        routing.shortlist = plan.shortlist
        routing.activated_candidates = tuple(
            sorted(
                {
                    candidate
                    for candidate in routing.assignments.values()
                    if candidate in self.imputer_registry
                }
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
                runtime_already_spent=sum(
                    result.runtime_seconds for result in candidates.values()
                ),
            )
        return self.finish_route(plan, candidates, pseudo_candidates)
