"""Whole-sequence imputer-selection pipelines for comparison baselines."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    RoutingResult,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.pipeline import BlockwiseFAIS, FAISResult, RoutePlan
from tsfm_fais.routing.blocks import detect_missing_blocks
from tsfm_fais.routing.graph import BlockGraph
from tsfm_fais.routing.models import RouterBundle
from tsfm_fais.routing.sequence_features import sequence_meta_features

_RANDOM_SERIES_METHODS = frozenset({"random_valid_series", "random_valid_block"})


def _stable_random_index(seed: int, candidate_ids: tuple[str, ...]) -> int:
    payload = f"{int(seed)}|random_valid_series|{'|'.join(candidate_ids)}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % len(candidate_ids)


def _model_feature_vector(plan: RoutePlan, feature_names: tuple[str, ...]) -> np.ndarray:
    values = sequence_meta_features(plan.batch, period=plan.period)
    return np.asarray([float(values.get(name, 0.0)) for name in feature_names], dtype=float)


def _candidate_is_whole_sequence_valid(
    plan: RoutePlan,
    candidate_id: str,
    result: CandidateResult,
    pipeline: BlockwiseFAIS,
) -> bool:
    if result.status in {CandidateStatus.FAILED, CandidateStatus.UNAVAILABLE}:
        return False
    missing = ~np.asarray(plan.batch.observed_mask, dtype=bool)
    if not missing.any() or not bool(
        np.asarray(result.native_valid_mask, dtype=bool)[missing].all()
    ):
        return False
    if any(block.end == plan.batch.shape[1] for block in plan.blocks):
        if not pipeline.imputer_registry.get_spec(candidate_id).supports_tail:
            return False
    return True


class WholeSeriesSelectorFAIS(BlockwiseFAIS):
    """Select or combine complete candidate tensors once per corrupted episode."""

    uses_registry_candidates = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.router is None:
            raise ValueError("whole-series selection requires a trained selector bundle")
        self.shortlist_size = max(self.shortlist_size, len(self._selector_router.candidate_ids))
        self.fallback_internal = ()
        self.fallback_tail = ()
        self._pending_online_feedback: dict[str, Any] | None = None

    @property
    def _selector_router(self) -> RouterBundle:
        router = self.router
        if router is None:
            raise RuntimeError("whole-series selector router is unavailable")
        return router

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
        """Prepare one whole-episode selector decision without block routing.

        Missing spans are detected only so the common artifact schema can
        record which coordinates were filled.  They do not enter candidate
        scoring, graph construction, shortlisting, pseudo masking, or
        historical backtesting.
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
        requested_budget = budget or BudgetSpec(max_candidates=self.shortlist_size)
        requested_budget = replace(
            requested_budget,
            max_candidates=max(requested_budget.max_candidates, self.shortlist_size),
        )
        period = item.metadata.get("period")
        training_medians = item.metadata.get("training_medians", self.training_medians)
        if not blocks:
            return RoutePlan(
                batch=batch,
                blocks=(),
                graph=BlockGraph((), ()),
                forecast_spec=forecast_spec,
                budget=requested_budget,
                seed=seed,
                period=period,
                correlation_source="none",
                candidate_ids=(),
                shortlist=(),
                costs={},
                prior_unary={},
                pseudo_batch=None,
                training_medians=training_medians,
                artifact_load_failures={},
                available_artifact_ids=frozenset(),
                allow_fallback_execution=False,
                backtest_batch=None,
                backtest_cutoff=None,
            )

        explicit_artifacts = available_artifact_ids is not None
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
            unknown_available = available.difference(self.imputer_registry.ids)
            if unknown_available:
                raise ValueError(
                    "available_artifact_ids contains unknown candidates: "
                    + ", ".join(sorted(unknown_available))
                )
            failures = dict(
                self.artifact_load_failures
                if artifact_load_failures is None
                else artifact_load_failures
            )

        trained_candidates = frozenset(self._selector_router.candidate_ids)
        unknown_trained = trained_candidates.difference(self.imputer_registry.ids)
        if unknown_trained:
            raise ValueError(
                "whole-series router contains unknown candidates: "
                + ", ".join(sorted(unknown_trained))
            )
        candidate_ids = tuple(
            spec.imputer_id
            for spec in self.imputer_registry.specs()
            if (
                spec.imputer_id in trained_candidates
                and self.imputer_registry.availability(spec.imputer_id).available
                and (spec.device == "any" or spec.device in requested_budget.allowed_devices)
                and (not spec.requires_period or bool(period))
                and (spec.fit_scope == "none" or spec.imputer_id in available)
            )
        )
        if not candidate_ids:
            raise RuntimeError("no whole-sequence imputation candidate is available")

        return RoutePlan(
            batch=batch,
            blocks=blocks,
            # RoutePlan requires this field, but sequence selectors do not
            # place the recorded missing spans into a routing graph.
            graph=BlockGraph((), ()),
            forecast_spec=forecast_spec,
            budget=requested_budget,
            seed=seed,
            period=period,
            correlation_source="none",
            candidate_ids=candidate_ids,
            shortlist=candidate_ids,
            costs={candidate_id: 0.0 for candidate_id in candidate_ids},
            prior_unary={},
            pseudo_batch=None,
            training_medians=training_medians,
            artifact_load_failures=failures,
            available_artifact_ids=available,
            allow_fallback_execution=not explicit_artifacts,
            backtest_batch=None,
            backtest_cutoff=None,
        )

    def _safe_complete_without_candidate(
        self,
        plan: RoutePlan,
    ) -> tuple[np.ndarray, dict[str, str], dict[str, dict[str, Any]]]:
        values = np.asarray(plan.batch.values[0], dtype=float).copy()
        assignments: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        medians = (
            None
            if plan.training_medians is None
            else np.asarray(plan.training_medians, dtype=float).reshape(-1)
        )
        for block in plan.blocks:
            observed = plan.batch.observed_mask[0, :, block.channel]
            observed_values = plan.batch.values[0, :, block.channel][observed]
            if medians is not None and block.channel < len(medians):
                fill = float(medians[block.channel])
                source = "train_median"
            elif observed_values.size:
                fill = float(np.median(observed_values))
                source = "context_median"
            else:
                fill = 0.0
                source = "zero"
            values[block.start : block.end, block.channel] = fill
            assignments[block.block_id] = source
            records[block.block_id] = {
                "reason": "no_whole_sequence_candidate_was_native_valid",
                "selected": source,
                "selection_scope": "whole_series",
            }
        return values, assignments, records

    def _scores(self, features: np.ndarray) -> np.ndarray:
        model = self._selector_router.prior
        scorer = getattr(model, "score", None)
        if callable(scorer):
            raw = scorer(features)
        else:
            predictor = getattr(model, "predict", None)
            if not callable(predictor):
                raise TypeError(f"{self.selector_method} selector has no score method")
            raw = predictor(features.reshape(1, -1))
        scores = np.asarray(raw, dtype=float).reshape(-1)
        if (
            scores.shape != (len(self._selector_router.candidate_ids),)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("whole-series selector returned invalid candidate scores")
        higher_is_better = bool(
            getattr(
                model,
                "higher_is_better",
                self.selector_method in {"metaod", "alors", "neuralucb"},
            )
        )
        if higher_is_better:
            scores = -scores
        return scores

    def _select_candidate(
        self,
        plan: RoutePlan,
        valid_ids: tuple[str, ...],
        features: np.ndarray,
    ) -> tuple[str, dict[str, float]]:
        method = self.selector_method
        if method in _RANDOM_SERIES_METHODS:
            selected = valid_ids[_stable_random_index(plan.seed, valid_ids)]
            return selected, {candidate_id: 0.0 for candidate_id in valid_ids}

        model = self._selector_router.prior
        if method == "neuralucb":
            selector = getattr(model, "select", None)
            if callable(selector):
                selected = selector(features, valid_ids)
                if isinstance(selected, (int, np.integer)):
                    selected = self._selector_router.candidate_ids[int(selected)]
                selected = str(selected)
                if selected not in valid_ids:
                    raise ValueError("NeuralUCB selected an unavailable action")
                scores = np.asarray(model.score(features), dtype=float).reshape(-1)
                return selected, {
                    candidate_id: float(scores[index])
                    for index, candidate_id in enumerate(self._selector_router.candidate_ids)
                }

        scores = self._scores(features)
        by_id = {
            candidate_id: float(scores[index])
            for index, candidate_id in enumerate(self._selector_router.candidate_ids)
        }
        selected = min(valid_ids, key=lambda candidate_id: (by_id[candidate_id], candidate_id))
        return selected, {candidate_id: by_id[candidate_id] for candidate_id in valid_ids}

    def finish_route(
        self,
        plan: RoutePlan,
        candidates: Mapping[str, CandidateResult],
        pseudo_candidates: Mapping[str, CandidateResult] | None = None,
        *,
        backtest_candidates: Mapping[str, CandidateResult] | None = None,
        fallback_candidates: Mapping[str, CandidateResult] | None = None,
    ) -> FAISResult:
        del pseudo_candidates, backtest_candidates, fallback_candidates
        mask = np.asarray(plan.batch.observed_mask[0], dtype=bool)
        if plan.is_noop:
            return FAISResult(
                values=np.asarray(plan.batch.values[0], dtype=float).copy(),
                routing=RoutingResult(
                    {},
                    (),
                    0.0,
                    metadata={
                        "selector_method": self.selector_method,
                        "selection_scope": "whole_series",
                        "routing_target_protocol": "sequence_imputation_quality_v1",
                        "selector_training_target": "imputation_loss",
                        "forecaster_independent_selection": True,
                        "uses_missing_block_graph": False,
                        "solver": "noop",
                        "requires_pseudo_candidates": False,
                        "paper_native_valid": True,
                        "paper_ineligibility_reason": None,
                    },
                ),
                candidates={},
                observed_mask=mask,
            )
        route_candidates = self._injected_results(
            plan.shortlist,
            candidates,
            plan.batch,
            "whole-series actual",
        )
        valid_ids = tuple(
            candidate_id
            for candidate_id in plan.shortlist
            if _candidate_is_whole_sequence_valid(
                plan,
                candidate_id,
                route_candidates[candidate_id],
                self,
            )
        )
        features = _model_feature_vector(plan, self._selector_router.feature_names)
        fallback_records: dict[str, dict[str, Any]] = {}
        weights: dict[str, float] = {}
        selected: str | None = None
        score_by_id: dict[str, float] = {}
        selected_native_valid = True
        if not valid_ids:
            values, assignments, fallback_records = self._safe_complete_without_candidate(plan)
            selected_native_valid = False
            fallback_blocks = tuple(assignments)
            activated: tuple[str, ...] = ()
            total_energy = 0.0
        elif self.selector_method == "dselect1":
            weight_fn = getattr(self._selector_router.prior, "weights", None)
            if not callable(weight_fn):
                raise TypeError("DSelect-1 selector has no weights method")
            raw_weights = np.asarray(weight_fn(features), dtype=float).reshape(-1)
            if raw_weights.shape != (len(self._selector_router.candidate_ids),):
                raise ValueError("DSelect-1 returned an invalid expert-weight vector")
            raw_weights = np.maximum(raw_weights, 0.0)
            reachable_mass = float(raw_weights.sum())
            valid_set = set(valid_ids)
            for index, candidate_id in enumerate(self._selector_router.candidate_ids):
                if candidate_id not in valid_set:
                    raw_weights[index] = 0.0
            if float(raw_weights.sum()) <= 0.0:
                raw_weights[self._selector_router.candidate_ids.index(sorted(valid_ids)[0])] = 1.0
                reachable_mass = 1.0
            elif len(valid_ids) != len(self._selector_router.candidate_ids):
                # Redistribute mass from unavailable experts while retaining the
                # paper gate's non-power-of-two reachable mass.
                raw_weights *= reachable_mass / raw_weights.sum()
            weights = {
                candidate_id: float(raw_weights[index])
                for index, candidate_id in enumerate(self._selector_router.candidate_ids)
                if raw_weights[index] > 0.0
            }
            values = np.zeros_like(plan.batch.values[0], dtype=float)
            for candidate_id, weight in weights.items():
                values += weight * route_candidates[candidate_id].values[0]
            values[mask] = plan.batch.values[0][mask]
            selected = max(weights, key=lambda candidate_id: (weights[candidate_id], candidate_id))
            assignments = {block.block_id: selected for block in plan.blocks}
            fallback_blocks = ()
            activated = tuple(weights)
            total_energy = 0.0
        else:
            selected, score_by_id = self._select_candidate(plan, valid_ids, features)
            values = np.asarray(route_candidates[selected].values[0], dtype=float).copy()
            values[mask] = plan.batch.values[0][mask]
            assignments = {block.block_id: selected for block in plan.blocks}
            fallback_blocks = ()
            activated = (selected,)
            total_energy = float(score_by_id.get(selected, 0.0))

        if not np.isfinite(values).all():
            raise ValueError("whole-series selector did not produce a finite completion")
        if selected is not None and self.selector_method == "neuralucb":
            self._pending_online_feedback = {
                "features": features.copy(),
                "candidate_id": selected,
                "prediction": values.copy(),
                "missing_mask": ~mask,
                "native_valid": selected_native_valid,
            }
        else:
            self._pending_online_feedback = None
        metadata = {
            "selector_method": (
                "random_valid_series"
                if self.selector_method == "random_valid_block"
                else self.selector_method
            ),
            "selector_implementation": self.selector_method,
            "selection_scope": "whole_series",
            "selection_count": 1,
            "selected_candidate": selected,
            "valid_whole_sequence_candidates": list(valid_ids),
            "invalid_whole_sequence_candidates": sorted(set(plan.shortlist) - set(valid_ids)),
            "candidate_scores": score_by_id,
            "expert_weights": weights,
            "routing_target_protocol": "sequence_imputation_quality_v1",
            "selector_training_target": "imputation_loss",
            "forecaster_independent_selection": True,
            "uses_missing_block_graph": False,
            "requires_pseudo_candidates": False,
            "solver": "whole_series_native_selector",
            "selected_action_native_valid": selected_native_valid,
            "paper_native_valid": selected_native_valid,
            "paper_ineligibility_reason": (
                None if selected_native_valid else "no_native_valid_whole_series_candidate"
            ),
        }
        routing = RoutingResult(
            assignments=assignments,
            shortlist=plan.shortlist,
            total_energy=total_energy,
            activated_candidates=activated,
            fallback_blocks=fallback_blocks,
            fallback_records=fallback_records,
            risk_energy=total_energy,
            metadata=metadata,
        )
        return FAISResult(
            values=values,
            routing=routing,
            # The stage owns the save_all_candidate_outputs policy.  Selection
            # diagnostics live in routing metadata; forcing every portfolio
            # tensor into each selector artifact would repeat TSFM evaluation.
            candidates={},
            observed_mask=mask,
        )

    def observe_outcome(self, clean_context: np.ndarray) -> float | None:
        """Reveal the chosen reconstruction reward to NeuralUCB after selection."""

        feedback = self._pending_online_feedback
        self._pending_online_feedback = None
        if feedback is None:
            return None
        truth = np.asarray(clean_context, dtype=float)
        prediction = np.asarray(feedback["prediction"], dtype=float)
        missing = np.asarray(feedback["missing_mask"], dtype=bool)
        actual = truth[missing]
        estimate = prediction[missing]
        denominator = np.abs(actual) + np.abs(estimate)
        relative_error = np.zeros_like(denominator, dtype=float)
        nonzero = denominator > 0.0
        relative_error[nonzero] = np.abs(estimate[nonzero] - actual[nonzero]) / denominator[nonzero]
        loss = float(np.mean(relative_error))
        reward = float(1.0 / (1.0 + np.clip(loss, 0.0, 1.0)))
        observer = getattr(self._selector_router.prior, "observe", None)
        if not callable(observer):
            raise TypeError("NeuralUCB selector has no observe method")
        observer(feedback["features"], feedback["candidate_id"], reward)
        return reward


__all__ = ["WholeSeriesSelectorFAIS"]
