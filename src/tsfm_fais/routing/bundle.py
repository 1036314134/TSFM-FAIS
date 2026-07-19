"""Composable learned router for candidate scoring and block assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    ImputerSpec,
    MissingBlock,
    RoutingResult,
    SeriesBatch,
)

from .blocks import assemble_routed_values, extract_missing_blocks
from .features import (
    RoutingFeatureExtractor,
    RoutingFeatureTable,
    proxy_pairwise_scores,
    proxy_unary_scores,
)
from .graph import BlockGraph, build_block_graph
from .lightgbm_model import LazyLightGBMRegressor
from .shortlist import CandidateShortlister
from .solver import RoutingProblem, solve_routing


class RouterBundle:
    """Own feature extraction, unary scoring, shortlisting, and graph solving."""

    def __init__(
        self,
        *,
        unary_model: Any | None = None,
        feature_extractor: RoutingFeatureExtractor | None = None,
        shortlister: CandidateShortlister | None = None,
        imputer_specs: Mapping[str, ImputerSpec] | None = None,
        solver_method: str = "auto",
        exact_state_limit: int = 100_000,
        beam_width: int = 64,
        pairwise_weight: float = 0.1,
        activation_penalty: float = 0.0,
        invalid_native_penalty: float = 1000.0,
    ) -> None:
        self.unary_model = unary_model
        self.feature_extractor = feature_extractor or RoutingFeatureExtractor()
        self.shortlister = shortlister or CandidateShortlister()
        self.imputer_specs = dict(imputer_specs or {})
        self.solver_method = solver_method
        self.exact_state_limit = int(exact_state_limit)
        self.beam_width = int(beam_width)
        self.pairwise_weight = float(pairwise_weight)
        self.activation_penalty = float(activation_penalty)
        self.invalid_native_penalty = float(invalid_native_penalty)

    def fit(
        self,
        table: RoutingFeatureTable,
        targets: Mapping[tuple[str, str], float] | np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RouterBundle:
        features, keys = table.flatten()
        if isinstance(targets, Mapping):
            try:
                target_array = np.asarray([targets[key] for key in keys], dtype=float)
            except KeyError as exc:
                raise ValueError(f"missing unary target for {exc.args[0]}") from exc
        else:
            target_array = np.asarray(targets, dtype=float).reshape(-1)
        if self.unary_model is None:
            self.unary_model = LazyLightGBMRegressor()
        self.unary_model.fit(
            features,
            target_array,
            sample_weight=sample_weight,
            feature_names=table.feature_names,
        )
        return self

    def predict_unary(
        self,
        table: RoutingFeatureTable,
    ) -> dict[tuple[str, str], float]:
        if self.unary_model is None:
            return proxy_unary_scores(table)
        features, keys = table.flatten()
        predictions = np.asarray(self.unary_model.predict(features), dtype=float).reshape(-1)
        if len(predictions) != len(keys) or not np.all(np.isfinite(predictions)):
            raise ValueError("unary model returned invalid predictions")
        return {key: float(value) for key, value in zip(keys, predictions, strict=True)}

    def route(
        self,
        batch: SeriesBatch,
        candidates: Mapping[str, CandidateResult],
        *,
        budget: BudgetSpec | None = None,
        blocks: Sequence[MissingBlock] | None = None,
        graph: BlockGraph | None = None,
        predicted_unary: Mapping[tuple[str, str], float] | None = None,
        predicted_pairwise: Mapping[tuple[str, str, str, str], float] | None = None,
    ) -> RoutingResult:
        budget = budget or BudgetSpec()
        selected_blocks = tuple(blocks) if blocks is not None else extract_missing_blocks(batch)
        if not selected_blocks:
            return RoutingResult(
                assignments={},
                shortlist=(),
                total_energy=0.0,
                metadata={"solver": "none", "reason": "no_missing_blocks"},
            )
        graph = graph or build_block_graph(selected_blocks)
        table = self.feature_extractor.transform(
            batch,
            selected_blocks,
            candidates,
            graph=graph,
            imputer_specs=self.imputer_specs,
        )
        unary = dict(predicted_unary or self.predict_unary(table))

        block_lookup = {block.block_id: block for block in selected_blocks}
        invalid_assignments: set[tuple[str, str]] = set()
        for (block_id, candidate_id), value in list(unary.items()):
            if block_id not in block_lookup or candidate_id not in candidates:
                continue
            block = block_lookup[block_id]
            selector = (block.batch_index, slice(block.start, block.end), block.channel)
            coverage = float(np.mean(candidates[candidate_id].native_valid_mask[selector]))
            if coverage < 1.0:
                invalid_assignments.add((block_id, candidate_id))
            unary[(block_id, candidate_id)] = float(
                value + self.invalid_native_penalty * (1.0 - coverage)
            )

        candidate_costs = {
            candidate_id: float(
                np.mean(
                    [unary[(block.block_id, candidate_id)] for block in selected_blocks]
                )
            )
            for candidate_id in candidates
        }
        shortlist_result = self.shortlister.select(
            tuple(candidates),
            candidate_costs,
            budget,
            results=candidates,
            specs=self.imputer_specs,
        )
        shortlist = shortlist_result.selected

        pairwise = dict(predicted_pairwise or proxy_pairwise_scores(graph, candidates))
        activation_costs = {
            candidate_id: self.activation_penalty
            * float(
                self.imputer_specs[candidate_id].cost_tier
                if candidate_id in self.imputer_specs
                else 1
            )
            for candidate_id in shortlist
        }
        problem = RoutingProblem(
            block_ids=tuple(block.block_id for block in selected_blocks),
            candidate_ids=shortlist,
            unary={
                (block.block_id, candidate_id): unary[(block.block_id, candidate_id)]
                for block in selected_blocks
                for candidate_id in shortlist
            },
            edges=graph.edges,
            pairwise=pairwise,
            pairwise_weight=self.pairwise_weight,
            activation_costs=activation_costs,
            max_active_candidates=budget.max_active_candidates,
            invalid_assignments=frozenset(invalid_assignments),
        )
        solved = solve_routing(
            problem,
            method=self.solver_method,
            exact_state_limit=self.exact_state_limit,
            beam_width=self.beam_width,
        )

        fallback_blocks: list[str] = []
        for block in selected_blocks:
            candidate = candidates[solved.assignments[block.block_id]]
            selector = (block.batch_index, slice(block.start, block.end), block.channel)
            if not np.all(candidate.native_valid_mask[selector]):
                fallback_blocks.append(block.block_id)

        candidate_costs = {
            candidate_id: float(
                self.imputer_specs[candidate_id].cost_tier
                if candidate_id in self.imputer_specs
                else 1.0
            )
            for candidate_id in shortlist
        }
        activated_cost = sum(
            candidate_costs[candidate_id]
            for candidate_id in solved.activated_candidates
        )
        cost_energy = sum(
            activation_costs.get(candidate_id, 0.0)
            for candidate_id in solved.activated_candidates
        )

        return RoutingResult(
            assignments=dict(solved.assignments),
            shortlist=shortlist,
            total_energy=solved.total_energy,
            predicted_unary=unary,
            predicted_pairwise=pairwise,
            activated_candidates=solved.activated_candidates,
            fallback_blocks=tuple(fallback_blocks),
            candidate_costs=candidate_costs,
            activated_cost=float(activated_cost),
            risk_energy=float(solved.total_energy - cost_energy),
            cost_energy=float(cost_energy),
            metadata={
                "solver": solved.method,
                "feature_names": table.feature_names,
                "shortlist_excluded": dict(shortlist_result.excluded),
                "estimated_candidate_runtime_seconds": shortlist_result.estimated_runtime_seconds,
            },
        )

    @staticmethod
    def assemble(
        batch: SeriesBatch,
        blocks: Sequence[MissingBlock],
        candidates: Mapping[str, CandidateResult],
        result: RoutingResult,
    ) -> np.ndarray:
        by_id = {block.block_id: block for block in blocks}
        for block_id, candidate_id in result.assignments.items():
            block = by_id[block_id]
            selector = (block.batch_index, slice(block.start, block.end), block.channel)
            if not np.all(candidates[candidate_id].native_valid_mask[selector]):
                raise RuntimeError(
                    f"candidate {candidate_id} has no native output for block {block_id}"
                )
        return assemble_routed_values(
            batch, tuple(blocks), candidates, result.assignments
        )
