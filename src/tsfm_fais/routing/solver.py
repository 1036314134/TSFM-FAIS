"""Candidate shortlisting and deterministic structured assignment."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tsfm_fais.contracts import BudgetSpec, MissingBlock, RoutingResult

from .blocks import BlockGraph
from .graph import BlockEdge

UnaryScores = Mapping[tuple[str, str], float]
PairwiseScores = Mapping[tuple[str, str, str, str], float]


@dataclass(frozen=True)
class RoutingProblem:
    """Finite block assignment problem used by the reusable solver API.

    ``pairwise`` accepts both ``(block, candidate, block, candidate)`` and the
    older ``(block, block, candidate, candidate)`` key layout.  Supporting
    both layouts keeps serialized router artifacts compatible with the
    function-based solver below.
    """

    block_ids: Sequence[str]
    candidate_ids: Sequence[str]
    unary: Mapping[tuple[str, str], float]
    edges: Sequence[BlockEdge] = ()
    pairwise: PairwiseScores = field(default_factory=dict)
    pairwise_weight: float = 1.0
    activation_costs: Mapping[str, float] = field(default_factory=dict)
    max_active_candidates: int | None = None
    invalid_assignments: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        block_ids = tuple(self.block_ids)
        candidate_ids = tuple(self.candidate_ids)
        if not block_ids or len(set(block_ids)) != len(block_ids):
            raise ValueError("block_ids must be non-empty and unique")
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids must be non-empty and unique")
        if not math.isfinite(self.pairwise_weight) or self.pairwise_weight < 0:
            raise ValueError("pairwise_weight must be a finite non-negative value")
        if self.max_active_candidates is not None and not (
            1 <= self.max_active_candidates <= len(candidate_ids)
        ):
            raise ValueError("max_active_candidates must be within candidate_ids")
        block_set = set(block_ids)
        candidate_set = set(candidate_ids)
        for block_id in block_ids:
            for candidate_id in candidate_ids:
                value = self.unary.get((block_id, candidate_id))
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(
                        f"unary score missing or non-finite for {(block_id, candidate_id)}"
                    )
        if any(
            block_id not in block_set or candidate_id not in candidate_set
            for block_id, candidate_id in self.invalid_assignments
        ):
            raise ValueError("invalid_assignments contains an unknown id")
        for edge in self.edges:
            if edge.left not in block_set or edge.right not in block_set:
                raise ValueError("edge refers to an unknown block")
        for candidate_id, value in dict(self.activation_costs).items():
            if candidate_id not in candidate_ids or not math.isfinite(float(value)):
                raise ValueError("activation_costs contain an invalid entry")
        object.__setattr__(self, "block_ids", block_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "unary", dict(self.unary))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "pairwise", dict(self.pairwise))
        object.__setattr__(self, "activation_costs", dict(self.activation_costs))
        object.__setattr__(self, "invalid_assignments", frozenset(self.invalid_assignments))


@dataclass(frozen=True)
class SolverResult:
    assignments: Mapping[str, str]
    total_energy: float
    activated_candidates: tuple[str, ...]
    method: str
    explored_states: int = 0


def _problem_pair_score(
    problem: RoutingProblem,
    left_block: str,
    left_candidate: str,
    right_block: str,
    right_candidate: str,
) -> float:
    keys = (
        (left_block, left_candidate, right_block, right_candidate),
        (right_block, right_candidate, left_block, left_candidate),
        (left_block, right_block, left_candidate, right_candidate),
        (right_block, left_block, right_candidate, left_candidate),
    )
    for key in keys:
        if key in problem.pairwise:
            return float(problem.pairwise[key])
    return 0.0


def assignment_energy(
    problem: RoutingProblem,
    assignments: Mapping[str, str],
) -> float:
    """Return the complete structured objective for one assignment."""

    if set(assignments) != set(problem.block_ids):
        raise ValueError("assignments must contain every block exactly once")
    if any(candidate not in problem.candidate_ids for candidate in assignments.values()):
        raise ValueError("assignment contains an unknown candidate")
    if any(key in problem.invalid_assignments for key in assignments.items()):
        return float("inf")
    active = set(assignments.values())
    if (
        problem.max_active_candidates is not None
        and len(active) > problem.max_active_candidates
    ):
        return float("inf")
    total = 0.0
    for block_id in problem.block_ids:
        total += float(problem.unary.get((block_id, assignments[block_id]), float("inf")))
    total += sum(float(problem.activation_costs.get(candidate_id, 0.0)) for candidate_id in active)
    for edge in problem.edges:
        total += (
            problem.pairwise_weight
            * float(edge.weight)
            * _problem_pair_score(
                problem,
                edge.left,
                assignments[edge.left],
                edge.right,
                assignments[edge.right],
            )
        )
    return float(total)


def _solver_result(
    problem: RoutingProblem,
    assignment_values: Sequence[str],
    energy: float,
    method: str,
    explored_states: int,
) -> SolverResult:
    assignments = dict(zip(problem.block_ids, assignment_values))
    return SolverResult(
        assignments=assignments,
        total_energy=float(energy),
        activated_candidates=tuple(sorted(set(assignment_values))),
        method=method,
        explored_states=explored_states,
    )


class ExhaustiveSolver:
    """Deterministic reference solver for small routing problems."""

    def __init__(self, max_states: int = 1_000_000) -> None:
        if max_states < 1:
            raise ValueError("max_states must be positive")
        self.max_states = int(max_states)

    def solve(self, problem: RoutingProblem) -> SolverResult:
        state_count = len(problem.candidate_ids) ** len(problem.block_ids)
        if state_count > self.max_states:
            raise ValueError(
                f"exhaustive routing requires {state_count} states, above {self.max_states}"
            )
        best_values: tuple[str, ...] | None = None
        best_energy = float("inf")
        explored = 0
        for values in itertools.product(problem.candidate_ids, repeat=len(problem.block_ids)):
            explored += 1
            if (
                problem.max_active_candidates is not None
                and len(set(values)) > problem.max_active_candidates
            ):
                continue
            assignments = dict(zip(problem.block_ids, values))
            energy = assignment_energy(problem, assignments)
            if (energy, values) < (best_energy, best_values or values):
                best_values = values
                best_energy = energy
        if best_values is None or not math.isfinite(best_energy):
            raise RuntimeError("no feasible routing assignment")
        return _solver_result(problem, best_values, best_energy, "exhaustive", explored)


@dataclass(frozen=True)
class _ProblemBeamState:
    values: tuple[str, ...]
    active: frozenset[str]
    energy: float


class BeamSearchSolver:
    """Deterministic left-to-right beam search over missing blocks."""

    def __init__(self, beam_width: int = 32) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self.beam_width = int(beam_width)

    def _increment(
        self,
        problem: RoutingProblem,
        state: _ProblemBeamState,
        block_id: str,
        candidate_id: str,
    ) -> float:
        unary = float(problem.unary.get((block_id, candidate_id), float("inf")))
        if (block_id, candidate_id) in problem.invalid_assignments:
            return float("inf")
        increment = unary
        if candidate_id not in state.active:
            increment += float(problem.activation_costs.get(candidate_id, 0.0))
        assigned = dict(zip(problem.block_ids[: len(state.values)], state.values))
        for edge in problem.edges:
            if edge.left == block_id and edge.right in assigned:
                other_block = edge.right
                other_candidate = assigned[other_block]
            elif edge.right == block_id and edge.left in assigned:
                other_block = edge.left
                other_candidate = assigned[other_block]
            else:
                continue
            increment += (
                problem.pairwise_weight
                * float(edge.weight)
                * _problem_pair_score(
                    problem,
                    block_id,
                    candidate_id,
                    other_block,
                    other_candidate,
                )
            )
        return float(increment)

    def solve(self, problem: RoutingProblem) -> SolverResult:
        beam = [_ProblemBeamState((), frozenset(), 0.0)]
        explored = 0
        for block_id in problem.block_ids:
            expanded: list[_ProblemBeamState] = []
            for state in beam:
                for candidate_id in problem.candidate_ids:
                    explored += 1
                    active = state.active | {candidate_id}
                    if (
                        problem.max_active_candidates is not None
                        and len(active) > problem.max_active_candidates
                    ):
                        continue
                    increment = self._increment(problem, state, block_id, candidate_id)
                    if not math.isfinite(increment):
                        continue
                    expanded.append(
                        _ProblemBeamState(
                            state.values + (candidate_id,),
                            frozenset(active),
                            state.energy + increment,
                        )
                    )
            if not expanded:
                raise RuntimeError(f"no feasible routing assignment for block {block_id}")
            expanded.sort(key=lambda state: (state.energy, state.values))
            beam = expanded[: self.beam_width]
        best = beam[0]
        # Recompute to keep this API and the reference objective identical.
        final_energy = assignment_energy(
            problem, dict(zip(problem.block_ids, best.values))
        )
        return _solver_result(problem, best.values, final_energy, "beam", explored)


def solve_routing(
    problem: RoutingProblem,
    *,
    method: str = "auto",
    exact_state_limit: int = 100_000,
    beam_width: int = 32,
) -> SolverResult:
    if exact_state_limit < 1:
        raise ValueError("exact_state_limit must be positive")
    normalized = method.lower()
    if normalized == "auto":
        state_count = len(problem.candidate_ids) ** len(problem.block_ids)
        normalized = "exhaustive" if state_count <= exact_state_limit else "beam"
    if normalized in {"exact", "exhaustive"}:
        return ExhaustiveSolver(max_states=exact_state_limit).solve(problem)
    if normalized == "beam":
        return BeamSearchSolver(beam_width=beam_width).solve(problem)
    raise ValueError("method must be one of auto, exhaustive, exact, or beam")


def greedy_shortlist(
    blocks: Sequence[MissingBlock],
    candidates: Sequence[str],
    unary: UnaryScores,
    costs: Mapping[str, float],
    max_candidates: int = 6,
    forced: Sequence[str] = ("locf", "linear_interp"),
) -> tuple[str, ...]:
    available = tuple(dict.fromkeys(candidates))
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    selected = [candidate for candidate in forced if candidate in available][:max_candidates]
    while len(selected) < min(max_candidates, len(available)):
        best: tuple[tuple[float, float, float, str], str] | None = None
        for candidate in available:
            if candidate in selected:
                continue
            improvement = 0.0
            for block in blocks:
                old = min(
                    (unary.get((block.block_id, current), float("inf")) for current in selected),
                    default=float("inf"),
                )
                new = unary.get((block.block_id, candidate), float("inf"))
                if old == float("inf"):
                    improvement += 0.0 if new == float("inf") else 1.0 / (1.0 + max(new, 0.0))
                else:
                    improvement += max(0.0, old - new)
            score = improvement / (1.0 + max(0.0, costs.get(candidate, 1.0)))
            risks = [
                float(unary.get((block.block_id, candidate), float("inf")))
                for block in blocks
            ]
            mean_risk = sum(risks) / len(risks) if risks else float("inf")
            key = (
                -score,
                mean_risk,
                float(costs.get(candidate, 1.0)),
                candidate,
            )
            entry = (key, candidate)
            if best is None or entry[0] < best[0]:
                best = entry
        if best is None:
            break
        selected.append(best[1])
    return tuple(selected)


def _pair_score(
    pairwise: PairwiseScores,
    left_block: str,
    right_block: str,
    left_candidate: str,
    right_candidate: str,
) -> float:
    direct = (left_block, right_block, left_candidate, right_candidate)
    reverse = (right_block, left_block, right_candidate, left_candidate)
    return float(pairwise.get(direct, pairwise.get(reverse, 0.0)))


@dataclass
class _BeamState:
    assignments: dict[str, str]
    used: frozenset[str]
    energy: float


def _incremental_energy(
    state: _BeamState,
    block: MissingBlock,
    candidate: str,
    graph: BlockGraph,
    unary: UnaryScores,
    pairwise: PairwiseScores,
    costs: Mapping[str, float],
    beta: float,
    cost_weight: float,
) -> float:
    value = float(unary.get((block.block_id, candidate), float("inf")))
    if candidate not in state.used:
        value += cost_weight * float(costs.get(candidate, 1.0))
    for edge in graph.edges:
        if edge.left == block.block_id and edge.right in state.assignments:
            other_block = edge.right
        elif edge.right == block.block_id and edge.left in state.assignments:
            other_block = edge.left
        else:
            continue
        value += beta * float(edge.weight) * _pair_score(
            pairwise,
            block.block_id,
            other_block,
            candidate,
            state.assignments[other_block],
        )
    return value


def beam_search(
    graph: BlockGraph,
    candidates: Sequence[str],
    unary: UnaryScores,
    pairwise: PairwiseScores | None = None,
    costs: Mapping[str, float] | None = None,
    budget: BudgetSpec | None = None,
    beam_width: int = 32,
    beta: float = 1.0,
    cost_weight: float = 0.05,
    invalid: set[tuple[str, str]] | None = None,
) -> RoutingResult:
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    pairwise = pairwise or {}
    costs = costs or {}
    invalid = invalid or set()
    budget = budget or BudgetSpec(max_candidates=max(1, len(candidates)))
    ordered = sorted(graph.blocks, key=lambda block: (-block.end, block.block_id))
    beam = [_BeamState(assignments={}, used=frozenset(), energy=0.0)]
    for block in ordered:
        expanded: list[_BeamState] = []
        for state in beam:
            for candidate in candidates:
                if (block.block_id, candidate) in invalid:
                    continue
                used = state.used | {candidate}
                if budget.max_active_candidates is not None and len(used) > budget.max_active_candidates:
                    continue
                increment = _incremental_energy(
                    state,
                    block,
                    candidate,
                    graph,
                    unary,
                    pairwise,
                    costs,
                    beta,
                    cost_weight,
                )
                if increment == float("inf"):
                    continue
                assignments = dict(state.assignments)
                assignments[block.block_id] = candidate
                expanded.append(_BeamState(assignments, frozenset(used), state.energy + increment))
        if not expanded:
            raise RuntimeError(f"no valid assignment for block {block.block_id}")
        expanded.sort(key=lambda state: (state.energy, tuple(sorted(state.assignments.items()))))
        beam = expanded[:beam_width]
    best = beam[0]
    candidate_costs = {
        candidate: float(costs.get(candidate, 1.0)) for candidate in candidates
    }
    activated_cost = sum(candidate_costs[candidate] for candidate in best.used)
    cost_energy = cost_weight * activated_cost
    return RoutingResult(
        assignments=best.assignments,
        shortlist=tuple(candidates),
        total_energy=best.energy,
        predicted_unary=dict(unary),
        predicted_pairwise=dict(pairwise),
        activated_candidates=tuple(sorted(best.used)),
        candidate_costs=candidate_costs,
        activated_cost=float(activated_cost),
        risk_energy=float(best.energy - cost_energy),
        cost_energy=float(cost_energy),
    )


def exhaustive_search(
    graph: BlockGraph,
    candidates: Sequence[str],
    unary: UnaryScores,
    pairwise: PairwiseScores | None = None,
    costs: Mapping[str, float] | None = None,
    budget: BudgetSpec | None = None,
    beta: float = 1.0,
    cost_weight: float = 0.05,
    invalid: set[tuple[str, str]] | None = None,
) -> RoutingResult:
    best: RoutingResult | None = None
    for assignment_values in itertools.product(candidates, repeat=len(graph.blocks)):
        mapping = {
            block.block_id: candidate
            for block, candidate in zip(graph.blocks, assignment_values)
        }
        if invalid and any((block, candidate) in invalid for block, candidate in mapping.items()):
            continue
        active = set(mapping.values())
        if budget and budget.max_active_candidates is not None and len(active) > budget.max_active_candidates:
            continue
        energy = sum(unary.get((block, candidate), float("inf")) for block, candidate in mapping.items())
        energy += cost_weight * sum((costs or {}).get(candidate, 1.0) for candidate in active)
        for edge in graph.edges:
            left, right = edge.left, edge.right
            energy += beta * float(edge.weight) * _pair_score(
                pairwise or {}, left, right, mapping[left], mapping[right]
            )
        result = RoutingResult(
            assignments=mapping,
            shortlist=tuple(candidates),
            total_energy=float(energy),
            activated_candidates=tuple(sorted(active)),
            candidate_costs={
                candidate: float((costs or {}).get(candidate, 1.0))
                for candidate in candidates
            },
            activated_cost=float(
                sum((costs or {}).get(candidate, 1.0) for candidate in active)
            ),
            risk_energy=float(
                energy
                - cost_weight
                * sum((costs or {}).get(candidate, 1.0) for candidate in active)
            ),
            cost_energy=float(
                cost_weight
                * sum((costs or {}).get(candidate, 1.0) for candidate in active)
            ),
        )
        if best is None or (result.total_energy, tuple(sorted(mapping.items()))) < (
            best.total_energy,
            tuple(sorted(best.assignments.items())),
        ):
            best = result
    if best is None:
        raise RuntimeError("no feasible exhaustive assignment")
    return best
