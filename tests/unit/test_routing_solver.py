from __future__ import annotations

from tsfm_fais.routing import (
    BeamSearchSolver,
    BlockEdge,
    ExhaustiveSolver,
    RoutingProblem,
    assignment_energy,
)


def make_problem():
    blocks = ("b0", "b1", "b2")
    candidates = ("a", "b")
    unary = {
        ("b0", "a"): 0.0,
        ("b0", "b"): 2.0,
        ("b1", "a"): 2.0,
        ("b1", "b"): 0.0,
        ("b2", "a"): 0.0,
        ("b2", "b"): 2.0,
    }
    pairwise = {
        ("b0", "a", "b1", "b"): 4.0,
        ("b1", "b", "b2", "a"): 4.0,
    }
    return RoutingProblem(
        blocks,
        candidates,
        unary,
        edges=(BlockEdge("b0", "b1"), BlockEdge("b1", "b2")),
        pairwise=pairwise,
        pairwise_weight=1.0,
    )


def test_exhaustive_and_wide_beam_find_same_assignment():
    problem = make_problem()
    exact = ExhaustiveSolver().solve(problem)
    beam = BeamSearchSolver(beam_width=16).solve(problem)
    assert exact.assignments == beam.assignments
    assert exact.total_energy == beam.total_energy
    assert exact.total_energy == assignment_energy(problem, exact.assignments)


def test_max_active_candidate_constraint_is_enforced():
    problem = make_problem()
    constrained = RoutingProblem(
        problem.block_ids,
        problem.candidate_ids,
        problem.unary,
        edges=problem.edges,
        pairwise=problem.pairwise,
        max_active_candidates=1,
    )
    result = ExhaustiveSolver().solve(constrained)
    assert len(result.activated_candidates) == 1
