from __future__ import annotations

import numpy as np

from tsfm_fais.contracts import BudgetSpec
from tsfm_fais.routing.blocks import build_block_graph, detect_missing_blocks
from tsfm_fais.routing.metrics import mase
from tsfm_fais.routing.solver import beam_search, exhaustive_search, greedy_shortlist


def test_block_detection_keeps_channels_separate_and_graphs_overlap():
    mask = np.ones((1, 12, 3), dtype=bool)
    mask[0, 2:5, 0] = False
    mask[0, 3:7, 1] = False
    mask[0, 9:11, 0] = False
    blocks = detect_missing_blocks(mask)
    assert len(blocks) == 3
    graph = build_block_graph(blocks, np.eye(3))
    assert len(graph.edges) == 2


def test_beam_matches_exhaustive_on_small_problem():
    mask = np.ones((1, 10, 2), dtype=bool)
    mask[0, 2:4, 0] = False
    mask[0, 5:8, 1] = False
    graph = build_block_graph(detect_missing_blocks(mask), np.ones((2, 2)))
    candidates = ("a", "b", "c")
    unary = {}
    for index, block in enumerate(graph.blocks):
        for candidate_index, candidate in enumerate(candidates):
            unary[(block.block_id, candidate)] = abs(candidate_index - index)
    pairwise = {
        (graph.blocks[0].block_id, graph.blocks[1].block_id, "a", "b"): -0.25
    }
    budget = BudgetSpec(max_candidates=3, max_active_candidates=2)
    beam = beam_search(
        graph, candidates, unary, pairwise, budget=budget, beam_width=32
    )
    exhaustive = exhaustive_search(
        graph, candidates, unary, pairwise, budget=budget
    )
    assert beam.assignments == exhaustive.assignments
    assert np.isclose(beam.total_energy, exhaustive.total_energy)


def test_shortlist_forces_safe_candidates_and_is_bounded():
    mask = np.ones((1, 8, 1), dtype=bool)
    mask[0, 2:4, 0] = False
    blocks = detect_missing_blocks(mask)
    candidates = ("locf", "linear_interp", "x", "y")
    unary = {(blocks[0].block_id, candidate): index for index, candidate in enumerate(candidates)}
    shortlist = greedy_shortlist(
        blocks, candidates, unary, {candidate: 1 for candidate in candidates}, max_candidates=3
    )
    assert shortlist[:2] == ("locf", "linear_interp")
    assert len(shortlist) == 3


def test_mase_is_zero_for_exact_forecast():
    history = np.arange(10, dtype=float)
    truth = np.arange(10, 13, dtype=float)
    assert mase(truth, truth.copy(), history) == 0.0

