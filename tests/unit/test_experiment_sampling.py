from __future__ import annotations

import pytest

from tsfm_fais.experiment_sampling import (
    candidate_subset,
    connected_subset,
    deterministic_subset,
    evenly_spaced_subset,
)
from tsfm_fais.contracts import MissingBlock
from tsfm_fais.routing.graph import BlockEdge


def test_deterministic_subset_is_stable_and_source_ordered():
    values = tuple(range(20))
    first = deterministic_subset(values, 5, "dataset", 7)
    second = deterministic_subset(values, 5, "dataset", 7)
    assert first == second
    assert first == tuple(sorted(first))
    assert len(first) == 5


def test_evenly_spaced_subset_covers_both_ends():
    assert evenly_spaced_subset(tuple(range(10)), 4) == (0, 3, 6, 9)
    assert evenly_spaced_subset(tuple(range(10)), 1) == (9,)


def test_candidate_subset_forces_baselines_and_rotates_pool():
    candidates = ("locf", "linear_interp", "a", "b", "c", "d")
    selected = candidate_subset(candidates, 4, "episode", 1)
    assert selected[:2] == ("locf", "linear_interp")
    assert len(selected) == 4
    assert selected == candidate_subset(candidates, 4, "episode", 1)
    with pytest.raises(ValueError, match="forced"):
        candidate_subset(candidates, 1, "episode")


def test_connected_subset_keeps_an_edge_when_quota_allows():
    blocks = tuple(MissingBlock(str(index), 0, index, index, index + 1) for index in range(5))
    edges = (BlockEdge("1", "4"),)
    selected = connected_subset(blocks, edges, 2, "episode")
    assert {block.block_id for block in selected} == {"1", "4"}
