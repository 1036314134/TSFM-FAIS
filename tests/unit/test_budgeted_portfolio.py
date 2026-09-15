import numpy as np
import pytest

from tsfm_fais.routing.budgeted_portfolio import median_portfolio, rank_pairwise_candidates


def test_rank_ties_keep_the_source_preference_and_single_action_contract():
    candidates = ("a", "b", "c")
    pairs = [(0, 1), (0, 2), (1, 2)]
    # A cycle produces tied votes; exact pair ties must produce the same source preference.
    ranks = rank_pairwise_candidates([[1, -1, 1], [0, 0, 0]], pairs, candidates, "c")
    np.testing.assert_array_equal(ranks, [[2, 0, 1], [2, 0, 1]])


def test_budget_endpoints_and_model_target_scope():
    points = np.array([[[[1, 90]], [[2, 20]], [[30, 3]]]], dtype=float)
    joint = np.array([[[2, 0, 1], [2, 0, 1]]])
    np.testing.assert_array_equal(median_portfolio(points, joint, 1, joint=True), points[:, 2])
    np.testing.assert_array_equal(
        median_portfolio(points, joint, 3, joint=True), np.median(points, axis=1)
    )
    independent = np.array([[[0, 1, 2], [2, 1, 0]]])
    np.testing.assert_array_equal(
        median_portfolio(points, independent, 1, joint=False), [[[1, 3]]]
    )
    with pytest.raises(ValueError, match="same context"):
        median_portfolio(points, independent, 1, joint=True)
    with pytest.raises(ValueError, match="complete rankings"):
        median_portfolio(points, np.zeros_like(joint), 3, joint=True)
