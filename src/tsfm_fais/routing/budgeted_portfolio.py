"""Rank observable pairwise votes and aggregate a fixed odd query budget."""

import numpy as np

from .pairwise_utility import hard_pairwise_choice


def rank_pairwise_candidates(preferences, pairs, candidates, preferred):
    winners, votes = hard_pairwise_choice(preferences, pairs, candidates, preferred)
    tie_order = np.array(
        sorted(range(len(candidates)), key=lambda i: (candidates[i] != preferred, candidates[i]))
    )
    ranked = tie_order[np.argsort(-votes[:, tie_order], axis=1, kind="stable")]
    np.testing.assert_array_equal(ranked[:, 0], winners)
    return ranked


def median_portfolio(points, rankings, budget, *, joint):
    """Points [N,A,H,K], action rankings [N,K,A]; joint models share each context choice."""
    points, rankings = np.asarray(points, float), np.asarray(rankings)
    if (
        points.ndim != 4
        or rankings.shape != (points.shape[0], points.shape[3], points.shape[1])
        or not np.issubdtype(rankings.dtype, np.integer)
        or not np.isfinite(points).all()
        or not isinstance(budget, int)
        or not 1 <= budget <= points.shape[1]
        or budget % 2 != 1
        or not np.all(np.sort(rankings, axis=2) == np.arange(points.shape[1]))
    ):
        raise ValueError(
            "finite predictions, complete rankings and an odd supported budget are required"
        )
    if joint and not np.all(rankings == rankings[:, :1]):
        raise ValueError("joint forecasts require the same context ranking for every target")
    selected = np.take_along_axis(
        points.transpose(0, 3, 1, 2), rankings[:, :, :budget, None], axis=2
    )
    return np.median(selected, axis=2).transpose(0, 2, 1)
