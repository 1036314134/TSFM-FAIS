import numpy as np
import pandas as pd

from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector, hard_pairwise_choice


def test_pairwise_voting_keeps_the_candidate_that_wins_every_comparison():
    # Summing probabilities would select b; pairwise votes correctly select a.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    preferences = np.array([[0.01, 0.01, 0.01, 0.49, 0.49, 0.2]])
    winner, votes = hard_pairwise_choice(preferences, pairs, ("a", "b", "c", "d"), "d")
    assert winner.tolist() == [0]
    assert votes[0, 0] == 3


def test_cyclic_votes_use_the_training_fixed_reference():
    winner, votes = hard_pairwise_choice(
        [[0.2, -0.2, 0.2]], [(0, 1), (0, 2), (1, 2)], ("a", "b", "c"), "c"
    )
    assert winner.tolist() == [2]
    np.testing.assert_array_equal(votes, [[1, 1, 1]])


def test_mistake_cost_overrides_majority_and_inference_ignores_outcomes():
    rows = []
    for index in range(60):
        for candidate, loss in zip(("a", "b"), (0, 1) if index < 54 else (100, 0), strict=True):
            rows.append(
                {
                    "episode_id": str(index),
                    "origin_id": str(index),
                    "family_id": "family",
                    "dataset_id": "dataset",
                    "candidate_id": candidate,
                    "static.signal": 0.0,
                    "loss": loss,
                }
            )
    frame = pd.DataFrame(rows)
    selector = PairwiseUtilitySelector(n_estimators=10).fit(frame)
    _, scores, _ = selector.pair_scores(frame.drop(columns="loss"))
    np.testing.assert_allclose(scores, 54 / (54 + 600) - 0.5, atol=1e-7)
    first = selector.select(frame.drop(columns="loss"))
    second = selector.select(frame.assign(loss=-frame.loss * 1000))
    assert first.candidate_id.tolist() == second.candidate_id.tolist() == ["b"] * 60
    regression = PairwiseUtilitySelector(mode="regression", n_estimators=10).fit(frame)
    _, regression_scores, _ = regression.pair_scores(frame.drop(columns="loss"))
    np.testing.assert_allclose(regression_scores, -(6 * 100 - 54) / 60, atol=1e-7)
    assert regression.select(frame.drop(columns="loss")).candidate_id.tolist() == ["b"] * 60
