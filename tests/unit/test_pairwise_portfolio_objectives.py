from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.aligned_portfolio import option_catalog, option_vectors
from tsfm_fais.routing.forecast_projection import projection_targets
from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector

ACTIONS = (
    "locf",
    "linear_interp",
    "seasonal_lag",
    "knn_multivariate",
    "saits",
    "timemixerpp",
    "guarded_direct",
)


@pytest.fixture
def matched(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_pairwise_portfolio

    return train_pairwise_portfolio


def test_member_and_median_costs_use_the_same_reference_and_different_targets(matched):
    points = np.array(
        [
            [
                [1.0, 1.0],
                [1.1, 1.1],
                [1.2, 1.2],
                [-10.0, 0.0],
                [0.0, -10.0],
                [20.0, 20.0],
                [30.0, 30.0],
            ]
        ]
    )
    _, members = option_catalog(ACTIONS)
    vectors = option_vectors(points, members)
    labels = projection_targets(vectors, np.zeros((1, 2)), anchor=vectors[:, -1])["direct_risk"]
    member = matched.triple_labels(labels, [list(group) for group in members], "member_risk")
    median = matched.triple_labels(labels, members, "median_risk")
    anchor_risk = np.square(vectors[:, -1]).mean(axis=1)
    expected_member = np.array(
        [[np.square(points[0, group]).mean() - anchor_risk[0] for group in members[7:42]]]
    )
    np.testing.assert_allclose(member, expected_member)
    np.testing.assert_allclose(
        median, np.square(vectors[:, 7:42]).mean(axis=2) - anchor_risk[:, None]
    )
    assert members[7 + member.argmin()] == (0, 1, 2)
    assert members[7 + median.argmin()] == (0, 3, 4)


def test_explicit_membership_prefix_preserves_default_pairwise_inputs():
    frame = pd.DataFrame(
        [
            {
                "episode_id": str(index),
                "origin_id": str(index),
                "family_id": "family",
                "dataset_id": "dataset",
                "candidate_id": candidate,
                "static.context": float(index),
                "member.signal": float(candidate == "a"),
                "loss": float(candidate == ("a" if index < 30 else "b")),
            }
            for index in range(60)
            for candidate in ("a", "b")
        ]
    )
    default = PairwiseUtilitySelector(n_estimators=5).fit(frame)
    expanded = PairwiseUtilitySelector(
        n_estimators=5, feature_prefixes=("static.", "response.", "member.")
    ).fit(frame)
    assert default.feature_names == ("static.context",)
    assert expanded.feature_names == ("member.signal", "static.context")
    expected = expanded.select(frame.drop(columns="loss"))
    actual = expanded.select(frame.assign(loss=frame.loss + 99999))
    assert expected.candidate_id.tolist() == actual.candidate_id.tolist()


def test_triple_rows_preserve_label_alignment(matched):
    names, _ = option_catalog(ACTIONS)
    feature_names = ["static." + str(index) for index in range(40)]
    decisions = pd.DataFrame(
        {"episode_id": ["first", "second"], "family_id": "f", "dataset_id": "d"}
    )
    features = np.broadcast_to(np.arange(43)[None, :, None], (2, 43, 40)).copy()
    frame = matched.triple_rows(
        decisions, features, {"feature_names": feature_names, "option_names": list(names)}, [1, 0]
    )
    assert frame.episode_id.drop_duplicates().tolist() == ["second", "first"]
    assert frame.candidate_id.tolist() == list(names[7:42]) * 2
    np.testing.assert_array_equal(frame["static.0"], np.tile(np.arange(7, 42), 2))
