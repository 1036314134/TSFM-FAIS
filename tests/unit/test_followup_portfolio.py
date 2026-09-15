import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.followup_portfolio import portfolio_feature_frame, selected_portfolio_points
from tsfm_fais.routing.preforecast import STATIC_FEATURES


@pytest.mark.parametrize("joint", [False, True])
def test_query_order_and_outcome_columns_do_not_change_portfolio_inputs(joint):
    actions = [
        "locf",
        "linear_interp",
        "seasonal_lag",
        "knn_multivariate",
        "saits",
        "timemixerpp",
        "guarded_direct",
    ]
    rng = np.random.default_rng(91)
    points = rng.normal(size=(7, 96, 2))
    rows = []
    for slot in [-1] if joint else [0, 1]:
        for action in actions:
            rows.append(
                {
                    "episode_id": f"case-{slot}",
                    "target_slot": slot,
                    "candidate_id": action,
                    **dict(zip(STATIC_FEATURES, rng.normal(size=21), strict=True)),
                }
            )
    frame = pd.DataFrame(rows)
    first, options, decisions, names = portfolio_feature_frame(
        frame, points, actions, np.zeros(2), joint=joint
    )
    reversed_actions = actions[::-1]
    second, other_options, _, _ = portfolio_feature_frame(
        frame.iloc[::-1].assign(future_mse=1e12, teacher_label=-1e12),
        points[::-1],
        reversed_actions,
        np.zeros(2),
        joint=joint,
    )
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(options, other_options)
    assert len(first) == (35 if joint else 70)
    assert "future_mse" not in first and "teacher_label" not in first
    # Check an actual portfolio against the original forecast bank, not the helper output.
    wanted = "median:" + "+".join(sorted(["locf", "saits", "guarded_direct"]))
    choices = decisions[["episode_id"]].assign(candidate_id=wanted)
    actual = selected_portfolio_points(choices, options, decisions, names, joint=joint)
    expected = np.median(
        points[[actions.index(name) for name in ("locf", "saits", "guarded_direct")]], axis=0
    )
    np.testing.assert_array_equal(actual, expected)
