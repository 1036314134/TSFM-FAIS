import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector
from tsfm_fais.routing.preforecast import (
    consensus_projection_costs,
    decision_keys,
    preforecast_inputs,
    projection_row_costs,
)


def test_current_forecasts_and_outcomes_cannot_enter_student_inputs():
    frame = pd.DataFrame(
        [
            {
                "episode_id": str(index),
                "candidate_id": action,
                "origin_id": str(index),
                "family_id": "family",
                "dataset_id": "data",
                "static.missing_fraction": 0.5,
                "response.leak": loss,
                "mae": loss,
                "mse": loss,
                "teacher_mse": loss,
                "loss": loss,
            }
            for index in range(60)
            for action, loss in (("a", 0.0), ("b", 1.0))
        ]
    )
    inputs = preforecast_inputs(frame)
    selector = PairwiseUtilitySelector(n_estimators=5).fit(inputs.assign(loss=frame.loss))
    assert selector.feature_names == ("static.missing_fraction",)
    poisoned = frame.assign(mae=-1e6, mse=-2e6, teacher_mse=1e6, loss=-9e6)
    poisoned["response.leak"] *= -100
    pd.testing.assert_frame_equal(inputs, preforecast_inputs(poisoned))
    assert (
        selector.select(inputs).candidate_id.tolist()
        == selector.select(preforecast_inputs(poisoned)).candidate_id.tolist()
    )
    with pytest.raises(ValueError, match="unaudited"):
        preforecast_inputs(frame.assign(**{"static.future_value": 1}))


def test_projection_costs_preserve_joint_and_separate_action_constraints():
    predictions = np.array([[[[0, 9]], [[1, 0]], [[9, 1]]]], dtype=float)
    costs = consensus_projection_costs(predictions)
    frame = pd.DataFrame(
        {"episode_index": [0, 0, 0], "candidate_id": ["b"] * 3, "target_slot": [-1, 0, 1]}
    )
    absolute, floor = projection_row_costs(frame, costs, ["a", "b", "c"])
    np.testing.assert_allclose(absolute, [0.5, 0, 1])
    np.testing.assert_allclose(floor, [0.5, 0, 0])
    assert np.all(absolute - floor >= 0)


def test_target_keys_keep_the_same_independent_history():
    frame = pd.DataFrame(
        {"episode_id": ["window", "window"], "target_slot": [0, 1], "origin_id": ["history"] * 2}
    )
    keyed = decision_keys(frame)
    assert keyed.episode_id.nunique() == 2
    assert keyed.source_episode_id.nunique() == keyed.origin_id.nunique() == 1
