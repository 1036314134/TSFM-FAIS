import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.forecast_response import (
    FORECAST_FEATURES,
    RESPONSE_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.utility import response_features


def test_candidate_forecasts_are_available_but_outcomes_and_clean_history_are_excluded():
    frame = pd.DataFrame(
        [
            {
                "episode_id": "window",
                "candidate_id": action,
                **dict.fromkeys(FORECAST_FEATURES, 0.0),
                "response.mean_change": float(index),
                "mae": index,
                "mse": index,
                "teacher_mse": index,
                "clean_history": index,
                "loss": index,
            }
            for index, action in enumerate(("a", "b"))
        ]
    )
    inputs = forecast_response_inputs(frame)
    assert inputs["response.mean_change"].tolist() == [0.0, 1.0]
    pd.testing.assert_frame_equal(
        inputs,
        forecast_response_inputs(
            frame.assign(mae=-100, mse=-200, teacher_mse=300, clean_history=400, loss=500)
        ),
    )
    with pytest.raises(ValueError, match="audited"):
        forecast_response_inputs(frame.assign(**{"response.current_truth": 1.0}))


def test_response_inventory_matches_the_exported_forecast_feature_contract():
    values = np.zeros((96, 2))
    features = response_features(values, values, values, np.zeros(2), np.ones(2))
    assert set(features) == set(RESPONSE_FEATURES)
