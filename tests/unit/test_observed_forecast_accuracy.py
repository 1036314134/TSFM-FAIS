from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.forecasting.accuracy import forecast_errors
from tsfm_fais.forecasting.observed_accuracy import observed_future_errors


def test_observed_scoring_equals_complete_future_scoring_when_all_labels_exist():
    prediction = np.arange(24.0).reshape(3, 4, 2)
    truth = np.arange(8.0).reshape(4, 2)
    scales = np.array([2.0, 10.0])
    expected = forecast_errors(prediction, truth, scales)
    actual, counts = observed_future_errors(
        prediction, truth, np.ones_like(truth, bool), scales, minimum_observed=2
    )
    for metric in expected:
        np.testing.assert_allclose(actual[metric], expected[metric])
    np.testing.assert_equal(counts, [4, 4])


def test_observed_scoring_preserves_target_weight_with_unequal_observation_counts():
    truth = np.array([[0.0, 0.0], [0.0, np.nan], [0.0, np.nan], [0.0, 0.0]])
    prediction = np.array([[[2.0, 10.0], [2.0, 1e12], [2.0, -1e12], [2.0, 10.0]]])
    scores, counts = observed_future_errors(
        prediction, truth, np.isfinite(truth), [2.0, 2.0], minimum_observed=2
    )
    np.testing.assert_equal(counts, [4, 2])
    np.testing.assert_allclose(scores["mae"], [[1.0, 5.0]])
    np.testing.assert_allclose(scores["mse"], [[1.0, 25.0]])
    assert scores["mae"].mean(axis=1)[0] == 3.0
    prediction[:, 1:3, 1] = 0.0
    second, _ = observed_future_errors(
        prediction, truth, np.isfinite(truth), [2.0, 2.0], minimum_observed=2
    )
    for metric in scores:
        np.testing.assert_equal(second[metric], scores[metric])


def test_imputed_future_values_cannot_replace_unavailable_source_labels():
    observed = np.array([[True], [False], [True]])
    with pytest.raises(ValueError, match="original observed mask"):
        observed_future_errors(
            np.zeros((1, 3, 1)), np.zeros((3, 1)), observed, [1], minimum_observed=2
        )


def test_ineligible_target_cannot_disappear_into_macro_averaging():
    truth = np.array([[0.0, 0.0], [0.0, np.nan]])
    with pytest.raises(ValueError, match="each target"):
        observed_future_errors(
            np.zeros((1, 2, 2)), truth, np.isfinite(truth), [1, 1], minimum_observed=2
        )
