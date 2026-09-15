from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.forecasting.accuracy import (
    PrefixStandardizer,
    forecast_errors,
    guarded_direct_forecast,
)


def test_standardizer_uses_population_std_and_preserves_constant_channel():
    scaler = PrefixStandardizer.fit(np.array([[1.0, 8.0], [3.0, 8.0], [np.nan, 8.0]]))
    np.testing.assert_allclose(scaler.mean, [2, 8])
    np.testing.assert_allclose(scaler.scale, [1, 1])
    np.testing.assert_array_equal(scaler.constant, [False, True])


def test_downstream_mae_mse_are_in_fixed_standardized_units():
    truth = np.array([[10.0, 100.0], [10.0, 100.0]])
    point = np.array([[[12.0, 110.0], [14.0, 130.0]]])
    result = forecast_errors(point, truth, np.array([2.0, 10.0]))
    np.testing.assert_allclose(result["mae"], [[1.5, 2]])
    np.testing.assert_allclose(result["mse"], [[2.5, 5]])
    np.testing.assert_allclose(result["raw_mae"], [[3, 20]])
    np.testing.assert_allclose(result["raw_mse"], [[10, 500]])


def test_forecast_truth_cannot_be_imputed_for_primary_scoring():
    with pytest.raises(ValueError, match="complete truth"):
        forecast_errors(np.zeros((1, 2, 1)), np.array([[1.0], [np.nan]]), np.ones(1))


def test_insufficient_history_cannot_define_a_standardizer():
    with pytest.raises(ValueError, match="two historical"):
        PrefixStandardizer.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_direct_fallback_respects_forecaster_dependency_and_uses_no_outcome():
    context = np.array([[1.0, np.nan, 4.0], [2.0, np.nan, 5.0]])
    direct = np.array([[100.0, 200.0], [100.0, 200.0]])
    fallback = np.array([[3.0, 6.0], [3.0, 6.0]])
    independent, flags = guarded_direct_forecast(context, [0, 1], direct, fallback, joint=False)
    np.testing.assert_equal(independent, [[100, 6], [100, 6]])
    np.testing.assert_equal(flags, [False, True])
    joint, flags = guarded_direct_forecast(context, [0, 1], direct, fallback, joint=True)
    np.testing.assert_equal(joint, fallback)
    np.testing.assert_equal(flags, [True, True])
