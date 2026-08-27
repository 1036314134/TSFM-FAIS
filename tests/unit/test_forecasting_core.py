from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.forecasting.base import ForecastAdapterSpec, NativeForecast
from tsfm_fais.forecasting.registry import ForecastRegistry
from tsfm_fais.forecasting.runner import ForecastRunner


class MockUnivariate:
    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        contexts = np.asarray(contexts)
        point = np.repeat(contexts[:, -1:], horizon, axis=1)
        return NativeForecast(
            point=point,
            quantiles=np.repeat(point[..., None], len(quantile_levels), axis=-1),
        )


class MockJoint:
    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        contexts = np.asarray(contexts)
        point = np.repeat(contexts[:, -1:, :], horizon, axis=1)
        return NativeForecast(
            point=point,
            quantiles=np.repeat(point[..., None], len(quantile_levels), axis=-1),
        )


def _registry():
    registry = ForecastRegistry()
    registry.register(
        ForecastAdapterSpec("uni", "independent_univariate", "x:y", "mock", "none", 16)
    )
    registry.register(ForecastAdapterSpec("joint", "joint_multivariate", "x:y", "mock", "none", 16))
    return registry


def test_univariate_runner_expands_targets_and_reassembles():
    values = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3)
    runner = ForecastRunner(_registry(), {"uni": MockUnivariate()})
    result = runner.predict(
        values,
        ForecastSpec("uni", "independent_univariate", horizon=4, target_indices=(0, 2)),
    )
    assert result.point.shape == (2, 4, 2)
    assert result.quantiles.shape == (2, 4, 2, 3)
    metrics = runner.resource_metrics()
    assert metrics["forecast_call_count"] == 1
    assert metrics["forecast_context_count"] == 2
    assert metrics["forecast_underlying_series_count"] == 4
    assert metrics["forecast_runtime_seconds"] >= 0.0
    assert metrics["peak_cuda_memory_allocated_bytes"] >= 0


def test_joint_runner_selects_target_columns_after_prediction():
    values = np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3)
    runner = ForecastRunner(_registry(), {"joint": MockJoint()})
    result = runner.predict(
        values,
        ForecastSpec("joint", "joint_multivariate", horizon=4, target_indices=(1,)),
    )
    assert result.point.shape == (2, 4, 1)


def test_forecaster_refuses_missing_context():
    values = np.ones((1, 8, 2))
    values[0, 2, 0] = np.nan
    runner = ForecastRunner(_registry(), {"uni": MockUnivariate()})
    with pytest.raises(ValueError, match="complete finite"):
        runner.predict(values, ForecastSpec("uni", "independent_univariate", horizon=2))
