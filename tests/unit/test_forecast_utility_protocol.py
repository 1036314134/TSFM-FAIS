from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.contracts import CandidateResult, ForecastResult, ForecastSpec
from tsfm_fais.forecasting.adapters import Chronos2Adapter
from tsfm_fais.forecasting.metrics import macro_mase, training_mase_scale
from tsfm_fais.forecasting.registry import default_forecast_registry
from tsfm_fais.forecasting.runner import ForecastRunner
from tsfm_fais.routing.teacher import TeacherBuilder


def test_prefix_scales_preserve_target_weights_when_context_changes():
    prefix = np.arange(12, dtype=float)[:, None] * np.array([[10.0, 1.0]])
    scales, lag = training_mase_scale(prefix, 1)
    assert lag == 1

    def forecast(contexts, spec):
        return ForecastResult(contexts[:, -1:, :], (0, 1))

    spec = ForecastSpec("mock", "joint_multivariate", 1, target_indices=(0, 1))
    candidates = {
        name: CandidateResult(name, np.tile(value, (1, 4, 1)), np.ones((1, 4, 2), bool))
        for name, value in {"a": [1.0, 2.0], "b": [2.0, 1.0]}.items()
    }
    builder = TeacherBuilder(forecast, mase_scales=dict(enumerate(scales)))
    truth = np.zeros((1, 1, 2))
    for context in [np.zeros((1, 4, 2)), np.arange(8).reshape(1, 4, 2) * 100]:
        losses = builder.candidate_losses_batched(context, truth, candidates, spec)
        assert losses["a"] == pytest.approx(1.05)
        assert losses["b"] == pytest.approx(0.6)
        for name, candidate in candidates.items():
            expected = macro_mase(forecast(candidate.values, spec).point, truth, scales)[0]
            assert losses[name] == pytest.approx(expected)


def test_prefix_seasonal_lag_does_not_depend_on_short_forecast_context():
    scales, lag = training_mase_scale(np.arange(300)[:, None].astype(float), 96)
    assert lag == 96
    np.testing.assert_allclose(scales, [96])
    builder = TeacherBuilder(lambda *_: None, seasonality=96, mase_scales={0: scales[0]})
    assert builder._scales(np.zeros((1, 96, 1)), (0,)) == {0: 96}


@pytest.mark.parametrize("scales", [{}, {0: 0}, {0: np.nan}, {0: -1}])
def test_teacher_rejects_invalid_fixed_scales(scales):
    with pytest.raises(ValueError, match="MASE scales"):
        TeacherBuilder(lambda *_: None, mase_scales=scales)


def test_teacher_rejects_missing_target_scale():
    builder = TeacherBuilder(lambda *_: None, mase_scales={0: 1})
    with pytest.raises(ValueError, match="all forecast targets"):
        builder._scales(np.zeros((1, 4, 2)), (0, 1))


class RecordingChronos:
    def __init__(self):
        self.received = []

    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, **kwargs):
        self.received.extend(inputs)
        return [
            np.ones((np.asarray(entry["target"]).shape[0], prediction_length, len(quantile_levels)))
            for entry in inputs
        ], None


def test_native_missing_is_explicit_and_preserves_observed_values_and_nans():
    backend = RecordingChronos()
    adapter = Chronos2Adapter(backend=backend)
    runner = ForecastRunner(default_forecast_registry(), {"chronos2": adapter})
    contexts = np.arange(16, dtype=float).reshape(1, 8, 2)
    contexts[0, 3:5, 0] = np.nan
    spec = ForecastSpec("chronos2", "joint_multivariate", 4, target_indices=(0, 1))
    with pytest.raises(ValueError, match="complete finite"):
        runner.predict(contexts, spec)
    result = runner.predict_missing(contexts, spec)
    np.testing.assert_equal(backend.received[0]["target"], contexts[0].T)
    assert result.point.shape == (1, 4, 2)
    contexts[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        runner.predict_missing(contexts, spec)


def test_native_missing_rejects_unsupported_adapter():
    runner = ForecastRunner(default_forecast_registry(), {"timesfm2p5": object()})
    with pytest.raises(ValueError, match="does not support native missing"):
        runner.predict_missing(
            np.full((1, 8, 2), np.nan),
            ForecastSpec("timesfm2p5", "independent_univariate", 4),
        )
