from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.forecasting.registry import default_forecast_registry
from tsfm_fais.forecasting.runner import ForecastRunner


def test_vendor_preprocessing_preserves_observations_and_interpolates_gaps():
    vendor = pytest.importorskip("timesfm.timesfm_2p5.timesfm_2p5_base")
    values = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    result = vendor.linear_interpolation(vendor.strip_leading_nans(values.copy()))
    np.testing.assert_equal(result, [1.0, 2.0, 3.0, 3.0])
    np.testing.assert_equal(values, [np.nan, 1.0, np.nan, 3.0, np.nan])


def test_vendor_adapter_passes_raw_missing_inputs_to_the_official_interface():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_timesfm_vendor_missing.py"
    spec = importlib.util.spec_from_file_location("timesfm_vendor_missing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RecordingTimesFM:
        def __init__(self):
            self.inputs = []

        def forecast(self, *, horizon, inputs):
            self.inputs.extend(inputs)
            return np.ones((len(inputs), horizon)), np.ones((len(inputs), horizon, 10))

    backend = RecordingTimesFM()
    adapter = module.TimesFMVendorMissingAdapter(backend=backend)
    runner = ForecastRunner(default_forecast_registry(), {"timesfm2p5": adapter})
    contexts = np.arange(16.0).reshape(1, 8, 2)
    contexts[0, :2, 0], contexts[0, 4, 1] = np.nan, np.nan
    forecast_spec = ForecastSpec("timesfm2p5", "independent_univariate", 4, target_indices=(0, 1))
    with pytest.raises(ValueError, match="complete finite"):
        runner.predict(contexts, forecast_spec)
    result = runner.predict_missing(contexts, forecast_spec)
    np.testing.assert_equal(np.asarray(backend.inputs).T, contexts[0])
    assert result.point.shape == (1, 4, 2)
