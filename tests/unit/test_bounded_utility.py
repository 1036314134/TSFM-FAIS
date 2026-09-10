from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.forecasting.metrics import macro_mase


def test_scaled_mae_difference_respects_prediction_distance_bound():
    rng = np.random.default_rng(721)
    truth, candidate, anchor = rng.normal(size=(3, 50, 96, 2))
    scales = np.array([0.01, 100.0])
    delta = macro_mase(candidate, truth, scales) - macro_mase(anchor, truth, scales)
    distance = np.mean(np.abs(candidate - anchor) / scales, axis=(1, 2))
    assert np.all(np.abs(delta) <= distance + 1e-10)
    np.testing.assert_array_equal(
        macro_mase(anchor, truth, scales) - macro_mase(anchor, truth, scales), 0
    )


def test_bounded_target_and_prediction_rescaling():
    path = Path(__file__).resolve().parents[2] / "scripts/analyze_utility_anchor_ablation.py"
    spec = importlib.util.spec_from_file_location("utility_anchor_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RecordingRegressor:
        def fit(self, matrix, target, sample_weight):
            self.target = target

        def predict(self, matrix):
            return np.array([2.0, -2.0, 0.5])

    base = RecordingRegressor()
    model = module.BoundedUtilityModel(base)
    matrix = pd.DataFrame({"response.mean_change": [0.0, 2.0, 4.0]})
    model.fit(matrix, np.array([0.0, -1.0, 1.0]), np.ones(3))
    np.testing.assert_allclose(base.target, [0.0, -0.5, 0.25])
    np.testing.assert_allclose(model.predict(matrix), [0.0, -2.0, 2.0])
    with pytest.raises(ValueError, match="bound"):
        model.fit(matrix, np.array([0.0, 3.0, 1.0]), np.ones(3))
