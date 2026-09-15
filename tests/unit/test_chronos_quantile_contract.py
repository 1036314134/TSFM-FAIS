from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.forecasting.accuracy import recover_legacy_chronos_median
from tsfm_fais.forecasting.adapters import Chronos2Adapter


class OfficialLayoutBackend:
    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, **kwargs):
        outputs = []
        for entry in inputs:
            values = np.asarray(entry["target"])
            n = values.shape[0] if values.ndim == 2 else 1
            outputs.append(
                100 * np.arange(n)[:, None, None]
                + 10 * np.arange(prediction_length)[None, :, None]
                + np.arange(len(quantile_levels))[None, None, :]
            )
        return outputs, None


@pytest.mark.parametrize("dimensions,horizon", [(3, 5), (3, 3), (5, 5), (2, 7)])
def test_chronos_axes_follow_sdk_contract_even_when_sizes_match(dimensions, horizon):
    targets = (dimensions - 1, 0)
    spec = ForecastSpec("chronos2", "joint_multivariate", horizon, target_indices=targets)
    result = Chronos2Adapter(backend=OfficialLayoutBackend()).predict(
        np.ones((2, 8, dimensions)), spec
    )
    expected = np.array(targets)[None, :] * 100 + np.arange(horizon)[:, None] * 10 + 1
    np.testing.assert_equal(result.point, np.stack([expected, expected]))
    np.testing.assert_equal(result.quantiles[..., 2] - result.quantiles[..., 0], 2)


def test_recovery_uses_the_retained_median_without_inventing_upper_quantiles():
    raw = (
        100 * np.arange(3)[None, :, None, None]
        + 10 * np.arange(5)[None, None, :, None]
        + np.arange(3)[None, None, None, :]
    )
    legacy = raw.transpose(0, 2, 3, 1)[:, :, [0, 1], :]
    recovered = recover_legacy_chronos_median(legacy, (0, 1), (0.1, 0.5, 0.9), 3)
    expected = np.array([[[10 * time + 1, 100 + 10 * time + 1] for time in range(5)]])
    np.testing.assert_equal(recovered, expected)
    with pytest.raises(ValueError, match="retain"):
        recover_legacy_chronos_median(legacy, (0, 2), (0.1, 0.5, 0.9), 3)


def test_gpu_repair_accepts_a_json_serialized_forecast_request():
    path = Path(__file__).parents[2] / "scripts/repair_chronos_layout_forecasts.py"
    spec = importlib.util.spec_from_file_location("chronos_layout_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module.request_from_json(
        {
            "model_id": "chronos2",
            "mode": "joint_multivariate",
            "horizon": 96,
            "context_length": 96,
            "target_indices": [0, 1],
            "quantile_levels": [0.1, 0.5, 0.9],
            "num_samples": 20,
        }
    )
    assert request.quantile_levels == (0.1, 0.5, 0.9)
    assert request.target_indices == (0, 1)
