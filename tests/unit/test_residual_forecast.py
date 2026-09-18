import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from peer_outage_core import PrefixRegression  # noqa: E402
from residual_forecast_experiment import decompose, simple_residuals  # noqa: E402


def fixture():
    random = np.random.default_rng(9301)
    prefix = random.normal(size=(640, 17))
    prefix[:, 0] = 0.8 * prefix[:, 11] + 0.2 * random.normal(size=640)
    prefix[:, 1] = -0.4 * prefix[:, 14] + 0.3 * random.normal(size=640)
    x = random.normal(size=(192, 17))
    x[-12:, :2] = np.nan
    x[80, 2] = np.nan
    return (
        PrefixRegression(prefix),
        {"context": x, "keep": np.ones(17, bool)},
        random.normal(size=(24, 17)),
    )


def test_residual_labels_require_all_original_predictor_observations():
    stats, data, future = fixture()
    result = decompose(stats, data, future)
    assert np.isnan(result["residuals"][-12:]).all()
    assert np.isnan(result["residuals"][80]).all()
    for slot, model in enumerate(json.loads(str(result["models"]))):
        features = model["features"]
        assert 0 not in features and 1 not in features
        z = (data["context"] - stats.mean) / stats.scale
        valid = np.isfinite(result["residuals"][:, slot])
        component = model["beta"][0] + z[valid][:, features] @ model["beta"][1:]
        np.testing.assert_allclose(
            component + result["residuals"][valid, slot], z[valid, slot], rtol=1e-12, atol=1e-12
        )


def test_predicted_target_values_cannot_enter_shared_component():
    stats, data, future = fixture()
    baseline = decompose(stats, data, future)
    changed = future.copy()
    changed[:, :2] = 1e20
    replay = decompose(stats, data, changed)
    for key in ("component", "residuals", "residual_means", "residual_scales"):
        np.testing.assert_array_equal(replay[key], baseline[key])


def test_residual_statistics_are_frozen_by_the_prefix():
    stats, data, future = fixture()
    baseline = decompose(stats, data, future)
    data["context"] = data["context"] * 100
    changed = decompose(stats, data, future * -50)
    np.testing.assert_array_equal(baseline["residual_means"], changed["residual_means"])
    np.testing.assert_array_equal(baseline["residual_scales"], changed["residual_scales"])


def test_no_available_covariates_reduces_to_target_residual():
    stats, data, future = fixture()
    data["context"][:, 2:] = np.nan
    result = decompose(stats, data, future)
    np.testing.assert_array_equal(result["component"], np.zeros((24, 2)))
    np.testing.assert_array_equal(
        result["residuals"], ((data["context"] - stats.mean) / stats.scale)[:, :2]
    )
    assert all(not m["features"] for m in json.loads(str(result["models"])))


def test_ar_forecast_accounts_for_gap_age_and_empty_residual_history():
    residuals = np.full((192, 2), np.nan)
    residuals[189, 0] = 2.0
    models = [
        {"residual_mean": 0.5, "residual_ar1": 0.4},
        {"residual_mean": 0.5, "residual_ar1": 0.4},
    ]
    last, ar = simple_residuals(
        {"residuals": residuals, "models": np.asarray(json.dumps(models))}, 3
    )
    np.testing.assert_array_equal(last[:, 0], np.full(3, 2.0))
    np.testing.assert_allclose(
        ar[:, 0], 0.5 + np.power(0.4, [3, 4, 5]) * 1.5, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_array_equal(last[:, 1], np.zeros(3))
    np.testing.assert_array_equal(ar[:, 1], np.zeros(3))
