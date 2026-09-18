import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from peer_outage_core import PrefixRegression, bridge_residual, case_context  # noqa: E402


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.9, 0.99])
def test_ar_bridge_matches_independent_gaussian_conditioning(phi):
    anchors = np.array([0, 10])
    residuals = np.array([1.5, -0.3])
    covariance = phi ** abs(anchors[:, None] - anchors[None, :])
    for t in (1, 4, 8):
        cross = phi ** abs(t - anchors)
        expected = cross @ np.linalg.solve(covariance, residuals)
        assert bridge_residual(phi, t, anchors, residuals) == pytest.approx(expected, abs=1e-12)
    assert bridge_residual(phi, 14, anchors, residuals) == pytest.approx(phi**4 * residuals[-1])


def test_hidden_targets_and_actual_future_cannot_change_prepared_context():
    full = np.arange(400 * 17, dtype=float).reshape(400, 17)
    changed = full.copy()
    changed[276:300, :2] = 1e100
    changed[300:] = -1e100
    row = {"origin": 300, "panel": "synthetic_outage_h24"}
    a, b = case_context(full, row), case_context(changed, row)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[-24:, :2]).all()
    np.testing.assert_array_equal(a[:, 2:], full[108:300, 2:])


def test_ridge_matches_augmented_least_squares_and_protects_observations():
    random = np.random.default_rng(31)
    prefix = random.normal(size=(640, 4))
    prefix[:, 0] = 1.4 + 0.7 * prefix[:, 2] - 0.2 * prefix[:, 3] + 0.05 * random.normal(size=640)
    model = PrefixRegression(prefix)
    fit = model.fit(0, [2, 3])
    x = np.column_stack([np.ones(640), model.z[:, [2, 3]]])
    penalty = np.diag([0, np.sqrt(0.001), np.sqrt(0.001)])
    expected = np.linalg.lstsq(
        np.vstack([x / np.sqrt(640), penalty]),
        np.r_[model.z[:, 0] / np.sqrt(640), np.zeros(3)],
        rcond=None,
    )[0]
    np.testing.assert_allclose(fit["beta"], expected, atol=1e-12, rtol=1e-12)
    context = prefix[-192:].copy()
    context[80:88, :2] = np.nan
    context[-24:, :2] = np.nan
    observed = np.isfinite(context[:, :2])
    for repaired in model.targets(context, np.zeros((192, 2))).values():
        np.testing.assert_array_equal(repaired[observed], context[:, :2][observed])
        assert np.isfinite(repaired).all()
    np.testing.assert_array_equal(model.prefix, prefix)


def test_empty_predictors_use_declared_fallback():
    random = np.random.default_rng(8)
    model = PrefixRegression(random.normal(size=(256, 4)))
    x = random.normal(size=(192, 4))
    x[-3:] = np.nan
    fallback = np.full((192, 2), 42.0)
    for repaired in model.targets(x, fallback).values():
        np.testing.assert_array_equal(repaired[-3:], fallback[-3:])
