import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from forecast_correlation_core import (  # noqa: E402
    future_covariances,
    pooled_point,
    transport_outputs,
)


def test_future_covariance_matches_independent_innovation_factor():
    a = np.array([[0.7, 0.1], [0.2, 0.6]])
    q = np.array([[0.3, 0.05], [0.05, 0.2]])
    p = np.array([[0.2, 0.03], [0.03, 0.4]])
    b, state = np.array([0.1, -0.2]), np.array([1.0, 2.0])
    mean, covariance, initial = future_covariances(a, q, state, p, b, 5)
    factor = np.zeros((10, 12))
    for t in range(5):
        factor[2 * t : 2 * t + 2, :2] = np.linalg.matrix_power(a, t + 1) @ np.linalg.cholesky(p)
        for k in range(t + 1):
            factor[2 * t : 2 * t + 2, 2 * (k + 1) : 2 * (k + 2)] = np.linalg.matrix_power(
                a, t - k
            ) @ np.linalg.cholesky(q)
    np.testing.assert_allclose(covariance, factor @ factor.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(initial, factor[:, :2] @ factor[:, :2].T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mean[0], a @ state + b, rtol=1e-12, atol=1e-12)


def test_equal_covariance_is_exact_half_and_precision_form_agrees():
    rng = np.random.default_rng(38)
    a, b = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
    cv, cf = a @ a.T + np.eye(6), b @ b.T + np.eye(6)
    f, v = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
    point, used, _ = pooled_point(f, v, cv, cf)
    expected = np.linalg.solve(
        np.linalg.inv(cv) + np.linalg.inv(used),
        np.linalg.solve(cv, v.ravel()) + np.linalg.solve(used, f.ravel()),
    )
    np.testing.assert_allclose(point.ravel(), expected, rtol=1e-12, atol=1e-12)
    half, _, _ = pooled_point(f, v, cv, cv)
    np.testing.assert_array_equal(half, 0.5 * f + 0.5 * v)
    np.testing.assert_allclose(np.trace(used), np.trace(cv), rtol=1e-12, atol=1e-12)


def test_zero_foundation_spread_has_registered_safe_fallback():
    f, v = np.ones((2, 2)), np.zeros((2, 2))
    point, covariance, fallback = pooled_point(f, v, np.eye(4), np.zeros((4, 4)))
    assert fallback
    np.testing.assert_array_equal(point, np.full((2, 2), 0.5))
    np.testing.assert_array_equal(covariance, np.eye(4))
    methods, _, _ = transport_outputs(f, f, np.zeros_like(f), v, np.eye(4), np.zeros((4, 4)))
    assert len(methods) == 6 and all(np.isfinite(x).all() for x in methods.values())
