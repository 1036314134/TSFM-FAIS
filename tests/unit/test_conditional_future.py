import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from conditional_future import (
    ForecastHull,
    condition_history,
    expected_risks,
    future_moments,
    gaussian_absolute,
    sample_futures,
    seasonal,
)  # noqa: E402

from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)  # noqa: E402


def process():
    return {
        "a": np.array([[0.6, 0.2], [0.1, 0.7]]),
        "q": np.array([[0.2, 0.03], [0.03, 0.1]]),
        "amplitude": np.array([0.5, 0.8]),
        "period": 12,
    }


def test_hindsight_gain_exists_without_conditional_single_selection_signal():
    from readout_conditional_risk import evaluate_decision

    points = np.array([[-1.0], [1.0]])
    future = np.random.default_rng(14541).normal(size=(20000, 1))
    hull = ForecastHull(points)
    metrics, _ = evaluate_decision(
        points, np.zeros(1), np.ones(1), future, hull.solve(np.zeros(1)), hull
    )
    np.testing.assert_allclose(metrics["conditional_single_mse"], 2.0)
    np.testing.assert_allclose(metrics["optimism_single_mse"], 2 * np.sqrt(2 / np.pi), atol=0.05)
    np.testing.assert_allclose(metrics["hindsight_single_independent_mse"], 2.0)
    np.testing.assert_allclose(metrics["conditional_convex_mse"], 1.0)
    assert metrics["hindsight_convex_independent_mse"] > metrics["conditional_convex_mse"]


def test_filter_matches_dense_gaussian_conditioning():
    model = process()
    a = model["a"]
    stationary = solve_discrete_lyapunov(a, model["q"])
    length = 4
    phase = 3
    context = seasonal(model, np.arange(phase, phase + length)) + np.arange(8).reshape(4, 2) / 10
    context[0, 1] = np.nan
    context[2, 0] = np.nan
    context[-1, 1] = np.nan
    covariance = np.empty((8, 8))
    for t in range(length):
        for s in range(length):
            block = (
                np.linalg.matrix_power(a, t - s) @ stationary
                if t >= s
                else stationary @ np.linalg.matrix_power(a, s - t).T
            )
            covariance[2 * t : 2 * t + 2, 2 * s : 2 * s + 2] = block
    observed = np.flatnonzero(np.isfinite(context.ravel()))
    residual = (context - seasonal(model, np.arange(phase, phase + length))).ravel()[observed]
    cross = covariance[-2:, observed]
    expected_mean = cross @ np.linalg.solve(covariance[np.ix_(observed, observed)], residual)
    expected_cov = covariance[-2:, -2:] - cross @ np.linalg.solve(
        covariance[np.ix_(observed, observed)], cross.T
    )
    mean, cov = condition_history(model, context, phase)
    scalar_mean, scalar_cov = condition_history(model, context, phase, scalar=True)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(mean, scalar_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cov, scalar_cov, rtol=1e-11, atol=1e-11)


def test_future_noise_changes_variance_not_mean_or_mse_ranking():
    model = process()
    mean = np.array([0.1, -0.2])
    covariance = np.array([[0.2, 0.01], [0.01, 0.1]])
    first, missing, innovation = future_moments(model, mean, covariance, 7, 8, 0.0)
    second, missing2, innovation2 = future_moments(model, mean, covariance, 7, 8, 2.0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(missing, missing2)
    assert (innovation == 0).all()
    points = np.stack([first + 0.2, first - 0.4, first + 0.7]).reshape(3, -1)
    risk1 = expected_risks(points, first.ravel(), (missing + innovation).ravel())[1]
    risk2 = expected_risks(points, second.ravel(), (missing2 + innovation2).ravel())[1]
    np.testing.assert_allclose(risk1 - risk1[0], risk2 - risk2[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gaussian_absolute(np.zeros(3), np.array([0.0, 1.0, 2.0])),
        np.array([0.0, 1.0, 2.0]) * np.sqrt(2 / np.pi),
    )


def test_sampler_matches_conditional_moments_and_zero_noise_control():
    model = process()
    mean = np.array([0.1, -0.2])
    covariance = np.array([[0.2, 0.01], [0.01, 0.1]])
    expected, missing, innovation = future_moments(model, mean, covariance, 4, 3, 0.5)
    samples = sample_futures(model, mean, covariance, 4, 3, 0.5, 40000, 14521)
    np.testing.assert_allclose(samples.mean(0), expected, rtol=0, atol=0.008)
    np.testing.assert_allclose(samples.var(0), missing + innovation, rtol=0.03, atol=0.002)
    deterministic = sample_futures(model, mean, np.zeros((2, 2)), 4, 3, 0.0, 5, 14521)
    np.testing.assert_array_equal(deterministic, np.repeat(deterministic[:1], 5, axis=0))


def test_cached_hull_matches_independent_face_solver():
    rng = np.random.default_rng(14531)
    points = rng.normal(size=(8, 12))
    targets = rng.normal(size=(13, 12))
    hull = ForecastHull(points)
    weight, prediction, gap, _ = hull.solve(targets)
    gram = forecast_geometry(points[None])[3]
    alignment = projection_targets(np.repeat(points[None], len(targets), axis=0), targets)[
        "raw_projection"
    ]
    reference = simplex_quadratic_weights(np.repeat(gram, len(targets), axis=0), alignment)[0]
    np.testing.assert_allclose(prediction, reference @ points, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(weight.sum(1), 1, rtol=0, atol=1e-10)
    assert gap.max() <= 1e-7
    duplicate = np.repeat(points[:1], 8, axis=0)
    _, same, _, _ = ForecastHull(duplicate).solve(targets)
    np.testing.assert_array_equal(same, np.repeat(points[:1], len(targets), axis=0))
