import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from future_dependence import (
    covariance_intervention,
    dense_future_covariance,
    flip_probabilities,
    matched_future_components,
    projected_variances,
)  # noqa: E402
from readout_future_dependence import dependence_summaries  # noqa: E402


def test_projected_covariance_matches_dense_future_law():
    model = {
        "a": np.array([[0.7, 0.1], [0.0, 0.8]]),
        "q": np.array([[0.2, 0.03], [0.03, 0.1]]),
        "period": 12,
        "amplitude": np.zeros(2),
    }
    posterior = np.array([[0.2, 0.02], [0.02, 0.1]])
    scale = np.array([1.3, 0.7])
    horizon = 5
    covariance = dense_future_covariance(model, posterior, horizon, scale)
    rng = np.random.default_rng(15501)
    for slot in (-1, 0, 1):
        indices = np.arange(2 * horizon) if slot == -1 else np.arange(slot, 2 * horizon, 2)
        directions = rng.normal(size=(8, len(indices)))
        selected = covariance[np.ix_(indices, indices)]
        correlated, diagonal = projected_variances(
            model, posterior, directions, np.diag(selected), scale, slot
        )
        expected = (
            4 * np.einsum("ai,ij,aj->a", directions, selected, directions) / len(indices) ** 2
        )
        np.testing.assert_allclose(correlated, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            diagonal,
            4 * np.sum(directions**2 * np.diag(selected), axis=1) / len(indices) ** 2,
            rtol=1e-12,
            atol=1e-12,
        )


def test_covariance_intervention_preserves_marginals_and_original_samples():
    model = {
        "a": np.array([[0.7, 0.1], [0.0, 0.8]]),
        "q": np.eye(2) * 0.1,
        "period": 12,
        "amplitude": np.zeros(2),
    }
    mean, variance, correlated, independent = matched_future_components(
        model,
        np.array([0.1, -0.2]),
        np.eye(2) * 0.2,
        0,
        np.zeros(2),
        np.ones(2),
        15502,
        samples=30000,
        horizon=4,
    )
    np.testing.assert_array_equal(
        covariance_intervention(mean, correlated, independent, 1.0), correlated
    )
    for rho in (0.0, 0.5, 1.0):
        values = covariance_intervention(mean, correlated, independent, rho)
        np.testing.assert_allclose(values.mean(0), mean, rtol=0, atol=0.009)
        np.testing.assert_allclose(values.var(0), variance, rtol=0.03, atol=0.002)


def test_pairwise_flip_probability_handles_ties_and_zero_variance():
    result = flip_probabilities(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]))
    np.testing.assert_allclose(result, [0.0, 0.5, 0.15865525393145707], rtol=1e-12, atol=1e-12)


def test_aggregation_preserves_conditions_and_distinguishes_ratio_definitions():
    rows, risks = [], {}
    for rho in (0.0, 0.5, 1.0):
        for process, fixed, multiplier in (("a", 1.0, 1.0), ("b", 3.0, 2.0)):
            for history in range(2):
                value = np.array([0.0, 2 * fixed]) if history == 0 else np.array([2 * fixed, 0.0])
                risks[len(rows)] = (value, value)
                rows.append(
                    {
                        "model_id": "chronos2",
                        "process": process,
                        "mechanism": "random_point",
                        "missing_rate": 0.3,
                        "noise": 1.0,
                        "pool_size": 7,
                        "target_slot": -1,
                        "correlation": rho,
                        "origin_id": f"{process}{history}",
                        "conditional_single_mae": 0.0,
                        "conditional_single_mse": 0.0,
                        "optimism_single_mae": multiplier * (rho + 0.5),
                        "optimism_single_mse": multiplier * (rho + 0.5),
                    }
                )
    panels, summary = dependence_summaries(pd.DataFrame(rows), risks)
    assert len(panels) == 6 and len(summary) == 3
    for row in summary.itertuples():
        a = row.correlation + 0.5
        assert row.attainable_single_mse == 2.0
        np.testing.assert_allclose(row.ratio_of_mean_gains_mse, 1.5 * a / (2 + 1.5 * a))
        np.testing.assert_allclose(
            row.mean_process_optimism_fraction_mse, (a / (1 + a) + 2 * a / (3 + 2 * a)) / 2
        )
