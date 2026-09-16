import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from observation_conditioning import conditional_forecasts  # noqa: E402


def test_no_observations_preserve_prior_and_do_not_invent_feedback():
    prior = np.tile(np.sin(np.arange(192) / 24)[:, None], (1, 2))
    quantiles = prior[..., None] + np.array([-1.0, 0.0, 1.0])
    context = np.full((96, 2), np.nan)
    result, records = conditional_forecasts(prior, quantiles, context)
    for prediction in result.values():
        np.testing.assert_array_equal(prediction, prior[96:])
    assert all(row["observed_count"] == 0 for row in records)
    assert np.isnan(context).all()


def test_two_observation_update_matches_explicit_two_by_two_inverse():
    prior = np.zeros((192, 2))
    quantiles = np.tile(np.array([-1.0, 0.0, 1.0]), (192, 2, 1))
    context = np.full((96, 2), np.nan)
    context[[0, 95], 0] = [2.0, -1.0]
    original = context.copy()
    result, records = conditional_forecasts(prior, quantiles, context)
    diagonal = 1.1 + 1e-8
    off_diagonal = 0.5 + 0.5 * np.exp(-95 / 96)
    determinant = diagonal**2 - off_diagonal**2
    weights = np.array([2 * diagonal + off_diagonal, -diagonal - 2 * off_diagonal]) / determinant
    future = np.arange(96, 192)
    cross = np.column_stack(
        [0.5 + 0.5 * np.exp(-future / 96), 0.5 + 0.5 * np.exp(-(future - 95) / 96)]
    )
    np.testing.assert_allclose(
        result["unit_conditioner"][:, 0], cross @ weights, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(result["unit_conditioner"][:, 1], prior[96:, 1])
    np.testing.assert_array_equal(context, original)
    assert np.all(result["last_innovation"][:, 0] == -1)
    assert np.all(result["mean_innovation"][:, 0] == 0.5)
    assert records[0]["observed_count"] == 2
    assert records[0]["minimum_remaining_variance"]["unit_conditioner"] >= 0


def test_unit_spread_equivalence_and_crossing_quantiles_leave_the_point_prior_intact():
    prior = np.zeros((192, 2))
    quantiles = np.tile(np.array([-ndtri(0.9), 0.0, ndtri(0.9)]), (192, 2, 1))
    context = np.full((96, 2), np.nan)
    context[::5] = np.array([0.5, -1.0])
    result, _ = conditional_forecasts(prior, quantiles, context)
    np.testing.assert_array_equal(result["unit_conditioner"], result["spread_conditioner"])
    permuted, _ = conditional_forecasts(prior, quantiles[..., [2, 0, 1]], context)
    for name in result:
        np.testing.assert_array_equal(result[name], permuted[name])
    np.testing.assert_array_equal(result["prior_slice"], np.zeros((96, 2)))
