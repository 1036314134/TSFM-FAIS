import numpy as np

from tsfm_fais.imputers.gaussian_bridge import conditional_ar1


def test_banded_conditioning_matches_dense_gaussian_conditional():
    rho = 0.83
    context = np.array([np.nan, 1.1, np.nan, np.nan, -0.4, np.nan])[:, None]
    missing, observed = (
        np.flatnonzero(np.isnan(context[:, 0])),
        np.flatnonzero(np.isfinite(context[:, 0])),
    )
    noise = np.zeros((len(missing), len(context), 1))
    noise[np.arange(len(missing)), missing, 0] = 1
    mean, samples = conditional_ar1(context, noise, rho=rho)
    kernel = rho ** np.abs(np.arange(len(context))[:, None] - np.arange(len(context)))
    cross = kernel[np.ix_(missing, observed)]
    reference_mean = cross @ np.linalg.solve(
        kernel[np.ix_(observed, observed)], context[observed, 0]
    )
    covariance = kernel[np.ix_(missing, missing)] - cross @ np.linalg.solve(
        kernel[np.ix_(observed, observed)], cross.T
    )
    transform = (samples[:, missing, 0] - mean[missing, 0]).T
    np.testing.assert_allclose(mean[missing, 0], reference_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(transform @ transform.T, covariance, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        samples[:, observed, 0], np.repeat(context[observed, 0][None], len(missing), axis=0)
    )


def test_antithetic_draws_preserve_mean_and_observations():
    context = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, 0.2]])
    noise = np.random.default_rng(6101).normal(size=(4, *context.shape))
    noise = np.stack([noise, -noise], axis=1).reshape(8, *context.shape)
    mean, samples = conditional_ar1(context, noise, rho=0.9)
    np.testing.assert_allclose(samples.mean(axis=0), mean, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(mean[np.isfinite(context)], context[np.isfinite(context)])


def test_full_missing_and_single_position_stationary_prior():
    noise = np.eye(5)[:, :, None]
    mean, samples = conditional_ar1(np.full((5, 1), np.nan), noise, rho=0.9)
    reference = 0.9 ** np.abs(np.arange(5)[:, None] - np.arange(5))
    np.testing.assert_array_equal(mean, 0)
    np.testing.assert_allclose(samples[:, :, 0].T @ samples[:, :, 0], reference, atol=1e-12)
    mean, samples = conditional_ar1([[np.nan]], [[[1.0]], [[-1.0]]], rho=0.99)
    np.testing.assert_array_equal(mean, [[0.0]])
    np.testing.assert_array_equal(samples[:, 0, 0], [1, -1])
