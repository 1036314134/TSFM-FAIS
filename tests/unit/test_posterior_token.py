import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from chronos.chronos_bolt import InstanceNorm, ResidualBlock
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from posterior_token_core import (  # noqa: E402
    conditional_uncertainty,
    gaussian_relu_mean,
    moment_embedding,
    posterior_normalization,
    static_samples,
    transformed_moments,
)


def test_conditional_covariance_excludes_numerical_jitter_and_observed_uncertainty():
    covariance = np.array([[1.000001, 0.4], [0.4, 2.000001]])
    context = np.array([[1.0, np.nan], [np.nan, np.nan], [2.0, 3.0]])
    posterior, roots, _ = conditional_uncertainty(context, covariance, np.ones(2, bool))
    expected = 2.0 - 0.4**2 / 1.000001
    np.testing.assert_allclose(posterior[0], [[0, 0], [0, expected]], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(posterior[1], covariance - 1e-6 * np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(posterior[2], np.zeros((2, 2)))
    np.testing.assert_allclose(roots @ roots.transpose(0, 2, 1), posterior, rtol=1e-12, atol=1e-12)
    constant = np.diag([1e-6, 1 + 1e-6])
    p, _, _ = conditional_uncertainty(np.full((1, 2), np.nan), constant, np.ones(2, bool))
    assert p[0, 0, 0] == 0


def test_fixed_samples_keep_original_encoded_observations_and_seed():
    context = np.array([[1.0, np.nan], [np.nan, 3.0], [2.0, np.nan]])
    observed = np.isfinite(context).T
    mean = np.nan_to_num(context).T.astype(np.float32)
    _, roots, _ = conditional_uncertainty(
        context, np.array([[1.000001, 0.2], [0.2, 1.000001]]), np.ones(2, bool)
    )
    samples, seed = static_samples(mean, roots, observed, "test")
    again, repeated = static_samples(mean, roots, observed, "test")
    assert seed == repeated
    np.testing.assert_array_equal(samples, again)
    for value in samples:
        np.testing.assert_array_equal(value[observed], mean[observed])


def test_posterior_normalization_uses_expected_population_variance():
    norm = InstanceNorm()
    mean = torch.tensor([[-1.0, 1.0]])
    variance = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    loc, scale, _, contribution = posterior_normalization(norm, mean, variance)
    torch.testing.assert_close(loc, torch.zeros((1, 1)), rtol=0, atol=0)
    torch.testing.assert_close(contribution, torch.full((1, 1), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(scale.square(), torch.full((1, 1), 1.5), rtol=1e-6, atol=1e-6)
    _, expected = norm(mean)
    actual = posterior_normalization(norm, mean, torch.zeros_like(variance))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_relu_moment_matches_closed_form_and_deterministic_limit():
    mean = np.array([-2.0, 0.0, 2.0])
    variance = np.array([0.1, 1.0, 4.0])
    scale = np.sqrt(variance)
    expected = scale * np.exp(-0.5 * (mean / scale) ** 2) / np.sqrt(2 * np.pi) + mean * ndtr(
        mean / scale
    )
    actual = gaussian_relu_mean(torch.tensor(mean), torch.tensor(variance)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    zero = gaussian_relu_mean(torch.tensor(mean), torch.zeros(3, dtype=torch.float64)).numpy()
    np.testing.assert_array_equal(zero, np.maximum(mean, 0))


def test_negative_tail_relu_expectation_is_nonnegative_and_accurate():
    values = np.linspace(-10, -5, 1001)
    expected = np.exp(-0.5 * values**2) / np.sqrt(2 * np.pi) + values * ndtr(values)
    actual = gaussian_relu_mean(torch.tensor(values), torch.ones(1001, dtype=torch.float64)).numpy()
    assert (actual >= 0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-30)


def test_arcsinh_quadrature_and_zero_variance_coordinates():
    norm = InstanceNorm(use_arcsinh=True)
    mean = torch.tensor([[-1.0, 0.5, 2.0]])
    variance = torch.tensor([[0.0, 0.2, 0.8]], dtype=torch.float64)
    loc, scale = torch.tensor([[0.1]]), torch.tensor([[1.2]])
    average, var = transformed_moments(norm, mean, variance, loc, scale)
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    values = np.arcsinh(
        (
            mean.numpy().astype(float)[..., None]
            - float(loc.item())
            + np.sqrt(2 * variance.numpy())[..., None] * nodes
        )
        / float(scale.item())
    )
    expected = (values * (weights / np.sqrt(np.pi))).sum(-1)
    np.testing.assert_allclose(average.numpy()[:, 1:], expected[:, 1:], rtol=1e-12, atol=1e-12)
    point, _ = norm(mean, (loc, scale))
    assert average[0, 0] == point[0, 0]
    assert var[0, 0] == 0 and torch.all(var >= 0)


def test_zero_variance_input_block_is_exact_and_moments_can_leave_point_curve():
    torch.manual_seed(37)
    block = ResidualBlock(in_dim=6, h_dim=8, out_dim=4, act_fn_name="relu", dropout_p=0).eval()
    patches = torch.randn(2, 3, 6)
    values = patches[..., 2:4].reshape(2, 6).double()
    actual = moment_embedding(block, patches, values, torch.zeros_like(values), "moment")
    torch.testing.assert_close(actual, block(patches), rtol=0, atol=0)
    # For phi(x)=(x,relu(x)), E phi(N(0,1)) has no single-x representation.
    expected_relu = gaussian_relu_mean(torch.tensor(0.0), torch.tensor(1.0)).item()
    assert expected_relu > 0 and torch.relu(torch.tensor(0.0)).item() == 0
