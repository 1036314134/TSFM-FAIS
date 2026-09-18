import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from dynamic_posterior_core import (  # noqa: E402
    complete_raw,
    covariance_root,
    dense_conditional_tail,
    fit_dynamics,
    posterior_samples,
    predictive_mixture,
    quantile_weights,
    smooth_history,
)


def model_and_history():
    model = {
        "a": np.array([[0.7, 0.1, 0.0], [0.0, 0.6, 0.1], [0.1, 0.0, 0.5]]),
        "b": np.array([0.1, -0.1, 0.0]),
        "q": np.array([[0.3, 0.1, 0.0], [0.1, 0.4, 0.1], [0.0, 0.1, 0.5]]),
        "initial_mean": np.zeros(3),
        "initial_covariance": np.eye(3),
    }
    z = np.array(
        [
            [0.2, np.nan, 0.7],
            [np.nan, -0.2, np.nan],
            [0.5, 0.1, 0.0],
            [np.nan, np.nan, 0.4],
            [0.3, np.nan, np.nan],
            [np.nan, -0.1, 0.8],
        ]
    )
    return model, z


def test_smoother_and_sampling_covariance_match_dense_gaussian_conditioning():
    model, z = model_and_history()
    state = smooth_history(z, model)
    expected_mean, expected_cov = dense_conditional_tail(
        model["initial_mean"], model["initial_covariance"], model["a"], model["b"], model["q"], z
    )
    np.testing.assert_allclose(state["smoothed_mean"], expected_mean, rtol=1e-10, atol=1e-10)
    length, dimension = z.shape
    covariance = np.zeros((length, dimension, length, dimension))
    covariance[-1, :, -1, :] = state["conditional_roots"][-1] @ state["conditional_roots"][-1].T
    for t in range(length - 2, -1, -1):
        gain, root = state["backward_gain"][t], state["conditional_roots"][t]
        covariance[t, :, t, :] = root @ root.T + gain @ covariance[t + 1, :, t + 1, :] @ gain.T
        for following in range(t + 1, length):
            covariance[t, :, following, :] = gain @ covariance[t + 1, :, following, :]
            covariance[following, :, t, :] = covariance[t, :, following, :].T
    np.testing.assert_allclose(
        covariance.reshape(length * dimension, -1), expected_cov, rtol=1e-10, atol=1e-10
    )


def test_pairing_reproducibility_and_original_observation_preservation():
    model, z = model_and_history()
    observed = np.isfinite(z)
    state = smooth_history(z, model)
    samples, seed = posterior_samples(state, observed, "case-a")
    again, repeated = posterior_samples(state, observed, "case-a")
    assert seed == repeated
    np.testing.assert_array_equal(samples, again)
    np.testing.assert_allclose(
        samples[0::2] + samples[1::2],
        np.broadcast_to(2 * state["smoothed_mean"], samples[0::2].shape),
        rtol=0,
        atol=1e-12,
    )
    mean, scale = np.array([9.1, -2.1, 120.0]), np.array([3.2, 0.7, 20.0])
    raw = z * scale + mean
    completed = complete_raw(raw, samples, mean, scale)
    for values in completed:
        np.testing.assert_array_equal(values[observed], raw[observed])
        assert np.isfinite(values).all()


def test_filter_is_causal_within_the_observed_history():
    model, z = model_and_history()
    initial = smooth_history(z, model)
    altered = z.copy()
    altered[-1] = [5.0, -3.0, 9.0]
    changed = smooth_history(altered, model)
    np.testing.assert_array_equal(initial["filtered_mean"][:-1], changed["filtered_mean"][:-1])
    assert not np.allclose(initial["smoothed_mean"][:-1], changed["smoothed_mean"][:-1])


def test_prefix_fit_uses_original_complete_transitions_and_is_stable():
    rng = np.random.default_rng(35)
    prefix = rng.normal(size=(400, 3))
    prefix[10:25, 1] = np.nan
    fitted = fit_dynamics(prefix)
    valid = np.isfinite(prefix).all(1)
    assert int(fitted["transition_pairs"]) == int((valid[:-1] & valid[1:]).sum())
    assert np.max(abs(np.linalg.eigvals(fitted["a"]))) <= 0.99 + 1e-12
    assert np.linalg.eigvalsh(fitted["q"]).min() > 0


def test_indefinite_covariance_is_rejected_without_relaxing_the_bound():
    with pytest.raises(ValueError, match="positive semidefinite"):
        covariance_root(np.diag([1.0, -1e-5]))


def test_quantile_mixture_uses_mass_and_has_an_exact_half_mass_rule():
    levels = [0.25, 0.5, 0.75]
    np.testing.assert_array_equal(quantile_weights(levels), [3, 2, 3])
    q = np.array([[[-10.0], [0.0], [1.0]], [[10.0], [20.0], [30.0]]])[:, :, :, None]
    mean, median = predictive_mixture(q, levels)
    np.testing.assert_array_equal(median, [[5.5]])
    np.testing.assert_array_equal(mean, [[8.3125]])
    assert median[0, 0] != np.median(q[:, 1, 0, 0])


def test_quantile_crossing_control_and_single_distribution_median():
    q = np.array([3.0, 1.0, 2.0]).reshape(1, 3, 1, 1)
    mean, median = predictive_mixture(q, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(mean, [[2.0]])
    np.testing.assert_array_equal(median, [[2.0]])
