"""Matched-marginal Gaussian future laws with variable temporal dependence."""

import numpy as np
from conditional_future import future_moments, sample_futures
from scipy.special import ndtr


def matched_future_components(
    model, posterior_mean, posterior_covariance, phase, center, scale, seed, samples=512, horizon=96
):
    mean, missing, innovation = future_moments(
        model, posterior_mean, posterior_covariance, phase, horizon, 1.0
    )
    mean = (mean[:, :2] - center) / scale
    variance = missing[:, :2] / scale**2 + innovation[:, :2] / scale**2
    correlated = sample_futures(
        model, posterior_mean, posterior_covariance, phase, horizon, 1.0, samples, seed
    )
    correlated = (correlated[:, :, :2] - center) / scale
    independent = (
        np.random.default_rng(seed + 500000).standard_normal(correlated.shape)
        * np.sqrt(variance)[None]
    )
    return mean, variance, correlated, independent


def covariance_intervention(mean, correlated, independent, correlation):
    if correlation == 1.0:
        return correlated
    if correlation == 0.0:
        return mean + independent
    if not 0 < correlation < 1:
        raise ValueError("the covariance mixture must be in [0,1]")
    return (
        mean + np.sqrt(correlation) * (correlated - mean) + np.sqrt(1 - correlation) * independent
    )


def projected_variances(model, posterior_covariance, directions, variance, scale, slot):
    count, coordinates = directions.shape
    horizon = coordinates // 2 if slot == -1 else coordinates
    raw = np.zeros((count, horizon, len(model["a"])))
    if slot == -1:
        raw[:, :, :2] = directions.reshape(count, horizon, 2) / scale
    else:
        raw[:, :, slot] = directions / scale[slot]
    accumulated = np.zeros((count, len(model["a"])))
    future = np.zeros(count)
    for time in range(horizon - 1, -1, -1):
        accumulated = raw[:, time] + accumulated @ model["a"]
        future += np.einsum("ai,ij,aj->a", accumulated, model["q"], accumulated)
    initial = accumulated @ model["a"]
    correlated = future + np.einsum("ai,ij,aj->a", initial, posterior_covariance, initial)
    diagonal = np.sum(directions**2 * variance, axis=1)
    return 4 * correlated / coordinates**2, 4 * diagonal / coordinates**2


def flip_probabilities(margins, variance):
    result = np.zeros_like(margins, dtype=float)
    positive = variance > 0
    result[positive] = ndtr(-margins[positive] / np.sqrt(variance[positive]))
    return result


def dense_future_covariance(model, posterior_covariance, horizon, scale):
    dimension = len(model["a"])
    covariance = posterior_covariance.copy()
    marginals = []
    for _ in range(horizon):
        covariance = model["a"] @ covariance @ model["a"].T + model["q"]
        marginals.append(covariance.copy())
    powers = [np.linalg.matrix_power(model["a"], lag) for lag in range(horizon)]
    result = np.empty((2 * horizon, 2 * horizon))
    if dimension < 2:
        raise ValueError("two forecast targets are required")
    for later in range(horizon):
        for earlier in range(later + 1):
            block = (powers[later - earlier] @ marginals[earlier])[:2, :2] / np.outer(scale, scale)
            result[2 * later : 2 * later + 2, 2 * earlier : 2 * earlier + 2] = block
            result[2 * earlier : 2 * earlier + 2, 2 * later : 2 * later + 2] = block.T
    return result
