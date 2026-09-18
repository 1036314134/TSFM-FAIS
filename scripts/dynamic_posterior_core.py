"""Fixed multivariate Gaussian dynamics and predictive-distribution quadrature."""

import hashlib
import math
from fractions import Fraction

import numpy as np


def regular_covariance(values):
    centered = values - values.mean(0)
    covariance = centered.T @ centered / len(values)
    return 0.95 * covariance + 0.05 * np.diag(np.diag(covariance)) + 1e-6 * np.eye(values.shape[1])


def fit_dynamics(prefix):
    mean, scale = np.nanmean(prefix, 0), np.nanstd(prefix, 0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    z = (prefix - mean) / scale
    complete = np.isfinite(z).all(1)
    pairs = complete[:-1] & complete[1:]
    if complete.sum() < 128 or pairs.sum() < 128:
        raise ValueError("insufficient originally complete prefix rows or transitions")
    x, y = z[:-1][pairs], z[1:][pairs]
    xm, ym = x.mean(0), y.mean(0)
    raw_a = np.linalg.solve(
        (x - xm).T @ (x - xm) / len(x) + 0.001 * np.eye(z.shape[1]), (x - xm).T @ (y - ym) / len(x)
    ).T
    radius = float(np.abs(np.linalg.eigvals(raw_a)).max())
    a = raw_a * min(1.0, 0.99 / max(radius, 1e-12))
    b = ym - a @ xm
    return {
        "mean": mean,
        "scale": scale,
        "a_unconstrained": raw_a,
        "a": a,
        "b": b,
        "q": regular_covariance(y - (x @ a.T + b)),
        "initial_mean": z[complete].mean(0),
        "initial_covariance": regular_covariance(z[complete]),
        "complete_rows": np.asarray(complete.sum()),
        "transition_pairs": np.asarray(pairs.sum()),
        "spectral_radius_unconstrained": np.asarray(radius),
    }


def condition_state(mean, covariance, observation):
    observed = np.flatnonzero(np.isfinite(observation))
    if not len(observed):
        return mean.copy(), covariance.copy()
    gain = np.linalg.solve(covariance[np.ix_(observed, observed)], covariance[:, observed].T).T
    result = mean + gain @ (observation[observed] - mean[observed])
    posterior = covariance - gain @ covariance[observed]
    posterior = (posterior + posterior.T) / 2
    result[observed] = observation[observed]
    posterior[observed, :] = 0
    posterior[:, observed] = 0
    return result, posterior


def covariance_root(covariance):
    value, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    tolerance = 1e-10 * max(1.0, float(abs(value).max()))
    if not np.isfinite(value).all() or value.min() < -tolerance:
        raise ValueError(
            "conditional covariance is not positive semidefinite within the fixed bound"
        )
    return vectors * np.sqrt(np.maximum(value, 0))[None]


def smooth_history(z, model):
    length, dimension = z.shape
    predicted = np.empty_like(z)
    filtered = np.empty_like(z)
    predicted_cov = np.empty((length, dimension, dimension))
    filtered_cov = np.empty_like(predicted_cov)
    static = np.empty_like(z)
    a, b, q = model["a"], model["b"], model["q"]
    for t in range(length):
        if t == 0:
            predicted[t], predicted_cov[t] = model["initial_mean"], model["initial_covariance"]
        else:
            predicted[t] = a @ filtered[t - 1] + b
            predicted_cov[t] = a @ filtered_cov[t - 1] @ a.T + q
        filtered[t], filtered_cov[t] = condition_state(predicted[t], predicted_cov[t], z[t])
        static[t], _ = condition_state(model["initial_mean"], model["initial_covariance"], z[t])
    smoothed = filtered.copy()
    gain = np.zeros_like(predicted_cov)
    roots = np.zeros_like(predicted_cov)
    roots[-1] = covariance_root(filtered_cov[-1])
    for t in range(length - 2, -1, -1):
        gain[t] = np.linalg.solve(predicted_cov[t + 1], a @ filtered_cov[t]).T
        smoothed[t] = filtered[t] + gain[t] @ (smoothed[t + 1] - predicted[t + 1])
        covariance = filtered_cov[t] - gain[t] @ predicted_cov[t + 1] @ gain[t].T
        roots[t] = covariance_root(covariance)
    observed = np.isfinite(z)
    smoothed[observed] = z[observed]
    roots[observed] = 0
    return {
        "predicted_mean": predicted,
        "predicted_covariance": predicted_cov,
        "filtered_mean": filtered,
        "filtered_covariance": filtered_cov,
        "smoothed_mean": smoothed,
        "static_mean": static,
        "backward_gain": gain,
        "conditional_roots": roots,
    }


def posterior_samples(state, observed, case_id):
    seed_hex = hashlib.sha256(f"r35|6103|{case_id}".encode()).hexdigest()[:16]
    random = np.random.default_rng(int(seed_hex, 16))
    length, dimension = observed.shape
    noise = random.standard_normal((8, length, dimension))
    deviation = np.zeros_like(noise)
    deviation[:, -1] = noise[:, -1] @ state["conditional_roots"][-1].T
    for t in range(length - 2, -1, -1):
        deviation[:, t] = (
            deviation[:, t + 1] @ state["backward_gain"][t].T
            + noise[:, t] @ state["conditional_roots"][t].T
        )
    deviation[:, observed] = 0
    paired = np.stack([deviation, -deviation], axis=1).reshape(16, length, dimension)
    samples = state["smoothed_mean"][None] + paired
    return samples, seed_hex


def complete_raw(context, values_z, mean, scale):
    result = values_z * scale + mean
    observed = np.isfinite(context)
    result[..., observed] = context[observed]
    return result


def quantile_weights(levels):
    values = [Fraction(str(v)).limit_denominator(1_000_000) for v in levels]
    if not values or values != sorted(set(values)) or values[0] <= 0 or values[-1] >= 1:
        raise ValueError("strictly ordered interior quantile levels are required")
    edges = [
        Fraction(0),
        *[(a + b) / 2 for a, b in zip(values[:-1], values[1:], strict=True)],
        Fraction(1),
    ]
    weights = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    denominator = math.lcm(*(w.denominator for w in weights))
    return np.asarray([int(w * denominator) for w in weights], dtype=np.int64)


def predictive_mixture(quantiles, levels):
    """Equal-history mixture: input [sample,quantile,horizon,target], output two point rules."""
    values = np.sort(np.asarray(quantiles, dtype=float), axis=1)
    weights = quantile_weights(levels)
    if values.ndim != 4 or values.shape[1] != len(weights) or not np.isfinite(values).all():
        raise ValueError("aligned finite conditional quantiles are required")
    nodes = values.reshape(-1, *values.shape[2:])
    mass = np.tile(weights, len(values))
    order = np.argsort(nodes, axis=0, kind="stable")
    ordered = np.take_along_axis(nodes, order, axis=0)
    masses = np.take_along_axis(np.broadcast_to(mass[:, None, None], nodes.shape), order, axis=0)
    cumulative = masses.cumsum(0)
    total = int(mass.sum())
    index = np.argmax(2 * cumulative >= total, axis=0)
    median = np.take_along_axis(ordered, index[None], axis=0)[0]
    halfway = 2 * np.take_along_axis(cumulative, index[None], axis=0)[0] == total
    following = np.take_along_axis(ordered, np.minimum(index + 1, len(nodes) - 1)[None], axis=0)[0]
    median = np.where(halfway, (median + following) / 2, median)
    mean = (nodes * mass[:, None, None]).sum(0) / total
    return mean, median


def dense_conditional_tail(initial_mean, initial_covariance, a, b, q, observations):
    """Independent block-Gaussian conditioning used only by tests and audits."""
    length, dimension = observations.shape
    means = np.empty((length, dimension))
    covariance = np.zeros((length, dimension, length, dimension))
    means[0], covariance[0, :, 0, :] = initial_mean, initial_covariance
    for t in range(1, length):
        means[t] = a @ means[t - 1] + b
        covariance[t, :, t, :] = a @ covariance[t - 1, :, t - 1, :] @ a.T + q
        for previous in range(t):
            covariance[t, :, previous, :] = a @ covariance[t - 1, :, previous, :]
            covariance[previous, :, t, :] = covariance[t, :, previous, :].T
    joint = covariance.reshape(length * dimension, length * dimension)
    flat_mean, flat_values = means.ravel(), observations.ravel()
    observed = np.flatnonzero(np.isfinite(flat_values))
    if not len(observed):
        return means, joint
    gain = np.linalg.solve(joint[np.ix_(observed, observed)], joint[:, observed].T).T
    posterior_mean = flat_mean + gain @ (flat_values[observed] - flat_mean[observed])
    posterior_cov = joint - gain @ joint[observed]
    posterior_mean[observed] = flat_values[observed]
    posterior_cov[observed, :] = 0
    posterior_cov[:, observed] = 0
    return posterior_mean.reshape(length, dimension), (posterior_cov + posterior_cov.T) / 2
