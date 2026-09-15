"""Controlled Gaussian histories with exactly conditioned forecasting risks."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    simplex_quadratic_weights,
)


def scenarios():
    coupled = np.array([[0.80, 0.12, 0.0], [0.0, 0.78, 0.12], [0.10, 0.0, 0.75]])
    cycle = np.roll(np.eye(7), 1, axis=1)
    return [
        {
            "name": "independent_ar3",
            "a": 0.9 * np.eye(3),
            "q": 0.04 * np.eye(3),
            "period": 24,
            "amplitude": np.zeros(3),
        },
        {
            "name": "coupled_ar3",
            "a": coupled,
            "q": np.diag([0.04, 0.06, 0.03]),
            "period": 24,
            "amplitude": np.zeros(3),
        },
        {
            "name": "seasonal_ar7",
            "a": 0.75 * np.eye(7) + 0.08 * cycle,
            "q": 0.04 * (0.6 * np.eye(7) + 0.4 * np.ones((7, 7))),
            "period": 48,
            "amplitude": np.linspace(1.0, 2.0, 7),
        },
    ]


def seasonal(model, times):
    t = np.asarray(times, float)[:, None]
    phase = np.arange(len(model["amplitude"]))[None] * 0.3
    return model["amplitude"][None] * (
        np.sin(2 * np.pi * t / model["period"] + phase)
        + 0.4 * np.cos(4 * np.pi * t / model["period"] + phase)
    )


def psd_root(matrix):
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < -1e-10 * max(1.0, float(abs(matrix).max())):
        raise ValueError("a conditional covariance is not positive semidefinite")
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def draw_history(model, length, phase, seed):
    rng = np.random.default_rng(seed)
    a, q = model["a"], model["q"]
    covariance = solve_discrete_lyapunov(a, q)
    state = psd_root(covariance) @ rng.standard_normal(len(a))
    root = psd_root(q)
    values = []
    for t in range(length):
        if t:
            state = a @ state + root @ rng.standard_normal(len(a))
        values.append(state.copy())
    return np.asarray(values) + seasonal(model, np.arange(phase, phase + length))


def condition_history(model, observed, phase, *, scalar=False):
    a, q = model["a"], model["q"]
    mean = np.zeros(len(a))
    covariance = solve_discrete_lyapunov(a, q)
    residual = np.asarray(observed, float) - seasonal(
        model, np.arange(phase, phase + len(observed))
    )
    for time, row in enumerate(residual):
        if time:
            mean = a @ mean
            covariance = a @ covariance @ a.T + q
        indices = np.flatnonzero(np.isfinite(row))
        if len(indices) == len(a):
            mean = row.copy()
            covariance = np.zeros_like(covariance)
        elif len(indices):
            if scalar:
                for index in indices:
                    gain = covariance[:, index] / covariance[index, index]
                    mean = mean + gain * (row[index] - mean[index])
                    covariance = covariance - np.outer(gain, covariance[index])
                    mean[index] = row[index]
                    covariance[index] = 0.0
                    covariance[:, index] = 0.0
            else:
                gain = np.linalg.solve(covariance[np.ix_(indices, indices)], covariance[indices]).T
                mean = mean + gain @ (row[indices] - mean[indices])
                covariance = covariance - gain @ covariance[indices]
                mean[indices] = row[indices]
                covariance[indices] = 0.0
                covariance[:, indices] = 0.0
        covariance = (covariance + covariance.T) / 2
    psd_root(covariance)
    return mean, covariance


def future_moments(model, mean, covariance, phase, horizon, noise):
    a, q = model["a"], model["q"]
    current = mean.copy()
    missing = covariance.copy()
    innovation = np.zeros_like(covariance)
    means, missing_variances, innovation_variances = [], [], []
    for _ in range(horizon):
        current = a @ current
        missing = a @ missing @ a.T
        innovation = a @ innovation @ a.T + noise**2 * q
        means.append(current.copy())
        missing_variances.append(np.maximum(np.diag(missing), 0.0))
        innovation_variances.append(np.maximum(np.diag(innovation), 0.0))
    means = np.asarray(means) + seasonal(model, np.arange(phase, phase + horizon))
    return means, np.asarray(missing_variances), np.asarray(innovation_variances)


def sample_futures(model, mean, covariance, phase, horizon, noise, samples, seed):
    rng = np.random.default_rng(seed)
    state = mean[None] + rng.standard_normal((samples, len(mean))) @ psd_root(covariance).T
    root = psd_root(model["q"])
    seasonal_values = seasonal(model, np.arange(phase, phase + horizon))
    output = np.empty((samples, horizon, len(mean)))
    for time in range(horizon):
        state = state @ model["a"].T + noise * (rng.standard_normal(state.shape) @ root.T)
        output[:, time] = state + seasonal_values[time]
    return output


def gaussian_absolute(error, sigma):
    error, sigma = np.broadcast_arrays(np.asarray(error, float), np.asarray(sigma, float))
    result = np.abs(error).copy()
    positive = sigma > 0
    z = np.divide(error, sigma, out=np.zeros_like(error), where=positive)
    result[positive] = (
        sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * z * z) + error * (2 * ndtr(z) - 1)
    )[positive]
    return result


def expected_risks(points, mean, variance):
    error = np.asarray(points, float) - np.asarray(mean, float)
    return gaussian_absolute(error, np.sqrt(np.maximum(variance, 0.0))).mean(-1), (
        error**2 + variance
    ).mean(-1)


class ForecastHull:
    """Reuse simplex-face factorizations while varying the realized future."""

    def __init__(self, points):
        self.points = np.asarray(points, float)
        self.count = len(points)
        self.anchor, self.delta, _, gram = forecast_geometry(self.points[None])
        self.anchor = self.anchor[0]
        self.delta = self.delta[0]
        self.gram = gram[0]
        self.scale = max(float(abs(self.gram).max()), 1e-12)
        normalized = self.gram / self.scale
        self.faces = []
        for size in range(1, self.count + 1):
            for face in combinations(range(self.count), size):
                indices = np.asarray(face)
                local = normalized[np.ix_(indices, indices)]
                system = np.zeros((size + 1, size + 1))
                system[:size, :size] = 2 * local
                system[:size, -1] = system[-1, :size] = 1
                self.faces.append(
                    (indices, local, system, np.linalg.pinv(system, rcond=1e-12, hermitian=True))
                )

    def solve(self, targets):
        targets = np.asarray(targets, float)
        if targets.ndim == 1:
            targets = targets[None]
        alignment = (targets - self.anchor) @ self.delta.T / self.points.shape[1]
        linear = alignment / self.scale
        weights = np.zeros((len(targets), self.count))
        weights[:, 0] = 1
        best = self.gram[0, 0] / self.scale - 2 * linear[:, 0]
        for indices, local, system, inverse in self.faces:
            rhs = np.c_[2 * linear[:, indices], np.ones(len(targets))]
            solution = rhs @ inverse.T
            candidate = solution[:, : len(indices)]
            residual = np.max(abs(solution @ system.T - rhs), axis=1) / np.maximum(
                np.max(abs(rhs), axis=1), 1.0
            )
            feasible = (candidate.min(1) >= -1e-9) & (residual <= 1e-8)
            candidate = np.maximum(candidate, 0.0)
            candidate /= np.maximum(candidate.sum(1, keepdims=True), 1e-30)
            objective = np.einsum("ni,ij,nj->n", candidate, local, candidate) - 2 * np.sum(
                candidate * linear[:, indices], axis=1
            )
            improved = feasible & (objective < best)
            chosen = np.flatnonzero(improved)
            weights[chosen] = 0.0
            weights[np.ix_(chosen, indices)] = candidate[chosen]
            best[chosen] = objective[chosen]
        gradient = 2 * (weights @ self.gram - alignment)
        scales = np.maximum(np.maximum(abs(alignment).max(1), abs(self.gram).max()), 1e-12)
        gap = (np.sum(weights * gradient, axis=1) - gradient.min(1)) / scales
        failed = np.flatnonzero(gap > 1e-7)
        if len(failed):
            weights[failed] = simplex_quadratic_weights(
                np.broadcast_to(self.gram, (len(failed), self.count, self.count)), alignment[failed]
            )[0]
        gradient = 2 * (weights @ self.gram - alignment)
        gap = (np.sum(weights * gradient, axis=1) - gradient.min(1)) / scales
        if (
            gap.max() > 1e-7
            or (weights < 0).any()
            or not np.allclose(weights.sum(1), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("conditional or hindsight hull weights failed optimality")
        prediction = self.points[0] + weights @ (self.points - self.points[:1])
        return weights, prediction, np.maximum(gap, 0.0), len(failed)
