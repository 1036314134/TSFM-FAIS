"""Fixed correlation transport between statistical and foundation forecasts."""

import numpy as np


def future_covariances(a, q, state_mean, state_covariance, b, horizon, targets=(0, 1)):
    dimension, count = len(a), len(targets)
    powers = [np.eye(dimension)]
    for _ in range(horizon):
        powers.append(a @ powers[-1])
    means, marginal = [], []
    mean, covariance = state_mean.copy(), state_covariance.copy()
    for _ in range(horizon):
        mean = a @ mean + b
        covariance = a @ covariance @ a.T + q
        covariance = (covariance + covariance.T) / 2
        means.append(mean[list(targets)].copy())
        marginal.append(covariance.copy())
    total = np.empty((horizon, count, horizon, count))
    initial = np.empty_like(total)
    for t in range(horizon):
        for s in range(t + 1):
            cross = powers[t - s] @ marginal[s]
            shared = powers[t + 1] @ state_covariance @ powers[s + 1].T
            total[t, :, s, :] = cross[np.ix_(targets, targets)]
            total[s, :, t, :] = total[t, :, s, :].T
            initial[t, :, s, :] = shared[np.ix_(targets, targets)]
            initial[s, :, t, :] = initial[t, :, s, :].T
    total, initial = total.reshape(horizon * count, -1), initial.reshape(horizon * count, -1)
    return np.stack(means), (total + total.T) / 2, (initial + initial.T) / 2


def pooled_point(foundation, statistical, covariance_v, covariance_f, match_trace=True):
    cv, cf = np.asarray(covariance_v, float), np.asarray(covariance_f, float).copy()
    fallback = bool(np.trace(cf) <= 1e-12)
    if fallback:
        cf = cv.copy()
    elif match_trace:
        cf *= np.trace(cv) / np.trace(cf)
    cf = (cf + cf.T) / 2
    if np.array_equal(cv, cf):
        point = 0.5 * foundation + 0.5 * statistical
    else:
        difference = (foundation - statistical).ravel()
        point = statistical + (cv @ np.linalg.solve(cv + cf, difference)).reshape(statistical.shape)
    return point, cf, fallback


def transport_outputs(
    foundation, mean_foundation, variance_foundation, statistical, covariance_v, initial_covariance
):
    diagonal = np.diag(covariance_v)
    if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError("statistical forecast covariance must have positive marginals")
    spread = np.maximum(np.asarray(variance_foundation, float).ravel(), 0)
    ratio = np.sqrt(spread / diagonal)
    transported = covariance_v * ratio[:, None] * ratio[None, :]
    complete = transported + initial_covariance
    definitions = {
        "correlation_transport": (foundation, covariance_v, complete, True),
        "transport_no_initial": (foundation, covariance_v, transported, True),
        "transport_independent_f": (
            foundation,
            covariance_v,
            np.diag(spread) + initial_covariance,
            True,
        ),
        "transport_unscaled": (foundation, covariance_v, complete, False),
        "transport_diagonal": (foundation, np.diag(diagonal), np.diag(np.diag(complete)), True),
        "transport_mean_center": (mean_foundation, covariance_v, complete, True),
    }
    outputs, covariances, fallbacks = {}, {}, {}
    for name, (center, cv, cf, match) in definitions.items():
        outputs[name], covariances[name], fallbacks[name] = pooled_point(
            center, statistical, cv, cf, match
        )
    return outputs, covariances, fallbacks
