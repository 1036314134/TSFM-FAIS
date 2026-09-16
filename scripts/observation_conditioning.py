"""Fixed, time-preserving conditional updates from original current observations."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri


def correlation():
    time = np.arange(192)
    return 0.5 + 0.5 * np.exp(-abs(time[:, None] - time[None, :]) / 96)


def conditional_forecasts(prior, quantiles, context):
    if prior.shape != (192, 2) or quantiles.shape != (192, 2, 3) or context.shape != (96, 2):
        raise ValueError("expected the registered prior, quantiles and two current targets")
    if not np.isfinite(prior).all() or not np.isfinite(quantiles).all():
        raise ValueError("predictive priors must be finite")
    result = {
        name: prior[96:].copy()
        for name in (
            "prior_slice",
            "last_innovation",
            "mean_innovation",
            "unit_conditioner",
            "spread_conditioner",
        )
    }
    kernel = correlation()
    records = []
    for slot in (0, 1):
        observed = np.flatnonzero(np.isfinite(context[:, slot]))
        residual = context[observed, slot] - prior[observed, slot]
        if not len(observed):
            records.append({"slot": slot, "observed_count": 0, "minimum_remaining_variance": None})
            continue
        result["last_innovation"][:, slot] += residual[-1]
        result["mean_innovation"][:, slot] += residual.mean()
        ordered = np.sort(quantiles[:, slot], axis=-1)
        spreads = np.maximum(0.05, (ordered[:, -1] - ordered[:, 0]) / (2 * ndtri(0.9)))
        minima = {}
        for name, scale in (("unit_conditioner", np.ones(192)), ("spread_conditioner", spreads)):
            covariance = kernel * scale[:, None] * scale[None, :]
            noise = 0.1 * scale[observed] ** 2 + 1e-8
            system = covariance[np.ix_(observed, observed)] + np.diag(noise)
            factor = cho_factor(system, lower=True, check_finite=True)
            cross = covariance[96:, observed]
            update = cross @ cho_solve(factor, residual)
            variance = np.diag(covariance)[96:] - np.sum(
                cross * cho_solve(factor, cross.T).T, axis=1
            )
            if variance.min() < -1e-8 or (variance - np.diag(covariance)[96:]).max() > 1e-8:
                raise ValueError("conditional covariance failed its algebraic check")
            result[name][:, slot] += update
            minima[name] = float(variance.min())
        records.append(
            {"slot": slot, "observed_count": len(observed), "minimum_remaining_variance": minima}
        )
    return result, records
