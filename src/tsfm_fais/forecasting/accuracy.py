"""Downstream MAE/MSE with one standardizer fitted on historical observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefixStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    constant: np.ndarray
    observed_count: np.ndarray

    @classmethod
    def fit(cls, prefix: np.ndarray) -> PrefixStandardizer:
        values = np.asarray(prefix, dtype=float)
        if values.ndim != 2 or np.isinf(values).any():
            raise ValueError("prefix must be [T,D], with missing values represented by NaN")
        counts = np.isfinite(values).sum(axis=0)
        if np.any(counts < 2):
            raise ValueError(
                "standardization needs at least two historical observations per channel"
            )
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=0)
        constant = std <= 1e-12
        return cls(mean, np.where(constant, 1.0, std), constant, counts)


def forecast_errors(
    point: np.ndarray, truth: np.ndarray, scales: np.ndarray
) -> dict[str, np.ndarray]:
    """Return [N,K] errors; raw and standardized errors use the same predictions."""
    prediction, future, scale = (
        np.asarray(point, float),
        np.asarray(truth, float),
        np.asarray(scales, float),
    )
    if (
        prediction.ndim != 3
        or future.shape != prediction.shape[1:]
        or scale.shape != (prediction.shape[2],)
    ):
        raise ValueError("expected predictions [N,H,K], truth [H,K], scales [K]")
    if (
        not np.isfinite(prediction).all()
        or not np.isfinite(future).all()
        or not np.all(np.isfinite(scale) & (scale > 0))
    ):
        raise ValueError(
            "evaluation requires finite predictions, complete truth, and positive scales"
        )
    residual = prediction - future[None]
    standardized = residual / scale
    return {
        "mae": np.mean(np.abs(standardized), axis=1),
        "mse": np.mean(standardized**2, axis=1),
        "raw_mae": np.mean(np.abs(residual), axis=1),
        "raw_mse": np.mean(residual**2, axis=1),
    }


def guarded_direct_forecast(context, targets, direct, reference, *, joint: bool):
    """Use only context observability to choose an existing fallback forecast."""
    empty = ~np.isfinite(np.asarray(context)).any(axis=0)
    fallback = np.full(len(targets), empty.any(), dtype=bool) if joint else empty[list(targets)]
    point = np.where(fallback[None, :], reference, direct)
    return point, fallback


def recover_legacy_chronos_median(cached_quantiles, saved_targets, quantile_levels, n_variates):
    """Recover retained medians after the documented D==Q axis collision.

    Only a point forecast is recoverable when the old adapter retained a subset
    of quantile indices as target indices. Missing quantiles are not synthesized.
    """
    values = np.asarray(cached_quantiles, dtype=float)
    median = np.flatnonzero(np.isclose(quantile_levels, 0.5))
    if (
        values.ndim != 4
        or n_variates != len(quantile_levels)
        or values.shape[1] == n_variates
        or len(median) != 1
    ):
        raise ValueError("cache does not match the recoverable D==Q collision")
    saved = list(saved_targets)
    if int(median[0]) not in saved or values.shape[2:] != (len(saved), n_variates):
        raise ValueError("the old target slice did not retain the median quantile")
    return np.take(values[:, :, saved.index(int(median[0])), :], saved, axis=-1)
