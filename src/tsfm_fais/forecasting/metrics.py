"""Forecast losses with scales frozen from historical training data."""

from __future__ import annotations

import numpy as np


def training_mase_scale(values: np.ndarray, period: int) -> tuple[np.ndarray, int]:
    """Return per-variate scales and the common lag used on a [T,D] prefix."""

    history = np.asarray(values, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2:
        raise ValueError("MASE scaling requires a [T,D] training prefix")
    requested = max(1, int(period))
    lags = (requested, 1) if requested != 1 and history.shape[0] > requested else (1,)
    for lag in lags:
        paired = np.isfinite(history[lag:]) & np.isfinite(history[:-lag])
        if np.any(np.sum(paired, axis=0) == 0):
            continue
        differences = np.abs(history[lag:] - history[:-lag])
        scale = np.asarray(
            [
                np.mean(differences[paired[:, channel], channel])
                for channel in range(history.shape[1])
            ],
            dtype=float,
        )
        if np.isfinite(scale).all():
            return np.maximum(scale, 1e-8), lag
    raise ValueError("MASE scaling has no observed lagged pair for at least one variate")


def macro_mase(point: np.ndarray, truth: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute one target-macro MASE per [N,H,K] forecast; require full truth."""

    predicted = np.asarray(point, dtype=float)
    observed = np.asarray(truth, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if predicted.ndim != 3 or predicted.shape != observed.shape:
        raise ValueError("point and truth must have matching [N,H,K] shapes")
    if scale.shape != (predicted.shape[2],) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("scales must be positive finite values, one per target")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("MASE requires finite predictions and fully observed truth")
    return np.mean(np.mean(np.abs(predicted - observed), axis=1) / scale[None, :], axis=1)
