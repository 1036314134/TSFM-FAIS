"""Downstream errors on explicitly observed native-missing future targets."""

from __future__ import annotations

import numpy as np

from .accuracy import forecast_errors


def observed_future_errors(point, truth, observed, scales, *, minimum_observed=48):
    """Return per-target errors and counts, rejecting ineligible evaluation windows.

    The supplied source mask must match finite truth exactly. Filled future
    labels cannot be passed as evaluation truth while keeping their source mask.
    Every target receives its own denominator, as in complete-future scoring.
    """
    prediction, future, mask, scale = (
        np.asarray(point, float),
        np.asarray(truth, float),
        np.asarray(observed),
        np.asarray(scales, float),
    )
    if (
        prediction.ndim != 3
        or future.shape != prediction.shape[1:]
        or mask.shape != future.shape
        or mask.dtype != np.bool_
        or scale.shape != (prediction.shape[2],)
        or prediction.shape[2] < 1
    ):
        raise ValueError("expected [A,H,K] predictions, [H,K] truth and boolean mask, [K] scales")
    if np.isinf(future).any() or not np.array_equal(np.isfinite(future), mask):
        raise ValueError(
            "truth must preserve the original observed mask; unavailable values are NaN"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("all forecast values must be finite, including unscored positions")
    counts = mask.sum(axis=0)
    if not 1 <= minimum_observed <= prediction.shape[1] or (counts < minimum_observed).any():
        raise ValueError("each target must satisfy the declared minimum observed future count")
    per_target = [
        forecast_errors(
            prediction[:, mask[:, target], target : target + 1],
            future[mask[:, target], target : target + 1],
            scale[target : target + 1],
        )
        for target in range(prediction.shape[2])
    ]
    errors = {
        metric: np.concatenate([scores[metric] for scores in per_target], axis=1)
        for metric in ("mae", "mse", "raw_mae", "raw_mse")
    }
    return errors, counts
