"""Forecast-space costs for explicitly identified reference predictions."""

import numpy as np


def forecast_reference_costs(predictions, reference):
    points = np.asarray(predictions, dtype=float)
    target = np.asarray(reference, dtype=float)
    if (
        points.ndim != 4
        or min(points.shape) < 1
        or target.shape != (points.shape[0], *points.shape[2:])
        or not np.isfinite(points).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError(
            "finite [N,A,H,K] predictions and an aligned [N,H,K] reference are required"
        )
    return np.square(points - target[:, None]).mean(axis=2)
