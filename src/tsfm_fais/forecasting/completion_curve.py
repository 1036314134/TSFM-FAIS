"""Observable projections onto a sampled scalar-completion response curve."""

from __future__ import annotations

import numpy as np


def nearest_curve_points(grid, predictions, target):
    """Return feasible grid and interpolated-input proposals without future labels.

    A polyline distance approximates the output curve. It is not a certified
    bound on the true nonlinear curve; callers must evaluate the proposed input.
    """
    grid = np.asarray(grid, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or predictions.shape[0] != len(grid)
        or predictions.shape[1:] != target.shape
        or not np.isfinite(grid).all()
        or not np.isfinite(predictions).all()
        or not np.isfinite(target).all()
        or not np.all(np.diff(grid) > 0)
    ):
        raise ValueError("an increasing grid and aligned finite forecast vectors are required")
    curve = predictions.reshape(len(grid), -1)
    goal = target.reshape(-1)
    if len(goal) == 0:
        raise ValueError("forecast vectors must be nonempty")
    errors = ((curve - goal) ** 2).mean(axis=1)
    grid_index = int(errors.argmin())
    directions = curve[1:] - curve[:-1]
    norm = (directions**2).sum(axis=1)
    fraction = np.divide(
        ((goal - curve[:-1]) * directions).sum(axis=1),
        norm,
        out=np.zeros_like(norm),
        where=norm > 0,
    ).clip(0, 1)
    approximation = curve[:-1] + fraction[:, None] * directions
    poly_errors = ((approximation - goal) ** 2).mean(axis=1)
    segment = int(poly_errors.argmin())
    return {
        "grid_index": grid_index,
        "grid_input": float(grid[grid_index]),
        "grid_mse": float(errors[grid_index]),
        "interpolated_input": float(grid[segment] + fraction[segment] * np.diff(grid)[segment]),
        "polyline_mse_approximation": float(poly_errors[segment]),
    }
