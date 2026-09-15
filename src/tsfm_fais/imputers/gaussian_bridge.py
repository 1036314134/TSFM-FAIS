"""Conditional stationary AR(1) completions in historical prefix units."""

from __future__ import annotations

import numpy as np


def conditional_ar1(context, noise, *, rho):
    """Condition a zero-mean, unit-variance AR(1) prior on exact observations.

    `noise` contains standard normal innovations in [draw,L,D] order. Supplying
    antithetic innovations makes the average completion equal its conditional
    mean. Each variable uses an independent prior; no outcomes enter this code.
    """
    context, noise = np.asarray(context, dtype=float), np.asarray(noise, dtype=float)
    if (
        context.ndim != 2
        or min(context.shape) < 1
        or noise.ndim != 3
        or noise.shape[0] < 1
        or noise.shape[1:] != context.shape
        or not np.isfinite(noise).all()
        or np.isinf(context).any()
        or not 0 <= rho < 1
    ):
        raise ValueError("finite innovations, a [L,D] context and 0 <= rho < 1 are required")
    length, dims = context.shape
    completed = context.copy()
    samples = np.repeat(context[None], len(noise), axis=0)
    variance = 1 - rho**2
    main = np.full(length, (1 + rho**2) / variance)
    main[[0, -1]] = 1 / variance
    if length == 1:
        main[0] = 1.0
    off = -rho / variance
    for dim in range(dims):
        missing = np.flatnonzero(np.isnan(context[:, dim]))
        for block in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
            if not len(block):
                continue
            diagonal, lower = main[block].copy(), np.zeros(len(block))
            rhs = np.zeros(len(block))
            if block[0] > 0:
                rhs[0] -= off * context[block[0] - 1, dim]
            if block[-1] < length - 1:
                rhs[-1] -= off * context[block[-1] + 1, dim]
            # LDL^T factorization of the missing-coordinate precision block.
            for index in range(1, len(block)):
                lower[index] = off / diagonal[index - 1]
                diagonal[index] -= lower[index] ** 2 * diagonal[index - 1]
                rhs[index] -= lower[index] * rhs[index - 1]
            if not np.all(diagonal > 0):
                raise ValueError("the conditional precision must be positive definite")
            mean = rhs / diagonal
            deviations = noise[:, block, dim] / np.sqrt(diagonal)[None]
            for index in range(len(block) - 2, -1, -1):
                mean[index] -= lower[index + 1] * mean[index + 1]
                deviations[:, index] -= lower[index + 1] * deviations[:, index + 1]
            completed[block, dim] = mean
            samples[:, block, dim] = mean[None] + deviations
    return completed, samples
