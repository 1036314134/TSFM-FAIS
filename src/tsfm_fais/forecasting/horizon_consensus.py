"""Select cached forecast segments using only their observable median consensus."""

import numpy as np


def consensus_medoid_segments(predictions, *, blocks, joint_targets):
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 4 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("finite [N,A,H,K] predictions are required")
    count, _, horizon, targets = values.shape
    if blocks < 1 or horizon % blocks:
        raise ValueError("equal forecast blocks must partition the horizon")
    width = horizon // blocks
    chunks = values.reshape(count, values.shape[1], blocks, width, targets)
    median = np.median(values, axis=1).reshape(count, blocks, width, targets)
    distance = (chunks - median[:, None]) ** 2
    if joint_targets:
        choices = distance.mean(axis=(3, 4)).argmin(axis=1)
    else:
        choices = distance.mean(axis=3).argmin(axis=1)
    result = np.empty((count, horizon, targets))
    for block in range(blocks):
        positions = slice(block * width, (block + 1) * width)
        if joint_targets:
            result[:, positions] = chunks[np.arange(count), choices[:, block], block]
        else:
            for target in range(targets):
                result[:, positions, target] = chunks[
                    np.arange(count), choices[:, block, target], block, :, target
                ]
    return result, choices
