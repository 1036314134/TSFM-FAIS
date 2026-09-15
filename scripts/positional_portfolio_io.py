"""Convert audited joint or independent decision rows into forecast-target trajectories."""

import numpy as np
import pandas as pd


def target_nodes(decisions, base_features, vectors, *, joint):
    if not joint:
        if set(decisions.target_slot) != {0, 1}:
            raise ValueError("two independent target slots are required")
        return decisions.copy(), np.asarray(base_features), np.asarray(vectors)
    if set(decisions.target_slot) != {-1} or vectors.shape[2] % 2:
        raise ValueError("joint vectors must contain the two interleaved targets")
    count, _, length = vectors.shape
    points = vectors.reshape(count, 7, length // 2, 2)
    ids = decisions.source_episode_id if "source_episode_id" in decisions else decisions.episode_id
    nodes = pd.concat(
        [
            decisions.assign(
                source_episode_id=ids, episode_id=ids + "|target=" + str(slot), target_slot=slot
            )
            for slot in (0, 1)
        ],
        ignore_index=True,
    )
    return (
        nodes,
        np.concatenate([base_features, base_features]),
        np.concatenate([points[:, :, :, 0], points[:, :, :, 1]]),
    )


def restore_positions(values, decisions, count, horizon):
    if values.shape != (len(decisions), horizon) or len(decisions) != 2 * count:
        raise ValueError("positional outputs must cover both target trajectories")
    result = np.empty((count, horizon, 2))
    for slot in (0, 1):
        positions = np.flatnonzero(decisions.target_slot.to_numpy() == slot)
        indices = decisions.iloc[positions].episode_index.to_numpy(int)
        if len(positions) != count or set(indices) != set(range(count)):
            raise ValueError("a target trajectory is missing or duplicated")
        result[indices, :, slot] = values[positions]
    if not np.isfinite(result).all():
        raise ValueError("all returned forecast coordinates must be finite")
    return result
