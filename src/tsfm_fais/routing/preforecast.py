"""Explicit pre-forecast feature boundary and observable projection costs."""

import numpy as np
import pandas as pd

STATIC_FEATURES = tuple(
    "static." + name
    for name in (
        "missing_fraction",
        "target_missing_fraction",
        "tail_missing_fraction",
        "recent_missing_fraction",
        "gap_mean_ratio",
        "gap_max_ratio",
        "dimension_log",
        "period_context_ratio",
        "native_coverage",
        "observed_mean",
        "observed_std",
        "mean_fill_change",
        "max_fill_change",
        "recent_fill_change",
        "variation",
        "curvature",
        "slope",
        "direct_missing",
        "empty_channel_fraction",
        "empty_target_fraction",
        "fallback_target_fraction",
    )
)
METADATA = (
    "episode_id",
    "source_episode_id",
    "episode_index",
    "origin_id",
    "model_id",
    "candidate_id",
    "target_slot",
    "family_id",
    "dataset_id",
    "item_id",
    "split",
)


def preforecast_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = {name for name in frame if name.startswith("static.")} - set(STATIC_FEATURES)
    if unknown:
        raise ValueError(f"unaudited static features: {sorted(unknown)}")
    features = [name for name in STATIC_FEATURES if name in frame]
    if not features or not {"episode_id", "candidate_id"} <= set(frame):
        raise ValueError("pre-forecast decisions require static features and candidate identities")
    return frame[[name for name in METADATA if name in frame] + features].copy()


def decision_keys(frame):
    result = frame.copy()
    result["source_episode_id"] = result.episode_id
    if result.target_slot.min() >= 0:
        result["episode_id"] = result.episode_id + "|target=" + result.target_slot.astype(str)
    return result


def consensus_projection_costs(predictions):
    """Per-target costs [N,A,K]; no actual future is an argument."""
    points = np.asarray(predictions, dtype=float)
    if points.ndim != 4 or min(points.shape) < 1 or not np.isfinite(points).all():
        raise ValueError("finite [N,A,H,K] predictions are required")
    teacher = np.median(points, axis=1, keepdims=True)
    return ((points - teacher) ** 2).mean(axis=2)


def projection_row_costs(frame, costs, actions):
    costs = np.asarray(costs, dtype=float)
    positions = frame.candidate_id.map({name: index for index, name in enumerate(actions)})
    if costs.ndim != 3 or costs.shape[1] != len(actions) or positions.isna().any():
        raise ValueError("projection costs and candidate identities do not agree")
    episode = frame.episode_index.to_numpy(dtype=int)
    action = positions.to_numpy(dtype=int)
    slots = frame.target_slot.to_numpy(dtype=int)
    if not (
        np.all((0 <= episode) & (episode < len(costs)))
        and np.all((-1 <= slots) & (slots < costs.shape[2]))
    ):
        raise ValueError("projection cost indices are outside the forecast array")
    absolute = np.empty(len(frame))
    floor = np.empty(len(frame))
    joint = slots == -1
    joint_costs = costs.mean(axis=2)
    absolute[joint] = joint_costs[episode[joint], action[joint]]
    floor[joint] = joint_costs.min(axis=1)[episode[joint]]
    absolute[~joint] = costs[episode[~joint], action[~joint], slots[~joint]]
    floor[~joint] = costs.min(axis=1)[episode[~joint], slots[~joint]]
    return absolute, floor
