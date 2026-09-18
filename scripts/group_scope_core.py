"""Matched candidate grouping and direct forecasting-variable controls."""

import numpy as np
from repair_proxy_core import validate


def comparison_queries(data):
    validate(data)
    count, length, targets = data["pool"].shape
    flat = np.ascontiguousarray(data["pool"].transpose(0, 2, 1).reshape(count * targets, length))
    native = np.ascontiguousarray(data["native"][:targets])
    anchored = np.ascontiguousarray(np.concatenate([native, flat], axis=0))
    return {
        "isolated_candidates": (flat, np.repeat(np.arange(count, dtype=np.int64), targets)),
        "joint_candidates": (flat.copy(), np.zeros(len(flat), dtype=np.int64)),
        "native_targets": (native, np.zeros(targets, dtype=np.int64)),
        "target_anchor_candidates": (anchored, np.zeros(len(anchored), dtype=np.int64)),
    }


def diagnostic_points(data, raw, horizon, median_index):
    count, _, targets = data["pool"].shape

    def candidates(name, offset=0):
        return (
            raw[name][offset:, median_index, :horizon]
            .astype(float)
            .reshape(count, targets, horizon)
            .transpose(0, 2, 1)
        )

    isolated = candidates("isolated_candidates")
    joint = candidates("joint_candidates")
    anchored = candidates("target_anchor_candidates", targets)
    native = raw["native_targets"][:targets, median_index, :horizon].T.astype(float)
    augmented = np.concatenate([isolated, native[None]], axis=0)
    points = {
        "isolated_" + name: isolated[index]
        for index, name in enumerate(data["pool_names"].tolist())
    }
    points.update(
        independent_proxy_mean=isolated.mean(0),
        independent_proxy_median=np.median(isolated, axis=0),
        independent_plus_native_mean=augmented.mean(0),
        independent_plus_native_median=np.median(augmented, axis=0),
        joint_reference_mean=joint.mean(0),
        joint_reference_median=np.median(joint, axis=0),
        native_targets_only=native,
        target_anchor_pool=raw["target_anchor_candidates"][
            :targets, median_index, :horizon
        ].T.astype(float),
        target_anchor_proxy_mean=anchored.mean(0),
        target_anchor_proxy_median=np.median(anchored, axis=0),
    )
    if len(points) != count + 10:
        raise ValueError("the fixed scope control set is incomplete")
    return points
