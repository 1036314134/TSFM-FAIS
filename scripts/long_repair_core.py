"""Extend fixed target repairs into the already registered native long history."""

import numpy as np


def selected_repairs(dataset, names):
    selected = (
        ["knn_multivariate", "gaussian", "local_ridge", "peer_ridge"]
        if dataset == "beijing"
        else ["median", "ffill", "linear", "seasonal24", "knn", "gaussian"]
    )
    if any(name not in names for name in selected):
        raise ValueError("a registered fixed repair is unavailable")
    return [(name, names.index(name)) for name in selected]


def repaired_long_context(native, repair):
    targets, window = repair.shape[1], repair.shape[0]
    if native.ndim != 2 or targets > len(native) or window > native.shape[1]:
        raise ValueError("repair and long-context geometry are incompatible")
    if not np.isfinite(repair).all():
        raise ValueError("the cached target repair must be complete")
    current = native[:targets, -window:]
    observed = np.isfinite(current)
    np.testing.assert_array_equal(repair.T[observed], current[observed])
    result = native.copy()
    result[:targets, -window:] = np.where(observed, current, repair.T)
    np.testing.assert_array_equal(result[:, :-window], native[:, :-window])
    np.testing.assert_array_equal(result[targets:], native[targets:])
    return np.ascontiguousarray(result, dtype=np.float32)
