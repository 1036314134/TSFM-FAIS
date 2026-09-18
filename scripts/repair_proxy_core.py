"""Frozen imputation trajectories as additional group-context series."""

import numpy as np

QUERY_NAMES = (
    "raw_plus_pool",
    "raw_plus_gaussian",
    "raw_plus_knn",
    "raw_plus_pool_median",
    "raw_plus_duplicate_pool",
    "raw_plus_duplicate_single",
    "proxy_pool_only",
)
POINT_NAMES = (
    *QUERY_NAMES[:-1],
    "proxy_pool_only_mean",
    "proxy_pool_only_median",
    "raw_plus_pool_proxy_mean",
    "raw_plus_pool_proxy_median",
)


def validate(data):
    native, pool = data["native"], data["pool"]
    count, length, targets = pool.shape
    if length != native.shape[1] or targets > len(native) or not np.isfinite(pool).all():
        raise ValueError("aligned finite target repair trajectories are required")
    observed = np.isfinite(native[:targets].T)
    for candidate in pool:
        np.testing.assert_array_equal(candidate[observed], native[:targets].T[observed])
    if not 0 <= int(data["gaussian_index"]) < count or not 0 <= int(data["knn_index"]) < count:
        raise ValueError("the fixed single-proxy controls are missing")
    return bool(observed.all())


def make_queries(data):
    if validate(data):
        return {}
    native, pool = data["native"], data["pool"]
    count, length, targets = pool.shape
    flat = np.ascontiguousarray(pool.transpose(0, 2, 1).reshape(count * targets, length))
    additions = {
        "raw_plus_pool": flat,
        "raw_plus_gaussian": pool[int(data["gaussian_index"])].T,
        "raw_plus_knn": pool[int(data["knn_index"])].T,
        "raw_plus_pool_median": np.median(pool, axis=0).T,
        "raw_plus_duplicate_pool": np.tile(native[:targets], (count, 1)),
        "raw_plus_duplicate_single": native[:targets],
    }
    result = {
        name: np.ascontiguousarray(np.concatenate([native, value], axis=0), dtype=np.float32)
        for name, value in additions.items()
    }
    result["proxy_pool_only"] = flat
    for name, context in result.items():
        if name != "proxy_pool_only":
            np.testing.assert_array_equal(context[: len(native)], native)
    return result


def read_points(data, quantiles, horizon, median_index):
    count, _, targets = data["pool"].shape
    points = {
        name: raw[:targets, median_index, :horizon].T.astype(float)
        for name, raw in quantiles.items()
        if name != "proxy_pool_only"
    }
    for query_name, prefix, offset in (
        ("raw_plus_pool", "raw_plus_pool_proxy", len(data["native"])),
        ("proxy_pool_only", "proxy_pool_only", 0),
    ):
        raw = quantiles[query_name][offset:, median_index, :horizon].astype(float)
        values = raw.reshape(count, targets, horizon).transpose(0, 2, 1)
        points[prefix + "_mean"] = values.mean(0)
        points[prefix + "_median"] = np.median(values, axis=0)
    if set(points) != set(POINT_NAMES):
        raise ValueError("registered repair-proxy point outputs are incomplete")
    return points
