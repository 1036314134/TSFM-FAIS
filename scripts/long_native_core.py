"""Causal native-history extension with exact preservation of the registered short input."""

import numpy as np

CONTEXT_LIMIT = 8192


def extend_visible_history(values, origin, current, limit=CONTEXT_LIMIT):
    if origin < len(current) or origin > len(values) or limit < len(current):
        raise ValueError("a complete current-history interval must precede the origin")
    if values.shape[1] != current.shape[1]:
        raise ValueError("historical variable identities changed")
    observed = np.isfinite(current)
    np.testing.assert_array_equal(
        current[observed], values[origin - len(current) : origin][observed]
    )
    start = max(0, origin - limit)
    history = np.array(values[start:origin], dtype=float, copy=True, order="C")
    history[-len(current) :] = current
    return history, start


def model_inputs(history, short, targets):
    selected, mean, scale = short["selected"], short["mean"], short["scale"]
    np.testing.assert_array_equal(selected[:targets], np.arange(targets))
    raw = np.array(history[:, selected].T, dtype=np.float32, order="C")
    standardized = np.array(
        ((history[:, selected] - mean[selected]) / scale[selected]).T, dtype=np.float32, order="C"
    )
    np.testing.assert_array_equal(standardized[:, -short["native"].shape[1] :], short["native"])
    return {
        "native_long_raw_peer": raw,
        "native_long_raw_targets": np.ascontiguousarray(raw[:targets]),
        "native_long_prefix_peer": standardized,
        "native_long_prefix_targets": np.ascontiguousarray(standardized[:targets]),
        "native_short_restoration": short["native"].copy(),
    }


def standardized_point(quantiles, name, targets, horizon, median_index, mean, scale):
    point = quantiles[:targets, median_index, :horizon].T.astype(float)
    if name.startswith("native_long_raw_"):
        point = (point - mean[:targets]) / scale[:targets]
    return point
