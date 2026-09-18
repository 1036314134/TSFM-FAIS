import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from long_native_core import extend_visible_history, model_inputs, standardized_point  # noqa: E402


def test_extension_excludes_future_and_preserves_every_current_hidden_cell():
    values = np.arange(300, dtype=float).reshape(100, 3)
    current = values[60:80].copy()
    current[-5:, :2] = np.nan
    first, start = extend_visible_history(values, 80, current, 50)
    changed = values.copy()
    changed[80:] = -999
    second, _ = extend_visible_history(changed, 80, current, 50)
    assert start == 30 and len(first) == 50
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[-20:], current)
    np.testing.assert_array_equal(first[:-20], values[30:60])


def test_raw_and_standardized_queries_share_time_variables_and_short_tail():
    values = np.arange(240, dtype=float).reshape(80, 3)
    current = values[60:].copy()
    current[-3:, 0] = np.nan
    mean, scale, selected = (
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([2.0, 3.0, 4.0]),
        np.asarray([0, 2]),
    )
    short = {
        "mean": mean,
        "scale": scale,
        "selected": selected,
        "native": np.array(
            ((current[:, selected] - mean[selected]) / scale[selected]).T,
            dtype=np.float32,
            order="C",
        ),
    }
    history, _ = extend_visible_history(values, 80, current, 50)
    queries = model_inputs(history, short, 1)
    assert len(queries) == 5 and all(q.flags.c_contiguous for q in queries.values())
    np.testing.assert_array_equal(queries["native_long_prefix_peer"][:, -20:], short["native"])
    np.testing.assert_array_equal(
        queries["native_long_raw_peer"][:, -20:], current[:, selected].T.astype(np.float32)
    )


def test_raw_forecasts_are_scored_in_the_original_prefix_units():
    q = np.full((2, 3, 4), 12, dtype=np.float32)
    result = standardized_point(
        q, "native_long_raw_peer", 2, 4, 1, np.asarray([10, 8]), np.asarray([2, 4])
    )
    np.testing.assert_array_equal(result, np.ones((4, 2)))


def test_changed_observations_and_invalid_time_boundaries_are_rejected():
    values = np.arange(60, dtype=float).reshape(20, 3)
    current = values[-5:].copy()
    current[0, 0] = -100
    with pytest.raises(AssertionError):
        extend_visible_history(values, 20, current, 10)
    with pytest.raises(ValueError):
        extend_visible_history(values, 4, values[:5], 10)
