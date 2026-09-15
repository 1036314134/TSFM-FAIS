import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from interval_gate_inputs import HAS_INDEX, WIDTH_INDEX, add_intervals  # noqa: E402


def test_interval_fields_follow_interleaved_target_and_episode_ids():
    rng = np.random.default_rng(9515)
    base = rng.normal(size=(4, 7, 33)).astype(np.float32)
    base[:, :, [HAS_INDEX, WIDTH_INDEX]] = 0
    original = base.copy()
    frame = pd.DataFrame({"episode_index": [2, 0, 2, 1], "target_slot": [1, -1, 0, 1]})
    joint = np.arange(21.0).reshape(3, 7)
    target = np.arange(42.0).reshape(3, 7, 2) + 100
    result = add_intervals(base, frame, joint, target)
    np.testing.assert_array_equal(
        result[:, :, WIDTH_INDEX],
        np.stack([target[2, :, 1], joint[0], target[2, :, 0], target[1, :, 1]]),
    )
    np.testing.assert_array_equal(result[:, :, HAS_INDEX], np.ones((4, 7)))
    keep = [index for index in range(33) if index not in (HAS_INDEX, WIDTH_INDEX)]
    np.testing.assert_array_equal(result[:, :, keep], original[:, :, keep])
    np.testing.assert_array_equal(base, original)


def test_incomplete_quantiles_cannot_silently_remove_a_source_decision():
    frame = pd.DataFrame({"episode_index": [0], "target_slot": [-1]})
    joint = np.zeros((1, 7))
    joint[0, 3] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        add_intervals(np.zeros((1, 7, 33)), frame, joint, np.zeros((1, 7, 2)))
