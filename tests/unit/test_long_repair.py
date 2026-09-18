import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from long_repair_core import repaired_long_context, selected_repairs  # noqa: E402


def test_only_registered_current_target_gaps_are_repaired():
    native = np.arange(60, dtype=np.float32).reshape(3, 20)
    native[0, 2] = np.nan
    native[0, -2:] = np.nan
    native[2, -3:] = np.nan
    repair = np.nan_to_num(native[:2, -5:].T, nan=-3)
    actual = repaired_long_context(native, repair)
    np.testing.assert_array_equal(actual[:, :-5], native[:, :-5])
    np.testing.assert_array_equal(actual[2], native[2])
    observed = np.isfinite(native)
    np.testing.assert_array_equal(actual[observed], native[observed])
    np.testing.assert_array_equal(actual[0, -2:], [-3, -3])


def test_complete_current_targets_are_an_exact_input_identity():
    native = np.arange(60, dtype=np.float32).reshape(3, 20)
    native[2, -1] = np.nan
    np.testing.assert_array_equal(repaired_long_context(native, native[:2, -5:].T), native)


def test_repairs_must_not_change_original_observations():
    native = np.arange(60, dtype=np.float32).reshape(3, 20)
    repair = native[:2, -5:].T.copy()
    repair[0, 0] += 1
    with pytest.raises(AssertionError):
        repaired_long_context(native, repair)


def test_the_fixed_source_pools_are_not_selected_using_outcome_errors():
    names = ["gaussian", "local_ridge", "knn_multivariate", "peer_ridge"]
    assert selected_repairs("beijing", names) == [
        ("knn_multivariate", 2),
        ("gaussian", 0),
        ("local_ridge", 1),
        ("peer_ridge", 3),
    ]
    with pytest.raises(ValueError):
        selected_repairs("hdb", names)
