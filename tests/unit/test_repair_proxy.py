import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from repair_proxy_core import POINT_NAMES, make_queries, read_points, validate  # noqa: E402


def example():
    native = np.arange(30, dtype=np.float32).reshape(3, 10)
    native[0, -2:] = np.nan
    native[1, -3:] = np.nan
    one = np.nan_to_num(native[:2].T, nan=1)
    two = np.nan_to_num(native[:2].T, nan=2)
    return {
        "native": native,
        "pool": np.stack([one, two]),
        "gaussian_index": np.asarray(0),
        "knn_index": np.asarray(1),
    }


def test_anchor_rows_and_missingness_remain_exact():
    data = example()
    queries = make_queries(data)
    assert len(queries) == 7
    for name, context in queries.items():
        if name != "proxy_pool_only":
            np.testing.assert_array_equal(context[:3], data["native"])
        assert context.dtype == np.float32 and context.flags.c_contiguous
    assert queries["raw_plus_pool"].shape == queries["raw_plus_duplicate_pool"].shape == (7, 10)


def test_repairs_cannot_modify_original_observations():
    data = example()
    data["pool"][0, 0, 0] = -999
    with pytest.raises(AssertionError):
        validate(data)


def test_joint_readouts_keep_candidate_and_target_indices_separate():
    data = example()
    requests = make_queries(data)
    raw = {
        name: np.broadcast_to(np.arange(len(context))[:, None, None], (len(context), 3, 4)).copy()
        for name, context in requests.items()
    }
    points = read_points(data, raw, 4, 1)
    assert set(points) == set(POINT_NAMES)
    np.testing.assert_array_equal(points["raw_plus_pool"][0], [0, 1])
    np.testing.assert_array_equal(points["raw_plus_pool_proxy_mean"][0], [4, 5])
    np.testing.assert_array_equal(points["proxy_pool_only_median"][0], [1, 2])


def test_complete_targets_use_the_common_native_fallback_despite_auxiliary_missingness():
    data = example()
    data["native"][:2] = data["pool"][0].T
    data["native"][2, -1] = np.nan
    data["pool"] = np.repeat(data["native"][:2].T[None], 2, axis=0)
    assert validate(data) and make_queries(data) == {}
