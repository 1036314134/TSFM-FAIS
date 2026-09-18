import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from group_scope_core import comparison_queries, diagnostic_points  # noqa: E402


def data():
    native = np.arange(30, dtype=np.float32).reshape(3, 10)
    native[0, -1] = np.nan
    native[1, -2:] = np.nan
    pool = np.stack([np.nan_to_num(native[:2].T, nan=value) for value in (1.0, 2.0, 3.0)])
    return {
        "native": native,
        "pool": pool,
        "pool_names": np.asarray(["a", "b", "c"]),
        "gaussian_index": np.asarray(0),
        "knn_index": np.asarray(1),
    }


def test_candidate_grouping_changes_only_group_ids():
    q = comparison_queries(data())
    np.testing.assert_array_equal(q["isolated_candidates"][0], q["joint_candidates"][0])
    np.testing.assert_array_equal(q["isolated_candidates"][1], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(q["joint_candidates"][1], np.zeros(6, dtype=np.int64))


def test_target_anchor_drops_other_direct_variables_and_preserves_target_nans():
    d = data()
    q = comparison_queries(d)
    np.testing.assert_array_equal(q["target_anchor_candidates"][0][:2], d["native"][:2])
    np.testing.assert_array_equal(q["target_anchor_candidates"][0][2:], q["joint_candidates"][0])
    np.testing.assert_array_equal(q["native_targets"][0], d["native"][:2])
    assert all(
        context.flags.c_contiguous and groups.dtype == np.int64 for context, groups in q.values()
    )


def test_readout_candidate_target_mapping_and_complete_target_queries():
    d = data()
    q = comparison_queries(d)
    raw = {
        name: np.broadcast_to(np.arange(len(context))[:, None, None], (len(context), 3, 4)).copy()
        for name, (context, _) in q.items()
    }
    points = diagnostic_points(d, raw, 4, 1)
    assert len(points) == 13
    np.testing.assert_array_equal(points["isolated_c"][0], [4, 5])
    np.testing.assert_array_equal(points["independent_proxy_median"][0], [2, 3])
    np.testing.assert_array_equal(points["target_anchor_pool"][0], [0, 1])
    np.testing.assert_array_equal(points["target_anchor_proxy_mean"][0], [4, 5])
    d["native"][:2] = d["pool"][0].T
    d["pool"] = np.repeat(d["native"][:2].T[None], 3, axis=0)
    assert len(comparison_queries(d)) == 4
