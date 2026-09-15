import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_source_quantiles import guarded_outputs, interval_quality  # noqa: E402


def test_joint_empty_covariate_switches_points_and_all_quantiles_together():
    context = np.array([[1, 2, np.nan], [3, 4, np.nan]])
    native = np.zeros((4, 2))
    fallback = np.full((4, 2), 10.0)
    native_q = np.stack([native - 1, native, native + 1], axis=-1)
    fallback_q = np.stack([fallback - 2, fallback, fallback + 2], axis=-1)
    point, quantiles, chosen = guarded_outputs(
        context, native, native_q, fallback, fallback_q, joint=True
    )
    np.testing.assert_array_equal(point, fallback)
    np.testing.assert_array_equal(quantiles, fallback_q)
    np.testing.assert_array_equal(chosen, [True, True])


def test_independent_empty_target_only_switches_that_target():
    context = np.array([[np.nan, 2, np.nan], [np.nan, 4, np.nan]])
    native, fallback = np.zeros((4, 2)), np.full((4, 2), 10.0)
    point, quantiles, chosen = guarded_outputs(
        context,
        native,
        np.repeat(native[..., None], 3, axis=-1),
        fallback,
        np.repeat(fallback[..., None], 3, axis=-1),
        joint=False,
    )
    np.testing.assert_array_equal(point, np.tile([10.0, 0.0], (4, 1)))
    np.testing.assert_array_equal(quantiles[..., 1], point)
    np.testing.assert_array_equal(chosen, [True, False])


def test_interval_width_uses_original_target_scales_and_keeps_crossings():
    quantiles = np.array([[[[1, 2, 5], [8, 6, 4]], [[2, 3, 6], [7, 5, 3]]]], float)
    original = quantiles.copy()
    result = interval_quality(quantiles, [2.0, 4.0])
    np.testing.assert_array_equal(result["width_by_target"], [[2.0, -1.0]])
    np.testing.assert_array_equal(result["width_joint"], [0.5])
    assert result["crossed"].sum() == 2
    assert result["reversed_endpoints"].sum() == 2
    np.testing.assert_array_equal(quantiles, original)


def test_nonfinite_quantile_is_reported_as_unavailable():
    quantiles = np.zeros((1, 4, 2, 3))
    quantiles[0, 1, 0, 2] = np.nan
    result = interval_quality(quantiles, np.ones(2))
    assert (~result["finite"]).sum() == 1
    assert np.isnan(result["width_by_target"][0, 0])
