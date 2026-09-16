import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_tail_bridge import direct_gaps, direct_guard  # noqa: E402
from evaluate_tail_bridge import aggregate, fixed_points  # noqa: E402
from forecast_matched_replay import guarded_long  # noqa: E402
from tail_bridge_core import bridge_slice, mask_tail, supported_gaps  # noqa: E402


def test_bridge_targets_original_times_without_compressing_internal_gaps():
    origin = 500
    context = np.tile(np.arange(origin - 96, origin, dtype=float)[:, None], (1, 3))
    context[20:27] = np.nan
    for gap in (0, 1, 8, 24, 48):
        missing = context.copy()
        if gap:
            missing[-gap:] = np.nan
        assert supported_gaps(missing, True) == direct_gaps(missing, True) == [gap, gap]
        cropped = missing[: 96 - gap]
        assert len(cropped) == 96 - gap
        assert np.isnan(cropped[20:27]).all()
        assert cropped[27, 0] == origin - 96 + 27
        # A timestamp forecaster exposes any off-by-one error in the output slice.
        requested = np.arange(origin - gap, origin + 96)[:, None]
        np.testing.assert_array_equal(
            bridge_slice(requested, gap)[:, 0], np.arange(origin, origin + 96)
        )
        if gap:
            assert requested[:96, 0][-1] != origin + 95


def test_joint_covariates_are_retained_and_independent_targets_use_separate_gaps():
    context = np.ones((96, 3))
    context[-28:, 0] = np.nan
    context[-27:, 1] = np.nan
    assert supported_gaps(context, True) == [0, 0]
    assert supported_gaps(context, False) == [28, 27]
    context[-3:, 2] = np.nan
    assert supported_gaps(context, True) == [3, 3]
    for joint in (True, False):
        assert supported_gaps(context, joint) == direct_gaps(context, joint)
    context[-60:, 0] = np.nan
    assert supported_gaps(context, False) == [0, 27]
    assert supported_gaps(np.full((96, 3), np.nan), True) == [0, 0]


def test_tail_masks_and_native_guard_preserve_observed_values():
    original = np.arange(288.0).reshape(96, 3)
    copies = [mask_tail(original, gap) for gap in (8, 24, 48)]
    for masked, gap in zip(copies, (8, 24, 48), strict=True):
        np.testing.assert_array_equal(masked[:-gap], original[:-gap])
        assert np.isnan(masked[-gap:]).all()
    assert np.isfinite(original).all()
    context = copies[1].copy()
    context[:, 2] = np.nan
    context[:5, 0] = np.nan
    defaults = np.array([2.0, 3.0, 4.0])
    for joint in (True, False):
        rebuilt = direct_guard(context, defaults, joint)
        np.testing.assert_array_equal(rebuilt, guarded_long(context, defaults, joint))
        np.testing.assert_array_equal(rebuilt[np.isfinite(context)], context[np.isfinite(context)])
    with pytest.raises(ValueError):
        mask_tail(context, 8)
    with pytest.raises(ValueError):
        bridge_slice(np.zeros((96, 2)), 8)


def test_fixed_control_uses_the_matching_output_budget_and_source_action_order():
    actions = [f"action{i}" for i in range(8)]
    bank = np.broadcast_to(np.arange(8.0)[:, None, None], (8, 96, 2)).copy()
    prediction = {
        "actions": np.asarray(actions),
        "normal": bank,
        "budget": bank + 100,
        "bridge": np.ones((96, 2)) * 7,
        "long": np.ones((2, 96, 2)) * 9,
    }
    control = {
        "actions": actions,
        "single_index": 3,
        "fixed_mae": {"weights": [0.5, 0.5, 0, 0, 0, 0, 0, 0]},
        "fixed_joint_weights": [0, 0, 0, 0, 0, 0, 0, 1.0],
    }
    points = fixed_points(prediction, control)
    assert len(points) == 29
    assert np.all(points["median8"] == 3.5)
    assert np.all(points["budget_median8"] == 103.5)
    assert np.all(points["source_fixed_mae"] == 0.5)
    assert np.all(points["budget_source_fixed_joint"] == 107)
    assert np.all(points["source_single_mae"] == 3)
    with pytest.raises(ValueError):
        fixed_points(prediction, {**control, "actions": actions[::-1]})


def test_group_aggregation_does_not_weight_repeated_masks_as_extra_histories():
    rows = []
    for base, values in (("a", [1.0, 2.0, 3.0]), ("b", [8.0])):
        for gap, value in zip((8, 24, 48), values, strict=False):
            rows.append(
                {
                    "panel": "synthetic",
                    "model_id": "chronos2",
                    "method": "bridge_native",
                    "group_id": "g",
                    "family_id": "f",
                    "dataset_id": "d",
                    "item_id": "s",
                    "base_id": base,
                    "gap": gap,
                    "mae": value,
                    "mse": value * 2,
                }
            )
    # Histories have unequal repetitions here to make accidental row weighting detectable.
    summary = aggregate(pd.DataFrame(rows))["summary"]
    pooled = summary[summary.evaluation_panel == "synthetic_all"].iloc[0]
    assert pooled.mae == 5.0
    assert pooled.mse == 10.0
