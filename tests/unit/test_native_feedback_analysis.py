from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_native_feedback.py"
SPEC = importlib.util.spec_from_file_location("native_feedback", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_prior_excludes_current_family_and_validation_outcomes():
    rows = []
    for family in ("source", "held_out"):
        for split in ("train", "validation"):
            for action, error in (("locf", 2.0), ("linear_interp", 1.0)):
                rows.append(
                    dict(
                        model_id="chronos2",
                        family_id=family,
                        dataset_id=family,
                        split=split,
                        target_slot=-1,
                        candidate_id=action,
                        mae=error,
                        mse=error,
                    )
                )
    frame = pd.DataFrame(rows)
    first = MODULE.source_fixed_choice(frame, "chronos2", "held_out", ["locf", "linear_interp"])
    assert first[0] == "linear_interp"
    excluded = (frame.family_id == "held_out") | (frame.split == "validation")
    frame.loc[excluded & (frame.candidate_id == "locf"), ["mae", "mse"]] = 0.0
    frame.loc[excluded & (frame.candidate_id == "linear_interp"), ["mae", "mse"]] = 1e9
    assert (
        MODULE.source_fixed_choice(frame, "chronos2", "held_out", ["locf", "linear_interp"])
        == first
    )


def history():
    return pd.DataFrame(
        [
            dict(
                candidate_id=action,
                target_slot=slot,
                observed_count=20,
                historical_mae=error,
                historical_mse=error,
            )
            for action, errors in (("guarded_direct", (0.0, 2.0)), ("locf", (2.0, 0.0)))
            for slot, error in enumerate(errors)
        ]
    )


def test_independent_targets_and_sequence_tie_use_valid_model_granularity():
    scales = {"mae": 1.0, "mse": 1.0}
    assert MODULE.restricted_choice(history(), "locf", scales, per_target=True) == (
        "guarded_direct",
        "locf",
    )
    assert MODULE.restricted_choice(history(), "locf", scales, per_target=False) == (
        "guarded_direct",
        "guarded_direct",
    )


def test_unobserved_feedback_falls_back_and_unequal_coverage_is_rejected():
    frame = history()
    frame.loc[frame.target_slot == 1, ["historical_mae", "historical_mse"]] = np.nan
    frame.loc[frame.target_slot == 1, "observed_count"] = 0
    scales = {"mae": 1.0, "mse": 1.0}
    assert MODULE.restricted_choice(frame, "locf", scales, per_target=True) == (
        "guarded_direct",
        "guarded_direct",
    )
    frame.loc[0, "observed_count"] = 19
    with pytest.raises(ValueError, match="same observed"):
        MODULE.restricted_choice(frame, "locf", scales, per_target=False)
