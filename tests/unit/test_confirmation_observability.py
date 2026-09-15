from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def audit_module():
    script = Path(__file__).parents[2] / "scripts/audit_r3_confirmation_sources.py"
    spec = importlib.util.spec_from_file_location("confirmation_observability", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_uses_later_context_and_two_completed_post_fit_probes():
    result = audit_module().confirmation_windows(np.ones((100, 3), bool), 4, 4, 2)
    assert result["prefix_end"] == 20
    assert result["temporal_boundary"] == 60
    assert result["eligible_window_count"] == 5
    origins = [row["origin"] for row in result["windows"]]
    assert origins == [64, 72, 80, 88, 96]
    for row in result["windows"]:
        assert row["origin"] - 4 >= 60
        assert row["origin"] - 2 * 4 - 4 >= 20
        assert row["recent_probe_observed_by_target"] == [[4, 4], [4, 4]]
    assert result["eligible_missing_context_count"] == 0


def test_confirmation_counts_native_target_and_auxiliary_missingness_separately():
    observed = np.ones((100, 3), bool)
    observed[60:64, 2] = False
    observed[68:72, 0] = False
    observed[80:83, 0] = False
    result = audit_module().confirmation_windows(observed, 4, 4, 2)
    assert result["eligible_window_count"] == 4
    assert result["eligible_missing_context_count"] == 2
    assert result["eligible_missing_target_context_count"] == 1
    rows = {row["origin"]: row for row in result["windows"]}
    assert rows[64]["context_empty_channel_count"] == 1
    assert rows[64]["context_empty_target_count"] == 0
    assert rows[72]["context_empty_target_count"] == 1
    assert rows[80]["exclusion_reason"] == "insufficient_future_observations"


def test_inadequate_prefix_blocks_confirmation_even_when_future_is_observed():
    observed = np.ones((100, 3), bool)
    observed[:19, 2] = False
    result = audit_module().confirmation_windows(observed, 4, 4, 2)
    assert not result["prefix_eligible"]
    assert result["eligible_window_count"] == 0
    assert all(
        row["exclusion_reason"] == "insufficient_prefix_observations" for row in result["windows"]
    )


def test_confirmation_audit_accepts_masks_without_reading_values_or_predictions():
    with pytest.raises(ValueError, match="boolean"):
        audit_module().confirmation_windows(np.ones((100, 3)), 4, 4, 2)
    with pytest.raises(ValueError, match="minimum future"):
        audit_module().confirmation_windows(np.ones((100, 3), bool), 4, 4, 5)
