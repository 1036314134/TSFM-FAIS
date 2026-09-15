import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from freeze_followup_cohort import evenly_spaced, select_windows, target_window_sha


def test_empty_and_small_sampling_keep_all_without_repetition():
    assert evenly_spaced([], 16) == []
    assert evenly_spaced([1, 2, 3], 16) == [1, 2, 3]
    result = evenly_spaced(list(range(100)), 16)
    assert len(set(result)) == 16 and result[0] == 0 and result[-1] == 99


def test_native_sampling_preserves_strata_and_eligibility():
    rows = [
        {"origin": i * 192 + 96, "eligible": i != 7, "context_has_missing": i % 2 == 1}
        for i in range(80)
    ]
    picked = select_windows({"windows": rows}, "new_native", np.ones((16000, 2), bool))
    assert len(picked) == 32 and all(row["eligible"] for row in picked)
    assert sum(row["context_has_missing"] for row in picked) == 16
    assert picked == sorted(picked, key=lambda row: row["origin"])


def test_duplicate_check_preserves_nan_mask_and_detects_changed_values():
    values = np.zeros((192, 2))
    copy = values.copy()
    copy[0, 0] = -0.0
    assert target_window_sha(values) == target_window_sha(copy)
    copy[0, 0] = np.nan
    assert target_window_sha(values) != target_window_sha(copy)
    values[0, 0] = np.nan
    assert target_window_sha(values) == target_window_sha(copy)
    copy[-1, 1] = 1
    assert target_window_sha(values) != target_window_sha(copy)
