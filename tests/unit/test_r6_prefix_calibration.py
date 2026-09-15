import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_r6_prefix_calibration import calibration_origins  # noqa: E402


def test_calibration_history_is_after_fit_and_before_prefix_end():
    prefix = np.ones((1200, 3))
    prefix[910, 2] = np.nan
    origins = calibration_origins(prefix, 600)
    assert len(origins) == 4
    assert all(600 <= origin - 96 < origin <= len(prefix) for origin in origins)
    assert all(not origin - 96 <= 910 < origin for origin in origins)
    assert all(right - left >= 96 for left, right in zip(origins, origins[1:], strict=False))


def test_single_complete_history_is_not_used_to_fit_local_weights():
    prefix = np.full((1100, 2), np.nan)
    prefix[600:696] = 1.0
    assert calibration_origins(prefix, 600) == []
