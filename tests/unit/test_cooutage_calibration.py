import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import cooutage_calibration_core as core  # noqa: E402


@pytest.mark.parametrize("age", [6, 24, 54])
@pytest.mark.parametrize("pattern", core.PATTERNS)
def test_group_intervention_clears_only_registered_historical_cells(age, pattern):
    full = np.arange(500 * 17, dtype=float).reshape(500, 17)
    full[123, 8] = np.nan
    row = {"origin": 300, "outage_age": age, "outage_pattern": pattern}
    x = core.calibration_context(full, row)
    hidden = np.zeros((192, 17), bool)
    hidden[-age:, : 2 if pattern == "targets_only" else 6] = True
    if pattern == "regional_pollutants":
        hidden[-age:, 11:17] = True
    original = full[108:300]
    assert np.isnan(x[hidden]).all()
    np.testing.assert_array_equal(x[~hidden], original[~hidden])
    changed = full.copy()
    changed[300:] = 1e30
    past = changed[108:300]
    past[hidden] = -1e30
    np.testing.assert_array_equal(core.calibration_context(changed, row), x)


def test_expansion_preserves_parent_history_labels_and_has_unique_cases(monkeypatch):
    rows = [
        {"case_id": "a", "station": "station", "origin": 4000, "outage_age": 6},
        {"case_id": "b", "station": "station", "origin": 4500, "outage_age": 24},
    ]
    monkeypatch.setattr(core, "base_population", lambda _: (rows, ["eligibility"]))
    expanded, eligibility = core.calibration_population({})
    assert len(expanded) == len({r["case_id"] for r in expanded}) == 6
    assert eligibility == ["eligibility"]
    for row in rows:
        group = [r for r in expanded if r["parent_case_id"] == row["case_id"]]
        assert [r["outage_pattern"] for r in group] == list(core.PATTERNS)
        assert group[0]["case_id"] == row["case_id"]
        assert all(
            r["origin"] == row["origin"] and r["outage_age"] == row["outage_age"] for r in group
        )
