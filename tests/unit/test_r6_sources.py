import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_r6_sources import eligibility, regular_grid


def test_unrecorded_hours_stay_unobserved_without_zero_filling():
    frame = pd.DataFrame({"a": [4.0, 8.0], "b": [6.0, 10.0]})
    values, _, present, audit = regular_grid(
        frame, pd.to_datetime(["2020-01-01 00:00", "2020-01-01 02:00"]), ["a", "b"], "h"
    )
    assert values.shape == (3, 2) and np.isnan(values[1]).all()
    assert present.tolist() == [True, False, True]
    assert audit["inserted_unobserved_bins"] == 1 and audit["original_missing_cells"] == 0


def test_one_second_clock_rounding_preserves_aggregated_observations():
    frame = pd.DataFrame({"a": [2.0, 4.0, 8.0], "b": [3.0, 5.0, 9.0]})
    times = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:59", "2020-01-01 00:01:00"])
    values, _, present, audit = regular_grid(frame, times, ["a", "b"], "min", minute_rounding=True)
    np.testing.assert_array_equal(values, [[2, 3], [6, 7]])
    assert present.all() and audit["duplicate_rounded_rows_aggregated"] == 1
    assert audit["maximum_rounding_seconds"] == 1
    with pytest.raises(ValueError, match="one-second"):
        regular_grid(
            frame, times + pd.Timedelta(seconds=5), ["a", "b"], "min", minute_rounding=True
        )


def test_dual_horizon_eligibility_rejects_missing_short_horizon_labels():
    values = np.ones((1200, 2))
    origin = int(0.6 * len(values)) + 96
    values[origin : origin + 60] = np.nan
    result = eligibility(values, np.ones(len(values), bool))
    assert result["eligible_window_count"] == 0
    row = result["windows"][0]
    assert row["future_observed_by_horizon"] == {"96": [36, 36], "192": [132, 132]}
    assert row["exclusion_reason"] == "insufficient_first96_future_observations"
