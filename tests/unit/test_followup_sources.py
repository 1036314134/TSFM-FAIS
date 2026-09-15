import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_followup_sources import AIR_COLUMNS, POWER_COLUMNS, parse_uci_frame


def test_air_sentinels_padding_and_missing_timestamp_are_not_observations():
    header = ";".join(("Date", "Time", *AIR_COLUMNS, "Unnamed: 15"))
    rows = [
        "01/01/2004;00.00.00;" + ";".join(["-200", "1,5", *(["3"] * 11), ""]),
        "01/01/2004;02.00.00;" + ";".join(["2"] * 13 + [""]),
        ";" * 15,
    ]
    frame = pd.read_csv(io.StringIO("\n".join([header, *rows])), sep=";", decimal=",")
    values, columns, times, audit = parse_uci_frame(frame, "uci_air_quality")
    assert tuple(columns) == AIR_COLUMNS
    assert values.shape == (3, 13) and len(times) == 3
    assert np.isnan(values[0, 0]) and values[0, 1] == 1.5
    assert np.isnan(values[1]).all()
    assert audit["empty_padding_rows_removed"] == audit["inserted_unobserved_timestamps"] == 1


def test_power_missing_measurements_keep_their_minute():
    frame = pd.DataFrame(
        [
            ["01/01/2007", "00:00:00", *([1.0] * 7)],
            ["01/01/2007", "00:01:00", *([np.nan] * 7)],
        ],
        columns=["Date", "Time", *POWER_COLUMNS],
    )
    values, _, _, audit = parse_uci_frame(frame, "uci_household_power")
    assert values.shape == (2, 7) and np.isnan(values[1]).all()
    assert audit["empty_padding_rows_removed"] == 0
    frame.loc[1, "Time"] = "00:00:00"
    with pytest.raises(ValueError, match="unique"):
        parse_uci_frame(frame, "uci_household_power")
