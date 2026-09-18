import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from collect_hdb_hourly import parse_snapshot, parse_time  # noqa: E402


def park(name, update, kinds):
    return {
        "carpark_number": name,
        "update_datetime": update,
        "carpark_info": [
            {"lot_type": kind, "lots_available": value, "total_lots": total}
            for kind, value, total in kinds
        ],
    }


def test_C_type_and_latest_timestamp_take_precedence_over_API_list_order():
    payload = {
        "items": [
            {
                "timestamp": "2025-06-01T00:59:00+08:00",
                "carpark_data": [
                    park("SB36", "2025-06-01T00:58:00", [("M", "66", "100"), ("C", "127", "279")]),
                    park("SB36", "2025-06-01T00:59:00", [("C", "125", "279")]),
                ],
            }
        ]
    }
    result = parse_snapshot(payload, parse_time("2025-06-01T01:00:00"))
    assert result["records"][0]["fresh_value"] == 125
    assert result["records"][0]["lot_type"] == "C"
    assert result["duplicate_C_identifiers"] == 1


def test_conflicting_latest_data_is_not_silently_selected():
    payload = {
        "items": [
            {
                "timestamp": "2025-06-01T00:59:00+08:00",
                "carpark_data": [
                    park("A", "2025-06-01T00:58:00", [("C", "1", "10")]),
                    park("A", "2025-06-01T00:58:00", [("C", "2", "10")]),
                ],
            }
        ]
    }
    result = parse_snapshot(payload, parse_time("2025-06-01T01:00:00"))
    assert result["records"][0]["fresh_value"] is None
    assert result["records"][0]["flags"] & 4


def test_future_and_stale_updates_are_distinct_from_fresh_measurements():
    payload = {
        "items": [
            {
                "timestamp": "2025-06-01T01:59:00+08:00",
                "carpark_data": [
                    park("FUTURE", "2025-06-01T02:01:00", [("C", "1", "10")]),
                    park("STALE", "2025-06-01T00:20:00", [("C", "2", "10")]),
                    park("NO_C", "2025-06-01T01:59:00", [("M", "8", "20")]),
                ],
            }
        ]
    }
    result = parse_snapshot(payload, parse_time("2025-06-01T02:00:00"))
    rows = {r["carpark_number"]: r for r in result["records"]}
    assert rows["FUTURE"]["fresh_value"] is None and rows["FUTURE"]["flags"] == 32
    assert rows["STALE"]["raw_asof"] == 2 and rows["STALE"]["fresh_value"] is None
    assert "NO_C" in result["present_identifiers"] and "NO_C" not in rows


def test_capacity_anomaly_is_preserved_and_unknown_capacity_does_not_erase_count():
    payload = {
        "items": [
            {
                "timestamp": "2025-06-01T00:59:00+08:00",
                "carpark_data": [
                    park("A", "2025-06-01T00:59:00", [("C", "12", "10")]),
                    park("B", "2025-06-01T00:59:00", [("C", "7", "unknown")]),
                ],
            }
        ]
    }
    result = parse_snapshot(payload, parse_time("2025-06-01T01:00:00"))
    rows = {r["carpark_number"]: r for r in result["records"]}
    assert rows["A"]["fresh_value"] == 12 and rows["A"]["over_capacity"]
    assert rows["B"]["fresh_value"] == 7 and rows["B"]["total_lots"] is None


def test_later_snapshot_cannot_enter_an_earlier_grid_point():
    with pytest.raises(ValueError, match="at or before"):
        parse_snapshot(
            {"items": [{"timestamp": "2025-06-01T02:00:00+08:00", "carpark_data": []}]},
            parse_time("2025-06-01T01:00:00"),
        )
