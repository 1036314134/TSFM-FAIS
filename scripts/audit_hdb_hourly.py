"""Reconstruct the unscored car-lot panel independently from archived official JSON."""

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import numpy as np

TZ = timezone(timedelta(hours=8))


def time_value(text):
    value = datetime.fromisoformat(text)
    return value.replace(tzinfo=TZ) if value.tzinfo is None else value.astimezone(TZ)


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def reconstruct(payload, grid):
    eligible = [row for row in payload["items"] if time_value(row["timestamp"]) <= grid]
    if not eligible:
        raise ValueError("no causal official snapshot")
    item = sorted(eligible, key=lambda row: time_value(row["timestamp"]), reverse=True)[0]
    snapshot = time_value(item["timestamp"])
    rows, present, invalid = defaultdict(list), set(), set()
    for entry in item["carpark_data"]:
        identifier = str(entry["carpark_number"])
        present.add(identifier)
        lots = [lot for lot in entry.get("carpark_info", []) if lot.get("lot_type") == "C"]
        if not lots:
            continue
        try:
            updated = time_value(entry["update_datetime"])
        except (TypeError, ValueError, KeyError):
            invalid.add(identifier)
            continue
        if updated > grid:
            invalid.add(identifier)
            continue
        for lot in lots:
            rows[identifier].append(
                (updated, number(lot.get("lots_available")), number(lot.get("total_lots")))
            )
    reconstructed = {}
    for identifier in rows.keys() | invalid:
        if not rows.get(identifier):
            reconstructed[identifier] = (math.nan, math.nan, math.nan, math.nan, 32, None)
            continue
        latest = sorted(rows[identifier], key=lambda row: row[0], reverse=True)[0][0]
        latest_rows = [row for row in rows[identifier] if row[0] == latest]
        amounts = sorted({r[1] for r in latest_rows if not math.isnan(r[1]) and r[1] >= 0})
        capacities = sorted({r[2] for r in latest_rows if not math.isnan(r[2])})
        amount = amounts[0] if len(amounts) == 1 else math.nan
        capacity = capacities[0] if len(capacities) == 1 else math.nan
        age = (grid - latest).total_seconds()
        flag = 0
        if (grid - snapshot).total_seconds() > 3600:
            flag += 1
        if age > 3600:
            flag += 2
        if len(amounts) > 1 or len(capacities) > 1:
            flag += 4
        if math.isnan(amount):
            flag += 8
        if amount > capacity:
            flag += 128
        fresh = amount if flag & 15 == 0 else math.nan
        reconstructed[identifier] = (amount, fresh, capacity, age, flag, latest.isoformat())
    return snapshot, present, reconstructed


def audit(base):
    started = perf_counter()
    marker = base / "audit_manifest.json"
    if marker.exists():
        raise ValueError("preserve completed HDB audits")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((base / "identity.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["identity"] != identity:
        raise ValueError("collection is not complete or identity changed")
    if manifest["forecast_calls"] or manifest["prediction_errors_read"]:
        raise ValueError("raw collection must remain unscored")
    expected_identity = {
        "lot_type": "C",
        "sampling_minutes": 60,
        "api_query_lag_seconds": 60,
        "freshness_seconds": 3600,
        "start": "2025-06-01T00:00:00+08:00",
    }
    if any(identity[k] != v for k, v in expected_identity.items()):
        raise ValueError("registered acquisition rules changed")
    if identity["mode"] == "registered_unscored_collection" and identity["hours"] != 1344:
        raise ValueError("formal collection grid changed")
    panel = base / "hourly_panel.npz"
    if hashlib.sha256(panel.read_bytes()).hexdigest() != manifest["panel_sha256"]:
        raise ValueError("hourly panel changed")
    with np.load(panel, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    ids = arrays["identifiers"].tolist()
    if ids != sorted(set(ids)) or len(ids) != manifest["series"]:
        raise ValueError("duplicate, reordered or unregistered series")
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    if len(manifest["original_records"]) != identity["hours"]:
        raise ValueError("an official snapshot is missing")
    union, flag_counts = set(), {str(2**k): 0 for k in range(8)}
    start = time_value(identity["start"])
    for index, saved in enumerate(manifest["original_records"]):
        grid = start + timedelta(hours=index)
        if saved["index"] != index or arrays["timestamps"][index] != grid.isoformat():
            raise ValueError("registered time grid changed")
        label = grid.strftime("%Y%m%dT%H%M%S")
        record = json.loads((base / "records" / f"{label}.json").read_text(encoding="utf-8"))
        expected_query = grid - timedelta(minutes=1)
        url = urlparse(record["request_url"])
        if url.scheme != "https" or url.netloc != "api.data.gov.sg":
            raise ValueError("unexpected acquisition host")
        parameter = parse_qs(url.query)["date_time"]
        if parameter != [expected_query.replace(tzinfo=None).isoformat()]:
            raise ValueError("API query time changed")
        if record["api_query_timestamp"] != expected_query.isoformat():
            raise ValueError("recorded API query time changed")
        if record["requested_timestamp"] != grid.isoformat() or record["index"] != index:
            raise ValueError("recorded grid time changed")
        body = gzip.decompress((base / saved["raw_path"]).read_bytes())
        sha = hashlib.sha256(body).hexdigest()
        if (
            sha != saved["raw_sha256"]
            or sha != record["raw_sha256"]
            or len(body) != record["bytes"]
        ):
            raise ValueError("archived official response changed")
        snapshot, present, rows = reconstruct(json.loads(body), grid)
        if (
            snapshot.isoformat() != saved["snapshot_timestamp"]
            or snapshot.isoformat() != record["snapshot_timestamp"]
        ):
            raise ValueError("selected snapshot time changed")
        if sorted(present) != record["present_identifiers"]:
            raise ValueError("car-park presence metadata changed")
        union.update(rows)
        expected = np.full((4, len(ids)), np.nan, dtype=np.float32)
        flags = np.asarray([16 if name in present else 64 for name in ids], dtype=np.uint8)
        saved_rows = {row["carpark_number"]: row for row in record["records"]}
        if set(saved_rows) != set(rows) or len(saved_rows) != len(record["records"]):
            raise ValueError("C-type record population changed")
        for identifier, values in rows.items():
            column = lookup[identifier]
            expected[:, column] = values[:4]
            flags[column] = values[4]
            row = saved_rows[identifier]
            converted = [
                number(row[k])
                for k in ("raw_asof", "fresh_value", "total_lots", "update_age_seconds")
            ]
            np.testing.assert_array_equal(converted, values[:4])
            if (
                row["lot_type"] != "C"
                or row["flags"] != values[4]
                or row["update_datetime"] != values[5]
            ):
                raise ValueError("record type, flags or update time changed")
        for key, values in zip(
            ("raw_asof", "fresh_values", "total_lots", "update_age_seconds"), expected, strict=True
        ):
            np.testing.assert_array_equal(arrays[key][index], values)
        np.testing.assert_array_equal(arrays["flags"][index], flags)
        for bit in flag_counts:
            flag_counts[bit] += int(((flags & int(bit)) != 0).sum())
        if (index + 1) % 168 == 0:
            print(json.dumps({"snapshots_audited": index + 1}), flush=True)
    if set(ids) != union:
        raise ValueError("series with no observed C type were included")
    for field, key in (("fresh_cells", "fresh_values"), ("raw_cells", "raw_asof")):
        if int(np.isfinite(arrays[key]).sum()) != manifest[field]:
            raise ValueError("collection support counts changed")
    result = {
        "status": "completed",
        "snapshots": identity["hours"],
        "series": len(ids),
        "cells_reconstructed": int(arrays["flags"].size),
        "numeric_difference": 0,
        "flag_counts": flag_counts,
        "forecast_calls": 0,
        "prediction_errors_read": False,
        "manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": perf_counter() - started,
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    audit(parser.parse_args().output_root.resolve())
