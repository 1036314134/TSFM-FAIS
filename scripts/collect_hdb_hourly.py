"""Collect an unscored, auditable hourly C-lot panel from official historical snapshots."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

TZ = timezone(timedelta(hours=8))
START = datetime(2025, 6, 1, tzinfo=TZ)
HOURS = 56 * 24


def parse_time(value):
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=TZ) if parsed.tzinfo is None else parsed.astimezone(TZ)


def parse_snapshot(payload, requested):
    candidates = []
    for item in payload.get("items", []):
        stamp = parse_time(item["timestamp"])
        if stamp <= requested:
            candidates.append((stamp, item))
    if not candidates:
        raise ValueError("no snapshot at or before the requested grid point")
    snapshot_time, item = max(candidates, key=lambda x: x[0])
    candidates_by_id, invalid_times, present, time_problem = {}, 0, set(), set()
    for park in item.get("carpark_data", []):
        identifier = str(park["carpark_number"])
        present.add(identifier)
        car_lots = [lot for lot in park.get("carpark_info", []) if lot.get("lot_type") == "C"]
        if not car_lots:
            continue
        try:
            updated = parse_time(park["update_datetime"])
        except (KeyError, ValueError, TypeError):
            invalid_times += 1
            time_problem.add(identifier)
            continue
        if updated > requested:
            invalid_times += 1
            time_problem.add(identifier)
            continue
        for lot in car_lots:
            try:
                available = float(lot["lots_available"])
            except (KeyError, ValueError, TypeError):
                available = math.nan
            try:
                capacity = float(lot["total_lots"])
            except (KeyError, ValueError, TypeError):
                capacity = math.nan
            candidates_by_id.setdefault(identifier, []).append((updated, available, capacity))
    result, conflicts, duplicates, over_capacity = [], 0, 0, 0
    for identifier, rows in sorted(candidates_by_id.items()):
        duplicates += int(len(rows) > 1)
        latest = max(r[0] for r in rows)
        selected = [r for r in rows if r[0] == latest]
        amounts = {r[1] for r in selected if math.isfinite(r[1]) and r[1] >= 0}
        capacities = {r[2] for r in selected if math.isfinite(r[2])}
        available = next(iter(amounts)) if len(amounts) == 1 else math.nan
        capacity = next(iter(capacities)) if len(capacities) == 1 else math.nan
        conflict = len(amounts) > 1 or len(capacities) > 1
        valid = math.isfinite(available) and available >= 0
        snapshot_age = (requested - snapshot_time).total_seconds()
        update_age = (requested - latest).total_seconds()
        stale = snapshot_age > 3600 or update_age > 3600
        flags = (
            int(snapshot_age > 3600)
            | (int(update_age > 3600) << 1)
            | (int(conflict) << 2)
            | (int(not valid) << 3)
        )
        anomaly = bool(valid and math.isfinite(capacity) and available > capacity)
        flags |= int(anomaly) << 7
        conflicts += int(conflict)
        over_capacity += int(anomaly)
        result.append(
            {
                "carpark_number": identifier,
                "lot_type": "C",
                "raw_asof": available if valid else None,
                "fresh_value": available if valid and not conflict and not stale else None,
                "total_lots": capacity if math.isfinite(capacity) else None,
                "update_datetime": latest.isoformat(),
                "update_age_seconds": update_age,
                "flags": flags,
                "over_capacity": anomaly,
            }
        )
    for identifier in sorted(time_problem - candidates_by_id.keys()):
        result.append(
            {
                "carpark_number": identifier,
                "lot_type": "C",
                "raw_asof": None,
                "fresh_value": None,
                "total_lots": None,
                "update_datetime": None,
                "update_age_seconds": None,
                "flags": 32,
                "over_capacity": False,
            }
        )
    return {
        "snapshot_timestamp": snapshot_time.isoformat(),
        "requested_timestamp": requested.isoformat(),
        "snapshot_age_seconds": (requested - snapshot_time).total_seconds(),
        "records": result,
        "present_identifiers": sorted(present),
        "invalid_or_future_updates": invalid_times,
        "duplicate_C_identifiers": duplicates,
        "conflicting_latest_C_records": conflicts,
        "over_capacity_records": over_capacity,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probe-hours", type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed HDB collections")
    count = args.probe_hours if args.probe_hours is not None else HOURS
    if not 1 <= count <= HOURS:
        raise ValueError("invalid registered collection length")
    identity = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "start": START.isoformat(),
        "hours": count,
        "lot_type": "C",
        "sampling_minutes": 60,
        "api_query_lag_seconds": 60,
        "freshness_seconds": 3600,
        "mode": "probe" if args.probe_hours is not None else "registered_unscored_collection",
    }
    if (output / "identity.json").exists() and json.loads(
        (output / "identity.json").read_text(encoding="utf-8")
    ) != identity:
        raise ValueError("partial collection definition changed")
    write_json(output / "identity.json", identity)
    lock = threading.Lock()
    next_request = [0.0]

    def permit():
        with lock:
            due = max(time.monotonic(), next_request[0])
            next_request[0] = due + 1.0
        time.sleep(max(0, due - time.monotonic()))

    def fetch(index):
        requested = START + timedelta(hours=index)
        query_time = requested - timedelta(minutes=1)
        label = requested.strftime("%Y%m%dT%H%M%S")
        path, record_path = (
            output / "raw" / f"{label}.json.gz",
            output / "records" / f"{label}.json",
        )
        if record_path.exists() and path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            body = gzip.decompress(path.read_bytes())
            if hashlib.sha256(body).hexdigest() != record["raw_sha256"]:
                raise ValueError("an archived raw response changed")
            return index, {
                k: v for k, v in record.items() if k not in ("records", "present_identifiers")
            } | {"record_path": str(record_path.relative_to(output))}
        url = "https://api.data.gov.sg/v1/transport/carpark-availability?" + urllib.parse.urlencode(
                {"date_time": query_time.replace(tzinfo=None).isoformat()}
        )
        errors = []
        for attempt in range(3):
            permit()
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    body = response.read(5_000_001)
                if len(body) > 5_000_000:
                    raise ValueError("unexpectedly large historical snapshot")
                parsed = parse_snapshot(json.loads(body), requested)
                record = {
                    "status": "completed",
                    "index": index,
                    "request_url": url,
                    "api_query_timestamp": query_time.isoformat(),
                    "retrieved_at": datetime.now(TZ).isoformat(),
                    "raw_sha256": hashlib.sha256(body).hexdigest(),
                    "raw_path": str(path.relative_to(output)),
                    "bytes": len(body),
                    "attempt_errors": errors,
                    **parsed,
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(body, mtime=0))
                write_json(record_path, record)
                return index, {
                    k: v for k, v in record.items() if k not in ("records", "present_identifiers")
                } | {"record_path": str(record_path.relative_to(output))}
            except (OSError, ValueError, KeyError, TypeError) as error:
                delay = min(30.0, 5.0 * (2**attempt))
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    try:
                        delay = min(
                            60.0, max(delay, float(error.headers.get("Retry-After", delay)))
                        )
                    except ValueError:
                        pass
                    with lock:
                        next_request[0] = max(next_request[0], time.monotonic() + delay)
                errors.append({"attempt": attempt + 1, "error": f"{type(error).__name__}: {error}"})
                if attempt < 2:
                    time.sleep(delay)
        return index, {
            "status": "failed",
            "index": index,
            "request_url": url,
            "requested_timestamp": requested.isoformat(),
            "errors": errors,
        }

    started, completed, failed = time.perf_counter(), {}, []
    state_path = output / "state.json"
    write_json(
        state_path,
        {
            "status": "running",
            "worker_pid": os.getpid(),
            "completed": 0,
            "total": count,
        },
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for future in concurrent.futures.as_completed(futures):
            index, record = future.result()
            if record["status"] == "completed":
                completed[index] = record
            else:
                failed.append(record)
            write_json(
                state_path,
                {
                    "status": "running",
                    "worker_pid": os.getpid(),
                    "completed": len(completed),
                    "failed": len(failed),
                    "total": count,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if (len(completed) + len(failed)) % 48 == 0:
                print(
                    json.dumps(
                        {"completed": len(completed), "failed": len(failed), "total": count}
                    ),
                    flush=True,
                )
    if failed:
        write_json(output / "failures.json", failed)
        write_json(
            state_path,
            {
                "status": "failed",
                "completed": len(completed),
                "failed": len(failed),
                "total": count,
            },
        )
        raise RuntimeError(
            "historical API failures are retained as acquisition failures; no incomplete observation panel was built"
        )
    all_identifiers = set()
    for record in completed.values():
        content = json.loads((output / record["record_path"]).read_text(encoding="utf-8"))
        all_identifiers.update(r["carpark_number"] for r in content["records"])
    identifiers = sorted(all_identifiers)
    lookup = {name: i for i, name in enumerate(identifiers)}
    shape = count, len(identifiers)
    raw, fresh, capacity, age = (np.full(shape, np.nan, dtype=np.float32) for _ in range(4))
    flags = np.full(shape, 64, dtype=np.uint8)
    for t, record in completed.items():
        content = json.loads((output / record["record_path"]).read_text(encoding="utf-8"))
        for identifier in content["present_identifiers"]:
            if identifier in lookup:
                flags[t, lookup[identifier]] = 16
        for row in content["records"]:
            j = lookup[row["carpark_number"]]
            raw[t, j] = np.nan if row["raw_asof"] is None else row["raw_asof"]
            fresh[t, j] = np.nan if row["fresh_value"] is None else row["fresh_value"]
            capacity[t, j] = np.nan if row["total_lots"] is None else row["total_lots"]
            age[t, j] = np.nan if row["update_age_seconds"] is None else row["update_age_seconds"]
            flags[t, j] = row["flags"]
    panel = output / "hourly_panel.npz"
    np.savez_compressed(
        panel,
        identifiers=np.asarray(identifiers),
        raw_asof=raw,
        fresh_values=fresh,
        total_lots=capacity,
        update_age_seconds=age,
        flags=flags,
        timestamps=np.asarray([(START + timedelta(hours=i)).isoformat() for i in range(count)]),
    )
    manifest = {
        "status": "completed",
        "identity": identity,
        "snapshots": count,
        "series": len(identifiers),
        "panel_path": str(panel),
        "panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "original_records": [
            {
                "index": i,
                "raw_path": completed[i]["raw_path"],
                "raw_sha256": completed[i]["raw_sha256"],
                "snapshot_timestamp": completed[i]["snapshot_timestamp"],
            }
            for i in range(count)
        ],
        "fresh_cells": int(np.isfinite(fresh).sum()),
        "raw_cells": int(np.isfinite(raw).sum()),
        "forecast_calls": 0,
        "prediction_errors_read": False,
        "independent_source_claim": False,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        state_path,
        {"status": "completed", "completed": count, "total": count, "series": len(identifiers)},
    )
    print(
        json.dumps({"status": "completed", "snapshots": count, "series": len(identifiers)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
