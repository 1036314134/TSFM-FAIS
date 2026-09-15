"""Parse the four new public sources and inspect observation-only eligibility."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_r3_confirmation_sources import confirmation_windows  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402

BEIJING = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
BIKE = ["casual", "registered", "temp", "atemp", "hum", "windspeed"]
OCCUPANCY = ["Temperature", "Humidity", "Light", "CO2"]


def regular_grid(frame, timestamps, columns, frequency, *, minute_rounding=False):
    timestamps = pd.DatetimeIndex(timestamps)
    if timestamps.isna().any():
        raise ValueError("every measurement row must have a valid timestamp")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("infinite measurements require source review")
    displacement = 0.0
    if minute_rounding:
        rounded = timestamps.round("min")
        displacement = float(np.abs((rounded - timestamps).total_seconds()).max())
        if displacement > 1:
            raise ValueError(
                "occupancy clock displacement exceeds the registered one-second tolerance"
            )
        timestamps = rounded
    elif not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("original timestamps must be unique and ordered")
    indexed = pd.DataFrame(values, index=timestamps, columns=columns)
    duplicate_rows = int(indexed.index.duplicated().sum())
    if minute_rounding:
        indexed = indexed.groupby(level=0, sort=True).mean()
    grid = pd.date_range(indexed.index.min(), indexed.index.max(), freq=frequency)
    if not indexed.index.isin(grid).all():
        raise ValueError("measurements are off the registered nominal time grid")
    present = np.asarray(grid.isin(indexed.index))
    values = indexed.reindex(grid).to_numpy(float)
    return (
        values,
        grid,
        present,
        {
            "original_rows": len(frame),
            "duplicate_rounded_rows_aggregated": duplicate_rows,
            "maximum_rounding_seconds": displacement,
            "inserted_unobserved_bins": int((~present).sum()),
            "original_missing_cells": int((~np.isfinite(indexed.to_numpy())).sum()),
            "timezone_policy": "use the provider's nominal local clock; do not infer a timezone",
        },
    )


def eligibility(values, present):
    observed = np.isfinite(values)
    result = confirmation_windows(observed, context_length=96, horizon=192, minimum_future=96)
    for row in result["windows"]:
        origin = row["origin"]
        counts96 = observed[origin : origin + 96, :2].sum(axis=0)
        row["future_observed_by_horizon"] = {
            "96": counts96.tolist(),
            "192": row["future_observed_by_target"],
        }
        row["eligible"] = bool(row["eligible"] and (counts96 >= 48).all())
        if row["exclusion_reason"] is None and not row["eligible"]:
            row["exclusion_reason"] = "insufficient_first96_future_observations"
        row["synthetic_eligible"] = bool(
            row["eligible"]
            and observed[origin - 96 : origin].all()
            and observed[origin : origin + 192, :2].all()
        )
        row["context_has_time_grid_gap"] = bool((~present[origin - 96 : origin]).any())
        row["context_has_source_na"] = bool(
            (~observed[origin - 96 : origin] & present[origin - 96 : origin, None]).any()
        )
    eligible = [row for row in result["windows"] if row["eligible"]]
    result.update(
        eligible_window_count=len(eligible),
        eligible_missing_context_count=sum(row["context_has_missing"] for row in eligible),
        eligible_missing_target_context_count=sum(
            row["context_target_has_missing"] for row in eligible
        ),
        eligible_complete_future_count=sum(row["complete_target_future"] for row in eligible),
        synthetic_eligible_window_count=sum(row["synthetic_eligible"] for row in eligible),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "method-freeze", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source inventories")
    method = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if method["status"] != "completed" or method["primary_method"] != "ensemble_gate":
        raise ValueError("freeze the selected method before new source preparation")
    output.mkdir(parents=True, exist_ok=True)
    sources, issues = [], []

    def save(
        dataset,
        item,
        frame,
        timestamps,
        columns,
        frequency,
        period,
        source,
        *,
        rounding=False,
        notes="",
    ):
        try:
            values, times, present, parsing = regular_grid(
                frame, timestamps, columns, frequency, minute_rounding=rounding
            )
            availability = eligibility(values, present)
        except ValueError as error:
            issues.append(
                {
                    "dataset_id": dataset,
                    "item_id": item,
                    "status": "requires_source_review",
                    "reason": str(error),
                }
            )
            return
        key = hashlib.sha256(f"{dataset}|{item}".encode()).hexdigest()[:24]
        path = output / "series" / f"{key}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, values, allow_pickle=False)
        observed_path = output / "series" / f"{key}_timestamp_present.npy"
        np.save(observed_path, present, allow_pickle=False)
        availability_path = output / "series" / f"{key}_eligibility.json"
        _write_json(availability_path, availability)
        record = {
            "dataset_id": dataset,
            "family_id": dataset,
            "item_id": item,
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
            "timestamp_present_path": str(observed_path.relative_to(output)),
            "timestamp_present_sha256": file_sha256(observed_path),
            "eligibility_path": str(availability_path.relative_to(output)),
            "eligibility_sha256": file_sha256(availability_path),
            "columns": columns,
            "target_names": columns[:2],
            "shape": list(values.shape),
            "frequency": frequency,
            "period": period,
            "start": times[0].isoformat(),
            "end": times[-1].isoformat(),
            "source": source,
            "parsing": parsing,
            "notes": notes,
            "eligible_windows": availability["eligible_window_count"],
            "eligible_missing_contexts": availability["eligible_missing_context_count"],
            "eligible_synthetic_histories": availability["synthetic_eligible_window_count"],
        }
        sources.append(record)
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "dataset_id",
                        "item_id",
                        "shape",
                        "eligible_windows",
                        "eligible_missing_contexts",
                        "eligible_synthetic_histories",
                    )
                }
            ),
            flush=True,
        )

    for dataset in ("beijing_multisite", "appliances", "bike_sharing", "occupancy"):
        record = json.loads((args.reference_root / f"{dataset}.json").read_text(encoding="utf-8"))
        path = args.reference_root / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a downloaded source archive changed")
        with zipfile.ZipFile(path) as archive:
            if dataset == "beijing_multisite":
                nested = archive.read("PRSA2017_Data_20130301-20170228.zip")
                with zipfile.ZipFile(io.BytesIO(nested)) as data:
                    files = sorted(
                        name
                        for name in data.namelist()
                        if Path(name).name.startswith("PRSA_Data_") and name.endswith(".csv")
                    )
                    if len(files) != 12:
                        raise ValueError("the official Beijing station inventory changed")
                    for member in files:
                        raw = data.read(member)
                        frame = pd.read_csv(io.BytesIO(raw))
                        if frame.station.nunique() != 1:
                            raise ValueError("a Beijing item contains multiple station identifiers")
                        source = {
                            **record,
                            "member": member,
                            "member_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                        save(
                            dataset,
                            str(frame.station.iloc[0]),
                            frame,
                            pd.to_datetime(frame[["year", "month", "day", "hour"]]),
                            BEIJING,
                            "h",
                            24,
                            source,
                        )
            elif dataset == "appliances":
                raw = archive.read("energydata_complete.csv")
                frame = pd.read_csv(io.BytesIO(raw))
                columns = [name for name in frame.columns if name not in {"date", "rv1", "rv2"}]
                if columns[:2] != ["Appliances", "lights"] or len(columns) != 26:
                    raise ValueError("the registered appliance measurements changed")
                save(
                    dataset,
                    "item_0",
                    frame,
                    pd.to_datetime(frame.date),
                    columns,
                    "10min",
                    144,
                    {**record, "member_sha256": hashlib.sha256(raw).hexdigest()},
                    notes="Two random variables excluded. Provider release includes hourly airport weather interpolated to ten-minute resolution; energy targets are measured at ten minutes.",
                )
            elif dataset == "bike_sharing":
                raw = archive.read("hour.csv")
                frame = pd.read_csv(io.BytesIO(raw))
                times = pd.to_datetime(frame.dteday) + pd.to_timedelta(frame.hr, unit="h")
                save(
                    dataset,
                    "item_0",
                    frame,
                    times,
                    BIKE,
                    "h",
                    24,
                    {**record, "member_sha256": hashlib.sha256(raw).hexdigest()},
                    notes="Retain supplied weather scaling. Counts in unrecorded hours are unknown, not assigned zero. The cause of missing nominal hours is not identified; derived count total is excluded.",
                )
            else:
                frames, members = [], []
                for member in ("datatraining.txt", "datatest.txt", "datatest2.txt"):
                    raw = archive.read(member)
                    frames.append(pd.read_csv(io.BytesIO(raw)))
                    members.append({"path": member, "sha256": hashlib.sha256(raw).hexdigest()})
                frame = pd.concat(frames, ignore_index=True)
                save(
                    dataset,
                    "item_0",
                    frame,
                    pd.to_datetime(frame.date),
                    OCCUPANCY,
                    "min",
                    1440,
                    {**record, "members": members},
                    rounding=True,
                    notes="Four measured environmental channels; classification label and target-derived humidity ratio excluded. Rounded-minute means define the numerical forecasting signal.",
                )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "method_freeze_sha256": file_sha256(args.method_freeze),
            "protocol_sha256": file_sha256(args.protocol),
            "sources": sources,
            "issues": issues,
            "forecaster_calls": 0,
            "information_boundary": "raw source parsing and observation-mask/time-grid inspection only; no forecasting outcomes evaluated",
            "limits": "inventory is not a frozen evaluation cohort; exclusions and exact source overlap still require review",
        },
    )


if __name__ == "__main__":
    main()
