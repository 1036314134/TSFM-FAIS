"""Download and prepare UCI Beijing Multi-Site Air Quality without filling NA values."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/501/beijing"
SOURCE_DOI = "10.24432/C5RK5G"
LICENSE = "CC BY 4.0"
VALUE_COLUMNS = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"preserved partial metadata already exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _download(target: Path) -> None:
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"preserved partial download already exists: {temporary}")
    curl = shutil.which("curl")
    if curl is not None:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "0",
                "--output",
                str(temporary),
                SOURCE_URL,
            ],
            check=True,
        )
    else:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "TSFM-FAIS-R2/1.0 (academic reproducibility)"},
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    os.replace(temporary, target)


def _read_archive(source: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    station_records: list[dict[str, object]] = []
    with zipfile.ZipFile(source) as outer_archive:
        nested_archives = [
            name
            for name in outer_archive.namelist()
            if Path(name).name == "PRSA2017_Data_20130301-20170228.zip"
        ]
        if len(nested_archives) != 1:
            raise ValueError("UCI bundle must contain exactly one PRSA2017 station archive")
        nested_bytes = outer_archive.read(nested_archives[0])
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.startswith("PRSA_Data_") and name.lower().endswith(".csv")
        )
        if len(members) != 12:
            raise ValueError(f"expected 12 station CSV files, found {len(members)}")
        for member in members:
            with archive.open(member) as handle:
                frame = pd.read_csv(handle, na_values=["NA"])
            required = {"year", "month", "day", "hour", "station", *VALUE_COLUMNS}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{member} lacks required columns: {sorted(missing)}")
            timestamp = pd.to_datetime(
                frame[["year", "month", "day", "hour"]],
                errors="raise",
            )
            station_values = tuple(map(str, frame["station"].dropna().unique()))
            if len(station_values) != 1:
                raise ValueError(f"{member} does not contain exactly one station")
            prepared = frame.loc[:, ["station", *VALUE_COLUMNS]].copy()
            prepared.insert(0, "timestamp", timestamp)
            if prepared["timestamp"].duplicated().any():
                raise ValueError(f"{member} contains duplicate timestamps")
            expected = pd.date_range(
                prepared["timestamp"].iloc[0],
                periods=len(prepared),
                freq="h",
            )
            if not prepared["timestamp"].equals(pd.Series(expected)):
                raise ValueError(f"{member} is not a regular hourly series")
            station_records.append(
                {
                    "station": station_values[0],
                    "rows": len(prepared),
                    "first_timestamp": prepared["timestamp"].iloc[0].isoformat(),
                    "last_timestamp": prepared["timestamp"].iloc[-1].isoformat(),
                    "native_missing_values": int(prepared[list(VALUE_COLUMNS)].isna().sum().sum()),
                }
            )
            frames.append(prepared)
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["station", "timestamp"], inplace=True, kind="stable")
    combined.reset_index(drop=True, inplace=True)
    return combined, station_records


def prepare(output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "beijing_multi_site_air_quality_data.zip"
    prepared = output / "beijing_multisite_native.csv"
    metadata = output / "dataset_provenance.json"

    _download(source)
    frame, stations = _read_archive(source)
    temporary_csv = prepared.with_suffix(prepared.suffix + ".part")
    if temporary_csv.exists():
        raise FileExistsError(f"preserved partial CSV already exists: {temporary_csv}")
    frame.to_csv(temporary_csv, index=False, na_rep="")
    os.replace(temporary_csv, prepared)

    missing_by_column = {column: int(frame[column].isna().sum()) for column in VALUE_COLUMNS}
    payload: dict[str, object] = {
        "schema_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "license": LICENSE,
        "preparation_protocol": "uci_beijing_native_numeric_11var_v1",
        "categorical_wind_direction_policy": "excluded_without_encoding",
        "missing_value_policy": "preserve_source_NA_without_filling",
        "source_archive": source.name,
        "source_archive_sha256": _sha256(source),
        "prepared_csv": prepared.name,
        "prepared_csv_sha256": _sha256(prepared),
        "row_count": len(frame),
        "station_count": int(frame["station"].nunique()),
        "value_columns": list(VALUE_COLUMNS),
        "target_columns": ["PM2.5", "PM10"],
        "native_missing_values": int(sum(missing_by_column.values())),
        "native_missing_values_by_column": missing_by_column,
        "stations": stations,
    }
    _write_json_atomic(metadata, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New versioned directory under artifacts/iclr27-r2/_data.",
    )
    args = parser.parse_args()
    payload = prepare(args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
