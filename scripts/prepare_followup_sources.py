"""Preserve new public trajectories and their original observation masks."""

import argparse
import gzip
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

AIR_COLUMNS = (
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
)
POWER_COLUMNS = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def parse_uci_frame(frame, dataset):
    """Drop only empty padding; preserve missing measurements on a regular time grid."""
    air = dataset == "uci_air_quality"
    if dataset not in {"uci_air_quality", "uci_household_power"}:
        raise ValueError("unknown UCI source")
    columns = AIR_COLUMNS if air else POWER_COLUMNS
    extras = [name for name in frame if name not in ("Date", "Time", *columns)]
    if any(frame[name].notna().any() for name in extras):
        raise ValueError("unexpected nonempty source columns")
    frame = frame.drop(columns=extras)
    padding = frame.isna().all(axis=1)
    removed_padding = int(padding.sum())
    frame = frame.loc[~padding].copy()
    if list(frame.columns) != ["Date", "Time", *columns]:
        raise ValueError("the original numerical column order changed")
    timestamps = pd.to_datetime(
        frame.Date + " " + frame.Time,
        format="%d/%m/%Y %H.%M.%S" if air else "%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    if (
        timestamps.isna().any()
        or timestamps.duplicated().any()
        or not timestamps.is_monotonic_increasing
    ):
        raise ValueError("timestamps must be unique, present and increasing")
    values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    sentinel_count = int((values == -200).sum()) if air else 0
    if air:
        values[values == -200] = np.nan
    if np.isinf(values).any():
        raise ValueError("infinite measurements are not accepted")
    frequency = "h" if air else "min"
    grid = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=frequency)
    indexed = pd.DataFrame(values, index=pd.DatetimeIndex(timestamps), columns=columns)
    if not indexed.index.isin(grid).all():
        raise ValueError("off-grid source timestamps require review")
    values = indexed.reindex(grid).to_numpy(float)
    return (
        values,
        columns,
        grid,
        {
            "original_measurement_rows": len(frame),
            "empty_padding_rows_removed": removed_padding,
            "inserted_unobserved_timestamps": len(grid) - len(frame),
            "sentinel_cells_to_missing": sentinel_count,
            "frequency": frequency,
        },
    )


def save_series(
    output, dataset, values, columns, *, frequency, period, source, timestamps=None, parsing=None
):
    path = output / f"{dataset}.npy"
    np.save(path, values, allow_pickle=False)
    observed = np.isfinite(values)
    eligibility = confirmation_windows(observed)
    _write_json(output / f"{dataset}_eligibility.json", eligibility)
    row = {
        "dataset_id": dataset,
        "family_id": dataset,
        "item_id": "item_0",
        "path": path.name,
        "sha256": file_sha256(path),
        "shape": list(values.shape),
        "columns": list(columns),
        "frequency": frequency,
        "period": period,
        "start": None if timestamps is None else timestamps[0].isoformat(),
        "end": None if timestamps is None else timestamps[-1].isoformat(),
        "calendar_timestamps_provided": timestamps is not None,
        "missing_cells_by_column": (~observed).sum(axis=0).tolist(),
        "source": source,
        "parsing": parsing or {},
        "eligibility_path": f"{dataset}_eligibility.json",
        "eligibility_sha256": file_sha256(output / f"{dataset}_eligibility.json"),
        "eligible_windows": eligibility["eligible_window_count"],
        "eligible_missing_contexts": eligibility["eligible_missing_context_count"],
    }
    print(
        json.dumps(
            {
                key: row[key]
                for key in ("dataset_id", "shape", "eligible_windows", "eligible_missing_contexts")
            }
        ),
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("uci-root", "solar-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source preparations")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, key, member in (
        ("uci_air_quality", "air_quality", "AirQualityUCI.csv"),
        ("uci_household_power", "household_power", "household_power_consumption.txt"),
    ):
        source = json.loads((args.uci_root / f"{key}.json").read_text(encoding="utf-8"))
        archive = args.uci_root / source["file"]
        if file_sha256(archive) != source["sha256"]:
            raise ValueError("a downloaded UCI archive changed")
        with zipfile.ZipFile(archive) as zipped, zipped.open(member) as handle:
            frame = pd.read_csv(
                handle,
                sep=";",
                decimal="," if key == "air_quality" else ".",
                na_values=["?"],
                low_memory=False,
            )
        values, columns, timestamps, parsing = parse_uci_frame(frame, dataset)
        records.append(
            save_series(
                output,
                dataset,
                values,
                columns,
                frequency=parsing["frequency"],
                period=24 if key == "air_quality" else 1440,
                source=source,
                timestamps=timestamps,
                parsing=parsing,
            )
        )
    source = json.loads((args.solar_root / "download_manifest.json").read_text(encoding="utf-8"))
    archive = args.solar_root / source["path"]
    if file_sha256(archive) != source["sha256"]:
        raise ValueError("the downloaded solar archive changed")
    with gzip.open(archive, "rt") as handle:
        values = np.loadtxt(handle, delimiter=",")
    if values.ndim != 2 or values.shape[1] != 137 or not np.isfinite(values).all():
        raise ValueError("the complete 137-variable solar source requires review")
    records.append(
        save_series(
            output,
            "solar_alabama",
            values,
            [f"plant_{i}" for i in range(137)],
            frequency="10min",
            period=144,
            source=source,
            parsing={"calendar_policy": "relative sequence positions; calendar start not invented"},
        )
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "datasets": records,
            "information_boundary": "raw measurements parsed; eligibility uses original masks only; no forecasting outcomes computed",
            "pretraining_caveat": "new to this method-development experiment does not imply absent from forecasting-model pretraining; Solar is among the MoTM reference sources",
        },
    )


if __name__ == "__main__":
    main()
