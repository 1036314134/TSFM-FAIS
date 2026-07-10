"""CSV and Arrow IPC loaders that preserve multivariate item boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from tsfm_fais.contracts import TimeSeriesItem

from .catalog import DatasetSpec

TIME_NAMES = {"timestamp", "datetime", "date", "time", "index"}


def _value_columns(frame: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    if spec.value_columns is not None:
        missing = [name for name in spec.value_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"missing configured value columns: {missing}")
        return list(spec.value_columns)
    excluded = {name.lower() for name in TIME_NAMES}
    if spec.timestamp_column:
        excluded.add(spec.timestamp_column.lower())
    if spec.item_id_column:
        excluded.add(spec.item_id_column.lower())
    return [
        str(name)
        for name in frame.columns
        if str(name).lower() not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]


def _validate_target_columns(spec: DatasetSpec, value_columns: Iterable[str]) -> None:
    if spec.target_columns == "all":
        return
    missing = set(spec.target_columns) - set(value_columns)
    if missing:
        raise ValueError(f"configured target columns are missing: {sorted(missing)}")


def load_csv(spec: DatasetSpec) -> list[TimeSeriesItem]:
    frame = pd.read_csv(spec.path)
    time_column = spec.timestamp_column
    if time_column is None:
        time_column = next(
            (str(name) for name in frame.columns if str(name).lower() in TIME_NAMES), None
        )
    if time_column is None or time_column not in frame.columns:
        raise ValueError(f"no timestamp column found in {spec.path}")
    columns = _value_columns(frame, spec)
    if len(columns) < 2:
        raise ValueError(f"dataset {spec.dataset_id} has fewer than two numeric value columns")
    _validate_target_columns(spec, columns)
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if spec.item_id_column:
        if spec.item_id_column not in frame.columns:
            raise ValueError(f"item id column {spec.item_id_column!r} is missing")
        if frame[spec.item_id_column].isna().any():
            raise ValueError(f"item id column {spec.item_id_column!r} contains missing values")
        groups = frame.groupby(spec.item_id_column, sort=False, dropna=False)
    else:
        groups = ((spec.dataset_id, frame),)
    items: list[TimeSeriesItem] = []
    for item_id, group in groups:
        timestamps = pd.DatetimeIndex(pd.to_datetime(group[time_column], errors="coerce"))
        values = group[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        items.append(
            TimeSeriesItem(
                item_id=str(item_id),
                values=values,
                variate_names=tuple(columns),
                start=timestamps[0] if len(timestamps) else pd.NaT,
                freq=spec.frequency,
                timestamps=timestamps,
                metadata={
                    "dataset_id": spec.dataset_id,
                    "family_id": spec.family_id,
                    "domain": spec.domain,
                    "period": spec.period,
                    "source_path": str(spec.path),
                    "implicit_regular_time": False,
                },
            )
        )
    return items


def _read_arrow_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            return pa.ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return pa.ipc.open_file(source).read_all()


def _coerce_target(target: Any, num_variates: int | None = None) -> np.ndarray:
    array = np.asarray(target, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Arrow target must represent [D,T], got {array.shape}")
    if num_variates is None or array.shape[0] == num_variates:
        return array.T
    if array.shape[1] == num_variates:
        return array
    raise ValueError(
        f"Arrow target shape {array.shape} does not contain D={num_variates} variates"
    )


def load_arrow(spec: DatasetSpec) -> list[TimeSeriesItem]:
    files = sorted(spec.path.glob("*.arrow")) if spec.path.is_dir() else [spec.path]
    if not files:
        raise FileNotFoundError(f"no Arrow IPC files found under {spec.path}")
    items: list[TimeSeriesItem] = []
    for path in files:
        for row in _read_arrow_table(path).to_pylist():
            if "target" not in row:
                raise ValueError(f"Arrow row in {path} has no target field")
            raw_names = row.get("variate_names")
            declared_dimensions = (
                len(raw_names) if raw_names is not None else spec.expected_num_variates
            )
            values = _coerce_target(row["target"], declared_dimensions)
            names = (
                tuple(map(str, raw_names))
                if raw_names is not None
                else tuple(f"target_{idx}" for idx in range(values.shape[1]))
            )
            _validate_target_columns(spec, names)
            start = pd.Timestamp(row.get("start"))
            freq = str(row.get("freq") or spec.frequency)
            raw_timestamps = row.get("timestamps")
            timestamps = (
                pd.DatetimeIndex(pd.to_datetime(raw_timestamps, errors="coerce"))
                if raw_timestamps is not None
                else None
            )
            items.append(
                TimeSeriesItem(
                    item_id=str(row.get("item_id", len(items))),
                    values=values,
                    variate_names=names,
                    start=start,
                    freq=freq,
                    timestamps=timestamps,
                    metadata={
                        "dataset_id": spec.dataset_id,
                        "family_id": spec.family_id,
                        "domain": spec.domain,
                        "period": spec.period,
                        "source_path": str(path),
                        "implicit_regular_time": timestamps is None,
                    },
                )
            )
    return items


def load_dataset(spec: DatasetSpec) -> list[TimeSeriesItem]:
    if spec.format == "csv":
        return load_csv(spec)
    if spec.format == "arrow":
        return load_arrow(spec)
    raise ValueError(f"unsupported dataset format: {spec.format}")
