"""Dataset admission checks for complete multivariate source data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from tsfm_fais.contracts import TimeSeriesItem

from .catalog import DatasetSpec


@dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    item_id: str | None = None


@dataclass
class DatasetAudit:
    dataset_id: str
    accepted: bool
    item_count: int
    total_values: int
    min_length: int
    max_length: int
    dimensions: tuple[int, ...]
    time_axis_verification: str
    content_sha256: str | None
    issues: list[AuditIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def _hash_paths(path: Path) -> str | None:
    paths = sorted(path.glob("*.arrow")) if path.is_dir() else [path]
    if not paths or any(not candidate.exists() for candidate in paths):
        return None
    digest = hashlib.sha256()
    for candidate in paths:
        digest.update(candidate.name.encode("utf-8"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_offset(value: str):
    text = value.strip()
    upper = text.upper()
    if upper.endswith("T") and upper[:-1].isdigit():
        text = f"{upper[:-1]}min"
    else:
        text = {
            "T": "min",
            "H": "h",
            "M": "ME",
            "Q": "QE",
            "Y": "YE",
            "A": "YE",
        }.get(upper, text)
    try:
        return to_offset(text)
    except ValueError:
        return to_offset(value)


def _frequency_key(value: str) -> tuple[str, int]:
    offset = _parse_offset(value)
    rule = str(offset.rule_code).upper()
    if rule == "W" or rule.startswith("W-"):
        return "week", int(offset.n)
    if rule in {"ME", "MS", "BME", "BMS", "SME", "SMS"}:
        return "month", int(offset.n)
    if rule.startswith(("QE-", "QS-", "BQE-", "BQS-")) or rule in {
        "QE",
        "QS",
        "BQE",
        "BQS",
    }:
        return "quarter", int(offset.n)
    if rule.startswith(("YE-", "YS-", "BYE-", "BYS-")) or rule in {
        "YE",
        "YS",
        "BYE",
        "BYS",
    }:
        return "year", int(offset.n)
    try:
        return "fixed", int(offset.nanos)
    except ValueError:
        return rule, int(offset.n)


def _frequencies_equivalent(left: str, right: str) -> bool:
    try:
        return _frequency_key(left) == _frequency_key(right)
    except (TypeError, ValueError):
        return False


def _frequency_matches(timestamps: pd.DatetimeIndex, declared: str) -> bool:
    if len(timestamps) < 3:
        return True
    inferred = pd.infer_freq(timestamps)
    if inferred is None:
        return False
    return _frequencies_equivalent(inferred, declared)


def audit_dataset(spec: DatasetSpec, items: Iterable[TimeSeriesItem]) -> DatasetAudit:
    materialized = list(items)
    issues: list[AuditIssue] = []
    dimensions: set[int] = set()
    lengths: list[int] = []
    total_values = 0
    time_axis_modes: set[str] = set()
    seen_item_ids: set[str] = set()
    reference_names: tuple[str, ...] | None = None
    for item in materialized:
        values = item.values
        lengths.append(values.shape[0])
        dimensions.add(values.shape[1])
        total_values += values.size
        if item.item_id in seen_item_ids:
            issues.append(
                AuditIssue("duplicate_item_id", "item identifiers are not unique", item.item_id)
            )
        seen_item_ids.add(item.item_id)
        if values.shape[0] == 0:
            issues.append(AuditIssue("empty_item", "item contains no time steps", item.item_id))
        if values.shape[1] < 2:
            issues.append(AuditIssue("not_multivariate", "D must be at least 2", item.item_id))
        if len(set(item.variate_names)) != len(item.variate_names):
            issues.append(
                AuditIssue("duplicate_variate", "variate names are not unique", item.item_id)
            )
        if reference_names is None:
            reference_names = item.variate_names
        elif item.variate_names != reference_names:
            issues.append(
                AuditIssue(
                    "inconsistent_variates",
                    "variate names or ordering differ across items",
                    item.item_id,
                )
            )
        if spec.target_columns != "all":
            unknown_targets = set(spec.target_columns) - set(item.variate_names)
            if unknown_targets:
                issues.append(
                    AuditIssue(
                        "target_column",
                        f"target columns are missing: {sorted(unknown_targets)}",
                        item.item_id,
                    )
                )
        if np.isnan(values).any():
            issues.append(AuditIssue("nan", "source values contain NaN", item.item_id))
        if np.isinf(values).any():
            issues.append(AuditIssue("infinite", "source values contain infinity", item.item_id))
        for sentinel in spec.sentinel_values:
            count = int(np.sum(values == sentinel))
            if count:
                issues.append(
                    AuditIssue(
                        "sentinel",
                        f"source values contain sentinel {sentinel:g} ({count} values)",
                        item.item_id,
                    )
                )
        finite = np.isfinite(values)
        if values.shape[0] > 0 and finite.all() and np.any(np.ptp(values, axis=0) == 0):
            issues.append(AuditIssue("constant", "one or more variates are constant", item.item_id))
        if pd.isna(item.start):
            issues.append(AuditIssue("invalid_start", "start timestamp is missing", item.item_id))
        if not _frequencies_equivalent(item.freq, spec.frequency):
            issues.append(
                AuditIssue(
                    "item_frequency",
                    f"item frequency {item.freq!r} differs from manifest {spec.frequency!r}",
                    item.item_id,
                )
            )
        if item.timestamps is None:
            time_axis_modes.add("implicit")
            if not spec.allow_implicit_regular_time:
                issues.append(
                    AuditIssue(
                        "implicit_time_not_allowed",
                        "source has start+frequency only; manifest approval is required",
                        item.item_id,
                    )
                )
        else:
            time_axis_modes.add("explicit")
            if item.timestamps.hasnans:
                issues.append(AuditIssue("invalid_time", "timestamp parsing failed", item.item_id))
            if item.timestamps.has_duplicates:
                issues.append(
                    AuditIssue("duplicate_time", "timestamps are duplicated", item.item_id)
                )
            if not item.timestamps.is_monotonic_increasing:
                issues.append(
                    AuditIssue("unordered_time", "timestamps are not increasing", item.item_id)
                )
            valid_axis = (
                len(item.timestamps) > 0
                and not item.timestamps.hasnans
                and not item.timestamps.has_duplicates
                and item.timestamps.is_monotonic_increasing
            )
            if valid_axis and item.timestamps[0] != item.start:
                issues.append(
                    AuditIssue(
                        "start_mismatch",
                        "start does not equal the first explicit timestamp",
                        item.item_id,
                    )
                )
            if valid_axis and not _frequency_matches(item.timestamps, spec.frequency):
                issues.append(
                    AuditIssue("frequency", "declared frequency does not match", item.item_id)
                )
    if not materialized:
        issues.append(AuditIssue("empty", "dataset contains no items"))
    if spec.expected_num_variates is not None and dimensions and dimensions != {
        spec.expected_num_variates
    }:
        issues.append(
            AuditIssue(
                "dimension",
                f"expected D={spec.expected_num_variates}, observed {sorted(dimensions)}",
            )
        )
    return DatasetAudit(
        dataset_id=spec.dataset_id,
        accepted=not issues,
        item_count=len(materialized),
        total_values=total_values,
        min_length=min(lengths, default=0),
        max_length=max(lengths, default=0),
        dimensions=tuple(sorted(dimensions)),
        time_axis_verification=(
            next(iter(time_axis_modes))
            if len(time_axis_modes) == 1
            else ("none" if not time_axis_modes else "mixed")
        ),
        content_sha256=_hash_paths(spec.path),
        issues=issues,
    )
