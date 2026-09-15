"""Nested sampling of independent histories without consulting outcome values."""

from __future__ import annotations

import hashlib
import math

import pandas as pd


def nested_origin_ids(frame: pd.DataFrame, fraction: float, seed: int) -> tuple[str, ...]:
    if not 0 < fraction <= 1 or frame.empty:
        raise ValueError("a nonempty frame and 0 < fraction <= 1 are required")
    columns = ["origin_id", "family_id", "dataset_id", "item_id"]
    metadata = frame[columns].drop_duplicates()
    if metadata.isna().any().any() or metadata.origin_id.duplicated().any():
        raise ValueError("each historical origin must identify exactly one series and family")
    selected = []
    for _, group in metadata.groupby(["family_id", "dataset_id", "item_id"], sort=True):
        origins = sorted(
            group.origin_id.astype(str),
            key=lambda origin: (hashlib.sha256(f"{seed}|{origin}".encode()).hexdigest(), origin),
        )
        selected.extend(origins[: max(1, math.ceil(fraction * len(origins)))])
    return tuple(sorted(selected))
