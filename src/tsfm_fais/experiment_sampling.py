"""Deterministic caps that keep large experiment grids balanced and reproducible."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from tsfm_fais.data import stable_seed

T = TypeVar("T")


def deterministic_subset(
    values: Sequence[T],
    limit: int | None,
    *seed_parts: object,
) -> tuple[T, ...]:
    """Select at most ``limit`` entries and retain their source ordering."""

    entries = tuple(values)
    if limit is None or len(entries) <= limit:
        return entries
    if limit < 1:
        raise ValueError("subset limit must be positive")
    rng = np.random.default_rng(stable_seed(*seed_parts))
    indices = sorted(map(int, rng.choice(len(entries), size=limit, replace=False)))
    return tuple(entries[index] for index in indices)


def evenly_spaced_subset(values: Sequence[T], limit: int | None) -> tuple[T, ...]:
    """Cap an ordered sequence while retaining coverage of its full range."""

    entries = tuple(values)
    if limit is None or len(entries) <= limit:
        return entries
    if limit < 1:
        raise ValueError("subset limit must be positive")
    if limit == 1:
        return (entries[-1],)
    indices = np.linspace(0, len(entries) - 1, num=limit, dtype=int)
    return tuple(entries[int(index)] for index in indices)


def candidate_subset(
    candidate_ids: Sequence[str],
    limit: int | None,
    *seed_parts: object,
    forced: Sequence[str] = ("locf", "linear_interp"),
) -> tuple[str, ...]:
    """Keep safety baselines and rotate remaining candidates deterministically."""

    available = tuple(dict.fromkeys(candidate_ids))
    forced_ids = tuple(candidate for candidate in forced if candidate in available)
    if limit is None or len(available) <= limit:
        return available
    if limit < len(forced_ids):
        raise ValueError("candidate limit is smaller than the forced candidate set")
    remaining = tuple(candidate for candidate in available if candidate not in forced_ids)
    selected = deterministic_subset(
        remaining,
        limit - len(forced_ids),
        *seed_parts,
        "candidate_subset",
    )
    selected_set = set((*forced_ids, *selected))
    return tuple(candidate for candidate in available if candidate in selected_set)


def connected_subset(
    values: Sequence[T],
    edges: Sequence[object],
    limit: int | None,
    *seed_parts: object,
) -> tuple[T, ...]:
    """Prefer one connected pair, then fill the remaining deterministic quota.

    Values and edge endpoints are matched through their ``block_id``/``left``/
    ``right`` attributes.  If no pair fits the quota, this reduces to the
    ordinary deterministic subset.
    """

    entries = tuple(values)
    if limit is None or len(entries) <= limit:
        return entries
    if limit < 2 or not edges:
        return deterministic_subset(entries, limit, *seed_parts)
    selected_edge = deterministic_subset(edges, 1, *seed_parts, "connected_edge")[0]
    selected_ids = {
        str(getattr(selected_edge, "left")),
        str(getattr(selected_edge, "right")),
    }
    connected = tuple(
        entry
        for entry in entries
        if str(getattr(entry, "block_id", "")) in selected_ids
    )
    if len(connected) != 2:
        return deterministic_subset(entries, limit, *seed_parts)
    remaining = tuple(entry for entry in entries if entry not in connected)
    fill_limit = limit - len(connected)
    fill = (
        ()
        if fill_limit == 0
        else deterministic_subset(
            remaining,
            fill_limit,
            *seed_parts,
            "connected_fill",
        )
    )
    selected = set((*connected, *fill))
    return tuple(entry for entry in entries if entry in selected)


__all__ = [
    "candidate_subset",
    "connected_subset",
    "deterministic_subset",
    "evenly_spaced_subset",
]
