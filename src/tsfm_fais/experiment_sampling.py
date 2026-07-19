"""Deterministic caps that keep large experiment grids balanced and reproducible."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

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
    edges: Sequence[Any],
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
        str(selected_edge.left),
        str(selected_edge.right),
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


def forecast_aware_block_subset(
    values: Sequence[T],
    edges: Sequence[Any],
    limit: int | None,
    target_indices: Sequence[int],
    forecast_mode: str,
    *seed_parts: object,
) -> tuple[T, ...]:
    """Select informative teacher blocks for the downstream forecast mode.

    Independent-univariate forecasters never receive non-target channels, so
    counterfactual changes on those channels are exactly uninformative.  Joint
    forecasters can use every channel, but direct target-channel blocks are
    retained before graph-neighbour coverage is added.
    """

    entries = tuple(values)
    if not entries:
        return ()
    if limit is not None and limit < 1:
        raise ValueError("subset limit must be positive")
    if forecast_mode not in {"joint_multivariate", "independent_univariate"}:
        raise ValueError("unsupported forecast mode")
    targets = tuple(dict.fromkeys(int(index) for index in target_indices))
    target_set = set(targets)
    direct = tuple(
        entry
        for entry in entries
        if int(getattr(entry, "channel", -1)) in target_set
    )

    if forecast_mode == "independent_univariate":
        direct_ids = {
            str(getattr(entry, "block_id", "")) for entry in direct
        }
        visible_edges = tuple(
            edge
            for edge in edges
            if str(getattr(edge, "left", "")) in direct_ids
            and str(getattr(edge, "right", "")) in direct_ids
        )
        return connected_subset(
            direct,
            visible_edges,
            limit,
            *seed_parts,
            "forecast_visible_blocks",
        )

    if limit is None or len(entries) <= limit:
        return entries

    selected: list[T] = []
    for channel in targets:
        channel_entries = tuple(
            entry
            for entry in direct
            if int(getattr(entry, "channel", -1)) == channel
        )
        if not channel_entries or len(selected) >= limit:
            continue
        selected.append(
            deterministic_subset(
                channel_entries,
                1,
                *seed_parts,
                "forecast_target_channel",
                channel,
            )[0]
        )

    selected_ids = {
        str(getattr(entry, "block_id", "")) for entry in selected
    }
    adjacent_ids: set[str] = set()
    for edge in edges:
        left = str(getattr(edge, "left", ""))
        right = str(getattr(edge, "right", ""))
        if left in selected_ids:
            adjacent_ids.add(right)
        if right in selected_ids:
            adjacent_ids.add(left)
    remaining = tuple(entry for entry in entries if entry not in selected)
    adjacent = tuple(
        entry
        for entry in remaining
        if str(getattr(entry, "block_id", "")) in adjacent_ids
    )
    adjacent_capacity = limit - len(selected)
    adjacent_fill = (
        ()
        if adjacent_capacity == 0
        else deterministic_subset(
            adjacent,
            min(len(adjacent), adjacent_capacity),
            *seed_parts,
            "forecast_adjacent_blocks",
        )
    )
    selected.extend(adjacent_fill)
    remaining = tuple(entry for entry in remaining if entry not in adjacent_fill)
    if len(selected) < limit:
        selected.extend(
            deterministic_subset(
                remaining,
                limit - len(selected),
                *seed_parts,
                "forecast_remaining_blocks",
            )
        )
    selected_set = set(selected)
    return tuple(entry for entry in entries if entry in selected_set)


__all__ = [
    "candidate_subset",
    "connected_subset",
    "deterministic_subset",
    "evenly_spaced_subset",
    "forecast_aware_block_subset",
]
