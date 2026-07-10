"""Deterministic synthetic missingness for clean multivariate contexts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from tsfm_fais.contracts import MissingBlock

MissingMechanism = Literal[
    "random_point",
    "independent_block",
    "synchronous_block",
    "staggered_correlated",
    "value_dependent",
    "tail_mixed",
]


@dataclass(frozen=True)
class MaskingSpec:
    mechanism: MissingMechanism
    missing_rate: float
    block_fraction: float = 0.1
    max_blocks: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.missing_rate < 1:
            raise ValueError("missing_rate must be between zero and one")
        if not 0 < self.block_fraction <= 0.5:
            raise ValueError("block_fraction must be in (0, 0.5]")
        if self.max_blocks < 1:
            raise ValueError("max_blocks must be positive")


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def _runs(mask: np.ndarray, pattern: str) -> list[MissingBlock]:
    blocks: list[MissingBlock] = []
    n, length, dimensions = mask.shape
    for batch_index in range(n):
        for channel in range(dimensions):
            missing = ~mask[batch_index, :, channel]
            padded = np.pad(missing.astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            for start, end in zip(starts, ends):
                blocks.append(
                    MissingBlock(
                        block_id=f"n{batch_index}:d{channel}:{start}-{end}",
                        batch_index=batch_index,
                        channel=channel,
                        start=int(start),
                        end=int(end),
                        pattern=pattern,
                    )
                )
    return blocks


def _choose_start(rng: np.random.Generator, length: int, block: int, tail: bool = False) -> int:
    if tail:
        return length - block
    return int(rng.integers(0, max(1, length - block + 1)))


def _place_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    block: int,
    max_blocks: int,
    *,
    synchronous: bool = False,
    tail: bool = False,
) -> None:
    _, length, dimensions = mask.shape
    attempts = 0
    placed_blocks = 0
    max_attempts = max(500, max_blocks * 20)
    while (
        int((~mask).sum()) < target
        and placed_blocks < max_blocks
        and attempts < max_attempts
    ):
        attempts += 1
        remaining = target - int((~mask).sum())
        channel = int(rng.integers(0, dimensions))
        channels = tuple(range(dimensions)) if synchronous else (channel,)
        current = min(block, remaining // len(channels))
        if current < 1:
            break
        start = _choose_start(
            rng, length, current, tail=tail and placed_blocks == 0
        )
        changed = False
        for selected in channels:
            before = int(mask[0, start : start + current, selected].sum())
            mask[0, start : start + current, selected] = False
            changed |= before > 0
        if changed:
            placed_blocks += 1


def _strongly_correlated_channels(
    values: np.ndarray,
    minimum_count: int = 2,
) -> tuple[int, ...]:
    dimensions = values.shape[1]
    if dimensions < 2:
        return (0,)
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(
        centered,
        norms[None, :],
        out=np.zeros_like(centered),
        where=norms[None, :] > 0,
    )
    correlation = np.abs(normalized.T @ normalized)
    np.fill_diagonal(correlation, -np.inf)
    left, right = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    selected = [int(left), int(right)]
    target_count = min(
        dimensions,
        max(2, int(minimum_count), int(np.ceil(np.sqrt(dimensions)))),
    )
    while len(selected) < target_count:
        remaining = [index for index in range(dimensions) if index not in selected]
        next_channel = max(
            remaining,
            key=lambda index: (
                float(np.max(correlation[index, selected])),
                -index,
            ),
        )
        selected.append(int(next_channel))
    return tuple(selected)


def _place_staggered_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    block: int,
    channels: Sequence[int],
) -> None:
    """Place channel-correlated blocks on shifted, non-overlapping grids."""

    _, length, _ = mask.shape
    selected = tuple(map(int, channels))
    if not selected:
        return
    stagger_width = max(1, min(block, length))
    offsets = rng.permutation(stagger_width)
    channel_offsets = {
        channel: int(offsets[index % stagger_width])
        for index, channel in enumerate(selected)
    }
    for base in range(0, length, block):
        remaining = target - int((~mask).sum())
        width = min(block, remaining // len(selected))
        if width < 1:
            break
        for channel in selected:
            start = min(length - width, base + channel_offsets[channel])
            mask[0, start : start + width, channel] = False
        if int((~mask).sum()) >= target:
            break
    deficit = target - int((~mask).sum())
    if 0 < deficit < len(selected):
        available = np.argwhere(mask[0][:, np.asarray(selected)])
        chosen = rng.choice(len(available), size=deficit, replace=False)
        for time_index, channel_offset in available[chosen]:
            mask[0, int(time_index), selected[int(channel_offset)]] = False


def _place_value_dependent_blocks(
    mask: np.ndarray,
    scores: np.ndarray,
    target: int,
    block: int,
) -> None:
    """Greedily cover high-score observations with contiguous local blocks."""

    _, length, _ = mask.shape
    ordered = np.argsort(np.asarray(scores), axis=None)[::-1]
    for flat_index in ordered:
        if int((~mask).sum()) >= target:
            break
        time_index, channel = np.unravel_index(int(flat_index), scores.shape)
        if not mask[0, time_index, channel]:
            continue
        width = min(block, target - int((~mask).sum()))
        start = min(length - width, max(0, int(time_index) - width // 2))
        mask[0, start : start + width, int(channel)] = False


def inject_missing(
    values: np.ndarray,
    spec: MaskingSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[MissingBlock, ...]]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError("inject_missing currently accepts one [L,D] context")
    if not np.isfinite(array).all():
        raise ValueError("synthetic missingness requires a complete finite context")
    rng = np.random.default_rng(seed)
    mask = np.ones(array.shape, dtype=bool)
    target = max(1, int(round(array.size * spec.missing_rate)))
    length, dimensions = array.shape[1:]
    block = max(1, min(length, int(round(length * spec.block_fraction))))
    if spec.mechanism == "random_point":
        flat = rng.choice(array.size, size=min(target, array.size - 1), replace=False)
        mask.flat[flat] = False
    elif spec.mechanism == "independent_block":
        _place_blocks(mask, rng, target, block, spec.max_blocks * dimensions)
    elif spec.mechanism == "synchronous_block":
        target = max(
            dimensions,
            min(
                array.size - dimensions,
                int(round(target / dimensions)) * dimensions,
            ),
        )
        _place_blocks(mask, rng, target, block, spec.max_blocks, synchronous=True)
    elif spec.mechanism == "staggered_correlated":
        correlated = _strongly_correlated_channels(
            array[0],
            minimum_count=int(np.ceil(2.0 * spec.missing_rate * dimensions)),
        )
        width = len(correlated)
        target = max(
            width,
            min(
                ((array.size - 1) // width) * width,
                int(round(target / width)) * width,
            ),
        )
        _place_staggered_blocks(mask, rng, target, block, correlated)
    elif spec.mechanism == "value_dependent":
        scores = np.abs(array[0] - np.median(array[0], axis=0))
        _place_value_dependent_blocks(mask, scores, target, block)
    elif spec.mechanism == "tail_mixed":
        _place_blocks(
            mask,
            rng,
            target,
            block,
            spec.max_blocks * dimensions,
            tail=True,
        )
    else:  # pragma: no cover - Literal and validation guard this branch
        raise ValueError(f"unsupported mechanism: {spec.mechanism}")
    if int((~mask).sum()) < target:
        raise RuntimeError(
            f"{spec.mechanism} could not place the requested structured missingness"
        )
    masked = array.copy()
    masked[~mask] = np.nan
    return masked[0], mask[0], tuple(_runs(mask, spec.mechanism))
