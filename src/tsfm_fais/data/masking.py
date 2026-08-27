"""Deterministic sequence-level missingness for multivariate series."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tsfm_fais.contracts import MissingBlock

MissingMechanism = Literal[
    "native",
    "random_point",
    "independent_block",
    "synchronous_block",
    "staggered_correlated",
    "value_dependent",
    "mixed_outage",
]


@dataclass(frozen=True)
class MaskingSpec:
    """Configuration for one full-series missingness realization."""

    mechanism: MissingMechanism
    missing_rate: float
    block_lengths: tuple[int, ...] = (6, 12, 24, 48)

    def __post_init__(self) -> None:
        if self.mechanism == "native":
            if self.missing_rate != 0.0:
                raise ValueError("native masking requires missing_rate=0")
        elif not 0 < self.missing_rate <= 0.5:
            raise ValueError("missing_rate must be in (0, 0.5]")
        if not self.block_lengths or any(length < 1 for length in self.block_lengths):
            raise ValueError("block_lengths must contain positive integers")
        if len(set(self.block_lengths)) != len(self.block_lengths):
            raise ValueError("block_lengths must be unique")


@dataclass(frozen=True)
class MaskedSeries:
    """A reusable missingness realization over one ``[T,D]`` item."""

    values: np.ndarray
    observed_mask: np.ndarray
    blocks: tuple[MissingBlock, ...]
    spec: MaskingSpec
    seed: int
    realization_id: str
    base_observed_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        observed = np.asarray(self.observed_mask, dtype=bool)
        if values.ndim != 2 or observed.shape != values.shape:
            raise ValueError("MaskedSeries values and mask must share shape [T,D]")
        if values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("MaskedSeries cannot be empty")
        if np.any(~np.isfinite(values[observed])):
            raise ValueError("observed values must be finite")
        base_observed = (
            np.ones_like(observed, dtype=bool)
            if self.base_observed_mask is None
            else np.asarray(self.base_observed_mask, dtype=bool)
        )
        if base_observed.shape != observed.shape:
            raise ValueError("base_observed_mask must match MaskedSeries values")
        if np.any(observed & ~base_observed):
            raise ValueError("final observations must be a subset of the base observations")
        normalized = values.copy()
        normalized[~observed] = np.nan
        if not (~observed).any():
            raise ValueError("MaskedSeries must contain at least one missing value")
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "base_observed_mask", base_observed)


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def _fenwick_tree(weights: np.ndarray) -> np.ndarray:
    tree = np.zeros(len(weights) + 1, dtype=np.int64)
    tree[1:] = np.asarray(weights, dtype=np.int64)
    for index in range(1, len(tree)):
        parent = index + (index & -index)
        if parent < len(tree):
            tree[parent] += tree[index]
    return tree


def _fenwick_add(tree: np.ndarray, position: int, delta: int) -> None:
    index = position + 1
    while index < len(tree):
        tree[index] += delta
        index += index & -index


def _fenwick_position(tree: np.ndarray, rank: int) -> int:
    """Return the zero-based position of a zero-based active rank."""

    index = 0
    bit = 1 << ((len(tree) - 1).bit_length() - 1)
    while bit:
        candidate = index + bit
        if candidate < len(tree) and int(tree[candidate]) <= rank:
            index = candidate
            rank -= int(tree[candidate])
        bit >>= 1
    return index


def _hide_uniform_visible_cells(
    observed: np.ndarray,
    visible_counts: np.ndarray,
    prefix_end: int,
    count: int,
    rng: np.random.Generator,
) -> None:
    """Sample the legacy row-major eligible set without rebuilding it per draw."""

    dimensions = observed.shape[1]
    prefix = observed[:prefix_end]
    initially_eligible = prefix & (visible_counts > 2)[None, :]
    tree = _fenwick_tree(initially_eligible.reshape(-1))
    eligible_count = int(initially_eligible.sum())
    for _ in range(count):
        if eligible_count == 0:
            raise ValueError("base missing rate leaves a variate without sufficient observations")
        rank = int(rng.integers(0, eligible_count))
        position = _fenwick_position(tree, rank)
        time_index, channel = divmod(position, dimensions)
        observed[time_index, channel] = False
        _fenwick_add(tree, position, -1)
        eligible_count -= 1
        visible_counts[channel] -= 1
        if visible_counts[channel] == 2:
            for remaining_time in np.flatnonzero(observed[:prefix_end, channel]):
                remaining_position = int(remaining_time) * dimensions + channel
                _fenwick_add(tree, remaining_position, -1)
                eligible_count -= 1


def no_complete_window_base_mask(
    values: np.ndarray,
    prefix_end: int,
    window_length: int,
    missing_rate: float,
    seed: int,
    *,
    source_observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Hide only visible prefix cells while ensuring every training window is incomplete."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape [T,D]")
    if not 0 < missing_rate <= 0.5:
        raise ValueError("base missing_rate must be in (0, 0.5]")
    if window_length < 2 or not window_length <= prefix_end <= array.shape[0]:
        raise ValueError("invalid prefix_end or window_length for the persistent base mask")
    source = (
        np.isfinite(array)
        if source_observed_mask is None
        else np.asarray(source_observed_mask, dtype=bool)
    )
    if source.shape != array.shape or np.any(source & ~np.isfinite(array)):
        raise ValueError("source_observed_mask must identify finite source values")
    observed = source.copy()
    visible_counts = np.sum(observed[:prefix_end], axis=0).astype(int)
    if np.any(visible_counts < 2):
        raise ValueError("each variate needs at least two visible prefix values")

    rng = np.random.default_rng(int(seed))
    additions = 0
    for start in range(prefix_end - window_length + 1):
        stop = start + window_length
        if np.any(~observed[start:stop]):
            continue
        candidates = np.argwhere(observed[start:stop])
        eligible = [
            (start + int(time_index), int(channel))
            for time_index, channel in candidates
            if visible_counts[int(channel)] > 2
        ]
        if not eligible:
            raise ValueError("cannot make every training window incomplete safely")
        latest_time = max(time_index for time_index, _ in eligible)
        latest = [
            (time_index, channel) for time_index, channel in eligible if time_index == latest_time
        ]
        time_index, channel = latest[int(rng.integers(0, len(latest)))]
        observed[time_index, channel] = False
        visible_counts[channel] -= 1
        additions += 1

    source_visible = int(source[:prefix_end].sum())
    target_additions = max(additions, int(round(source_visible * missing_rate)))
    _hide_uniform_visible_cells(
        observed,
        visible_counts,
        prefix_end,
        target_additions - additions,
        rng,
    )

    if any(
        np.all(observed[start : start + window_length])
        for start in range(prefix_end - window_length + 1)
    ):
        raise RuntimeError("persistent base mask left a complete training window")
    if not np.array_equal(observed[prefix_end:], source[prefix_end:]):
        raise RuntimeError("persistent base mask changed values outside the fit prefix")
    return observed


def extract_missing_blocks(
    observed_mask: np.ndarray,
    pattern: str,
) -> tuple[MissingBlock, ...]:
    """Extract maximal channel-wise runs from a 2-D or 3-D mask."""

    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3:
        raise ValueError("observed_mask must have shape [T,D] or [N,T,D]")
    blocks: list[MissingBlock] = []
    for batch_index in range(mask.shape[0]):
        for channel in range(mask.shape[2]):
            missing = ~mask[batch_index, :, channel]
            padded = np.pad(missing.astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            for start, end in zip(starts, ends, strict=True):
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
    return tuple(blocks)


def _target_count(size: int, rate: float) -> int:
    return min(size - 1, max(1, int(round(size * rate))))


def _available(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.reshape(-1))


def _hide_random(
    mask: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> None:
    available = _available(mask)
    count = min(int(count), len(available))
    if count < 1:
        return
    selected = rng.choice(available, size=count, replace=False)
    mask.reshape(-1)[selected] = False


def _draw_block_length(
    rng: np.random.Generator,
    lengths: Sequence[int],
    time_length: int,
    remaining: int,
) -> int:
    eligible = tuple(min(time_length, length) for length in lengths if length <= remaining)
    if not eligible:
        return min(time_length, remaining)
    return int(eligible[int(rng.integers(0, len(eligible)))])


def _place_independent_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    lengths: Sequence[int],
) -> None:
    time_length, dimensions = mask.shape
    attempts = 0
    maximum_attempts = max(1000, target * 20)
    missing_count = int((~mask).sum())
    while missing_count < target and attempts < maximum_attempts:
        attempts += 1
        remaining = target - missing_count
        width = _draw_block_length(rng, lengths, time_length, remaining)
        channel = int(rng.integers(0, dimensions))
        start = int(rng.integers(0, time_length - width + 1))
        selector = mask[start : start + width, channel]
        observed_indices = np.flatnonzero(selector)
        if not len(observed_indices):
            continue
        take = min(remaining, len(observed_indices))
        mask[start + observed_indices[:take], channel] = False
        missing_count += take
    _hide_random(mask, rng, target - missing_count)


def _place_synchronous_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    lengths: Sequence[int],
) -> None:
    time_length, _ = mask.shape
    attempts = 0
    missing_count = int((~mask).sum())
    while missing_count < target and attempts < max(500, target * 10):
        attempts += 1
        remaining = target - missing_count
        width = _draw_block_length(rng, lengths, time_length, max(1, remaining))
        start = int(rng.integers(0, time_length - width + 1))
        for time_index in range(start, start + width):
            if missing_count >= target:
                break
            visible = np.flatnonzero(mask[time_index])
            if not len(visible):
                continue
            take = min(target - missing_count, len(visible))
            mask[time_index, visible[:take]] = False
            missing_count += take
    _hide_random(mask, rng, target - missing_count)


def _place_complete_synchronous_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    lengths: Sequence[int],
) -> None:
    """Preserve the established complete-series synchronous-mask realization."""

    time_length, dimensions = mask.shape
    synchronous_times = target // dimensions
    time_mask = np.ones(time_length, dtype=bool)
    attempts = 0
    missing_times = 0
    while missing_times < synchronous_times and attempts < max(500, target * 10):
        attempts += 1
        remaining = synchronous_times - missing_times
        width = _draw_block_length(rng, lengths, time_length, remaining)
        start = int(rng.integers(0, time_length - width + 1))
        available = np.flatnonzero(time_mask[start : start + width])
        take = min(remaining, len(available))
        time_mask[start + available[:take]] = False
        missing_times += take
    if missing_times < synchronous_times:
        available = np.flatnonzero(time_mask)
        selected = rng.choice(
            available,
            size=synchronous_times - missing_times,
            replace=False,
        )
        time_mask[selected] = False
        missing_times = synchronous_times
    mask[~time_mask, :] = False
    _hide_random(mask, rng, target - missing_times * dimensions)


def _strongly_correlated_channels(
    values: np.ndarray,
    minimum_count: int = 2,
) -> tuple[int, ...]:
    array = np.asarray(values, dtype=float)
    dimensions = array.shape[1]
    if dimensions < 2:
        return (0,)
    centered = array - np.mean(array, axis=0, keepdims=True)
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
        selected.append(
            int(
                max(
                    remaining,
                    key=lambda index: (
                        float(np.max(correlation[index, selected])),
                        -index,
                    ),
                )
            )
        )
    return tuple(selected)


def _place_staggered_blocks(
    mask: np.ndarray,
    rng: np.random.Generator,
    target: int,
    lengths: Sequence[int],
    channels: Sequence[int],
) -> None:
    time_length = mask.shape[0]
    selected = tuple(map(int, channels))
    attempts = 0
    missing_count = int((~mask).sum())
    while missing_count < target and attempts < max(1000, target * 20):
        attempts += 1
        remaining = target - missing_count
        nominal = _draw_block_length(
            rng,
            lengths,
            time_length,
            max(1, int(np.ceil(remaining / len(selected)))),
        )
        base = int(rng.integers(0, time_length - nominal + 1))
        maximum_shift = max(1, nominal // 2)
        for channel_offset, channel in enumerate(selected):
            if missing_count >= target:
                break
            shift = channel_offset % (2 * maximum_shift + 1) - maximum_shift
            start = min(time_length - nominal, max(0, base + shift))
            available = np.flatnonzero(mask[start : start + nominal, channel])
            take = min(target - missing_count, len(available))
            mask[start + available[:take], channel] = False
            missing_count += take
    _hide_random(mask, rng, target - missing_count)


def _place_value_dependent_blocks(
    mask: np.ndarray,
    scores: np.ndarray,
    target: int,
    lengths: Sequence[int],
) -> None:
    time_length = mask.shape[0]
    ordered = np.argsort(np.asarray(scores), axis=None)[::-1]
    length_index = 0
    missing_count = int((~mask).sum())
    for flat_index in ordered:
        if missing_count >= target:
            break
        time_index, channel = np.unravel_index(int(flat_index), scores.shape)
        if not mask[time_index, channel]:
            continue
        remaining = target - missing_count
        width = min(time_length, remaining, lengths[length_index % len(lengths)])
        length_index += 1
        start = min(time_length - width, max(0, int(time_index) - width // 2))
        available = np.flatnonzero(mask[start : start + width, int(channel)])
        take = min(remaining, len(available))
        mask[start + available[:take], int(channel)] = False
        missing_count += take


def _value_scores(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    center = np.median(calibration, axis=0)
    mad = np.median(np.abs(calibration - center[None, :]), axis=0)
    fallback = np.std(calibration, axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.abs(values - center[None, :]) / scale[None, :]


def _complete_visible(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape [T,D] with at least two rows")
    medians = np.nanmedian(array, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError(f"{name} has a variate without visible values")
    return np.where(np.isfinite(array), array, medians[None, :])


def mask_time_series(
    values: np.ndarray,
    spec: MaskingSpec,
    seed: int,
    *,
    calibration_values: np.ndarray | None = None,
    base_observed_mask: np.ndarray | None = None,
) -> MaskedSeries:
    """Generate one deterministic mask before any forecasting window is cut."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape [T,D]")
    if array.size < 2:
        raise ValueError("sequence-level masking requires at least two values")
    base_observed = (
        np.isfinite(array)
        if base_observed_mask is None
        else np.asarray(base_observed_mask, dtype=bool)
    )
    if base_observed.shape != array.shape or np.any(base_observed & ~np.isfinite(array)):
        raise ValueError("base_observed_mask must identify finite source values")
    if not base_observed.any() or np.any(np.sum(base_observed, axis=0) < 2):
        raise ValueError("every variate needs at least two base-observed values")
    raw_calibration = (
        np.where(base_observed, array, np.nan)
        if calibration_values is None
        else np.asarray(calibration_values, dtype=float)
    )
    if raw_calibration.ndim != 2 or raw_calibration.shape[1] != array.shape[1]:
        raise ValueError("calibration_values must align with the sequence variates")
    calibration = _complete_visible(raw_calibration, "calibration_values")
    scoring_values = _complete_visible(
        np.where(base_observed, array, np.nan),
        "values",
    )

    rng = np.random.default_rng(int(seed))
    observed = base_observed.copy()
    base_missing_count = int((~base_observed).sum())
    additional_target = (
        0
        if spec.mechanism == "native"
        else _target_count(int(base_observed.sum()), spec.missing_rate)
    )
    if spec.mechanism == "synchronous_block" and base_missing_count == 0:
        dimensions = array.shape[1]
        additional_target = max(
            dimensions,
            min(
                int(base_observed.sum()) - dimensions,
                int(round(additional_target / dimensions)) * dimensions,
            ),
        )
    target = base_missing_count + additional_target
    lengths = tuple(sorted(min(array.shape[0], length) for length in spec.block_lengths))

    if spec.mechanism == "native":
        pass
    elif spec.mechanism == "random_point":
        _hide_random(observed, rng, additional_target)
    elif spec.mechanism == "independent_block":
        _place_independent_blocks(observed, rng, target, lengths)
    elif spec.mechanism == "synchronous_block":
        if base_missing_count:
            _place_synchronous_blocks(observed, rng, target, lengths)
        else:
            _place_complete_synchronous_blocks(observed, rng, target, lengths)
    elif spec.mechanism == "staggered_correlated":
        channels = _strongly_correlated_channels(
            calibration,
            minimum_count=int(np.ceil(2.0 * spec.missing_rate * array.shape[1])),
        )
        _place_staggered_blocks(observed, rng, target, lengths, channels)
    elif spec.mechanism == "value_dependent":
        _place_value_dependent_blocks(
            observed,
            _value_scores(scoring_values, calibration),
            target,
            lengths,
        )
        _hide_random(observed, rng, target - int((~observed).sum()))
    elif spec.mechanism == "mixed_outage":
        block_target = base_missing_count + int(round(0.7 * additional_target))
        _place_independent_blocks(observed, rng, block_target, lengths)
        _hide_random(observed, rng, target - int((~observed).sum()))
    else:  # pragma: no cover - protected by the typed configuration
        raise ValueError(f"unsupported mechanism: {spec.mechanism}")

    missing_count = int((~observed).sum())
    additional_missing_count = missing_count - base_missing_count
    if additional_missing_count != additional_target:
        raise RuntimeError(
            f"{spec.mechanism} placed {additional_missing_count} additional cells; "
            f"expected {additional_target}"
        )
    masked = array.copy()
    masked[~observed] = np.nan
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(observed, dtype=np.uint8).tobytes())
    if base_missing_count:
        digest.update(np.ascontiguousarray(base_observed, dtype=np.uint8).tobytes())
    digest.update(str(int(seed)).encode("ascii"))
    realization_id = digest.hexdigest()[:20]
    metadata = {
        "protocol": (
            "native_observation_mask_v1"
            if spec.mechanism == "native"
            else "base_plus_sequence_mask_v1"
            if base_missing_count
            else "sequence_mask_v2"
        ),
        "target_missing_rate": float(spec.missing_rate),
        "realized_missing_rate": missing_count / float(array.size),
        "missing_count": missing_count,
        "base_missing_count": base_missing_count,
        "base_missing_rate": base_missing_count / float(array.size),
        "additional_missing_count": additional_missing_count,
        "additional_missing_rate_visible": (additional_missing_count / float(base_observed.sum())),
        "total_count": int(array.size),
        "block_lengths": list(spec.block_lengths),
    }
    return MaskedSeries(
        values=masked,
        observed_mask=observed,
        blocks=extract_missing_blocks(observed, spec.mechanism),
        spec=spec,
        seed=int(seed),
        realization_id=realization_id,
        base_observed_mask=base_observed,
        metadata=metadata,
    )
