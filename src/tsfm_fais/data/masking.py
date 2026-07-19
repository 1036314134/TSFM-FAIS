"""Deterministic sequence-level missingness for complete multivariate series."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tsfm_fais.contracts import MissingBlock

MissingMechanism = Literal[
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
        if not 0 < self.missing_rate <= 0.5:
            raise ValueError("missing_rate must be in (0, 0.5]")
        if not self.block_lengths or any(length < 1 for length in self.block_lengths):
            raise ValueError("block_lengths must contain positive integers")
        if len(set(self.block_lengths)) != len(self.block_lengths):
            raise ValueError("block_lengths must be unique")


@dataclass(frozen=True)
class MaskedSeries:
    """A reusable missingness realization over one complete ``[T,D]`` item."""

    values: np.ndarray
    observed_mask: np.ndarray
    blocks: tuple[MissingBlock, ...]
    spec: MaskingSpec
    seed: int
    realization_id: str
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
        normalized = values.copy()
        normalized[~observed] = np.nan
        if not (~observed).any():
            raise ValueError("MaskedSeries must contain at least one missing value")
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "observed_mask", observed)


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


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


def mask_time_series(
    values: np.ndarray,
    spec: MaskingSpec,
    seed: int,
    *,
    calibration_values: np.ndarray | None = None,
) -> MaskedSeries:
    """Generate one deterministic mask before any forecasting window is cut."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape [T,D]")
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("sequence-level masking requires complete finite values")
    calibration = array if calibration_values is None else np.asarray(calibration_values, dtype=float)
    if (
        calibration.ndim != 2
        or calibration.shape[1] != array.shape[1]
        or calibration.shape[0] < 2
        or not np.isfinite(calibration).all()
    ):
        raise ValueError("calibration_values must be complete [C,D] data")

    rng = np.random.default_rng(int(seed))
    observed = np.ones(array.shape, dtype=bool)
    target = _target_count(array.size, spec.missing_rate)
    lengths = tuple(sorted(min(array.shape[0], length) for length in spec.block_lengths))

    if spec.mechanism == "random_point":
        _hide_random(observed, rng, target)
    elif spec.mechanism == "independent_block":
        _place_independent_blocks(observed, rng, target, lengths)
    elif spec.mechanism == "synchronous_block":
        dimensions = array.shape[1]
        target = max(
            dimensions,
            min(
                array.size - dimensions,
                int(round(target / dimensions)) * dimensions,
            ),
        )
        _place_synchronous_blocks(observed, rng, target, lengths)
    elif spec.mechanism == "staggered_correlated":
        channels = _strongly_correlated_channels(
            calibration,
            minimum_count=int(np.ceil(2.0 * spec.missing_rate * array.shape[1])),
        )
        _place_staggered_blocks(observed, rng, target, lengths, channels)
    elif spec.mechanism == "value_dependent":
        _place_value_dependent_blocks(
            observed,
            _value_scores(array, calibration),
            target,
            lengths,
        )
        _hide_random(observed, rng, target - int((~observed).sum()))
    elif spec.mechanism == "mixed_outage":
        block_target = int(round(0.7 * target))
        _place_independent_blocks(observed, rng, block_target, lengths)
        _hide_random(observed, rng, target - int((~observed).sum()))
    else:  # pragma: no cover - protected by the typed configuration
        raise ValueError(f"unsupported mechanism: {spec.mechanism}")

    missing_count = int((~observed).sum())
    if missing_count != target:
        raise RuntimeError(
            f"{spec.mechanism} placed {missing_count} cells; expected {target}"
        )
    masked = array.copy()
    masked[~observed] = np.nan
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(observed, dtype=np.uint8).tobytes())
    digest.update(str(int(seed)).encode("ascii"))
    realization_id = digest.hexdigest()[:20]
    metadata = {
        "protocol": "sequence_mask_v2",
        "target_missing_rate": float(spec.missing_rate),
        "realized_missing_rate": missing_count / float(array.size),
        "missing_count": missing_count,
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
        metadata=metadata,
    )
