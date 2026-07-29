"""Inference-only features for whole-sequence imputer selection.

The extractor deliberately accepts only the incomplete context, its observation
mask, and an optional known period.  It therefore cannot consume dataset,
missingness-mechanism, forecaster, future, or hidden-ground-truth information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsfm_fais.contracts import SeriesBatch

_SUMMARY_SUFFIXES = ("mean", "std", "min", "max")

SEQUENCE_FEATURE_NAMES = (
    "log_length",
    "log_channels",
    "period_known",
    "period_ratio",
    "observed_fraction",
    "missing_fraction",
    "channel_with_observations_fraction",
    "channel_with_two_observations_fraction",
    "channel_missing_mean",
    "channel_missing_std",
    "channel_missing_min",
    "channel_missing_max",
    "time_missing_mean",
    "time_missing_std",
    "time_missing_min",
    "time_missing_max",
    "fully_missing_time_fraction",
    "block_count_per_channel",
    "block_length_ratio_mean",
    "block_length_ratio_std",
    "block_length_ratio_min",
    "block_length_ratio_max",
    "block_start_ratio_mean",
    "block_start_ratio_std",
    "block_start_ratio_min",
    "block_start_ratio_max",
    "leading_block_fraction",
    "tail_block_fraction",
    "internal_block_fraction",
    "observed_mean",
    "observed_std",
    "observed_median",
    "observed_iqr",
    "channel_mean_mean",
    "channel_mean_std",
    "channel_mean_min",
    "channel_mean_max",
    "channel_std_mean",
    "channel_std_std",
    "channel_std_min",
    "channel_std_max",
    "channel_iqr_mean",
    "channel_iqr_std",
    "channel_iqr_min",
    "channel_iqr_max",
    "consecutive_difference_scale_mean",
    "consecutive_difference_scale_std",
    "consecutive_difference_scale_min",
    "consecutive_difference_scale_max",
    "normalized_trend_mean",
    "normalized_trend_std",
    "normalized_trend_min",
    "normalized_trend_max",
    "lag1_autocorrelation_mean",
    "lag1_autocorrelation_std",
    "lag1_autocorrelation_min",
    "lag1_autocorrelation_max",
    "lag1_autocorrelation_valid_fraction",
    "period_autocorrelation_mean",
    "period_autocorrelation_std",
    "period_autocorrelation_min",
    "period_autocorrelation_max",
    "period_autocorrelation_valid_fraction",
    "spectral_entropy_mean",
    "spectral_entropy_std",
    "spectral_entropy_min",
    "spectral_entropy_max",
    "spectral_dominant_frequency_mean",
    "spectral_dominant_frequency_std",
    "spectral_dominant_frequency_min",
    "spectral_dominant_frequency_max",
    "spectral_low_frequency_power_mean",
    "spectral_low_frequency_power_std",
    "spectral_low_frequency_power_min",
    "spectral_low_frequency_power_max",
    "spectral_valid_fraction",
    "cross_correlation_mean",
    "cross_correlation_std",
    "cross_correlation_min",
    "cross_correlation_max",
    "cross_absolute_correlation_mean",
    "cross_absolute_correlation_std",
    "cross_absolute_correlation_min",
    "cross_absolute_correlation_max",
    "cross_correlation_valid_fraction",
)


def _as_single_sequence(
    values: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mask = np.asarray(observed_mask, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("values and observed_mask must have the same shape")
    if array.ndim == 1:
        array = array[:, None]
        mask = mask[:, None]
    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("sequence features require exactly one batch item")
        array = array[0]
        mask = mask[0]
    elif array.ndim != 2:
        raise ValueError("values must have shape [L], [L,D], or [1,L,D]")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("values must contain at least one timestep and one channel")
    if np.any(~np.isfinite(array[mask])):
        raise ValueError("observed values must be finite")
    return array, mask


def _summary(values: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array)),
    )


def _append_summary(output: dict[str, float], prefix: str, values: list[float]) -> None:
    for suffix, value in zip(_SUMMARY_SUFFIXES, _summary(values), strict=True):
        output[f"{prefix}_{suffix}"] = value


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    length, channels = mask.shape
    for channel in range(channels):
        missing = ~mask[:, channel]
        padded = np.concatenate(([False], missing, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        runs.extend(
            (channel, int(start), int(end)) for start, end in zip(starts, ends, strict=True)
        )
    return runs


def _masked_autocorrelation(
    values: np.ndarray,
    mask: np.ndarray,
    lag: int,
) -> tuple[float, bool]:
    if lag < 1 or lag >= len(values):
        return 0.0, False
    valid = mask[:-lag] & mask[lag:]
    if np.count_nonzero(valid) < 3:
        return 0.0, False
    left = values[:-lag][valid]
    right = values[lag:][valid]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1e-12:
        return 0.0, False
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)), True


def _spectral_features(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, bool]:
    length = len(values)
    if length < 4 or np.count_nonzero(mask) < 4:
        return 0.0, 0.0, 0.0, False
    centered = np.zeros(length, dtype=float)
    centered[mask] = values[mask] - float(np.mean(values[mask]))
    power = np.square(np.abs(np.fft.rfft(centered)))[1:]
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= 1e-12:
        return 0.0, 0.0, 0.0, False
    probabilities = power / total
    nonzero = probabilities > 0.0
    if len(probabilities) > 1:
        entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
        entropy /= float(np.log(len(probabilities)))
    else:
        entropy = 0.0
    dominant_bin = int(np.argmax(power)) + 1
    dominant_frequency = dominant_bin / float(length)
    low_frequency_bins = max(1, int(np.ceil(len(power) / 4.0)))
    low_frequency_power = float(np.sum(power[:low_frequency_bins]) / total)
    return (
        float(np.clip(entropy, 0.0, 1.0)),
        dominant_frequency,
        float(np.clip(low_frequency_power, 0.0, 1.0)),
        True,
    )


def extract_sequence_features(
    values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    period: int | None = None,
) -> dict[str, float]:
    """Return a deterministic, fixed-schema feature dictionary for one context.

    Values at locations where ``observed_mask`` is false are never read.  A 3-D
    input is accepted only when its batch dimension is one, which prevents
    accidental aggregation across independent selection tasks.
    """

    array, mask = _as_single_sequence(values, observed_mask)
    if period is not None and (isinstance(period, bool) or int(period) != period or period < 1):
        raise ValueError("period must be a positive integer")
    known_period = None if period is None else int(period)
    length, channels = array.shape
    missing = ~mask
    observed_values = array[mask]

    output: dict[str, float] = {
        "log_length": float(np.log1p(length)),
        "log_channels": float(np.log1p(channels)),
        "period_known": float(known_period is not None),
        "period_ratio": float((known_period or 0) / length),
        "observed_fraction": float(np.mean(mask)),
        "missing_fraction": float(np.mean(missing)),
        "channel_with_observations_fraction": float(np.mean(np.any(mask, axis=0))),
        "channel_with_two_observations_fraction": float(np.mean(np.sum(mask, axis=0) >= 2)),
    }
    _append_summary(output, "channel_missing", list(np.mean(missing, axis=0)))
    _append_summary(output, "time_missing", list(np.mean(missing, axis=1)))
    output["fully_missing_time_fraction"] = float(np.mean(np.all(missing, axis=1)))

    runs = _missing_runs(mask)
    output["block_count_per_channel"] = float(len(runs) / channels)
    _append_summary(
        output, "block_length_ratio", [(end - start) / length for _, start, end in runs]
    )
    _append_summary(output, "block_start_ratio", [start / length for _, start, _ in runs])
    run_count = max(1, len(runs))
    output["leading_block_fraction"] = float(sum(start == 0 for _, start, _ in runs) / run_count)
    output["tail_block_fraction"] = float(sum(end == length for _, _, end in runs) / run_count)
    output["internal_block_fraction"] = float(
        sum(start > 0 and end < length for _, start, end in runs) / run_count
    )

    if observed_values.size:
        lower, upper = np.percentile(observed_values, (25.0, 75.0))
        output.update(
            {
                "observed_mean": float(np.mean(observed_values)),
                "observed_std": float(np.std(observed_values)),
                "observed_median": float(np.median(observed_values)),
                "observed_iqr": float(upper - lower),
            }
        )
    else:
        output.update(
            {
                "observed_mean": 0.0,
                "observed_std": 0.0,
                "observed_median": 0.0,
                "observed_iqr": 0.0,
            }
        )

    channel_means: list[float] = []
    channel_stds: list[float] = []
    channel_iqrs: list[float] = []
    difference_scales: list[float] = []
    trends: list[float] = []
    lag1_values: list[float] = []
    period_values: list[float] = []
    spectral_entropies: list[float] = []
    spectral_frequencies: list[float] = []
    spectral_low_power: list[float] = []
    lag1_valid = 0
    period_valid = 0
    spectral_valid = 0
    normalized_time = np.linspace(0.0, 1.0, length)

    for channel in range(channels):
        channel_mask = mask[:, channel]
        channel_values = array[:, channel]
        visible = channel_values[channel_mask]
        if visible.size:
            lower, upper = np.percentile(visible, (25.0, 75.0))
            center = float(np.mean(visible))
            scale = float(np.std(visible))
            channel_means.append(center)
            channel_stds.append(scale)
            channel_iqrs.append(float(upper - lower))
        else:
            center = 0.0
            scale = 0.0
            channel_means.append(0.0)
            channel_stds.append(0.0)
            channel_iqrs.append(0.0)

        consecutive = channel_mask[:-1] & channel_mask[1:]
        if np.any(consecutive):
            difference_scales.append(float(np.median(np.abs(np.diff(channel_values)[consecutive]))))
        else:
            difference_scales.append(0.0)

        if np.count_nonzero(channel_mask) >= 2:
            time_visible = normalized_time[channel_mask]
            time_centered = time_visible - float(np.mean(time_visible))
            denominator = float(np.dot(time_centered, time_centered))
            slope = (
                float(np.dot(time_centered, visible - center) / denominator)
                if denominator > 1e-12
                else 0.0
            )
            trends.append(slope / max(scale, 1e-12))
        else:
            trends.append(0.0)

        lag1, valid = _masked_autocorrelation(channel_values, channel_mask, 1)
        lag1_values.append(lag1)
        lag1_valid += int(valid)
        seasonal, valid = _masked_autocorrelation(
            channel_values,
            channel_mask,
            known_period or length,
        )
        period_values.append(seasonal)
        period_valid += int(valid and known_period is not None)
        entropy, frequency, low_power, valid = _spectral_features(channel_values, channel_mask)
        spectral_entropies.append(entropy)
        spectral_frequencies.append(frequency)
        spectral_low_power.append(low_power)
        spectral_valid += int(valid)

    _append_summary(output, "channel_mean", channel_means)
    _append_summary(output, "channel_std", channel_stds)
    _append_summary(output, "channel_iqr", channel_iqrs)
    _append_summary(output, "consecutive_difference_scale", difference_scales)
    _append_summary(output, "normalized_trend", trends)
    _append_summary(output, "lag1_autocorrelation", lag1_values)
    output["lag1_autocorrelation_valid_fraction"] = float(lag1_valid / channels)
    _append_summary(output, "period_autocorrelation", period_values)
    output["period_autocorrelation_valid_fraction"] = float(period_valid / channels)
    _append_summary(output, "spectral_entropy", spectral_entropies)
    _append_summary(output, "spectral_dominant_frequency", spectral_frequencies)
    _append_summary(output, "spectral_low_frequency_power", spectral_low_power)
    output["spectral_valid_fraction"] = float(spectral_valid / channels)

    correlations: list[float] = []
    total_pairs = channels * (channels - 1) // 2
    for left in range(channels):
        for right in range(left + 1, channels):
            overlap = mask[:, left] & mask[:, right]
            if np.count_nonzero(overlap) < 3:
                continue
            left_values = array[overlap, left]
            right_values = array[overlap, right]
            left_values = left_values - np.mean(left_values)
            right_values = right_values - np.mean(right_values)
            denominator = float(
                np.sqrt(np.dot(left_values, left_values) * np.dot(right_values, right_values))
            )
            if denominator > 1e-12:
                correlations.append(
                    float(np.clip(np.dot(left_values, right_values) / denominator, -1.0, 1.0))
                )
    _append_summary(output, "cross_correlation", correlations)
    _append_summary(output, "cross_absolute_correlation", [abs(value) for value in correlations])
    output["cross_correlation_valid_fraction"] = float(
        len(correlations) / total_pairs if total_pairs else 0.0
    )

    if tuple(output) != SEQUENCE_FEATURE_NAMES:
        raise RuntimeError("sequence feature schema was constructed in an unexpected order")
    return {
        name: float(np.nan_to_num(value, nan=0.0, posinf=1e12, neginf=-1e12))
        for name, value in output.items()
    }


def sequence_meta_features(
    batch: SeriesBatch,
    period: int | None = None,
) -> dict[str, float]:
    """Extract one task-level feature dictionary from a single-item batch."""

    if not isinstance(batch, SeriesBatch):
        raise TypeError("batch must be a SeriesBatch")
    return extract_sequence_features(
        batch.values,
        batch.observed_mask,
        period=period,
    )


@dataclass(frozen=True)
class SequenceFeatureExtractor:
    """Small reusable wrapper around :func:`extract_sequence_features`."""

    period: int | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SEQUENCE_FEATURE_NAMES

    def transform(self, values: np.ndarray, observed_mask: np.ndarray) -> dict[str, float]:
        return extract_sequence_features(values, observed_mask, period=self.period)

    def vector(self, values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        features = self.transform(values, observed_mask)
        return np.asarray([features[name] for name in self.feature_names], dtype=float)


__all__ = [
    "SEQUENCE_FEATURE_NAMES",
    "SequenceFeatureExtractor",
    "extract_sequence_features",
    "sequence_meta_features",
]
