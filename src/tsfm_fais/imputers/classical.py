"""Deterministic and probabilistic per-channel imputers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tsfm_fais.contracts import SeriesBatch

from .base import BaseImputer, NativeImputation


def _linear_fill_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    if not observed.any():
        return np.full_like(values, np.nan)
    time = np.arange(len(values), dtype=float)
    return np.interp(time, time[observed], values[observed])


def _apply_per_channel(
    batch: SeriesBatch,
    function: Any,
) -> np.ndarray:
    output = np.array(batch.values, copy=True)
    for sample in range(batch.shape[0]):
        for channel in range(batch.shape[2]):
            output[sample, :, channel] = function(output[sample, :, channel])
    return output


class LOCFImputer(BaseImputer):
    """Last-observation-carried-forward with a leading-edge first value."""

    imputer_id = "locf"

    def _impute_native(self, batch: SeriesBatch, artifact: Any, seed: int) -> np.ndarray:
        def fill(values: np.ndarray) -> np.ndarray:
            out = np.array(values, copy=True)
            observed = np.flatnonzero(np.isfinite(out))
            if not observed.size:
                return out
            out[: observed[0]] = out[observed[0]]
            last = out[observed[0]]
            for index in range(observed[0], len(out)):
                if np.isfinite(out[index]):
                    last = out[index]
                else:
                    out[index] = last
            return out

        return _apply_per_channel(batch, fill)


class LinearInterpolationImputer(BaseImputer):
    imputer_id = "linear_interp"

    def _impute_native(self, batch: SeriesBatch, artifact: Any, seed: int) -> np.ndarray:
        return _apply_per_channel(batch, _linear_fill_1d)


class SeasonalLagImputer(BaseImputer):
    """Fill from the robust median of strictly historical same-phase values."""

    imputer_id = "seasonal_lag"

    def __init__(self, period: int | None = 24) -> None:
        if period is not None and period < 2:
            raise ValueError("period must be at least 2")
        self.period = period

    def _fit(self, train_batch: SeriesBatch, metadata: Mapping[str, Any]) -> dict[str, Any]:
        period = self.period or metadata.get("period") or train_batch.metadata.get("period")
        if period is None or int(period) < 2:
            raise ValueError("seasonal_lag requires a period of at least 2")
        period = int(period)
        profiles = np.full((train_batch.shape[2], period), np.nan, dtype=float)
        for channel in range(train_batch.shape[2]):
            for phase in range(period):
                phase_values = train_batch.values[:, phase::period, channel]
                finite = phase_values[np.isfinite(phase_values)]
                if finite.size:
                    profiles[channel, phase] = float(np.median(finite))
        return {"period": period, "profiles": profiles}

    def _impute_native(
        self, batch: SeriesBatch, artifact: Mapping[str, Any] | None, seed: int
    ) -> np.ndarray:
        artifact = artifact or {}
        period = artifact.get("period") or batch.metadata.get("period") or self.period
        if period is None or int(period) < 2:
            raise ValueError("seasonal_lag requires a fitted artifact or configured period")
        period = int(period)
        profiles = artifact.get("profiles")
        output = np.array(batch.values, copy=True)
        for sample in range(batch.shape[0]):
            for channel in range(batch.shape[2]):
                source = batch.values[sample, :, channel]
                missing_indices = np.flatnonzero(~np.isfinite(source))
                for index in missing_indices:
                    historical = source[index % period : index : period]
                    historical = historical[np.isfinite(historical)]
                    if historical.size:
                        output[sample, index, channel] = float(np.median(historical))
                    elif profiles is not None:
                        value = np.asarray(profiles)[channel, index % period]
                        if np.isfinite(value):
                            output[sample, index, channel] = float(value)
        return output


def _positive_or(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) and value > 0 else float(default)


def _kalman_local_trend(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    n_steps = len(raw)
    observed = np.isfinite(raw)
    if n_steps == 0 or not observed.any():
        return np.array(raw, copy=True)
    filled = _linear_fill_1d(raw)
    differences = np.diff(filled)
    data_variance = _positive_or(float(np.var(filled)), 1.0)
    difference_variance = _positive_or(float(np.var(differences)), data_variance * 0.01)
    observation_variance = _positive_or(difference_variance * 0.1, data_variance * 0.01)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag(
        [
            _positive_or(difference_variance * 0.05, data_variance * 0.001),
            _positive_or(difference_variance * 0.005, data_variance * 0.0001),
        ]
    )
    observation = np.array([[1.0, 0.0]])
    identity = np.eye(2)
    slope = float(np.median(differences)) if differences.size else 0.0
    state = np.array([filled[0], slope if np.isfinite(slope) else 0.0])
    covariance = np.diag([data_variance, difference_variance + 1e-6])
    predicted_state = np.zeros((n_steps, 2))
    predicted_covariance = np.zeros((n_steps, 2, 2))
    filtered_state = np.zeros((n_steps, 2))
    filtered_covariance = np.zeros((n_steps, 2, 2))

    for time in range(n_steps):
        if time == 0:
            state_prediction, covariance_prediction = state, covariance
        else:
            state_prediction = transition @ filtered_state[time - 1]
            covariance_prediction = (
                transition @ filtered_covariance[time - 1] @ transition.T + process
            )
        predicted_state[time] = state_prediction
        predicted_covariance[time] = covariance_prediction
        if observed[time]:
            innovation = raw[time] - float((observation @ state_prediction)[0])
            scale = float(
                (observation @ covariance_prediction @ observation.T)[0, 0]
                + observation_variance
            )
            scale = max(scale, 1e-12) if np.isfinite(scale) else 1e-12
            gain = (covariance_prediction @ observation.T / scale).reshape(2)
            filtered_state[time] = state_prediction + gain * innovation
            filtered_covariance[time] = (
                identity - gain[:, None] @ observation
            ) @ covariance_prediction
        else:
            filtered_state[time] = state_prediction
            filtered_covariance[time] = covariance_prediction
        filtered_covariance[time] = (
            filtered_covariance[time] + filtered_covariance[time].T
        ) / 2.0

    smoothed_state = filtered_state.copy()
    for time in range(n_steps - 2, -1, -1):
        gain = (
            filtered_covariance[time]
            @ transition.T
            @ np.linalg.pinv(predicted_covariance[time + 1])
        )
        smoothed_state[time] = filtered_state[time] + gain @ (
            smoothed_state[time + 1] - predicted_state[time + 1]
        )
    output = filled.copy()
    output[~observed] = smoothed_state[~observed, 0]
    return output


class KalmanLocalTrendImputer(BaseImputer):
    imputer_id = "kalman_local_trend"

    def _impute_native(self, batch: SeriesBatch, artifact: Any, seed: int) -> np.ndarray:
        return _apply_per_channel(batch, _kalman_local_trend)


def _fit_ar_coefficients(filled: np.ndarray, max_lag: int) -> tuple[np.ndarray, float, float]:
    n_steps = len(filled)
    mean = float(np.mean(filled)) if n_steps else 0.0
    if n_steps < 4:
        return np.array([0.8]), mean, 1.0
    standard_deviation = float(np.std(filled))
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-8:
        standard_deviation = 1.0
    normalized = (filled - mean) / standard_deviation
    order = min(max_lag, max(1, n_steps // 10))
    features = np.asarray(
        [normalized[t - order : t][::-1] for t in range(order, n_steps)], dtype=float
    )
    targets = normalized[order:]
    try:
        coefficients = np.linalg.lstsq(features, targets, rcond=None)[0]
    except np.linalg.LinAlgError:
        coefficients = np.r_[0.8, np.zeros(order - 1)]
    if not np.all(np.isfinite(coefficients)):
        coefficients = np.r_[0.8, np.zeros(order - 1)]
    absolute_sum = float(np.sum(np.abs(coefficients)))
    if absolute_sum >= 0.98:
        coefficients *= 0.98 / absolute_sum
    residual = targets - features @ coefficients
    residual_variance = _positive_or(float(np.var(residual)), 0.01)
    return coefficients, mean, standard_deviation * np.sqrt(residual_variance)


def _kalman_ar(values: np.ndarray, max_lag: int) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    observed = np.isfinite(raw)
    if not observed.any():
        return np.array(raw, copy=True)
    filled = _linear_fill_1d(raw)
    coefficients, mean, residual_scale = _fit_ar_coefficients(filled, max_lag)
    standard_deviation = float(np.std(filled))
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-8:
        standard_deviation = 1.0
    normalized_raw = (raw - mean) / standard_deviation
    normalized_filled = (filled - mean) / standard_deviation
    order = len(coefficients)
    transition = np.zeros((order, order))
    transition[0] = coefficients
    if order > 1:
        transition[1:, :-1] = np.eye(order - 1)
    process = np.zeros((order, order))
    process[0, 0] = _positive_or((residual_scale / standard_deviation) ** 2, 0.01)
    observation = np.zeros((1, order))
    observation[0, 0] = 1.0
    observation_variance = max(process[0, 0] * 0.1, 1e-5)
    identity = np.eye(order)
    state = np.zeros(order)
    initial = normalized_filled[:order]
    state[: len(initial)] = initial[::-1]
    covariance = np.eye(order) * max(float(np.var(normalized_filled)), 1.0)
    n_steps = len(raw)
    predicted_state = np.zeros((n_steps, order))
    predicted_covariance = np.zeros((n_steps, order, order))
    filtered_state = np.zeros((n_steps, order))
    filtered_covariance = np.zeros((n_steps, order, order))

    for time in range(n_steps):
        if time == 0:
            state_prediction, covariance_prediction = state, covariance
        else:
            state_prediction = transition @ filtered_state[time - 1]
            covariance_prediction = (
                transition @ filtered_covariance[time - 1] @ transition.T + process
            )
        predicted_state[time] = state_prediction
        predicted_covariance[time] = covariance_prediction
        if observed[time]:
            innovation = normalized_raw[time] - float((observation @ state_prediction)[0])
            scale = float(
                (observation @ covariance_prediction @ observation.T)[0, 0]
                + observation_variance
            )
            scale = max(scale, 1e-12) if np.isfinite(scale) else 1e-12
            gain = (covariance_prediction @ observation.T / scale).reshape(order)
            filtered_state[time] = state_prediction + gain * innovation
            filtered_covariance[time] = (
                identity - gain[:, None] @ observation
            ) @ covariance_prediction
        else:
            filtered_state[time] = state_prediction
            filtered_covariance[time] = covariance_prediction
        filtered_covariance[time] = (
            filtered_covariance[time] + filtered_covariance[time].T
        ) / 2.0

    smoothed_state = filtered_state.copy()
    for time in range(n_steps - 2, -1, -1):
        gain = (
            filtered_covariance[time]
            @ transition.T
            @ np.linalg.pinv(predicted_covariance[time + 1])
        )
        smoothed_state[time] = filtered_state[time] + gain @ (
            smoothed_state[time + 1] - predicted_state[time + 1]
        )
    output = filled.copy()
    output[~observed] = mean + standard_deviation * smoothed_state[~observed, 0]
    return output


class KalmanARImputer(BaseImputer):
    imputer_id = "kalman_ar"

    def __init__(self, max_lag: int = 3) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be positive")
        self.max_lag = int(max_lag)

    def _impute_native(self, batch: SeriesBatch, artifact: Any, seed: int) -> np.ndarray:
        return _apply_per_channel(batch, lambda values: _kalman_ar(values, self.max_lag))


def _stl_baseline(filled: np.ndarray, period: int | None) -> np.ndarray:
    if period is None or period < 2 or len(filled) < 2 * period:
        return filled.copy()
    try:
        from statsmodels.tsa.seasonal import STL

        result = STL(filled, period=period, robust=True).fit()
        return np.asarray(result.seasonal + result.trend, dtype=float)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        kernel = np.ones(period, dtype=float) / period
        padded = np.pad(filled, (period // 2, period - 1 - period // 2), mode="edge")
        trend = np.convolve(padded, kernel, mode="valid")
        detrended = filled - trend
        profile = np.array(
            [np.median(detrended[phase::period]) for phase in range(period)], dtype=float
        )
        profile -= np.mean(profile)
        return trend + profile[np.arange(len(filled)) % period]


class STLKalmanImputer(BaseImputer):
    """STL baseline followed by local-trend Kalman residual smoothing."""

    imputer_id = "stl_kalman"

    def __init__(self, period: int | None = 24) -> None:
        if period is not None and period < 2:
            raise ValueError("period must be at least 2")
        self.period = period

    def _impute_native(
        self, batch: SeriesBatch, artifact: Mapping[str, Any] | None, seed: int
    ) -> np.ndarray:
        artifact = artifact or {}
        period = artifact.get("period") or batch.metadata.get("period") or self.period

        def fill(values: np.ndarray) -> np.ndarray:
            observed = np.isfinite(values)
            if not observed.any():
                return np.array(values, copy=True)
            initial = _linear_fill_1d(values)
            baseline = _stl_baseline(initial, int(period) if period is not None else None)
            residual = values - baseline
            residual_filled = _kalman_local_trend(residual)
            output = np.array(values, copy=True)
            output[~observed] = (baseline + residual_filled)[~observed]
            return output

        return _apply_per_channel(batch, fill)


def _rbf_kernel(
    left: np.ndarray, right: np.ndarray, length_scale: float, variance: float
) -> np.ndarray:
    distance = (left[:, None] - right[None, :]) ** 2
    return variance * np.exp(-0.5 * distance / max(length_scale, 1e-8) ** 2)


class GPRBFImputer(BaseImputer):
    imputer_id = "gp_rbf"

    def __init__(self, max_train_points: int = 512, noise: float = 1e-4) -> None:
        if max_train_points < 2 or noise <= 0:
            raise ValueError("max_train_points must be >=2 and noise must be positive")
        self.max_train_points = int(max_train_points)
        self.noise = float(noise)

    def _impute_native(
        self, batch: SeriesBatch, artifact: Any, seed: int
    ) -> NativeImputation:
        rng = np.random.default_rng(seed)
        output = np.array(batch.values, copy=True)
        uncertainty = np.zeros(batch.shape, dtype=float)
        for sample in range(batch.shape[0]):
            for channel in range(batch.shape[2]):
                values = batch.values[sample, :, channel]
                observed_indices = np.flatnonzero(np.isfinite(values))
                missing_indices = np.flatnonzero(~np.isfinite(values))
                if missing_indices.size == 0:
                    continue
                if observed_indices.size < 2:
                    output[sample, missing_indices, channel] = np.nan
                    uncertainty[sample, missing_indices, channel] = np.nan
                    continue
                if observed_indices.size > self.max_train_points:
                    distance = np.min(
                        np.abs(observed_indices[:, None] - missing_indices[None, :]), axis=1
                    )
                    local_count = self.max_train_points // 2
                    local = observed_indices[np.argsort(distance)[:local_count]]
                    remaining = np.setdiff1d(observed_indices, local)
                    random_count = min(self.max_train_points - local.size, remaining.size)
                    sampled = rng.choice(remaining, size=random_count, replace=False)
                    selected = np.sort(np.r_[local, sampled])
                else:
                    selected = observed_indices
                denominator = max(len(values) - 1, 1)
                x_train = selected / denominator
                x_missing = missing_indices / denominator
                y_raw = values[selected]
                y_mean = float(np.mean(y_raw))
                y_scale = float(np.std(y_raw))
                if not np.isfinite(y_scale) or y_scale < 1e-8:
                    y_scale = 1.0
                y_train = (y_raw - y_mean) / y_scale
                spacing = float(np.median(np.diff(x_train))) if selected.size > 2 else 0.01
                length_scale = max(spacing * 10.0, 0.03)
                variance = _positive_or(float(np.var(y_train)), 1.0)
                covariance = _rbf_kernel(x_train, x_train, length_scale, variance)
                covariance[np.diag_indices_from(covariance)] += self.noise + 1e-8
                try:
                    cholesky = np.linalg.cholesky(covariance)
                    alpha = np.linalg.solve(
                        cholesky.T, np.linalg.solve(cholesky, y_train)
                    )
                    cross = _rbf_kernel(x_missing, x_train, length_scale, variance)
                    prediction = y_mean + y_scale * (cross @ alpha)
                    projected = np.linalg.solve(cholesky, cross.T)
                    variance_prediction = np.maximum(
                        variance - np.sum(projected * projected, axis=0), 0.0
                    ) * (y_scale**2)
                except np.linalg.LinAlgError:
                    prediction = np.full(missing_indices.size, np.nan)
                    variance_prediction = np.full(missing_indices.size, np.nan)
                output[sample, missing_indices, channel] = prediction
                uncertainty[sample, missing_indices, channel] = variance_prediction
        return NativeImputation(output, uncertainty, {"seed": int(seed)})


__all__ = [
    "GPRBFImputer",
    "KalmanARImputer",
    "KalmanLocalTrendImputer",
    "LOCFImputer",
    "LinearInterpolationImputer",
    "STLKalmanImputer",
    "SeasonalLagImputer",
]
