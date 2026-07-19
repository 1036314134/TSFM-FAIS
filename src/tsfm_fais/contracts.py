"""Shared contracts for data, imputation, forecasting, and routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd


class CandidateStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TimeSeriesItem:
    """One aligned multivariate trajectory."""

    item_id: str
    values: np.ndarray
    variate_names: tuple[str, ...]
    start: pd.Timestamp
    freq: str
    timestamps: pd.DatetimeIndex | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"TimeSeriesItem.values must have shape [T,D], got {values.shape}")
        if values.shape[1] < 1:
            raise ValueError("TimeSeriesItem.values must contain at least one variate")
        if values.shape[1] != len(self.variate_names):
            raise ValueError("variate_names length must match values.shape[1]")
        if any(not str(name).strip() for name in self.variate_names):
            raise ValueError("variate_names cannot contain empty names")
        if self.timestamps is not None and len(self.timestamps) != values.shape[0]:
            raise ValueError("timestamps length must match values.shape[0]")
        if not str(self.item_id).strip():
            raise ValueError("item_id cannot be empty")
        if not str(self.freq).strip():
            raise ValueError("freq cannot be empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        if self.timestamps is not None:
            object.__setattr__(self, "timestamps", pd.DatetimeIndex(self.timestamps))


@dataclass(frozen=True)
class SeriesBatch:
    """Fixed-length windows with an explicit observation mask."""

    values: np.ndarray
    observed_mask: np.ndarray
    item_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        mask = np.asarray(self.observed_mask, dtype=bool)
        if values.ndim != 3:
            raise ValueError(f"SeriesBatch.values must have shape [N,L,D], got {values.shape}")
        if mask.shape != values.shape:
            raise ValueError("observed_mask must have the same shape as values")
        if np.any(~np.isfinite(values[mask])):
            raise ValueError("observed values must be finite")
        normalized = values.copy()
        normalized[~mask] = np.nan
        if self.item_ids and len(self.item_ids) != values.shape[0]:
            raise ValueError("item_ids length must match batch size")
        object.__setattr__(self, "values", normalized)
        object.__setattr__(self, "observed_mask", mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape


@dataclass(frozen=True, order=True)
class MissingBlock:
    block_id: str
    batch_index: int
    channel: int
    start: int
    end: int
    pattern: str = "unknown"

    def __post_init__(self) -> None:
        if self.batch_index < 0 or self.channel < 0:
            raise ValueError("batch_index and channel must be non-negative")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("MissingBlock uses a non-empty [start,end) interval")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ImputerSpec:
    imputer_id: str
    family: str
    mode: Literal["per_channel", "joint_multivariate"]
    factory: str
    version: str = "1"
    source: str = "project"
    fit_scope: Literal["none", "dataset", "online"] = "none"
    supports_tail: bool = True
    requires_period: bool = False
    stochastic: bool = False
    device: Literal["cpu", "gpu", "any"] = "cpu"
    cost_tier: int = 1
    max_fit_variates: int | None = None
    optional_extra: str | None = None
    dependencies: tuple[str, ...] = ()
    default_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.imputer_id:
            raise ValueError("imputer_id cannot be empty")
        if self.cost_tier < 1:
            raise ValueError("cost_tier must be positive")
        if self.max_fit_variates is not None and self.max_fit_variates < 2:
            raise ValueError("max_fit_variates must be at least two")


@dataclass
class CandidateResult:
    imputer_id: str
    values: np.ndarray
    native_valid_mask: np.ndarray
    uncertainty: np.ndarray | None = None
    runtime_seconds: float = 0.0
    peak_memory_bytes: int = 0
    status: CandidateStatus = CandidateStatus.SUCCESS
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_against(self, batch: SeriesBatch) -> None:
        values = np.asarray(self.values, dtype=float)
        valid = np.asarray(self.native_valid_mask, dtype=bool)
        if values.shape != batch.shape or valid.shape != batch.shape:
            raise ValueError("candidate values and valid mask must match the input batch")
        if np.any(~np.isfinite(values)):
            raise ValueError("candidate output must be finite after safe completion")
        if not np.array_equal(values[batch.observed_mask], batch.values[batch.observed_mask]):
            raise ValueError("candidate changed an observed value")
        if self.uncertainty is not None and np.asarray(self.uncertainty).shape != batch.shape:
            raise ValueError("uncertainty must match the input batch")


@dataclass(frozen=True)
class ForecastSpec:
    model_id: str
    mode: Literal["joint_multivariate", "independent_univariate"]
    horizon: int
    context_length: int | None = None
    target_indices: tuple[int, ...] | None = None
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    num_samples: int = 100

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.context_length is not None and self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.target_indices is not None and len(set(self.target_indices)) != len(
            self.target_indices
        ):
            raise ValueError("target_indices must be unique")
        if self.target_indices is not None:
            if not self.target_indices:
                raise ValueError("target_indices cannot be empty")
            if any(index < 0 for index in self.target_indices):
                raise ValueError("target_indices must be non-negative")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not self.quantile_levels:
            raise ValueError("quantile_levels cannot be empty")
        if any(not 0 < level < 1 for level in self.quantile_levels):
            raise ValueError("quantile_levels must lie strictly between zero and one")
        if tuple(sorted(set(self.quantile_levels))) != self.quantile_levels:
            raise ValueError("quantile_levels must be unique and increasing")


@dataclass
class ForecastResult:
    point: np.ndarray
    target_indices: tuple[int, ...]
    quantiles: np.ndarray | None = None
    samples: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float)
        if self.point.ndim != 3:
            raise ValueError("point forecasts must have shape [N,H,K]")
        if not np.isfinite(self.point).all():
            raise ValueError("point forecasts must be finite")
        if self.point.shape[2] != len(self.target_indices):
            raise ValueError("target_indices length must match forecast target dimension")
        if self.quantiles is not None:
            self.quantiles = np.asarray(self.quantiles, dtype=float)
            if self.quantiles.ndim != 4 or self.quantiles.shape[:3] != self.point.shape:
                raise ValueError("quantiles must have shape [N,H,K,Q]")
            if not np.isfinite(self.quantiles).all():
                raise ValueError("quantile forecasts must be finite")
        if self.samples is not None:
            self.samples = np.asarray(self.samples, dtype=float)
            if self.samples.ndim != 4:
                raise ValueError("samples must have shape [N,S,H,K]")
            if (
                self.samples.shape[0] != self.point.shape[0]
                or self.samples.shape[2:] != self.point.shape[1:]
            ):
                raise ValueError("sample forecast dimensions do not match point forecasts")
            if not np.isfinite(self.samples).all():
                raise ValueError("sample forecasts must be finite")


@dataclass(frozen=True)
class BudgetSpec:
    max_candidates: int = 6
    max_active_candidates: int | None = None
    max_runtime_seconds: float | None = None
    max_memory_bytes: int | None = None
    allowed_devices: tuple[str, ...] = ("cpu", "gpu")

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.max_active_candidates is not None and not (
            1 <= self.max_active_candidates <= self.max_candidates
        ):
            raise ValueError("max_active_candidates must be within max_candidates")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.max_memory_bytes is not None and self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if not self.allowed_devices:
            raise ValueError("allowed_devices cannot be empty")
        if len(set(self.allowed_devices)) != len(self.allowed_devices):
            raise ValueError("allowed_devices must be unique")
        if any(device not in {"cpu", "gpu"} for device in self.allowed_devices):
            raise ValueError("allowed_devices entries must be 'cpu' or 'gpu'")


@dataclass
class RoutingResult:
    assignments: dict[str, str]
    shortlist: tuple[str, ...]
    total_energy: float
    predicted_unary: dict[tuple[str, str], float] = field(default_factory=dict)
    predicted_pairwise: dict[tuple[str, str, str, str], float] = field(default_factory=dict)
    activated_candidates: tuple[str, ...] = ()
    fallback_blocks: tuple[str, ...] = ()
    candidate_costs: dict[str, float] = field(default_factory=dict)
    activated_cost: float = 0.0
    risk_energy: float = 0.0
    cost_energy: float = 0.0
    fallback_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
