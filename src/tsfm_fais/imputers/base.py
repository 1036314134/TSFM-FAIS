"""Common imputer interfaces and candidate-output assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tsfm_fais.contracts import CandidateResult, CandidateStatus, SeriesBatch


class ImputerDependencyError(ImportError):
    """Raised when an optional imputer dependency is not installed."""


@runtime_checkable
class ImputerProtocol(Protocol):
    """Lifecycle implemented by every imputation candidate."""

    imputer_id: str

    def fit(
        self,
        train_batch: SeriesBatch,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Fit using training-fold data and return a frozen artifact."""

    def impute(
        self,
        batch: SeriesBatch,
        artifact: Any,
        seed: int = 0,
    ) -> CandidateResult:
        """Impute a batch without updating the fitted artifact."""


@dataclass(frozen=True)
class NativeImputation:
    """Raw algorithm output before common safety completion."""

    values: np.ndarray
    uncertainty: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None


def _channel_defaults(batch: SeriesBatch) -> np.ndarray:
    defaults = np.zeros(batch.shape[2], dtype=float)
    for channel in range(batch.shape[2]):
        values = batch.values[:, :, channel]
        observed = batch.observed_mask[:, :, channel]
        finite = values[observed & np.isfinite(values)]
        if finite.size:
            defaults[channel] = float(np.median(finite))
    return defaults


def deterministic_safe_values(batch: SeriesBatch) -> np.ndarray:
    """Return a finite deterministic completion used only after native failure.

    Each trajectory/channel is interpolated on its time index. A channel median
    is used when that individual trajectory has no observation, and zero is used
    only when the entire batch has no observation for the channel.
    """

    safe = np.array(batch.values, dtype=float, copy=True)
    defaults = _channel_defaults(batch)
    time = np.arange(batch.shape[1], dtype=float)
    for sample in range(batch.shape[0]):
        for channel in range(batch.shape[2]):
            observed = batch.observed_mask[sample, :, channel]
            if observed.any():
                safe[sample, :, channel] = np.interp(
                    time,
                    time[observed],
                    batch.values[sample, observed, channel],
                )
            else:
                safe[sample, :, channel] = defaults[channel]
    safe[batch.observed_mask] = batch.values[batch.observed_mask]
    return safe


def assemble_candidate_result(
    imputer_id: str,
    batch: SeriesBatch,
    native_values: np.ndarray,
    *,
    uncertainty: np.ndarray | None = None,
    runtime_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    failure_reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateResult:
    """Preserve observations, complete numerical failures, and retain validity."""

    native = np.asarray(native_values, dtype=float)
    if native.shape != batch.shape:
        raise ValueError(
            f"{imputer_id} returned shape {native.shape}; expected {batch.shape}"
        )

    native_valid = np.isfinite(native)
    native_valid[batch.observed_mask] = True
    values = np.array(native, copy=True)
    values[batch.observed_mask] = batch.values[batch.observed_mask]

    invalid = ~np.isfinite(values)
    if invalid.any():
        safe = deterministic_safe_values(batch)
        values[invalid] = safe[invalid]

    missing = ~batch.observed_mask
    missing_count = int(missing.sum())
    valid_missing_count = int((native_valid & missing).sum())
    if missing_count == 0 or valid_missing_count == missing_count:
        status = CandidateStatus.SUCCESS
    elif valid_missing_count == 0:
        status = CandidateStatus.FAILED
        failure_reason = failure_reason or "candidate produced no native missing values"
    else:
        status = CandidateStatus.PARTIAL
        failure_reason = failure_reason or "candidate produced partial native output"

    uncertainty_array: np.ndarray | None = None
    if uncertainty is not None:
        uncertainty_array = np.asarray(uncertainty, dtype=float)
        if uncertainty_array.shape != batch.shape:
            raise ValueError("uncertainty must have the same shape as the input batch")
        uncertainty_array = np.where(
            np.isfinite(uncertainty_array), np.maximum(uncertainty_array, 0.0), np.nan
        )
        uncertainty_array[batch.observed_mask] = 0.0

    result = CandidateResult(
        imputer_id=imputer_id,
        values=values,
        native_valid_mask=native_valid,
        uncertainty=uncertainty_array,
        runtime_seconds=max(0.0, float(runtime_seconds)),
        peak_memory_bytes=max(0, int(peak_memory_bytes)),
        status=status,
        failure_reason=failure_reason,
        metadata=dict(metadata or {}),
    )
    result.validate_against(batch)
    return result


def failed_candidate_result(
    imputer_id: str,
    batch: SeriesBatch,
    reason: str,
    *,
    status: CandidateStatus = CandidateStatus.FAILED,
    runtime_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateResult:
    """Build a shape-safe result for an exception or unavailable dependency."""

    values = deterministic_safe_values(batch)
    valid = np.array(batch.observed_mask, copy=True)
    result = CandidateResult(
        imputer_id=imputer_id,
        values=values,
        native_valid_mask=valid,
        runtime_seconds=max(0.0, float(runtime_seconds)),
        peak_memory_bytes=max(0, int(peak_memory_bytes)),
        status=status,
        failure_reason=reason,
        metadata=dict(metadata or {}),
    )
    result.validate_against(batch)
    return result


class BaseImputer:
    """Small template implementation shared by local candidates."""

    imputer_id = "base"

    def fit(
        self,
        train_batch: SeriesBatch,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._fit(train_batch, metadata or {})

    def _fit(self, train_batch: SeriesBatch, metadata: Mapping[str, Any]) -> Any:
        return None

    def impute(
        self,
        batch: SeriesBatch,
        artifact: Any = None,
        seed: int = 0,
    ) -> CandidateResult:
        started = perf_counter()
        native = self._impute_native(batch, artifact, seed)
        if isinstance(native, NativeImputation):
            output = native
        else:
            output = NativeImputation(values=np.asarray(native, dtype=float))
        return assemble_candidate_result(
            self.imputer_id,
            batch,
            output.values,
            uncertainty=output.uncertainty,
            runtime_seconds=perf_counter() - started,
            metadata=output.metadata,
        )

    def _impute_native(
        self,
        batch: SeriesBatch,
        artifact: Any,
        seed: int,
    ) -> NativeImputation | np.ndarray:
        raise NotImplementedError

