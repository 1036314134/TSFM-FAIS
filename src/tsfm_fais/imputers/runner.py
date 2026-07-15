"""Failure-isolating execution for registered imputation candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from tsfm_fais.contracts import BudgetSpec, CandidateResult, CandidateStatus, SeriesBatch

from .base import ImputerDependencyError, failed_candidate_result
from .registry import DEFAULT_REGISTRY, ImputerRegistry


def _resident_memory() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


class CandidateRunner:
    """Run candidates independently so one optional method cannot abort a batch."""

    def __init__(self, registry: ImputerRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    def fit(
        self,
        imputer_id: str,
        train_batch: SeriesBatch,
        metadata: Mapping[str, Any] | None = None,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        availability = self.registry.availability(imputer_id)
        if not availability.available:
            missing = ", ".join(availability.missing)
            raise ImputerDependencyError(
                f"{imputer_id} is unavailable; missing dependencies: {missing}"
            )
        imputer = self.registry.create(imputer_id, **dict(params or {}))
        return imputer.fit(train_batch, metadata or {})

    fit_candidate = fit

    def run(
        self,
        imputer_id: str,
        batch: SeriesBatch,
        artifact: Any = None,
        *,
        seed: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> CandidateResult:
        started = perf_counter()
        memory_before = _resident_memory()
        availability = self.registry.availability(imputer_id)
        if not availability.available:
            missing = ", ".join(availability.missing)
            return failed_candidate_result(
                imputer_id,
                batch,
                f"missing dependencies: {missing}",
                status=CandidateStatus.UNAVAILABLE,
                runtime_seconds=perf_counter() - started,
                metadata={"missing_dependencies": availability.missing},
            )
        try:
            imputer = self.registry.create(imputer_id, **dict(params or {}))
            result = imputer.impute(batch, artifact, seed)
            if not isinstance(result, CandidateResult):
                raise TypeError("imputer must return CandidateResult")
            result.runtime_seconds = perf_counter() - started
            result.peak_memory_bytes = max(
                result.peak_memory_bytes, _resident_memory() - memory_before, 0
            )
            result.validate_against(batch)
        except ImputerDependencyError as error:
            return failed_candidate_result(
                imputer_id,
                batch,
                str(error),
                status=CandidateStatus.UNAVAILABLE,
                runtime_seconds=perf_counter() - started,
            )
        except Exception as error:
            return failed_candidate_result(
                imputer_id,
                batch,
                f"{type(error).__name__}: {error}",
                runtime_seconds=perf_counter() - started,
            )
        return result

    impute = run

    def fit_and_run(
        self,
        imputer_id: str,
        train_batch: SeriesBatch,
        batch: SeriesBatch,
        *,
        metadata: Mapping[str, Any] | None = None,
        seed: int = 0,
        params: Mapping[str, Any] | None = None,
    ) -> CandidateResult:
        try:
            artifact = self.fit(
                imputer_id, train_batch, metadata, params=params
            )
        except ImputerDependencyError as error:
            return failed_candidate_result(
                imputer_id,
                batch,
                str(error),
                status=CandidateStatus.UNAVAILABLE,
            )
        except Exception as error:
            return failed_candidate_result(
                imputer_id, batch, f"fit failed: {type(error).__name__}: {error}"
            )
        return self.run(imputer_id, batch, artifact, seed=seed, params=params)

    def run_many(
        self,
        imputer_ids: Sequence[str],
        batch: SeriesBatch,
        artifacts: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        artifact_failures: Mapping[str, str] | None = None,
        budget: BudgetSpec | None = None,
        runtime_already_spent: float = 0.0,
    ) -> dict[str, CandidateResult]:
        if runtime_already_spent < 0:
            raise ValueError("runtime_already_spent cannot be negative")
        limit = budget.max_candidates if budget is not None else len(imputer_ids)
        selected = tuple(imputer_ids[:limit])
        artifacts = artifacts or {}
        params = params or {}
        artifact_failures = artifact_failures or {}
        results: dict[str, CandidateResult] = {}
        elapsed = float(runtime_already_spent)
        for imputer_id in selected:
            spec = self.registry.get_spec(imputer_id)
            if budget is not None:
                supported_device = spec.device == "any" or spec.device in budget.allowed_devices
                if not supported_device:
                    results[imputer_id] = failed_candidate_result(
                        imputer_id,
                        batch,
                        f"device {spec.device!r} is excluded by the budget",
                        status=CandidateStatus.UNAVAILABLE,
                    )
                    continue
                if (
                    budget.max_runtime_seconds is not None
                    and elapsed >= budget.max_runtime_seconds
                ):
                    results[imputer_id] = failed_candidate_result(
                        imputer_id,
                        batch,
                        "runtime budget exhausted before candidate execution",
                        status=CandidateStatus.UNAVAILABLE,
                    )
                    continue
            artifact_failure = artifact_failures.get(imputer_id)
            if artifact_failure is not None:
                results[imputer_id] = failed_candidate_result(
                    imputer_id,
                    batch,
                    f"artifact load failed: {artifact_failure}",
                    metadata={"artifact_load_failure": artifact_failure},
                )
                continue
            result = self.run(
                imputer_id,
                batch,
                artifacts.get(imputer_id),
                seed=seed,
                params=params.get(imputer_id),
            )
            elapsed += result.runtime_seconds
            if (
                budget is not None
                and budget.max_memory_bytes is not None
                and result.peak_memory_bytes > budget.max_memory_bytes
            ):
                result.status = CandidateStatus.FAILED
                result.failure_reason = (
                    f"peak memory {result.peak_memory_bytes} exceeds budget "
                    f"{budget.max_memory_bytes}"
                )
                result.native_valid_mask[~batch.observed_mask] = False
            results[imputer_id] = result
        return results


ImputerRunner = CandidateRunner


__all__ = ["CandidateRunner", "ImputerRunner"]
