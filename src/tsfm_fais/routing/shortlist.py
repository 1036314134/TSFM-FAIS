"""Budget-aware candidate shortlist construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ImputerSpec,
)


@dataclass(frozen=True)
class ShortlistResult:
    selected: tuple[str, ...]
    excluded: Mapping[str, str] = field(default_factory=dict)
    estimated_runtime_seconds: float = 0.0


class CandidateShortlister:
    """Filter unavailable candidates, then rank lower predicted cost first."""

    def __init__(self, *, fallback_candidate: str | None = None) -> None:
        self.fallback_candidate = fallback_candidate

    @staticmethod
    def _exclusion_reason(
        candidate_id: str,
        budget: BudgetSpec,
        result: CandidateResult | None,
        spec: ImputerSpec | None,
    ) -> str | None:
        if result is not None and result.status in {
            CandidateStatus.FAILED,
            CandidateStatus.UNAVAILABLE,
        }:
            return f"status={result.status.value}"
        if spec is not None and spec.device != "any" and spec.device not in budget.allowed_devices:
            return f"device={spec.device}"
        if (
            budget.max_memory_bytes is not None
            and result is not None
            and result.peak_memory_bytes > budget.max_memory_bytes
        ):
            return "memory_budget"
        return None

    def select(
        self,
        candidate_ids: Sequence[str],
        predicted_costs: Mapping[str, float],
        budget: BudgetSpec,
        *,
        results: Mapping[str, CandidateResult] | None = None,
        specs: Mapping[str, ImputerSpec] | None = None,
    ) -> ShortlistResult:
        unique = tuple(dict.fromkeys(candidate_ids))
        if not unique:
            raise ValueError("candidate shortlist cannot be empty")
        missing_scores = [candidate for candidate in unique if candidate not in predicted_costs]
        if missing_scores:
            raise ValueError(f"missing predicted costs for candidates: {missing_scores}")

        excluded: dict[str, str] = {}
        feasible: list[str] = []
        for candidate_id in unique:
            reason = self._exclusion_reason(
                candidate_id,
                budget,
                (results or {}).get(candidate_id),
                (specs or {}).get(candidate_id),
            )
            if reason:
                excluded[candidate_id] = reason
            else:
                feasible.append(candidate_id)
        if not feasible:
            raise RuntimeError("no candidate satisfies status and resource constraints")

        feasible.sort(key=lambda candidate: (float(predicted_costs[candidate]), candidate))
        if self.fallback_candidate in feasible:
            feasible.remove(self.fallback_candidate)
            feasible.insert(0, self.fallback_candidate)

        selected: list[str] = []
        cumulative_runtime = 0.0
        for candidate_id in feasible:
            if len(selected) >= budget.max_candidates:
                excluded[candidate_id] = "candidate_budget"
                continue
            runtime = max(0.0, float((results or {}).get(candidate_id, _ZERO_RESULT).runtime_seconds))
            if (
                budget.max_runtime_seconds is not None
                and cumulative_runtime + runtime > budget.max_runtime_seconds
            ):
                excluded[candidate_id] = "runtime_budget"
                continue
            selected.append(candidate_id)
            cumulative_runtime += runtime
        if not selected:
            raise RuntimeError("runtime budget excludes every feasible candidate")
        return ShortlistResult(tuple(selected), excluded, cumulative_runtime)


_ZERO_RESULT = CandidateResult(
    imputer_id="__zero__",
    values=__import__("numpy").empty((0, 0, 0)),
    native_valid_mask=__import__("numpy").empty((0, 0, 0), dtype=bool),
)


def shortlist_candidates(
    candidate_ids: Sequence[str],
    predicted_costs: Mapping[str, float],
    budget: BudgetSpec,
    *,
    results: Mapping[str, CandidateResult] | None = None,
    specs: Mapping[str, ImputerSpec] | None = None,
) -> tuple[str, ...]:
    return CandidateShortlister().select(
        candidate_ids,
        predicted_costs,
        budget,
        results=results,
        specs=specs,
    ).selected
