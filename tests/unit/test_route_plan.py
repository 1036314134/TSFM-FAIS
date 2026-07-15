from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsfm_fais import RoutePlan as PublicRoutePlan
from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    ImputerSpec,
    TimeSeriesItem,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner, ImputerRegistry
from tsfm_fais.imputers.base import failed_candidate_result
from tsfm_fais.pipeline import BlockwiseFAIS, FAISResult, RoutePlan
from tsfm_fais.routing.models import RouterBundle


def _item(*, tail: bool = False) -> tuple[TimeSeriesItem, np.ndarray]:
    time = np.arange(16, dtype=float)
    values = np.column_stack((time, np.sin(time), np.cos(time)))
    item = TimeSeriesItem(
        item_id="route-plan",
        values=values,
        variate_names=("trend", "sin", "cos"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"period": 4, "training_correlation": np.eye(3)},
    )
    mask = np.ones_like(values, dtype=bool)
    mask[4:7, 0] = False
    if tail:
        mask[-3:, 1] = False
    return item, mask


def _spec() -> ForecastSpec:
    return ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0,),
    )


def _manual(
    pipeline: BlockwiseFAIS,
    item: TimeSeriesItem,
    mask: np.ndarray,
    budget: BudgetSpec,
    *,
    seed: int,
) -> tuple[RoutePlan, FAISResult]:
    plan = pipeline.prepare_route(item, mask, _spec(), budget, seed=seed)
    actual = pipeline.candidate_runner.run_many(
        plan.shortlist,
        plan.batch,
        pipeline.imputer_artifacts,
        seed=seed,
        budget=plan.budget,
    )
    pseudo: dict[str, CandidateResult] = {}
    if pipeline.router is not None:
        assert plan.pseudo_batch is not None
        pseudo = pipeline.candidate_runner.run_many(
            plan.shortlist,
            plan.pseudo_batch,
            pipeline.imputer_artifacts,
            seed=seed,
            budget=plan.budget,
            runtime_already_spent=sum(
                result.runtime_seconds for result in actual.values()
            ),
        )
    return plan, pipeline.finish_route(plan, actual, pseudo)


def _assert_equivalent(left: FAISResult, right: FAISResult) -> None:
    np.testing.assert_allclose(left.values, right.values)
    np.testing.assert_array_equal(left.observed_mask, right.observed_mask)
    assert left.metadata == right.metadata
    assert left.routing.assignments == right.routing.assignments
    assert left.routing.shortlist == right.routing.shortlist
    assert left.routing.activated_candidates == right.routing.activated_candidates
    assert left.routing.candidate_costs == right.routing.candidate_costs
    assert left.routing.fallback_blocks == right.routing.fallback_blocks
    assert left.routing.fallback_records == right.routing.fallback_records
    assert left.routing.metadata == right.routing.metadata
    assert left.routing.predicted_unary == right.routing.predicted_unary
    assert left.routing.predicted_pairwise == right.routing.predicted_pairwise
    assert np.isclose(left.routing.risk_energy, right.routing.risk_energy)
    assert np.isclose(left.routing.cost_energy, right.routing.cost_energy)
    assert np.isclose(left.routing.total_energy, right.routing.total_energy)
    assert tuple(left.candidates) == tuple(right.candidates)
    for candidate_id in left.candidates:
        left_candidate = left.candidates[candidate_id]
        right_candidate = right.candidates[candidate_id]
        assert left_candidate.status is right_candidate.status
        np.testing.assert_allclose(left_candidate.values, right_candidate.values)
        np.testing.assert_array_equal(
            left_candidate.native_valid_mask,
            right_candidate.native_valid_mask,
        )


def test_public_impute_matches_manual_route_phases_for_light_candidates() -> None:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )
    item, mask = _item(tail=True)
    budget = BudgetSpec(max_candidates=2, max_active_candidates=1)

    wrapped = BlockwiseFAIS(imputer_registry=registry).impute(
        item, mask, _spec(), budget, seed=31
    )
    plan, phased = _manual(
        BlockwiseFAIS(imputer_registry=registry),
        item,
        mask,
        budget,
        seed=31,
    )

    assert plan.candidate_ids == ("locf", "linear_interp")
    assert PublicRoutePlan is RoutePlan
    assert plan.shortlist == wrapped.routing.shortlist
    _assert_equivalent(wrapped, phased)


class _RecordingRanker:
    def __init__(self, column: int) -> None:
        self.column = column
        self.inputs: list[np.ndarray] = []

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        self.inputs.append(values.copy())
        return values[:, self.column]


def _router() -> RouterBundle:
    feature_names = (
        "proxy_mae",
        "proxy_rmse",
        "mean_uncertainty",
        "candidate_id::locf",
        "candidate_id::linear_interp",
    )
    return RouterBundle(
        prior=_RecordingRanker(4),  # type: ignore[arg-type]
        unary=_RecordingRanker(2),  # type: ignore[arg-type]
        pairwise=SimpleNamespace(model=None),  # type: ignore[arg-type]
        feature_names=feature_names,
        candidate_ids=("locf", "linear_interp"),
        metadata={"unary_risk_scale": 2.0},
    )


class _SeededRunMany:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        imputer_ids,
        batch,
        artifacts=None,
        *,
        seed=0,
        params=None,
        budget=None,
        runtime_already_spent=0.0,
    ):
        del artifacts, params
        self.calls.append(
            {
                "ids": tuple(imputer_ids),
                "seed": seed,
                "budget": budget,
                "runtime_already_spent": runtime_already_spent,
                "mask": batch.observed_mask.copy(),
            }
        )
        results: dict[str, CandidateResult] = {}
        for index, candidate_id in enumerate(imputer_ids):
            rng = np.random.default_rng(seed + index + 1)
            values = batch.values.copy()
            hidden = ~batch.observed_mask
            values[hidden] = rng.normal(loc=index + 1.0, scale=0.1, size=hidden.sum())
            uncertainty = np.full(batch.shape, 0.25 + index, dtype=float)
            results[candidate_id] = CandidateResult(
                candidate_id,
                values,
                np.ones(batch.shape, dtype=bool),
                uncertainty=uncertainty,
                runtime_seconds=0.2 + 0.1 * index,
            )
        return results


def _mock_pipeline() -> tuple[BlockwiseFAIS, _SeededRunMany]:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )
    pipeline = BlockwiseFAIS(router=_router(), imputer_registry=registry)
    runner = _SeededRunMany()
    pipeline.candidate_runner.run_many = runner  # type: ignore[method-assign]
    return pipeline, runner


def test_route_phases_preserve_seeded_results_r1_uncertainty_and_budget() -> None:
    item, mask = _item()
    budget = BudgetSpec(
        max_candidates=2,
        max_active_candidates=1,
        max_runtime_seconds=9.0,
        max_memory_bytes=123456,
        allowed_devices=("cpu",),
    )
    wrapped_pipeline, wrapped_runner = _mock_pipeline()
    phased_pipeline, phased_runner = _mock_pipeline()

    wrapped = wrapped_pipeline.impute(item, mask, _spec(), budget, seed=17)
    plan, phased = _manual(
        phased_pipeline, item, mask, budget, seed=17
    )

    _assert_equivalent(wrapped, phased)
    assert plan.pseudo_batch is not None
    assert not np.array_equal(plan.pseudo_batch.observed_mask, plan.batch.observed_mask)
    assert len(wrapped_runner.calls) == len(phased_runner.calls) == 2
    for calls in (wrapped_runner.calls, phased_runner.calls):
        assert calls[0]["budget"] is budget
        assert calls[1]["budget"] is budget
        assert calls[0]["runtime_already_spent"] == 0.0
        assert np.isclose(float(calls[1]["runtime_already_spent"]), 0.5)
    unary = wrapped_pipeline.router.unary  # type: ignore[union-attr]
    assert isinstance(unary, _RecordingRanker)
    assert unary.inputs
    uncertainty_column = unary.inputs[0][:, 2]
    assert np.any(uncertainty_column > 0)


def test_route_phases_preserve_candidate_failures_and_fallback(monkeypatch) -> None:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )
    item, mask = _item(tail=True)

    def fail_many(imputer_ids, batch, *_args, **_kwargs):
        return {
            candidate_id: failed_candidate_result(
                candidate_id, batch, "synthetic candidate failure"
            )
            for candidate_id in imputer_ids
        }

    wrapped_pipeline = BlockwiseFAIS(imputer_registry=registry)
    phased_pipeline = BlockwiseFAIS(imputer_registry=registry)
    monkeypatch.setattr(wrapped_pipeline.candidate_runner, "run_many", fail_many)
    monkeypatch.setattr(phased_pipeline.candidate_runner, "run_many", fail_many)
    budget = BudgetSpec(max_candidates=2)

    wrapped = wrapped_pipeline.impute(item, mask, _spec(), budget, seed=9)
    _, phased = _manual(phased_pipeline, item, mask, budget, seed=9)

    _assert_equivalent(wrapped, phased)
    assert wrapped.routing.fallback_blocks
    assert all(
        candidate.status is CandidateStatus.FAILED
        for candidate in wrapped.candidates.values()
    )


def test_route_phases_preserve_runtime_budget_exhaustion() -> None:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )
    item, mask = _item()
    budget = BudgetSpec(max_candidates=2, max_runtime_seconds=1e-12)

    wrapped = BlockwiseFAIS(imputer_registry=registry).impute(
        item, mask, _spec(), budget, seed=3
    )
    plan, phased = _manual(
        BlockwiseFAIS(imputer_registry=registry),
        item,
        mask,
        budget,
        seed=3,
    )

    _assert_equivalent(wrapped, phased)
    assert plan.budget.max_runtime_seconds == 1e-12
    assert wrapped.candidates["locf"].status is CandidateStatus.SUCCESS
    assert wrapped.candidates["linear_interp"].status is CandidateStatus.UNAVAILABLE


def test_prepare_route_rebuilds_shortlist_from_explicit_available_artifacts(
    monkeypatch,
) -> None:
    registry = ImputerRegistry(
        (
            DEFAULT_REGISTRY.get_spec("locf"),
            ImputerSpec(
                imputer_id="fitted",
                family="test",
                mode="joint_multivariate",
                factory="tsfm_fais.imputers.structured:KNNMultivariateImputer",
                fit_scope="dataset",
            ),
        )
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forced_candidates = ("locf",)
    pipeline.shortlist_size = 2
    monkeypatch.setattr(
        pipeline,
        "_ensure_item_artifacts",
        lambda _item: (_ for _ in ()).throw(AssertionError("unexpected eager load")),
    )
    item, mask = _item()

    before = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=2),
        available_artifact_ids=("fitted",),
    )
    after = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=2),
        available_artifact_ids=(),
        artifact_load_failures={"fitted": "synthetic load failure"},
    )

    assert before.candidate_ids == ("locf", "fitted")
    assert "fitted" in before.shortlist
    assert after.candidate_ids == ("locf",)
    assert after.shortlist == ("locf",)
    assert after.artifact_load_failures == {"fitted": "synthetic load failure"}


def test_finish_route_uses_precomputed_non_shortlist_fallback(monkeypatch) -> None:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )
    pipeline = BlockwiseFAIS(
        imputer_registry=registry,
        fallback_internal=("linear_interp", "locf", "train_median"),
    )
    pipeline.forced_candidates = ("locf",)
    pipeline.shortlist_size = 1
    item, mask = _item()
    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=1),
        seed=5,
        available_artifact_ids=(),
    )
    assert plan.shortlist == ("locf",)
    assert plan.allow_fallback_execution is False
    failed = failed_candidate_result("locf", plan.batch, "route failure")
    linear = CandidateRunner(registry).run("linear_interp", plan.batch, seed=5)
    monkeypatch.setattr(
        pipeline.candidate_runner,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fallback should use precomputed output")
        ),
    )

    result = pipeline.finish_route(
        plan,
        {"locf": failed, "linear_interp": linear},
        fallback_candidates={"linear_interp": linear},
    )

    assert set(result.routing.assignments.values()) == {"linear_interp"}
    assert tuple(result.candidates) == ("locf", "linear_interp")
    assert next(iter(result.routing.fallback_records.values()))["attempts"] == (
        "linear_interp",
    )


def test_finish_route_rejects_missing_or_mismatched_injected_results() -> None:
    registry = ImputerRegistry((DEFAULT_REGISTRY.get_spec("locf"),))
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forced_candidates = ("locf",)
    item, mask = _item()
    plan = pipeline.prepare_route(
        item, mask, _spec(), BudgetSpec(max_candidates=1)
    )

    try:
        pipeline.finish_route(plan, {})
    except ValueError as error:
        assert "missing actual candidate results" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing candidate output was accepted")

    invalid = CandidateResult(
        "different",
        plan.batch.values.copy(),
        np.ones(plan.batch.shape, dtype=bool),
    )
    try:
        pipeline.finish_route(plan, {"locf": invalid})
    except ValueError as error:
        assert "contains 'different'" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("mismatched candidate output was accepted")
