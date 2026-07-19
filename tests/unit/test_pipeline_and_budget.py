from __future__ import annotations

import numpy as np
import pandas as pd

from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    ImputerSpec,
    MissingBlock,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers import CandidateRunner, ImputerRegistry
from tsfm_fais.imputers.base import failed_candidate_result
from tsfm_fais.pipeline import BlockwiseFAIS, _pairwise_free_search
from tsfm_fais.routing.graph import BlockGraph
from tsfm_fais.routing.solver import exhaustive_search


def test_candidate_runner_enforces_device_and_runtime_budgets() -> None:
    values = np.arange(12, dtype=float).reshape(1, 6, 2)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2:4, 0] = False
    batch = SeriesBatch(values, mask)
    gpu_registry = ImputerRegistry(
        [
            ImputerSpec(
                imputer_id="locf",
                family="persistence",
                mode="per_channel",
                factory="tsfm_fais.imputers.classical:LOCFImputer",
                device="gpu",
            )
        ]
    )
    excluded = CandidateRunner(gpu_registry).run_many(
        ("locf",), batch, budget=BudgetSpec(allowed_devices=("cpu",))
    )
    assert excluded["locf"].status is CandidateStatus.UNAVAILABLE

    from tsfm_fais.imputers import DEFAULT_REGISTRY

    timed = CandidateRunner(DEFAULT_REGISTRY).run_many(
        ("locf", "linear_interp"),
        batch,
        budget=BudgetSpec(max_candidates=2, max_runtime_seconds=1e-12),
    )
    assert timed["locf"].status is CandidateStatus.SUCCESS
    assert timed["linear_interp"].status is CandidateStatus.UNAVAILABLE
    already_exhausted = CandidateRunner(DEFAULT_REGISTRY).run_many(
        ("locf",),
        batch,
        budget=BudgetSpec(max_runtime_seconds=0.5),
        runtime_already_spent=0.5,
    )
    assert already_exhausted["locf"].status is CandidateStatus.UNAVAILABLE


def test_candidate_validation_failure_is_isolated(monkeypatch) -> None:
    values = np.arange(12, dtype=float).reshape(1, 6, 2)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2:4, 0] = False
    batch = SeriesBatch(values, mask)

    class InvalidImputer:
        imputer_id = "locf"

        def impute(self, batch, artifact, seed):
            del artifact, seed
            return CandidateResult(
                "locf",
                np.zeros((1, 1, 1)),
                np.zeros((1, 1, 1), dtype=bool),
            )

    from tsfm_fais.imputers import DEFAULT_REGISTRY

    runner = CandidateRunner(DEFAULT_REGISTRY)
    monkeypatch.setattr(runner.registry, "create", lambda *args, **kwargs: InvalidImputer())
    result = runner.run("locf", batch)
    assert result.status is CandidateStatus.FAILED
    assert "must match the input batch" in result.failure_reason


def test_pipeline_forwards_configured_candidate_runtime_params(monkeypatch) -> None:
    values = np.arange(24, dtype=float).reshape(12, 2)
    item = TimeSeriesItem(
        item_id="params",
        values=values,
        variate_names=("a", "b"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )
    mask = np.ones_like(values, dtype=bool)
    mask[4:7, 0] = False
    configured = {"csdi": {"num_samples": 5}}
    pipeline = BlockwiseFAIS(candidate_params=configured)
    native = pipeline.candidate_runner.run_many
    calls = []

    def recording(*args, **kwargs):
        calls.append(kwargs.get("params"))
        kwargs["params"] = {}
        return native(*args, **kwargs)

    monkeypatch.setattr(pipeline.candidate_runner, "run_many", recording)
    pipeline.impute(
        item,
        mask,
        ForecastSpec("mock", "independent_univariate", 2, target_indices=(0,)),
        BudgetSpec(max_candidates=2),
    )

    assert calls == [configured]


def test_pipeline_uses_explicit_median_fallback_when_all_native_outputs_fail() -> None:
    time = np.arange(10, dtype=float)
    values = np.stack((time, 100.0 + time), axis=1)
    item = TimeSeriesItem(
        item_id="fallback",
        values=values,
        variate_names=("missing_channel", "observed_channel"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )
    mask = np.ones_like(values, dtype=bool)
    mask[:, 0] = False
    result = BlockwiseFAIS().impute(
        item,
        mask,
        ForecastSpec(
            model_id="mock",
            mode="independent_univariate",
            horizon=2,
            target_indices=(0,),
        ),
        BudgetSpec(max_candidates=2),
    )
    assert result.routing.fallback_blocks
    assert set(result.routing.assignments.values()) == {"context_median"}
    record = next(iter(result.routing.metadata["fallback_records"].values()))
    assert result.routing.fallback_records == result.routing.metadata["fallback_records"]
    assert record["kind"] == "tail"
    assert record["attempts"] == ("locf", "train_median")
    assert np.isfinite(result.values).all()
    np.testing.assert_array_equal(result.values[mask], values[mask])


def test_internal_fallback_follows_configured_order(monkeypatch) -> None:
    time = np.arange(12, dtype=float)
    values = np.stack((time, time**2), axis=1)
    item = TimeSeriesItem(
        item_id="ordered-fallback",
        values=values,
        variate_names=("a", "b"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"training_correlation": np.eye(2)},
    )
    mask = np.ones_like(values, dtype=bool)
    mask[4:7, 0] = False
    pipeline = BlockwiseFAIS()
    native_run = pipeline.candidate_runner.run

    def fail_locf(imputer_id, batch, artifact=None, *, seed=0, params=None):
        if imputer_id == "locf":
            return failed_candidate_result("locf", batch, "synthetic failure")
        return native_run(imputer_id, batch, artifact, seed=seed, params=params)

    monkeypatch.setattr(pipeline.candidate_runner, "run", fail_locf)
    result = pipeline.impute(
        item,
        mask,
        ForecastSpec("mock", "independent_univariate", 2, target_indices=(0,)),
        BudgetSpec(max_candidates=1),
    )
    assert set(result.routing.assignments.values()) == {"linear_interp"}
    record = next(iter(result.routing.metadata["fallback_records"].values()))
    assert record["kind"] == "internal"
    assert record["attempts"] == ("linear_interp",)
    assert result.routing.metadata["correlation_source"] == "training"
    assert result.routing.candidate_costs
    assert result.routing.activated_cost == 1.0
    assert np.isclose(result.routing.cost_energy, 0.0)
    assert np.isclose(
        result.routing.total_energy,
        result.routing.risk_energy + result.routing.cost_energy,
    )


def test_pseudo_blocks_do_not_overlap_in_time_across_channels() -> None:
    values = np.arange(3 * 60, dtype=float).reshape(1, 60, 3)
    batch = SeriesBatch(values, np.ones_like(values, dtype=bool))
    pseudo = BlockwiseFAIS()._pseudo_batch(batch, seed=17, max_blocks=8)
    newly_hidden = batch.observed_mask & ~pseudo.observed_mask
    assert np.all(newly_hidden.sum(axis=2) <= 1)


def test_pseudo_blocks_cover_priority_channels_with_matched_lengths() -> None:
    values = np.arange(3 * 96, dtype=float).reshape(1, 96, 3)
    batch = SeriesBatch(values, np.ones_like(values, dtype=bool))
    blocks = (
        MissingBlock("b0", 0, 0, 20, 32),
        MissingBlock("b2", 0, 2, 40, 46),
    )

    pseudo = BlockwiseFAIS()._pseudo_batch(
        batch,
        seed=17,
        max_blocks=2,
        target_blocks=blocks,
        priority_channels=(2, 0),
    )
    newly_hidden = batch.observed_mask & ~pseudo.observed_mask

    assert newly_hidden[..., 0].sum() > 0
    assert newly_hidden[..., 2].sum() > 0
    assert np.all(newly_hidden.sum(axis=2) <= 1)


def test_univariate_forecaster_uses_fast_selection_for_invisible_blocks() -> None:
    values = np.arange(36, dtype=float).reshape(12, 3)
    item = TimeSeriesItem(
        item_id="invisible",
        values=values,
        variate_names=("target", "other_a", "other_b"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )
    mask = np.ones_like(values, dtype=bool)
    mask[4:7, 2] = False

    result = BlockwiseFAIS().impute(
        item,
        mask,
        ForecastSpec("mock", "independent_univariate", 2, target_indices=(0,)),
        BudgetSpec(max_candidates=6),
    )

    assert result.routing.shortlist == ("locf", "linear_interp")
    assert set(result.routing.assignments.values()) == {"linear_interp"}
    assert result.routing.metadata["forecast_irrelevant_block_count"] == 1


def test_pipeline_skips_pairwise_features_for_large_block_sets() -> None:
    length, dimensions = 4, 32
    values = np.arange(length * dimensions, dtype=float).reshape(length, dimensions)
    item = TimeSeriesItem(
        item_id="many-blocks",
        values=values,
        variate_names=tuple(f"v{index}" for index in range(dimensions)),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"training_correlation": np.eye(dimensions)},
    )
    mask = np.ones_like(values, dtype=bool)
    mask[1, :] = False
    result = BlockwiseFAIS(max_pairwise_blocks=10).impute(
        item,
        mask,
        ForecastSpec("mock", "independent_univariate", 1, target_indices=(0,)),
        BudgetSpec(max_candidates=2),
    )
    assert result.routing.metadata["pairwise_skipped_for_scale"] is True
    assert result.routing.metadata["solver"] == "pairwise_free_subset"
    assert result.routing.metadata["block_graph_edges"] <= 3 * dimensions
    assert np.isfinite(result.values).all()


def test_pairwise_free_search_matches_exact_assignment_with_active_limit() -> None:
    blocks = tuple(MissingBlock(f"b{index}", 0, index, 0, 1) for index in range(4))
    candidates = ("a", "b", "c")
    unary = {
        (block.block_id, candidate): float((block.channel + candidate_index) % 4)
        for block in blocks
        for candidate_index, candidate in enumerate(candidates)
    }
    costs = {"a": 1.0, "b": 3.0, "c": 2.0}
    invalid = {("b0", "a"), ("b3", "c")}
    budget = BudgetSpec(max_candidates=3, max_active_candidates=2)

    scalable = _pairwise_free_search(
        blocks, candidates, unary, costs, budget, 0.2, invalid
    )
    exact = exhaustive_search(
        BlockGraph(blocks, ()),
        candidates,
        unary,
        costs=costs,
        budget=budget,
        cost_weight=0.2,
        invalid=invalid,
    )

    assert scalable.assignments == exact.assignments
    assert np.isclose(scalable.total_energy, exact.total_energy)


def test_fallback_does_not_consume_active_candidate_budget() -> None:
    time = np.arange(12, dtype=float)
    values = np.stack((time, 10.0 + time), axis=1)
    item = TimeSeriesItem(
        item_id="mixed-fallback",
        values=values,
        variate_names=("fully_missing", "partly_missing"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )
    mask = np.ones_like(values, dtype=bool)
    mask[:, 0] = False
    mask[4:7, 1] = False
    result = BlockwiseFAIS().impute(
        item,
        mask,
        ForecastSpec("mock", "independent_univariate", 2, target_indices=(1,)),
        BudgetSpec(max_candidates=2, max_active_candidates=1),
    )
    assert len(result.routing.fallback_blocks) == 1
    assert len(result.routing.activated_candidates) <= 1
    assert np.isfinite(result.values).all()
