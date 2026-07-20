from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsfm_fais.contracts import BudgetSpec, ForecastSpec, TimeSeriesItem
from tsfm_fais.imputers import DEFAULT_REGISTRY, ImputerRegistry
from tsfm_fais.pipeline import BlockwiseFAIS
from tsfm_fais.routing.models import RouterBundle


class SeedAwareRanker:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[str, str], ...], int]] = []

    def predict(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover - guard
        raise AssertionError("context-aware ranker must not use legacy predict")

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: tuple[tuple[str, str], ...],
        seed: int,
    ) -> np.ndarray:
        assert len(features) == len(keys)
        self.calls.append((keys, seed))
        preferred = "locf" if seed % 2 == 0 else "linear_interp"
        return np.asarray(
            [1.0 if candidate_id == preferred else 0.0 for _, candidate_id in keys]
        )


class LegacyRanker:
    def __init__(self, column: int) -> None:
        self.column = column
        self.calls = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray(features, dtype=float)[:, self.column]


def _registry() -> ImputerRegistry:
    return ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id)
        for candidate_id in ("locf", "linear_interp")
    )


def _item() -> tuple[TimeSeriesItem, np.ndarray]:
    values = np.arange(12, dtype=float)[:, None]
    item = TimeSeriesItem(
        item_id="baseline-route",
        values=values,
        variate_names=("value",),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"training_correlation": np.eye(1)},
    )
    observed = np.ones_like(values, dtype=bool)
    observed[4:7, 0] = False
    return item, observed


def _spec() -> ForecastSpec:
    return ForecastSpec(
        model_id="mock",
        mode="independent_univariate",
        horizon=2,
        target_indices=(0,),
    )


def _baseline_pipeline() -> tuple[BlockwiseFAIS, SeedAwareRanker]:
    prior = SeedAwareRanker()
    router = RouterBundle(
        prior=prior,  # type: ignore[arg-type]
        unary=SimpleNamespace(),  # type: ignore[arg-type]
        pairwise=SimpleNamespace(model=None),  # type: ignore[arg-type]
        feature_names=("candidate_id::locf", "candidate_id::linear_interp"),
        candidate_ids=("locf", "linear_interp"),
        metadata={
            "requires_pseudo_candidates": False,
            "selector_method": "random_valid_block",
            "unary_risk_scale": 1.0,
        },
    )
    return BlockwiseFAIS(router=router, imputer_registry=_registry()), prior


def test_baseline_selector_is_seeded_and_skips_pseudo_candidates(monkeypatch) -> None:
    pipeline, ranker = _baseline_pipeline()
    item, observed = _item()
    calls: list[np.ndarray] = []
    native_run_many = pipeline.candidate_runner.run_many

    def recording_run_many(imputer_ids, batch, *args, **kwargs):
        calls.append(batch.observed_mask.copy())
        return native_run_many(imputer_ids, batch, *args, **kwargs)

    monkeypatch.setattr(pipeline.candidate_runner, "run_many", recording_run_many)
    even_first = pipeline.impute(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=12,
    )
    even_second = pipeline.impute(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=12,
    )
    odd = pipeline.impute(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=13,
    )

    assert len(calls) == 3
    assert all(np.array_equal(mask, observed[None, ...]) for mask in calls)
    assert set(even_first.routing.assignments.values()) == {"locf"}
    assert even_first.routing.assignments == even_second.routing.assignments
    assert set(odd.routing.assignments.values()) == {"linear_interp"}
    assert [seed for _, seed in ranker.calls] == [12, 12, 13]
    assert even_first.routing.metadata["selector_method"] == "random_valid_block"
    assert even_first.routing.metadata["requires_pseudo_candidates"] is False
    np.testing.assert_array_equal(even_first.values[observed], item.values[observed])


def test_baseline_selector_excludes_native_invalid_preferred_candidate() -> None:
    pipeline, _ = _baseline_pipeline()
    item, observed = _item()
    plan = pipeline.prepare_route(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=12,
    )
    assert plan.pseudo_batch is None
    candidates = pipeline.candidate_runner.run_many(
        plan.shortlist,
        plan.batch,
        seed=12,
        budget=plan.budget,
    )
    block = plan.blocks[0]
    candidates["locf"].native_valid_mask[
        block.batch_index,
        block.start : block.end,
        block.channel,
    ] = False

    result = pipeline.finish_route(plan, candidates)

    assert set(result.routing.assignments.values()) == {"linear_interp"}
    assert not result.routing.fallback_blocks


def test_baseline_noop_preserves_selector_method() -> None:
    pipeline, ranker = _baseline_pipeline()
    item, _ = _item()
    observed = np.ones_like(item.values, dtype=bool)

    result = pipeline.impute(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=12,
    )

    assert not ranker.calls
    assert result.routing.assignments == {}
    assert result.routing.metadata == {
        "selector_method": "random_valid_block",
        "requires_pseudo_candidates": False,
        "solver": "noop",
    }


def test_legacy_router_keeps_predict_and_pseudo_candidate_behavior(monkeypatch) -> None:
    prior = LegacyRanker(1)
    unary = LegacyRanker(1)
    router = RouterBundle(
        prior=prior,  # type: ignore[arg-type]
        unary=unary,  # type: ignore[arg-type]
        pairwise=SimpleNamespace(model=None),  # type: ignore[arg-type]
        feature_names=("candidate_id::locf", "candidate_id::linear_interp"),
        candidate_ids=("locf", "linear_interp"),
        metadata={"unary_risk_scale": 1.0},
    )
    pipeline = BlockwiseFAIS(router=router, imputer_registry=_registry())
    item, observed = _item()
    masks: list[np.ndarray] = []
    native_run_many = pipeline.candidate_runner.run_many

    def recording_run_many(imputer_ids, batch, *args, **kwargs):
        masks.append(batch.observed_mask.copy())
        return native_run_many(imputer_ids, batch, *args, **kwargs)

    monkeypatch.setattr(pipeline.candidate_runner, "run_many", recording_run_many)
    result = pipeline.impute(
        item,
        observed,
        _spec(),
        BudgetSpec(max_candidates=2),
        seed=17,
    )

    assert len(masks) == 2
    assert not np.array_equal(masks[0], masks[1])
    assert prior.calls == 1
    assert unary.calls == 1
    assert result.routing.metadata["selector_method"] == "b_fais"
    assert result.routing.metadata["requires_pseudo_candidates"] is True
