from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais import RoutePlan as PublicRoutePlan
from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    CandidateStatus,
    ForecastResult,
    ForecastSpec,
    ImputerSpec,
    MissingBlock,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner, ImputerRegistry
from tsfm_fais.imputers.base import failed_candidate_result
from tsfm_fais.pipeline import (
    BlockwiseFAIS,
    FAISResult,
    RoutePlan,
    _anchor_period_is_eligible,
    _blend_candidate_global_priors,
    _blend_routing_evidence,
    _candidate_signal_scores,
    _candidate_switch_penalties,
    _forecast_consensus_inputs,
    _forecast_consensus_scores,
    _forecast_medoid_candidate,
    _historical_backtest_candidate,
    _merge_forecast_consensus_config,
    _proxy_outlier_candidates,
    _proxy_weighted_consensus_weights,
    _pseudo_calibrated_convex_weight,
    _regularized_forecast_consensus_candidate,
    _safe_prior_consensus_override,
    _select_extrapolation_anchor,
    _top_k_consensus_weights,
)
from tsfm_fais.routing.graph import BlockEdge, BlockGraph
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


def test_forecast_medoid_candidate_uses_scaled_prediction_consensus() -> None:
    values = {
        "left": np.zeros((1, 4, 1), dtype=float),
        "middle": np.ones((1, 4, 1), dtype=float),
        "outlier": np.full((1, 4, 1), 10.0),
    }

    def predictor(contexts: np.ndarray, spec: ForecastSpec) -> ForecastResult:
        point = np.repeat(contexts[:, :1, :], spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=(0,))

    selected, scores = _forecast_medoid_candidate(
        values,
        _spec(),
        predictor,
        np.asarray([2.0]),
    )

    assert selected == "middle"
    assert scores["middle"] < scores["left"] < scores["outlier"]


def test_forecast_consensus_scores_preserve_target_specific_evidence() -> None:
    values = {
        "a": np.tile(np.asarray([[[0.0, 10.0]]]), (1, 4, 1)),
        "b": np.tile(np.asarray([[[1.0, 0.0]]]), (1, 4, 1)),
        "c": np.tile(np.asarray([[[10.0, 1.0]]]), (1, 4, 1)),
    }
    spec = ForecastSpec(
        "mock",
        "joint_multivariate",
        horizon=2,
        target_indices=(0, 1),
    )

    def predictor(contexts: np.ndarray, forecast_spec: ForecastSpec) -> ForecastResult:
        point = np.repeat(contexts[:, -1:, :], forecast_spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=(0, 1))

    aggregate_scores, target_scores = _forecast_consensus_scores(
        values,
        spec,
        predictor,
        np.ones(2, dtype=float),
    )

    assert min(target_scores, key=lambda candidate_id: target_scores[candidate_id][0]) == "b"
    assert min(target_scores, key=lambda candidate_id: target_scores[candidate_id][1]) == "c"
    assert min(aggregate_scores, key=aggregate_scores.get) == "b"


def test_forecast_consensus_inputs_projects_and_remaps_targets() -> None:
    values = {
        "a": np.arange(30, dtype=float).reshape(1, 5, 6),
        "b": np.arange(30, 60, dtype=float).reshape(1, 5, 6),
    }
    spec = ForecastSpec(
        "mock",
        "joint_multivariate",
        horizon=2,
        target_indices=(1, 4),
    )

    projected, projected_spec, indices = _forecast_consensus_inputs(
        values,
        spec,
        "targets_only",
    )

    assert indices == (1, 4)
    assert projected_spec.target_indices == (0, 1)
    assert projected_spec.mode == "joint_multivariate"
    np.testing.assert_array_equal(projected["a"], values["a"][:, :, [1, 4]])
    np.testing.assert_array_equal(projected["b"], values["b"][:, :, [1, 4]])
    assert values["a"].shape == (1, 5, 6)


def test_forecast_consensus_inputs_keeps_targets_and_top_correlates() -> None:
    values = {"a": np.arange(30, dtype=float).reshape(1, 5, 6)}
    correlation = np.eye(6)
    correlation[1, 2] = correlation[2, 1] = 0.9
    correlation[4, 5] = correlation[5, 4] = -0.8
    spec = ForecastSpec(
        "mock",
        "joint_multivariate",
        horizon=2,
        target_indices=(1, 4),
    )

    projected, projected_spec, indices = _forecast_consensus_inputs(
        values,
        spec,
        "targets_with_correlates",
        correlation=correlation,
        max_context_variates=4,
    )

    assert indices == (1, 2, 4, 5)
    assert projected_spec.target_indices == (0, 2)
    np.testing.assert_array_equal(projected["a"], values["a"][:, :, [1, 2, 4, 5]])


def test_regularized_forecast_consensus_blends_supported_training_prior() -> None:
    selected_medoid, medoid_scores = _regularized_forecast_consensus_candidate(
        {"forecast_best": 0.0, "prior_best": 1.0},
        {"forecast_best": 1.0, "prior_best": 0.0},
        0.0,
    )
    selected_prior, prior_scores = _regularized_forecast_consensus_candidate(
        {"forecast_best": 0.0, "prior_best": 1.0},
        {"forecast_best": 1.0, "prior_best": 0.0},
        0.8,
    )

    assert selected_medoid == "forecast_best"
    assert medoid_scores["forecast_best"] < medoid_scores["prior_best"]
    assert selected_prior == "prior_best"
    assert prior_scores["prior_best"] < prior_scores["forecast_best"]


def test_top_k_consensus_weights_are_stable_and_uniform() -> None:
    weights = _top_k_consensus_weights(
        {"later": 0.0, "first": 0.0, "excluded": 2.0},
        top_k=2,
    )

    assert weights == {"first": 0.5, "later": 0.5}
    with pytest.raises(ValueError, match="at least two"):
        _top_k_consensus_weights({"first": 0.0}, top_k=1)


def test_top_k_consensus_weights_admits_close_third_candidate() -> None:
    close = _top_k_consensus_weights(
        {"first": 0.05, "second": 0.10, "third": 0.109, "fourth": 0.11},
        top_k=2,
        third_candidate_relative_gap=0.1,
    )
    distant = _top_k_consensus_weights(
        {"first": 0.05, "second": 0.10, "third": 0.111},
        top_k=2,
        third_candidate_relative_gap=0.1,
    )

    assert close == pytest.approx({"first": 1 / 3, "second": 1 / 3, "third": 1 / 3})
    assert distant == {"first": 0.5, "second": 0.5}
    with pytest.raises(ValueError, match="top-k two"):
        _top_k_consensus_weights(
            {"first": 0.0, "second": 0.1, "third": 0.2},
            top_k=3,
            third_candidate_relative_gap=0.1,
        )


def test_proxy_weighted_consensus_weights_use_inverse_error_power() -> None:
    weighted = _proxy_weighted_consensus_weights(
        {"low_error": 0.5, "high_error": 0.5},
        {"low_error": 1.0, "high_error": 4.0},
        power=0.5,
    )
    missing_proxy = _proxy_weighted_consensus_weights(
        {"low_error": 0.5, "high_error": 0.5},
        {"low_error": 1.0},
        power=0.5,
    )

    assert weighted == pytest.approx({"low_error": 2 / 3, "high_error": 1 / 3})
    assert missing_proxy == {"low_error": 0.5, "high_error": 0.5}
    with pytest.raises(ValueError, match="finite and non-negative"):
        _proxy_weighted_consensus_weights(
            {"low_error": 0.5, "high_error": 0.5},
            {"low_error": 1.0, "high_error": 4.0},
            power=float("inf"),
        )


def test_pseudo_calibrated_convex_weight_uses_ridge_and_safe_fallback() -> None:
    truth = np.asarray([[[0.0], [0.25], [0.75], [1.0]]])
    pseudo_mask = np.ones_like(truth, dtype=bool)
    pseudo_mask[:, 1:3, :] = False
    primary = CandidateResult(
        "primary",
        np.zeros_like(truth),
        np.ones_like(truth, dtype=bool),
    )
    alternative = CandidateResult(
        "alternative",
        np.ones_like(truth),
        np.ones_like(truth, dtype=bool),
    )

    weight, diagnostics = _pseudo_calibrated_convex_weight(
        truth,
        pseudo_mask,
        primary,
        alternative,
        channel=0,
        prior_weight=0.2,
        prior_strength=2.0,
        min_points=2,
    )
    fallback, fallback_diagnostics = _pseudo_calibrated_convex_weight(
        truth,
        pseudo_mask,
        primary,
        alternative,
        channel=0,
        prior_weight=0.2,
        prior_strength=2.0,
        min_points=3,
    )

    assert weight == pytest.approx(0.35)
    assert diagnostics["unconstrained_weight"] == pytest.approx(0.5)
    assert diagnostics["sample_count"] == 2
    assert diagnostics["applied"] is True
    assert fallback == 0.2
    assert fallback_diagnostics["source"] == "prior"
    assert fallback_diagnostics["applied"] is False


def test_safe_prior_consensus_override_requires_both_gates() -> None:
    medoid_scores = {"medoid": 0.0, "prior": 0.2, "other": 1.0}
    priors = {"medoid": 0.5, "prior": 0.0, "other": 0.8}

    selected, diagnostics = _safe_prior_consensus_override(
        "medoid",
        medoid_scores,
        priors,
        max_medoid_penalty=0.3,
        min_prior_margin=0.02,
    )
    high_penalty, _ = _safe_prior_consensus_override(
        "medoid",
        {"medoid": 0.0, "prior": 0.4, "other": 1.0},
        priors,
        max_medoid_penalty=0.3,
        min_prior_margin=0.02,
    )
    low_margin, _ = _safe_prior_consensus_override(
        "medoid",
        medoid_scores,
        {"medoid": 0.01, "prior": 0.0, "other": 0.8},
        max_medoid_penalty=0.3,
        min_prior_margin=0.02,
    )
    disabled, disabled_diagnostics = _safe_prior_consensus_override(
        "medoid",
        medoid_scores,
        priors,
        max_medoid_penalty=None,
        min_prior_margin=0.02,
    )

    assert selected == "prior"
    assert diagnostics["applied"] is True
    assert diagnostics["prior_margin"] == pytest.approx(0.5)
    assert diagnostics["medoid_penalty"] == pytest.approx(0.2)
    assert high_penalty == "medoid"
    assert low_margin == "medoid"
    assert disabled == "medoid"
    assert disabled_diagnostics == {"configured": False, "applied": False}


def test_runtime_forecast_consensus_overrides_frozen_router_metadata() -> None:
    merged = _merge_forecast_consensus_config(
        {
            "mode": "medoid",
            "candidates": ("locf", "linear_interp"),
            "context_mode": "targets_only",
            "prior_weight_by_model": {"chronos2": 0.0},
        },
        {
            "mode": "disabled",
            "candidates": (),
            "context_mode": "native",
            "prior_weight_by_model": {},
            "prior_override_max_medoid_penalty_by_model": {"chronos2": 0.3},
            "prior_override_min_margin_by_model": {"chronos2": 0.02},
        },
    )

    assert merged["mode"] == "disabled"
    assert merged["candidates"] == ()
    assert merged["context_mode"] == "native"
    assert merged["prior_weight_by_model"] == {}
    assert merged["prior_override_max_medoid_penalty_by_model"] == {"chronos2": 0.3}
    assert merged["prior_override_min_margin_by_model"] == {"chronos2": 0.02}


def test_candidate_signal_scores_aggregate_router_and_proxy_evidence() -> None:
    blocks = (
        MissingBlock("b0", 0, 0, 1, 2),
        MissingBlock("b1", 0, 0, 4, 5),
    )
    unary = {
        (blocks[0].block_id, "a"): 1.0,
        (blocks[1].block_id, "a"): 3.0,
        (blocks[0].block_id, "b"): 4.0,
        (blocks[1].block_id, "b"): 2.0,
    }

    router_scores = _candidate_signal_scores(
        "router_risk",
        ("a", "b", "incomplete"),
        blocks,
        unary,
        {},
    )
    proxy_scores = _candidate_signal_scores(
        "proxy_min",
        ("a", "b"),
        blocks,
        {},
        {"a": 0.25, "b": 0.5},
    )

    assert router_scores == {"a": 2.0, "b": 3.0}
    assert proxy_scores == {"a": 0.25, "b": 0.5}


def test_anchor_period_gate_uses_declared_period_and_context() -> None:
    assert _anchor_period_is_eligible(24, 96, 0.25) == (True, 0.25)
    assert _anchor_period_is_eligible(48, 96, 0.25) == (False, 0.5)
    assert _anchor_period_is_eligible(None, 96, 0.25) == (False, None)
    assert _anchor_period_is_eligible(None, 96, None) == (True, None)


def test_historical_backtest_candidate_uses_only_observed_context_suffix() -> None:
    values = {
        "zero": np.zeros((1, 4, 1), dtype=float),
        "one": np.ones((1, 4, 1), dtype=float),
    }
    clean = np.asarray([[[0.0], [0.0], [1.0], [1.0]]])
    observed = np.ones_like(clean, dtype=bool)

    def predictor(contexts: np.ndarray, spec: ForecastSpec) -> ForecastResult:
        point = np.repeat(contexts[:, -1:, :], spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=(0,))

    selected, scores = _historical_backtest_candidate(
        values,
        _spec(),
        predictor,
        np.asarray([1.0]),
        clean,
        observed,
        cutoff=2,
        min_observed_per_target=2,
    )

    assert selected == "one"
    assert scores["one"] == 0.0
    assert scores["zero"] == 1.0


def test_value_median_consensus_aggregates_valid_forecast_blocks() -> None:
    item, mask = _item()
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forecast_consensus_mode = "value_median"
    pipeline.forecast_consensus_candidates = ("locf", "linear_interp")
    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=2),
        available_artifact_ids=(),
    )
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, fill in (("locf", 1.0), ("linear_interp", 3.0)):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        candidates[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones(plan.batch.shape, dtype=bool),
        )

    result = pipeline.finish_route(plan, candidates)

    assert np.all(result.values[4:7, 0] == 2.0)
    assert result.routing.activated_candidates == ("linear_interp", "locf")
    assert result.routing.metadata["ensemble_block_assignments"]
    assert all(
        assignment.startswith("value_median[") for assignment in result.routing.assignments.values()
    )


def test_value_topk_mean_renormalizes_after_block_invalidity() -> None:
    item, mask = _item()
    candidate_ids = ("locf", "linear_interp", "seasonal_lag")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=3),
        available_artifact_ids=(),
    )
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, fill in zip(candidate_ids, (1.0, 3.0, 9.0), strict=True):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        native_valid = np.ones(plan.batch.shape, dtype=bool)
        if candidate_id == "seasonal_lag":
            native_valid[~plan.batch.observed_mask] = False
        candidates[candidate_id] = CandidateResult(candidate_id, values, native_valid)
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        None,
        {
            "active": True,
            "mode": "value_topk_mean",
            "aggregation": "weighted_mean",
            "ensemble_candidates": list(candidate_ids),
            "ensemble_weights": {
                "locf": 0.25,
                "linear_interp": 0.25,
                "seasonal_lag": 0.5,
            },
        },
    )

    result = pipeline.finish_route(plan, candidates)

    np.testing.assert_allclose(result.values[4:7, 0], 2.0)
    assert result.routing.activated_candidates == ("linear_interp", "locf")
    assert all(
        assignment.startswith("value_topk_mean[")
        for assignment in result.routing.assignments.values()
    )
    for weights in result.routing.metadata["ensemble_block_weights"].values():
        assert weights == {"locf": 0.5, "linear_interp": 0.5}


def test_target_specific_value_topk_mean_scores_each_forecast_target() -> None:
    item, mask = _item()
    mask[8:10, 1] = False
    candidate_ids = ("locf", "linear_interp", "seasonal_lag")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )

    def predictor(contexts: np.ndarray, spec: ForecastSpec) -> ForecastResult:
        point = np.repeat(contexts[:, -1:, list(spec.target_indices or ())], spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=tuple(spec.target_indices or ()))

    pipeline = BlockwiseFAIS(imputer_registry=registry, forecast_predictor=predictor)
    pipeline.forecast_consensus_mode = "value_topk_mean"
    pipeline.forecast_consensus_candidates = candidate_ids
    pipeline.forecast_consensus_ensemble_top_k = 2
    pipeline.forecast_consensus_context_mode = "native"
    pipeline.forecast_consensus_model_granularities = {"mock": "target"}
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0, 1),
    )
    plan = pipeline.prepare_route(
        item,
        mask,
        spec,
        BudgetSpec(max_candidates=3),
        available_artifact_ids=(),
    )
    plan.batch.metadata["mase_scale"] = [1.0, 1.0, 1.0]
    terminal_values = {
        "locf": (0.0, 10.0),
        "linear_interp": (1.0, 0.0),
        "seasonal_lag": (10.0, 1.0),
    }
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, (first, second) in terminal_values.items():
        values = plan.batch.values.copy()
        values[:, :, 0] = first
        values[:, :, 1] = second
        candidates[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones(plan.batch.shape, dtype=bool),
        )

    selected, diagnostics = pipeline._forecast_consensus_anchor(
        plan.batch,
        plan.blocks,
        candidate_ids,
        candidates,
        spec,
        set(),
        proxy_scores={},
    )

    assert selected is None
    assert diagnostics["ensemble_weights_by_target"] == {
        "0": {"linear_interp": 0.5, "locf": 0.5},
        "1": {"linear_interp": 0.5, "seasonal_lag": 0.5},
    }
    assert diagnostics["ensemble_candidates"] == [
        "linear_interp",
        "locf",
        "seasonal_lag",
    ]


def test_target_specific_value_topk_mean_assembles_each_target_pair() -> None:
    item, mask = _item()
    mask[8:10, 1] = False
    candidate_ids = ("locf", "linear_interp", "seasonal_lag")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0, 1),
    )
    plan = pipeline.prepare_route(
        item,
        mask,
        spec,
        BudgetSpec(max_candidates=3),
        available_artifact_ids=(),
    )
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, fill in zip(candidate_ids, (1.0, 3.0, 9.0), strict=True):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        candidates[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones(plan.batch.shape, dtype=bool),
        )
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        None,
        {
            "active": True,
            "mode": "value_topk_mean",
            "aggregation": "weighted_mean",
            "ensemble_candidates": list(candidate_ids),
            "ensemble_weights": {},
            "ensemble_weights_by_target": {
                "0": {"locf": 0.25, "linear_interp": 0.75},
                "1": {"linear_interp": 0.5, "seasonal_lag": 0.5},
            },
        },
    )

    result = pipeline.finish_route(plan, candidates)

    np.testing.assert_allclose(result.values[4:7, 0], 2.5)
    np.testing.assert_allclose(result.values[8:10, 1], 6.0)
    assignments_by_channel = {
        block.channel: result.routing.assignments[block.block_id] for block in plan.blocks
    }
    assert assignments_by_channel == {
        0: "value_topk_mean[locf,linear_interp]",
        1: "value_topk_mean[linear_interp,seasonal_lag]",
    }


def test_candidate_shrinkage_uses_only_native_forecast_visible_blocks() -> None:
    item, mask = _item()
    mask[8:10, 1] = False
    candidate_ids = ("locf", "linear_interp")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forecast_consensus_candidate_shrinkage_id = "linear_interp"
    pipeline.forecast_consensus_candidate_shrinkage_weight = 0.25
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "locf",
        {
            "active": True,
            "selected_candidate": "locf",
            "selected_candidates_by_target": {"0": "locf", "1": "linear_interp"},
            "anchor_weight_override": None,
        },
    )
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0, 1),
    )
    plan = pipeline.prepare_route(
        item,
        mask,
        spec,
        BudgetSpec(max_candidates=2),
        available_artifact_ids=(),
    )
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, fill in (("locf", 1.0), ("linear_interp", 3.0)):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        native = np.ones(plan.batch.shape, dtype=bool)
        if candidate_id == "linear_interp":
            native[0, 8:10, 1] = False
        candidates[candidate_id] = CandidateResult(candidate_id, values, native)

    result = pipeline.finish_route(plan, candidates)

    np.testing.assert_allclose(result.values[4:7, 0], 1.5)
    assert result.routing.assignments["n0:d0:4-7"] == ("candidate_shrink[locf,linear_interp]")
    untouched = result.routing.metadata["candidate_shrinkage_block_assignments"]["n0:d1:8-10"]
    assert untouched == {
        "primary_assignment": "locf",
        "primary_candidate_id": "linear_interp",
        "fallback_candidate_id": None,
        "candidate_id": "linear_interp",
        "weight": 0.0,
        "applied": False,
        "used_fallback": False,
        "reason": "candidate_not_native_valid",
    }
    assert result.routing.activated_candidates == ("linear_interp", "locf")


def test_candidate_shrinkage_uses_validity_gated_fallback() -> None:
    item, mask = _item()
    mask[8:10, 1] = False
    candidate_ids = ("locf", "linear_interp", "seasonal_lag")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forecast_consensus_candidate_shrinkage_id = "seasonal_lag"
    pipeline.forecast_consensus_candidate_shrinkage_weight = 0.9
    pipeline.forecast_consensus_candidate_shrinkage_fallback_id = "linear_interp"
    pipeline.forecast_consensus_candidate_shrinkage_fallback_weight = 0.75
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "locf",
        {
            "active": True,
            "selected_candidate": "locf",
            "anchor_weight_override": None,
        },
    )
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0, 1),
    )
    plan = pipeline.prepare_route(
        item,
        mask,
        spec,
        BudgetSpec(max_candidates=3),
        available_artifact_ids=(),
    )
    candidates: dict[str, CandidateResult] = {}
    for candidate_id, fill in (
        ("locf", 1.0),
        ("linear_interp", 3.0),
        ("seasonal_lag", 5.0),
    ):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        native = np.ones(plan.batch.shape, dtype=bool)
        if candidate_id == "seasonal_lag":
            native[0, 8:10, 1] = False
        candidates[candidate_id] = CandidateResult(candidate_id, values, native)

    result = pipeline.finish_route(plan, candidates)

    np.testing.assert_allclose(result.values[4:7, 0], 4.6)
    np.testing.assert_allclose(result.values[8:10, 1], 2.5)
    fallback_record = result.routing.metadata["candidate_shrinkage_block_assignments"]["n0:d1:8-10"]
    assert fallback_record == {
        "primary_assignment": "locf",
        "primary_candidate_id": "seasonal_lag",
        "fallback_candidate_id": "linear_interp",
        "candidate_id": "linear_interp",
        "weight": 0.75,
        "applied": True,
        "used_fallback": True,
        "reason": None,
    }
    assert result.routing.metadata["candidate_shrinkage_fallback_id"] == ("linear_interp")
    assert result.routing.activated_candidates == (
        "linear_interp",
        "locf",
        "seasonal_lag",
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
            runtime_already_spent=sum(result.runtime_seconds for result in actual.values()),
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
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    item, mask = _item(tail=True)
    budget = BudgetSpec(max_candidates=2, max_active_candidates=1)

    wrapped = BlockwiseFAIS(imputer_registry=registry).impute(item, mask, _spec(), budget, seed=31)
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


def test_refined_unary_uses_channel_matched_proxy_error() -> None:
    truth = np.zeros((1, 6, 2), dtype=float)
    batch = SeriesBatch(truth, np.ones_like(truth, dtype=bool))
    pseudo_mask = np.ones_like(truth, dtype=bool)
    pseudo_mask[:, 1, 0] = False
    pseudo_mask[:, 2, 1] = False
    pseudo_batch = SeriesBatch(truth.copy(), pseudo_mask)
    blocks = (
        MissingBlock("channel-0", 0, 0, 3, 4),
        MissingBlock("channel-1", 0, 1, 4, 5),
    )
    candidate_ids = ("locf", "linear_interp")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    router.metadata["evidence_blend"] = {
        "mock": {"r0": 0.0, "r1": 0.0, "proxy": 1.0, "global_prior": 0.0}
    }
    pipeline = BlockwiseFAIS(imputer_registry=registry, router=router)
    actual: dict[str, CandidateResult] = {}
    pseudo: dict[str, CandidateResult] = {}
    for candidate_id, channel_errors in (
        ("locf", (1.0, 9.0)),
        ("linear_interp", (5.0, 2.0)),
    ):
        actual[candidate_id] = CandidateResult(
            candidate_id,
            truth.copy(),
            np.ones_like(truth, dtype=bool),
        )
        values = truth.copy()
        values[:, 1, 0] = channel_errors[0]
        values[:, 2, 1] = channel_errors[1]
        pseudo[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones_like(truth, dtype=bool),
        )
    fallback = {
        (block.block_id, candidate_id): 0.0 for block in blocks for candidate_id in candidate_ids
    }

    risks, proxy_risks = pipeline._refined_unary(
        batch,
        pseudo_batch,
        blocks,
        candidate_ids,
        actual,
        pseudo,
        _spec(),
        period=None,
        fallback=fallback,
    )

    assert risks[("channel-0", "locf")] < risks[("channel-0", "linear_interp")]
    assert risks[("channel-1", "linear_interp")] < risks[("channel-1", "locf")]
    assert proxy_risks[("channel-0", "locf")] == 1.0
    assert proxy_risks[("channel-0", "linear_interp")] == 5.0
    assert proxy_risks[("channel-1", "locf")] == 9.0
    assert proxy_risks[("channel-1", "linear_interp")] == 2.0


def test_refined_unary_uses_frozen_operational_features() -> None:
    truth = np.zeros((1, 6, 2), dtype=float)
    batch = SeriesBatch(truth, np.ones_like(truth, dtype=bool))
    pseudo_mask = np.ones_like(truth, dtype=bool)
    pseudo_mask[:, 1, 0] = False
    pseudo_batch = SeriesBatch(truth.copy(), pseudo_mask)
    block = MissingBlock("channel-0", 0, 0, 3, 4)
    candidate_ids = ("locf", "linear_interp")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    router.feature_names = (
        "runtime_seconds",
        "peak_memory_mb",
        "candidate_id::locf",
        "candidate_id::linear_interp",
    )
    router.unary = _RecordingRanker(0)  # type: ignore[assignment]
    router.metadata.update(
        {
            "candidate_runtime_seconds": {"locf": 0.25, "linear_interp": 0.5},
            "candidate_peak_memory_mb": {"locf": 2.0, "linear_interp": 4.0},
        }
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry, router=router)
    actual: dict[str, CandidateResult] = {}
    pseudo: dict[str, CandidateResult] = {}
    for index, candidate_id in enumerate(candidate_ids):
        actual[candidate_id] = CandidateResult(
            candidate_id,
            truth.copy(),
            np.ones_like(truth, dtype=bool),
        )
        pseudo[candidate_id] = CandidateResult(
            candidate_id,
            truth.copy(),
            np.ones_like(truth, dtype=bool),
            runtime_seconds=100.0 + index,
            peak_memory_bytes=(1000 + index) * 1024**2,
        )
    fallback = {(block.block_id, candidate_id): 0.0 for candidate_id in candidate_ids}

    pipeline._refined_unary(
        batch,
        pseudo_batch,
        (block,),
        candidate_ids,
        actual,
        pseudo,
        _spec(),
        period=None,
        fallback=fallback,
    )

    unary = router.unary
    assert isinstance(unary, _RecordingRanker)
    np.testing.assert_array_equal(
        unary.inputs[0][:, :2],
        np.asarray([[0.25, 2.0], [0.5, 4.0]]),
    )


def test_finish_route_blends_consensus_with_proxy_block_choice() -> None:
    item, mask = _item()
    candidate_ids = ("locf", "linear_interp")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    pipeline = BlockwiseFAIS(imputer_registry=registry, router=router)
    pipeline.forecast_consensus_proxy_blend_weight = 0.5
    pipeline.forecast_consensus_model_proxy_blend_weights = {}
    pipeline.forecast_consensus_proxy_blend_min_relative_margin = 0.0
    pipeline.forecast_consensus_model_proxy_blend_min_relative_margins = {}
    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=2),
        available_artifact_ids=(),
    )
    assert plan.pseudo_batch is not None
    actual: dict[str, CandidateResult] = {}
    pseudo: dict[str, CandidateResult] = {}
    for candidate_id, fill in (("locf", 1.0), ("linear_interp", 3.0)):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        actual[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones(plan.batch.shape, dtype=bool),
        )
        pseudo_values = plan.pseudo_batch.values.copy()
        pseudo_values[~plan.pseudo_batch.observed_mask] = fill
        pseudo[candidate_id] = CandidateResult(
            candidate_id,
            pseudo_values,
            np.ones(plan.pseudo_batch.shape, dtype=bool),
        )
    unary = {
        (block.block_id, candidate_id): float(candidate_id != "locf")
        for block in plan.blocks
        for candidate_id in candidate_ids
    }
    proxy = {
        (block.block_id, candidate_id): float(candidate_id != "linear_interp")
        for block in plan.blocks
        for candidate_id in candidate_ids
    }
    pipeline._refined_unary = lambda *args, **kwargs: (unary, proxy)  # type: ignore[method-assign]
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "locf",
        {
            "active": True,
            "mode": "medoid",
            "selected_candidate": "locf",
            "selected_candidates_by_target": {},
        },
    )

    result = pipeline.finish_route(plan, actual, pseudo)

    np.testing.assert_allclose(result.values[4:7, 0], 2.0)
    assert result.routing.activated_candidates == ("linear_interp", "locf")
    assert all(
        assignment == "selector_blend[locf,linear_interp]"
        for assignment in result.routing.assignments.values()
    )
    blend = next(iter(result.routing.metadata["proxy_blend_block_assignments"].values()))
    assert blend["primary_assignment"] == "locf"
    assert blend["proxy_candidate"] == "linear_interp"
    assert blend["primary_proxy_risk"] == 1.0
    assert blend["proxy_risk"] == 0.0
    assert blend["proxy_risk_margin"] == 1.0
    assert blend["proxy_risk_relative_margin"] == 1.0
    assert blend["proxy_gate_applied"] is True
    assert blend["proxy_weight"] == 0.5

    pipeline.forecast_consensus_proxy_blend_min_relative_margin = 2.0
    gated = pipeline.finish_route(plan, actual, pseudo)
    np.testing.assert_allclose(gated.values[4:7, 0], 1.0)
    gated_blend = next(iter(gated.routing.metadata["proxy_blend_block_assignments"].values()))
    assert gated_blend["proxy_gate_applied"] is False
    assert gated_blend["configured_proxy_weight"] == 0.5
    assert gated_blend["proxy_weight"] == 0.0

    proxy_mask = plan.pseudo_batch.observed_mask | ~plan.batch.observed_mask
    newly_hidden = ~proxy_mask
    calibrated_pseudo: dict[str, CandidateResult] = {}
    for candidate_id, fill, offset in (
        ("locf", 1.0, -1.0),
        ("linear_interp", 3.0, 3.0),
    ):
        pseudo_values = plan.pseudo_batch.values.copy()
        pseudo_values[~plan.pseudo_batch.observed_mask] = fill
        pseudo_values[newly_hidden] = plan.batch.values[newly_hidden] + offset
        calibrated_pseudo[candidate_id] = CandidateResult(
            candidate_id,
            pseudo_values,
            np.ones(plan.pseudo_batch.shape, dtype=bool),
        )
    pipeline.forecast_consensus_proxy_blend_min_relative_margin = 0.0
    pipeline.forecast_consensus_pseudo_weight_calibration = "convex_l2"
    pipeline.forecast_consensus_pseudo_weight_prior_strength = 0.0
    pipeline.forecast_consensus_pseudo_weight_min_points = 1
    calibrated = pipeline.finish_route(plan, actual, calibrated_pseudo)
    np.testing.assert_allclose(calibrated.values[4:7, 0], 1.5)
    calibrated_blend = next(
        iter(calibrated.routing.metadata["proxy_blend_block_assignments"].values())
    )
    assert calibrated_blend["proxy_weight"] == pytest.approx(0.25)
    assert calibrated_blend["proxy_weight_calibration"]["applied"] is True


def test_finish_route_calibrates_top_two_weights_by_channel() -> None:
    item, mask = _item()
    candidate_ids = ("locf", "linear_interp")
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry, router=_router())
    pipeline.forecast_consensus_pseudo_weight_calibration = "convex_l2"
    pipeline.forecast_consensus_pseudo_weight_prior_strength = 0.0
    pipeline.forecast_consensus_pseudo_weight_min_points = 1
    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=2),
        available_artifact_ids=(),
    )
    assert plan.pseudo_batch is not None
    proxy_mask = plan.pseudo_batch.observed_mask | ~plan.batch.observed_mask
    newly_hidden = ~proxy_mask
    actual: dict[str, CandidateResult] = {}
    pseudo: dict[str, CandidateResult] = {}
    for candidate_id, fill, offset in (
        ("locf", 1.0, -1.0),
        ("linear_interp", 3.0, 3.0),
    ):
        values = plan.batch.values.copy()
        values[~plan.batch.observed_mask] = fill
        actual[candidate_id] = CandidateResult(
            candidate_id,
            values,
            np.ones(plan.batch.shape, dtype=bool),
        )
        pseudo_values = plan.pseudo_batch.values.copy()
        pseudo_values[~plan.pseudo_batch.observed_mask] = fill
        pseudo_values[newly_hidden] = plan.batch.values[newly_hidden] + offset
        pseudo[candidate_id] = CandidateResult(
            candidate_id,
            pseudo_values,
            np.ones(plan.pseudo_batch.shape, dtype=bool),
        )
    unary = {
        (block.block_id, candidate_id): 0.0
        for block in plan.blocks
        for candidate_id in candidate_ids
    }
    pipeline._refined_unary = lambda *args, **kwargs: (unary, {})  # type: ignore[method-assign]
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        None,
        {
            "active": True,
            "mode": "value_topk_mean",
            "aggregation": "weighted_mean",
            "ensemble_candidates": list(candidate_ids),
            "ensemble_weights": {"locf": 0.5, "linear_interp": 0.5},
        },
    )

    result = pipeline.finish_route(plan, actual, pseudo)

    np.testing.assert_allclose(result.values[4:7, 0], 1.5)
    weights = next(iter(result.routing.metadata["ensemble_block_weights"].values()))
    assert weights == pytest.approx({"locf": 0.75, "linear_interp": 0.25})
    calibrations = result.routing.metadata["forecast_consensus"]["ensemble_weight_calibration"]
    assert calibrations["0"]["applied"] is True


def test_candidate_global_prior_blend_uses_neutral_risk_for_unsupported_method() -> None:
    risks = {
        ("b0", "ranker_best"): 0.0,
        ("b0", "prior_best"): 2.0,
        ("b0", "unsupported"): 0.0,
    }

    blended = _blend_candidate_global_priors(
        risks,
        {"ranker_best": 2.0, "prior_best": -2.0},
        weight=0.5,
        scale=2.0,
    )

    assert blended[("b0", "ranker_best")] == pytest.approx(1.0)
    assert blended[("b0", "prior_best")] == pytest.approx(1.0)
    assert blended[("b0", "unsupported")] == pytest.approx(0.5)


def test_routing_evidence_blend_normalizes_each_block() -> None:
    keys = (("b0", "a"), ("b0", "b"), ("b1", "a"), ("b1", "b"))
    blended = _blend_routing_evidence(
        dict(zip(keys, (0.0, 2.0, 10.0, 0.0), strict=True)),
        dict(zip(keys, (0.0, 3.0, 4.0, 0.0), strict=True)),
        dict(zip(keys, (9.0, 1.0, 2.0, 8.0), strict=True)),
        {"a": -1.0, "b": 1.0},
        weights={"r0": 0.1, "r1": 0.2, "proxy": 0.7, "global_prior": 0.0},
        scale=2.0,
    )

    assert blended[("b0", "b")] < blended[("b0", "a")]
    assert blended[("b1", "a")] < blended[("b1", "b")]
    assert max(blended.values()) <= 2.0


def test_candidate_switch_penalty_only_connects_forecast_visible_blocks() -> None:
    blocks = (
        MissingBlock("target-0", 0, 0, 2, 4),
        MissingBlock("target-1", 0, 0, 7, 9),
        MissingBlock("irrelevant", 0, 1, 2, 4),
    )
    graph = BlockGraph(
        blocks,
        (
            BlockEdge("target-0", "target-1", 0.5),
            BlockEdge("target-0", "irrelevant", 1.0),
        ),
    )
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0,),
    )

    penalties = _candidate_switch_penalties(
        graph,
        ("a", "b"),
        spec,
        weight=0.02,
    )

    assert penalties == {
        ("target-0", "target-1", "a", "b"): 0.02,
        ("target-0", "target-1", "b", "a"): 0.02,
    }


def test_proxy_outlier_gate_rejects_only_extreme_pseudo_error() -> None:
    rejected, threshold = _proxy_outlier_candidates(
        {
            "locf": 2.0,
            "linear": 2.5,
            "knn": 1.5,
            "gpvae": 3.0,
            "csdi": 200.0,
        },
        multiplier=5.0,
    )

    assert threshold is not None
    assert 3.0 < threshold < 200.0
    assert rejected == frozenset({"csdi"})


def test_extrapolation_anchor_uses_calibrated_proxy_threshold() -> None:
    calibrations = {
        "mock": {
            "toy": {
                "candidates": ["gpvae", "knn_multivariate"],
                "proxy_log_ratio_threshold": 0.1,
                "max_training_origin": 100,
            }
        }
    }
    spec = _spec()

    selected, diagnostics = _select_extrapolation_anchor(
        calibrations,
        spec,
        {"dataset_id": "toy", "forecast_origin": 101},
        {"gpvae": 2.0, "knn_multivariate": 3.0},
        ("gpvae", "knn_multivariate"),
    )
    within_range, within_diagnostics = _select_extrapolation_anchor(
        calibrations,
        spec,
        {"dataset_id": "toy", "forecast_origin": 100},
        {"gpvae": 2.0, "knn_multivariate": 3.0},
        ("gpvae", "knn_multivariate"),
    )

    assert selected == "gpvae"
    assert diagnostics["active"] is True
    assert diagnostics["reason"] == "calibrated_proxy_threshold"
    assert within_range is None
    assert within_diagnostics["reason"] == "within_training_origin_range"

    cross_dataset = {
        "mock": {
            "__all__": {
                "candidates": ["gpvae", "knn_multivariate"],
                "proxy_log_ratio_threshold": -0.5,
                "max_training_origin": None,
            }
        }
    }
    selected_cross, cross_diagnostics = _select_extrapolation_anchor(
        cross_dataset,
        spec,
        {"dataset_id": "unseen", "forecast_origin": 1},
        {"gpvae": 2.0, "knn_multivariate": 3.0},
        ("gpvae", "knn_multivariate"),
    )

    assert selected_cross == "knn_multivariate"
    assert cross_diagnostics["scope"] == "cross_dataset"
    assert cross_diagnostics["active"] is True


def test_finish_route_forces_calibrated_anchor_on_forecast_visible_blocks() -> None:
    item, mask = _item()
    item.metadata.update({"dataset_id": "toy", "forecast_origin": 200})
    router = _router()
    router.metadata["candidate_anchor_calibrations"] = {
        "mock": {
            "toy": {
                "candidates": ["locf", "linear_interp"],
                "proxy_log_ratio_threshold": 1e6,
                "max_training_origin": 100,
            }
        }
    }
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    pipeline = BlockwiseFAIS(router=router, imputer_registry=registry)

    _, result = _manual(
        pipeline,
        item,
        mask,
        BudgetSpec(max_candidates=2),
        seed=29,
    )

    assert set(result.routing.assignments.values()) == {"locf"}
    calibration = result.routing.metadata["candidate_anchor_calibration"]
    assert calibration["active"] is True
    assert calibration["selected_candidate"] == "locf"
    assert result.routing.metadata["candidate_anchor_block_assignments"]


def test_finish_route_applies_target_specific_consensus_anchors() -> None:
    item, mask = _item()
    mask[8:10, 1] = False
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline._forecast_consensus_anchor = lambda *args, **kwargs: (  # type: ignore[method-assign]
        "locf",
        {
            "active": True,
            "selected_candidate": "locf",
            "selection_granularity": "target",
            "selected_candidates_by_target": {"0": "locf", "1": "linear_interp"},
            "anchor_weight_override": None,
        },
    )
    spec = ForecastSpec(
        "mock",
        "independent_univariate",
        horizon=2,
        target_indices=(0, 1),
    )
    plan = pipeline.prepare_route(
        item,
        mask,
        spec,
        BudgetSpec(max_candidates=2),
        seed=31,
    )
    candidates = pipeline.candidate_runner.run_many(
        plan.shortlist,
        plan.batch,
        seed=31,
        budget=plan.budget,
    )

    result = pipeline.finish_route(plan, candidates)
    assignments_by_channel = {
        block.channel: result.routing.assignments[block.block_id] for block in plan.blocks
    }

    assert assignments_by_channel == {0: "locf", 1: "linear_interp"}


def test_prepare_route_forces_reliable_model_specific_prior_candidates() -> None:
    candidate_ids = (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "kalman_local_trend",
    )
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    router.candidate_ids = candidate_ids
    router.metadata.update(
        {
            "candidate_global_priors": {
                "mock": {
                    "locf": 0.0,
                    "linear_interp": 0.1,
                    "seasonal_lag": -2.0,
                    "kalman_local_trend": -1.0,
                }
            },
            "candidate_global_support": {
                "mock": {
                    "locf": 4,
                    "linear_interp": 4,
                    "seasonal_lag": 3,
                    "kalman_local_trend": 2,
                }
            },
            "candidate_global_prior_weight": 0.5,
            "candidate_global_prior_min_support": 2,
            "candidate_global_prior_forced_count": 2,
        }
    )
    item, mask = _item()

    plan = BlockwiseFAIS(router=router, imputer_registry=registry).prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=4),
        available_artifact_ids=(),
    )

    assert plan.shortlist == (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "kalman_local_trend",
    )


def test_prepare_route_forces_dataset_prior_forecast_consensus_candidates() -> None:
    candidate_ids = (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "kalman_local_trend",
    )
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    router.candidate_ids = candidate_ids
    router.metadata.update(
        {
            "candidate_global_priors": {
                "mock": {
                    "locf": 0.0,
                    "linear_interp": 0.1,
                    "seasonal_lag": -1.0,
                    "kalman_local_trend": -2.0,
                }
            },
            "candidate_global_support": {
                "mock": {candidate_id: 4 for candidate_id in candidate_ids}
            },
        }
    )
    pipeline = BlockwiseFAIS(router=router, imputer_registry=registry)
    pipeline.forecast_consensus_mode = "medoid"
    pipeline.forecast_consensus_candidates = ("locf", "linear_interp")
    pipeline.forecast_consensus_dataset_prior_candidates = 2
    pipeline.forecast_consensus_model_prior_candidates = {"mock": 2}
    item, mask = _item()

    plan = pipeline.prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=4),
        available_artifact_ids=(),
    )

    assert plan.shortlist[:2] == ("locf", "linear_interp")
    assert set(plan.shortlist) == set(candidate_ids)


def test_dataset_prior_overrides_supported_model_prior() -> None:
    router = _router()
    router.metadata.update(
        {
            "candidate_global_priors": {"mock": {"locf": -1.0, "linear_interp": 1.0}},
            "candidate_global_support": {"mock": {"locf": 10, "linear_interp": 10}},
            "candidate_dataset_priors": {"mock": {"toy": {"locf": 2.0, "linear_interp": -2.0}}},
            "candidate_dataset_support": {"mock": {"toy": {"locf": 3, "linear_interp": 3}}},
        }
    )
    pipeline = BlockwiseFAIS(router=router)

    priors = pipeline._reliable_candidate_global_priors(
        _spec(),
        ("locf", "linear_interp"),
        {"dataset_id": "toy"},
    )

    assert priors == {"locf": 2.0, "linear_interp": -2.0}


def test_prepare_route_keeps_available_family_diverse_shortlist_anchors() -> None:
    candidate_ids = (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "kalman_local_trend",
    )
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in candidate_ids
    )
    router = _router()
    router.candidate_ids = candidate_ids
    router.metadata["shortlist_anchor_candidates"] = [
        "seasonal_lag",
        "kalman_local_trend",
    ]
    item, mask = _item()

    plan = BlockwiseFAIS(router=router, imputer_registry=registry).prepare_route(
        item,
        mask,
        _spec(),
        BudgetSpec(max_candidates=4),
        available_artifact_ids=(),
    )

    assert plan.shortlist == candidate_ids


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
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
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
    plan, phased = _manual(phased_pipeline, item, mask, budget, seed=17)

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
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
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
        candidate.status is CandidateStatus.FAILED for candidate in wrapped.candidates.values()
    )


def test_route_phases_preserve_runtime_budget_exhaustion() -> None:
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    item, mask = _item()
    budget = BudgetSpec(max_candidates=2, max_runtime_seconds=1e-12)

    wrapped = BlockwiseFAIS(imputer_registry=registry).impute(item, mask, _spec(), budget, seed=3)
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
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
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
    assert next(iter(result.routing.fallback_records.values()))["attempts"] == ("linear_interp",)


def test_finish_route_rejects_missing_or_mismatched_injected_results() -> None:
    registry = ImputerRegistry((DEFAULT_REGISTRY.get_spec("locf"),))
    pipeline = BlockwiseFAIS(imputer_registry=registry)
    pipeline.forced_candidates = ("locf",)
    item, mask = _item()
    plan = pipeline.prepare_route(item, mask, _spec(), BudgetSpec(max_candidates=1))

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
