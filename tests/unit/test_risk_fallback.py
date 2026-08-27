from __future__ import annotations

import inspect

import numpy as np
import pytest

from tsfm_fais.contracts import MissingBlock, SeriesBatch
from tsfm_fais.pipeline import BlockwiseFAIS
from tsfm_fais.risk_fallback import (
    RiskScoreProtocol,
    RiskThreshold,
    ThresholdEpisode,
    apply_whole_episode_fallback,
    channel_proxy_mae,
    derived_pipeline_runtime,
    deterministic_pseudo_observed_mask,
    discrete_cvar,
    enumerate_thresholds,
    normalized_proxy_regret,
    score_block,
    score_episode,
    select_robust_candidate,
    select_threshold,
    source_routing_actual_ids,
    threshold_switches,
    weighted_empirical_quantile,
)


def test_pseudo_mask_matches_existing_route_procedure() -> None:
    observed = np.ones((1, 24, 3), dtype=bool)
    observed[0, 5:8, 0] = False
    observed[0, 14:18, 2] = False
    blocks = (
        MissingBlock("n0:d0:5-8", 0, 0, 5, 8),
        MissingBlock("n0:d2:14-18", 0, 2, 14, 18),
    )
    batch = SeriesBatch(np.arange(72, dtype=float).reshape(1, 24, 3), observed)
    expected = BlockwiseFAIS()._pseudo_batch(  # noqa: SLF001 - exact compatibility gate
        batch,
        4101,
        max_blocks=8,
        target_blocks=blocks,
        priority_channels=(0, 1),
    )
    actual = deterministic_pseudo_observed_mask(
        observed,
        4101,
        max_blocks=8,
        target_blocks=blocks,
        priority_channels=(0, 1),
    )
    assert np.array_equal(actual, expected.observed_mask)
    assert np.array_equal(
        actual,
        deterministic_pseudo_observed_mask(
            observed, 4101, max_blocks=8, target_blocks=blocks, priority_channels=(0, 1)
        ),
    )
    assert not np.any(actual & ~observed)


def test_channel_proxy_mae_requires_native_finite_channel_evidence() -> None:
    truth = np.arange(12, dtype=float).reshape(1, 4, 3)
    observed = np.ones_like(truth, dtype=bool)
    pseudo = observed.copy()
    pseudo[0, 1:3, 1] = False
    values = truth.copy()
    values[0, 1:3, 1] += (1.0, 3.0)
    native = np.ones_like(observed)
    assert channel_proxy_mae(
        truth,
        values,
        native,
        observed,
        pseudo,
        channel=1,
        candidate_status="success",
    ) == pytest.approx(2.0)
    assert (
        channel_proxy_mae(
            truth, values, native, observed, pseudo, channel=0, candidate_status="success"
        )
        is None
    )
    native[0, 1, 1] = False
    assert (
        channel_proxy_mae(
            truth, values, native, observed, pseudo, channel=1, candidate_status="success"
        )
        is None
    )
    native[:] = True
    assert (
        channel_proxy_mae(
            truth, values, native, observed, pseudo, channel=1, candidate_status="failed"
        )
        is None
    )


def test_block_and_episode_scores_cover_visibility_safety_and_unavailable() -> None:
    target = MissingBlock("target", 0, 0, 1, 2)
    safety_block = MissingBlock("safety", 0, 0, 2, 3)
    unavailable_block = MissingBlock("unavailable", 0, 0, 3, 4)
    hidden = MissingBlock("hidden", 0, 2, 1, 2)
    protocol = RiskScoreProtocol("independent_univariate", (0,))
    direct = score_block(
        target,
        protocol,
        selected_candidate_id="b",
        shortlist=("a", "b", "c"),
        proxy_mae_by_candidate={"a": 1.0, "b": 2.0, "c": 3.0},
        direct_native_valid=True,
        safety_result_used=False,
    )
    assert direct.risk == pytest.approx(0.5)
    irrelevant = score_block(
        hidden,
        protocol,
        selected_candidate_id="a",
        shortlist=("a",),
        proxy_mae_by_candidate={"a": 1.0},
        direct_native_valid=True,
        safety_result_used=False,
    )
    assert irrelevant.status == "irrelevant"
    safety = score_block(
        safety_block,
        protocol,
        selected_candidate_id="train_median",
        shortlist=("a",),
        proxy_mae_by_candidate={"a": 1.0},
        direct_native_valid=False,
        safety_result_used=True,
    )
    assert safety.risk == 1.0
    unavailable = score_block(
        unavailable_block,
        protocol,
        selected_candidate_id="a",
        shortlist=("a",),
        proxy_mae_by_candidate={"a": None},
        direct_native_valid=True,
        safety_result_used=False,
    )
    assert unavailable.risk is None
    assert score_episode((direct, irrelevant, safety, unavailable)).score == 1.0
    assert score_episode((irrelevant, unavailable)).score is None


def test_regret_bounds_and_strict_threshold_action() -> None:
    assert normalized_proxy_regret(4.0, (4.0, 4.0)) == (0.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        normalized_proxy_regret(5.0, (1.0, 2.0))
    threshold = RiskThreshold("finite", 0.5)
    assert not threshold_switches(0.5, threshold)
    assert threshold_switches(0.5000000001, threshold)
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    observed = np.array([[True, False], [False, True]])
    robust = np.array([[9.0, 20.0], [30.0, 9.0]])
    native = np.ones_like(observed)
    unchanged = apply_whole_episode_fallback(
        source, observed, robust, native, score=0.5, threshold=threshold
    )
    assert not unchanged.switched
    assert np.array_equal(unchanged.values, source)
    switched = apply_whole_episode_fallback(
        source, observed, robust, native, score=0.6, threshold=threshold
    )
    assert switched.switched
    assert np.array_equal(switched.values[observed], source[observed])
    assert np.array_equal(switched.values[~observed], robust[~observed])
    native[1, 0] = False
    rejected = apply_whole_episode_fallback(
        source, observed, robust, native, score=0.6, threshold=threshold
    )
    assert not rejected.switched


def test_runtime_counts_shortlist_and_fallback_attempts_exactly_once() -> None:
    actual = source_routing_actual_ids(
        ("shortlisted",),
        {"block": {"attempts": ["attempted", "train_median"]}},
        ("shortlisted", "attempted", "new"),
    )
    assert actual == ("attempted", "shortlisted")
    assert (
        derived_pipeline_runtime(
            10.0,
            switched=True,
            robust_candidate_id="shortlisted",
            robust_candidate_runtime_seconds=3.0,
            source_actual_candidate_ids=actual,
        )
        == 10.0
    )
    assert (
        derived_pipeline_runtime(
            10.0,
            switched=True,
            robust_candidate_id="attempted",
            robust_candidate_runtime_seconds=3.0,
            source_actual_candidate_ids=actual,
        )
        == 10.0
    )
    assert (
        derived_pipeline_runtime(
            10.0,
            switched=True,
            robust_candidate_id="new",
            robust_candidate_runtime_seconds=3.0,
            source_actual_candidate_ids=actual,
        )
        == 13.0
    )


def _row(dataset: str, episode: str, candidate: str, loss: float, valid: bool = True):
    return {
        "dataset_id": dataset,
        "episode_id": episode,
        "candidate_id": candidate,
        "imputation_loss": loss,
        "native_valid": valid,
        "candidate_status": "success" if valid else "failed",
    }


def test_robust_candidate_uses_dataset_equal_metrics_and_lexical_tie() -> None:
    plans = {"d1": ("e1", "e2"), "d2": ("e3", "e4")}
    rows = [
        _row("d1", "e1", "a", 0.2),
        _row("d1", "e2", "a", 0.2),
        _row("d2", "e3", "a", 0.2),
        _row("d2", "e4", "a", 0.2),
        _row("d1", "e1", "b", 0.2),
        _row("d1", "e2", "b", 0.2),
        _row("d2", "e3", "b", 0.2),
        _row("d2", "e4", "b", 0.2),
        _row("d1", "e1", "c", 0.0),
        _row("d1", "e2", "c", 0.0, False),
        _row("d2", "e3", "c", 0.0),
        _row("d2", "e4", "c", 0.0),
    ]
    result = select_robust_candidate(rows, plans, ("a", "b", "c"), availability_floor=0.95)
    assert result.selected_candidate_id == "a"
    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    assert not by_id["c"].eligible
    assert by_id["a"].dataset_equal_mean_loss == pytest.approx(0.2)
    assert by_id["b"].dataset_equal_mean_loss == pytest.approx(0.2)


def test_tail_helpers_and_exhaustive_thresholds() -> None:
    assert discrete_cvar(tuple(range(10)), 0.9) == 9.0
    assert discrete_cvar(tuple(range(11)), 0.9) == pytest.approx(9.5)
    assert weighted_empirical_quantile((1.0, 2.0, 3.0), (0.1, 0.8, 0.1), 0.9) == 2.0
    thresholds = enumerate_thresholds((None, 0.2, 0.1, 0.2))
    assert [item.kind for item in thresholds] == ["always", "finite", "finite", "disabled"]
    assert [item.value for item in thresholds[1:3]] == [0.1, 0.2]


def test_threshold_selection_uses_cell_equal_statistics_and_freezes_delta() -> None:
    episodes = (
        ThresholdEpisode("a1", "f1", "d1", 0.8, True, 4.0, 0.5, 1.0),
        ThresholdEpisode("a2", "f1", "d1", 0.2, True, 1.0, 1.0, 0.4),
        ThresholdEpisode("b1", "f2", "d2", 0.8, True, 4.0, 0.5, 1.0),
        ThresholdEpisode("b2", "f2", "d2", 0.2, True, 1.0, 1.0, 0.4),
    )
    result = select_threshold(episodes, expected_cells=(("f1", "d1"), ("f2", "d2")))
    assert result.selected_threshold == RiskThreshold("finite", 0.2)
    assert result.cell_count == 2
    assert result.episode_count == 4
    assert result.delta_ett == pytest.approx(3.0)
    selected = next(
        item for item in result.candidates if item.threshold == result.selected_threshold
    )
    assert selected.switched_count == 2
    assert selected.mean_mase == pytest.approx(0.75)


def test_threshold_selects_disabled_when_no_active_candidate_improves_tail() -> None:
    episodes = (
        ThresholdEpisode("a", "f", "d", 0.5, True, 1.0, 2.0, 0.5),
        ThresholdEpisode("b", "f", "d", None, False, 1.0, 0.0, 0.5),
    )
    result = select_threshold(episodes, expected_cells=(("f", "d"),))
    assert result.selected_threshold.kind == "disabled"


def test_threshold_mean_constraint_excludes_tail_improvement_above_two_percent() -> None:
    episodes = tuple(
        ThresholdEpisode(
            f"e{index}",
            "f",
            "d",
            0.5,
            True,
            100.0 if index == 0 else 1.0,
            0.0 if index == 0 else 2.5,
            1.0,
        )
        for index in range(100)
    )
    result = select_threshold(episodes, expected_cells=(("f", "d"),))
    always = next(item for item in result.candidates if item.threshold.kind == "always")
    assert always.cvar90_mase < result.disabled_cvar90_mase
    assert always.mean_mase > 1.02 * result.disabled_mean_mase
    assert "mean_mase_exceeds_1.02_times_disabled" in always.ineligibility_reasons
    assert result.selected_threshold.kind == "disabled"


def test_threshold_tie_prefers_finite_value_over_always() -> None:
    episodes = (
        ThresholdEpisode("minimum", "f", "d", 0.1, False, 1.0, 0.0, 0.5),
        ThresholdEpisode("high", "f", "d", 0.9, True, 10.0, 0.0, 0.5),
    )
    result = select_threshold(episodes, expected_cells=(("f", "d"),))
    assert result.selected_threshold == RiskThreshold("finite", 0.1)


def test_protocol_has_no_forecast_model_identity_input() -> None:
    assert set(RiskScoreProtocol.__dataclass_fields__) == {
        "forecast_mode",
        "target_indices",
        "max_pseudo_blocks",
        "tolerance",
    }
    unavailable_inputs = {
        "clean_future",
        "forecast_outputs",
        "forecast_loss",
        "forecaster_id",
        "model_revision",
    }
    for function in (
        deterministic_pseudo_observed_mask,
        channel_proxy_mae,
        score_block,
        score_episode,
        apply_whole_episode_fallback,
    ):
        assert unavailable_inputs.isdisjoint(inspect.signature(function).parameters)
