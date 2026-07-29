from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.config import load_config
from tsfm_fais.contracts import (
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner, ImputerRegistry
from tsfm_fais.label_resume import LabelEpisodeExpectation, validate_label_rows
from tsfm_fais.pipeline import BlockwiseFAIS
from tsfm_fais.routing.hybrid_lstm_sequence import hybrid_window_label_rows
from tsfm_fais.routing.models import PairwiseRiskModel, RouterBundle
from tsfm_fais.routing.sequence_pipeline import WholeSeriesSelectorFAIS
from tsfm_fais.stage_execution import (
    _build_sequence_label_episode_rows,
    _fit_router_bundle,
    _prepare_dselect_training_rows,
    _sequence_imputation_metrics,
    _sequence_training_matrices,
)


@dataclass
class _HighScoreWins:
    scores: np.ndarray

    def score(self, features: np.ndarray) -> np.ndarray:
        assert features.ndim == 1
        return np.asarray(self.scores, dtype=float)


class _NoArtifactManager:
    def __init__(self, registry: ImputerRegistry) -> None:
        self.registry = registry

    def acquire(self, candidate_ids: tuple[str, ...]):
        del candidate_ids
        return {}, {}, ()

    def release(self, artifacts, ephemeral_ids) -> None:
        del artifacts, ephemeral_ids


class _RewardRecorder:
    def __init__(self) -> None:
        self.observed: tuple[np.ndarray, str, float] | None = None

    def observe(self, features: np.ndarray, candidate_id: str, reward: float) -> None:
        self.observed = (np.asarray(features, dtype=float), candidate_id, float(reward))


def _candidate(candidate_id: str, batch: SeriesBatch, fill: float) -> CandidateResult:
    values = np.where(batch.observed_mask, batch.values, fill)
    return CandidateResult(
        imputer_id=candidate_id,
        values=values,
        native_valid_mask=np.ones(batch.shape, dtype=bool),
        status=CandidateStatus.SUCCESS,
    )


def _heterogeneous_whole_series_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = (
        ("e0", 0.0, (("locf", 0.1), ("linear_interp", 0.2), ("brits", 0.3))),
        ("e1", 1.0, (("locf", 0.3), ("linear_interp", 0.1))),
        ("e2", 2.0, (("locf", 0.2), ("brits", 0.1))),
    )
    target = [0.25, 0.75]
    for episode_id, signal, candidates in groups:
        for candidate_index, (candidate_id, loss) in enumerate(candidates):
            row: dict[str, object] = {
                "episode_id": episode_id,
                "dataset_id": "dataset",
                "family_id": "family",
                "forecaster_id": "imputation",
                "group_id": f"imputation::{episode_id}::__sequence__",
                "block_id": "__sequence__",
                "candidate_id": candidate_id,
                "label_scope": "whole_series",
                "prior_features": {"signal": signal} if candidate_index == 0 else {},
                "unary_features": {},
                "imputation_loss": loss,
                "imputation_mae": loss,
                "imputation_rmse": loss,
                "imputation_reward": 1.0 / (1.0 + loss),
                "native_valid": True,
                "candidate_status": "success",
                "dselect_expert_values": [loss, 1.0 - loss],
            }
            if candidate_index == 0:
                row["dselect_target_values"] = target
            rows.append(row)
    return rows


def _hybrid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode_id, clean in (
        ("h0", np.arange(1, 9, dtype=float)),
        ("h1", np.square(np.arange(1, 9, dtype=float))),
    ):
        values = clean[None, :, None]
        observed = np.ones_like(values, dtype=bool)
        observed[0, (2, 5), 0] = False
        rows.extend(
            hybrid_window_label_rows(
                SeriesBatch(values=values, observed_mask=observed),
                clean[:, None],
                episode_id=episode_id,
                dataset_id="dataset",
                family_id="family",
                item_id="item",
                forecast_origin=8,
                period=None,
                window_size=8,
            )
        )
    return rows


def test_sequence_metrics_use_all_hidden_context_positions() -> None:
    clean = np.asarray([[[1.0], [2.0], [3.0], [4.0]]])
    observed = np.asarray([[[True], [False], [True], [False]]])
    batch = SeriesBatch(values=clean, observed_mask=observed)
    result = _candidate("locf", batch, 3.0)

    loss, mae, rmse, reward, valid = _sequence_imputation_metrics(
        clean,
        result,
        ~observed,
    )

    expected_errors = np.asarray([1.0, -1.0])
    assert valid is True
    assert np.isclose(mae, np.mean(np.abs(expected_errors)))
    assert np.isclose(rmse, np.sqrt(np.mean(expected_errors**2)))
    assert np.isclose(loss, np.mean([1.0 / 5.0, 1.0 / 7.0]))
    assert np.isclose(reward, 1.0 / (1.0 + loss))


def test_sequence_label_rows_form_one_forecaster_independent_group() -> None:
    clean = np.arange(1, 9, dtype=float)[:, None]
    observed = np.ones_like(clean, dtype=bool)
    observed[[2, 3], 0] = False
    context = SeriesBatch(
        values=clean[None, ...],
        observed_mask=observed[None, ...],
        item_ids=("item",),
        metadata={"period": 4},
    )
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    episode = SimpleNamespace(
        context=context,
        clean_context=clean,
        item_id="item",
        forecast_origin=8,
        seed=7,
    )
    dataset = SimpleNamespace(
        dataset_id="dataset",
        family_id="family",
        period=4,
    )

    built = _build_sequence_label_episode_rows(
        load_config("configs/main_rolling_train_baselines.yaml"),
        dataset,
        "dataset__item__8__random_point__0.2__22",
        episode,
        ("metaod", "dselect1"),
        registry.ids,
        _NoArtifactManager(registry),
        CandidateRunner(registry),
        ("cpu",),
    )

    assert len(built.unary_rows) == 2
    assert {row["label_scope"] for row in built.unary_rows} == {"whole_series"}
    assert {row["forecaster_id"] for row in built.unary_rows} == {"imputation"}
    assert {row["group_id"] for row in built.unary_rows} == {
        "imputation::dataset__item__8__random_point__0.2__22::__sequence__"
    }
    assert all("forecast_loss" not in row for row in built.unary_rows)
    assert "dselect_target_values" in built.unary_rows[0]
    assert all("dselect_expert_values" in row for row in built.unary_rows)
    assert len(built.unary_rows[0]["dselect_target_values"]) == 2
    expectation = LabelEpisodeExpectation(
        artifact_index=0,
        forecaster_id="imputation",
        episode_id="dataset__item__8__random_point__0.2__22",
        dataset_id="dataset",
        family_id="family",
        item_id="item",
        forecast_origin=8,
        sampling_cell={},
        dataset_plan_sha256="0" * 64,
        candidate_ids=built.candidate_ids,
        block_ids=built.block_ids,
    )
    validated = validate_label_rows(
        expectation,
        built.unary_rows,
        (),
        outcome="labeled",
    )
    assert validated["unary_row_count"] == 2


def test_sequence_training_matrices_preserve_union_and_mark_missing_candidates_invalid() -> None:
    rows = _heterogeneous_whole_series_rows()

    contexts, losses, validity, candidates, feature_names, ordered_rows = (
        _sequence_training_matrices(rows, label_scope="whole_series")
    )

    assert candidates == ("brits", "linear_interp", "locf")
    assert feature_names == ("signal",)
    np.testing.assert_array_equal(contexts[:, 0], [0.0, 1.0, 2.0])
    assert np.isnan(losses[1, 0])
    assert np.isnan(losses[2, 1])
    np.testing.assert_array_equal(
        validity,
        [
            [True, True, True],
            [False, True, True],
            [True, False, True],
        ],
    )
    assert len(ordered_rows) == len(rows)


def test_dselect_stage_rows_are_dense_and_reference_is_order_independent() -> None:
    rows = _heterogeneous_whole_series_rows()
    _, _, _, candidates, _, ordered_rows = _sequence_training_matrices(
        rows,
        label_scope="whole_series",
    )

    dense = _prepare_dselect_training_rows(ordered_rows, candidates)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in dense:
        grouped.setdefault(str(row["group_id"]), []).append(row)

    assert all(
        tuple(str(row["candidate_id"]) for row in group) == candidates for group in grouped.values()
    )
    for group in grouped.values():
        assert group[0]["prior_features"]
        assert group[0]["dselect_target_values"] == [0.25, 0.75]
    missing_brits = grouped["imputation::e1::__sequence__"][0]
    assert missing_brits["candidate_id"] == "brits"
    assert missing_brits["native_valid"] is False
    assert missing_brits["dselect_expert_values"] == [0.0, 0.0]


@pytest.mark.parametrize(
    ("selector_method", "selector_params"),
    (
        (
            "metaod",
            {"latent_dim": 2, "epochs": 1, "n_estimators": 2, "max_depth": 2},
        ),
        (
            "dselect1",
            {"epochs": 1, "batch_size": 2, "torch_threads": 1},
        ),
        (
            "neuralucb",
            {
                "hidden_size": 2,
                "training_steps": 1,
                "retrain_interval": 10,
            },
        ),
        (
            "alors",
            {
                "latent_dim": 2,
                "epochs": 1,
                "n_estimators": 2,
                "ndcg_cutoff": 2,
            },
        ),
        (
            "hybrid_lstm",
            {
                "window_size": 8,
                "epochs": 1,
                "batch_size": 2,
                "filters": 4,
                "kernel_size": 3,
                "hidden_size": 4,
                "dense_size": 4,
                "dropout": 0.0,
                "l1_strength": 0.0,
                "learning_rate": 1e-3,
                "torch_threads": 1,
            },
        ),
        ("random_valid_block", {}),
    ),
)
def test_all_sequence_routers_train_with_heterogeneous_whole_series_candidate_pools(
    tmp_path,
    selector_method: str,
    selector_params: dict[str, object],
) -> None:
    pytest.importorskip("torch")
    rows = [*_heterogeneous_whole_series_rows(), *_hybrid_rows()]

    bundle = _fit_router_bundle(
        rows,
        [],
        tmp_path / selector_method,
        {"split": "rolling_origin", "ranker_target": "imputation_loss"},
        selector_method=selector_method,
        selector_params=selector_params,
        seed=11,
    )

    expected_method = (
        "random_valid_series" if selector_method == "random_valid_block" else selector_method
    )
    assert bundle.metadata["selector_method"] == expected_method
    assert bundle.metadata["forecaster_independent_selection"] is True
    if selector_method == "hybrid_lstm":
        assert len(bundle.candidate_ids) == 10
    else:
        assert bundle.candidate_ids == ("brits", "linear_interp", "locf")
    if selector_method == "neuralucb":
        selected = [bundle.candidate_ids[index] for index in bundle.prior.selected_actions]
        assert selected[1] != "brits"
        assert selected[2] != "linear_interp"


def test_whole_series_pipeline_selects_once_and_ignores_forecaster_mode() -> None:
    clean = np.arange(8, dtype=float)[:, None]
    observed = np.ones_like(clean, dtype=bool)
    observed[[1, 2, 6], 0] = False
    batch = SeriesBatch(values=clean[None, ...], observed_mask=observed[None, ...])
    item = TimeSeriesItem(
        item_id="item",
        values=batch.values[0],
        variate_names=("x",),
        start=pd.Timestamp("2020-01-01"),
        freq="h",
    )
    router = RouterBundle(
        prior=_HighScoreWins(np.asarray([0.0, 1.0])),
        unary=None,
        pairwise=PairwiseRiskModel(),
        feature_names=(),
        candidate_ids=("locf", "linear_interp"),
        metadata={
            "selector_method": "metaod",
            "requires_pseudo_candidates": False,
        },
    )
    pipeline = WholeSeriesSelectorFAIS(router=router)
    candidates = {
        "locf": _candidate("locf", batch, -1.0),
        "linear_interp": _candidate("linear_interp", batch, 7.0),
    }
    joint = ForecastSpec(
        model_id="chronos2",
        mode="joint_multivariate",
        horizon=2,
        target_indices=(0,),
    )
    univariate = ForecastSpec(
        model_id="timesfm2p5",
        mode="independent_univariate",
        horizon=2,
        target_indices=(0,),
    )

    first_plan = pipeline.prepare_route(item, observed, joint, seed=17)
    second_plan = pipeline.prepare_route(item, observed, univariate, seed=17)
    first = pipeline.finish_route(first_plan, candidates)
    second = pipeline.finish_route(second_plan, candidates)

    assert set(first.routing.assignments.values()) == {"linear_interp"}
    assert first.routing.assignments == second.routing.assignments
    np.testing.assert_array_equal(first.values, second.values)
    assert first.routing.metadata["selection_count"] == 1
    assert first.routing.metadata["uses_missing_block_graph"] is False
    assert first.candidates == {}


def test_neuralucb_online_reward_matches_sequence_label_asymape() -> None:
    recorder = _RewardRecorder()
    router = RouterBundle(
        prior=recorder,
        unary=None,
        pairwise=PairwiseRiskModel(),
        feature_names=("signal",),
        candidate_ids=("locf",),
        metadata={"selector_method": "neuralucb"},
    )
    pipeline = WholeSeriesSelectorFAIS(router=router)
    pipeline._pending_online_feedback = {
        "prediction": np.asarray([[3.0], [3.0]]),
        "missing_mask": np.asarray([[True], [True]]),
        "features": np.asarray([0.5]),
        "candidate_id": "locf",
    }

    reward = pipeline.observe_outcome(np.asarray([[2.0], [4.0]]))

    expected_loss = np.mean([1.0 / 5.0, 1.0 / 7.0])
    expected_reward = 1.0 / (1.0 + expected_loss)
    assert np.isclose(reward, expected_reward)
    assert recorder.observed is not None
    assert recorder.observed[1] == "locf"
    assert np.isclose(recorder.observed[2], expected_reward)


def test_whole_series_prepare_route_bypasses_block_routing(monkeypatch) -> None:
    clean = np.column_stack((np.arange(8, dtype=float), np.arange(20, 28, dtype=float)))
    observed = np.ones_like(clean, dtype=bool)
    # The only missing coordinates are outside the forecast target.  A
    # whole-series selector must still retain its complete candidate pool.
    observed[[1, 2, 6], 1] = False
    item = TimeSeriesItem(
        item_id="item",
        values=clean,
        variate_names=("target", "other"),
        start=pd.Timestamp("2020-01-01"),
        freq="h",
        metadata={"period": 4, "training_correlation": np.eye(2)},
    )
    router = RouterBundle(
        prior=_HighScoreWins(np.asarray([0.0, 1.0])),
        unary=None,
        pairwise=PairwiseRiskModel(),
        feature_names=(),
        candidate_ids=("locf", "linear_interp"),
        metadata={
            "selector_method": "metaod",
            "requires_pseudo_candidates": False,
        },
    )
    pipeline = WholeSeriesSelectorFAIS(router=router)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("whole-series preparation entered block routing")

    monkeypatch.setattr(BlockwiseFAIS, "prepare_route", forbidden)
    monkeypatch.setattr(BlockwiseFAIS, "_heuristic_unary", forbidden)
    monkeypatch.setattr(BlockwiseFAIS, "_pseudo_batch", forbidden)
    monkeypatch.setattr("tsfm_fais.pipeline.build_block_graph", forbidden)
    monkeypatch.setattr("tsfm_fais.pipeline.greedy_shortlist", forbidden)

    plan = pipeline.prepare_route(
        item,
        observed,
        ForecastSpec(
            model_id="timesfm2p5",
            mode="independent_univariate",
            horizon=2,
            target_indices=(0,),
        ),
        available_artifact_ids=(),
        seed=23,
    )

    assert plan.candidate_ids == ("locf", "linear_interp")
    assert plan.shortlist == plan.candidate_ids
    assert plan.prior_unary == {}
    assert plan.graph.blocks == ()
    assert plan.graph.edges == ()
    assert plan.correlation_source == "none"
    assert plan.pseudo_batch is None
    assert plan.backtest_batch is None
    assert plan.backtest_cutoff is None


def test_sequence_router_training_records_imputation_protocol(tmp_path) -> None:
    rows = []
    for episode, signal in (("e0", 0.0), ("e1", 1.0)):
        for candidate_id, loss in (("locf", 0.2 + signal), ("linear_interp", 1.2 - signal)):
            rows.append(
                {
                    "episode_id": episode,
                    "dataset_id": "dataset",
                    "family_id": "family",
                    "forecaster_id": "imputation",
                    "group_id": f"imputation::{episode}::__sequence__",
                    "block_id": "__sequence__",
                    "candidate_id": candidate_id,
                    "label_scope": "whole_series",
                    "prior_features": {"signal": signal},
                    "unary_features": {"signal": signal},
                    "imputation_loss": loss,
                    "imputation_mae": loss,
                    "imputation_rmse": loss,
                    "imputation_reward": 1.0 / (1.0 + loss),
                    "native_valid": True,
                }
            )

    bundle = _fit_router_bundle(
        rows,
        [],
        tmp_path / "router",
        {"split": "rolling_origin", "ranker_target": "imputation_loss"},
        selector_method="random_valid_block",
        seed=11,
    )

    assert bundle.metadata["selector_method"] == "random_valid_series"
    assert bundle.metadata["selection_scope"] == "whole_series"
    assert bundle.metadata["selector_training_target"] == "imputation_loss"
    assert bundle.metadata["uses_missing_block_graph"] is False
    assert bundle.metadata["forecaster_independent_selection"] is True
    assert bundle.candidate_ids == ("linear_interp", "locf")
