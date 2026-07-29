from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.contracts import ForecastSpec, SeriesBatch, TimeSeriesItem
from tsfm_fais.routing.hybrid_lstm_sequence import (
    HYBRID_LSTM_PAPER_CANDIDATES,
    HYBRID_LSTM_THESIS_CANDIDATES,
    HybridLSTMSequencePipeline,
    HybridLSTMSequenceSelector,
    _encode_windows,
    _paper_impute_1d,
    hybrid_window_label_rows,
)
from tsfm_fais.routing.models import PairwiseRiskModel, RouterBundle
from tsfm_fais.stage_execution import _desired_actual_mode


def _batch(clean: np.ndarray, hidden: tuple[int, ...]) -> SeriesBatch:
    values = np.asarray(clean, dtype=float)[None, :, None]
    observed = np.ones_like(values, dtype=bool)
    observed[0, list(hidden), 0] = False
    return SeriesBatch(values=values, observed_mask=observed, item_ids=("item",))


def _rows(
    batch: SeriesBatch,
    clean: np.ndarray,
    window_size: int = 8,
    episode_id: str = "episode",
):
    return hybrid_window_label_rows(
        batch,
        np.asarray(clean, dtype=float)[:, None],
        episode_id=episode_id,
        dataset_id="dataset",
        family_id="family",
        item_id="item",
        forecast_origin=len(clean),
        period=None,
        window_size=window_size,
    )


def _small_selector(**overrides) -> HybridLSTMSequenceSelector:
    params = {
        "window_size": 8,
        "epochs": 1,
        "batch_size": 4,
        "filters": 4,
        "kernel_size": 3,
        "hidden_size": 4,
        "dense_size": 4,
        "dropout": 0.0,
        "l1_strength": 0.0,
        "learning_rate": 1e-3,
    }
    params.update(overrides)
    return HybridLSTMSequenceSelector.from_params(params, seed=7)


def test_paper_candidate_constants_distinguish_the_pix2pix_thesis_study() -> None:
    assert HYBRID_LSTM_PAPER_CANDIDATES == (
        "mean",
        "median",
        "linear",
        "cubic",
        "akima",
        "polynomial_5",
        "spline_5",
        "moving_mean_3",
        "backfill",
        "forward_fill",
    )
    assert HYBRID_LSTM_THESIS_CANDIDATES == (*HYBRID_LSTM_PAPER_CANDIDATES, "pix2pix")


def test_hybrid_network_input_is_one_raw_value_channel() -> None:
    values = np.asarray([[10.0, np.nan, -3.0]], dtype=float)
    observed = np.asarray([[True, False, True]], dtype=bool)

    encoded = _encode_windows(values, observed)

    assert encoded.shape == (1, 1, 3)
    np.testing.assert_array_equal(encoded[0, 0], [10.0, 0.0, -3.0])


def test_polynomial_candidate_uses_fifth_degree_lagrange_interpolation() -> None:
    positions = np.arange(10, dtype=float)
    clean = 0.01 * positions**5 - 0.2 * positions**3 + 2.0 * positions + 7.0
    incomplete = clean.copy()
    incomplete[[2, 7]] = np.nan

    completed, native_valid = _paper_impute_1d("polynomial_5", incomplete)

    assert native_valid is True
    np.testing.assert_allclose(completed, clean, rtol=0.0, atol=1e-9)


def test_hybrid_window_rows_use_fixed_univariate_windows_and_paper_methods() -> None:
    clean = np.arange(12, dtype=float)
    batch = _batch(clean, (2, 6, 9))
    rows = _rows(batch, clean)

    # [0,8) followed by an end-aligned [4,12), with ten methods per window.
    assert len(rows) == 2 * len(HYBRID_LSTM_PAPER_CANDIDATES)
    assert {row["candidate_id"] for row in rows} == set(HYBRID_LSTM_PAPER_CANDIDATES)
    assert {(row["window_start"], row["window_end"]) for row in rows} == {(0, 8), (4, 12)}
    assert {row["label_scope"] for row in rows} == {"hybrid_window"}
    assert all(str(row["group_id"]).startswith("imputation::episode::") for row in rows)
    assert all(np.isfinite(float(row["imputation_loss"])) for row in rows)
    assert all(np.isfinite(float(row["imputation_mae"])) for row in rows)
    assert all(np.isfinite(float(row["imputation_rmse"])) for row in rows)
    assert all(0.0 <= float(row["imputation_reward"]) <= 1.0 for row in rows)

    first = rows[0]["prior_features"]
    assert first["hybrid_observed_002"] == 0.0
    assert first["hybrid_value_002"] == 0.0
    assert first["hybrid_padding_007"] == 1.0
    assert rows[0]["prior_features"] is not rows[0]["unary_features"]
    assert rows[0]["prior_features"]
    assert all(row["prior_features"] == {} for row in rows[1:10])
    assert all(row["unary_features"] == {} for row in rows)


def test_hybrid_window_rows_skip_windows_without_missing_values() -> None:
    clean = np.arange(12, dtype=float)
    batch = _batch(clean, (2,))

    rows = _rows(batch, clean)

    assert len(rows) == len(HYBRID_LSTM_PAPER_CANDIDATES)
    assert {(row["window_start"], row["window_end"]) for row in rows} == {(0, 8)}


def test_label_rows_reject_clean_context_that_disagrees_with_observations() -> None:
    clean = np.arange(8, dtype=float)
    batch = _batch(clean, (3,))
    wrong = clean.copy()
    wrong[0] = 99.0
    with pytest.raises(ValueError, match="agree with observed"):
        _rows(batch, wrong)


def test_selector_fits_rows_and_completes_without_using_missing_blocks() -> None:
    pytest.importorskip("torch")
    first = np.arange(8, dtype=float)
    second = np.square(np.arange(8, dtype=float))
    first_batch = _batch(first, (2, 5))
    second_batch = _batch(second, (2, 5))
    rows = (
        *_rows(first_batch, first, episode_id="first"),
        *_rows(second_batch, second, episode_id="second"),
    )

    selector = _small_selector().fit_from_rows(rows)
    assert selector.fitted_window_count == 2
    logits = selector.predict_logits(
        first_batch.values[0, :, 0][None, :],
        observed_mask=first_batch.observed_mask[0, :, 0][None, :],
    )
    assert logits.shape == (1, len(HYBRID_LSTM_PAPER_CANDIDATES))
    assert np.isfinite(logits).all()

    completed = selector.complete(first_batch)
    assert completed.shape == first_batch.shape
    assert np.isfinite(completed).all()
    np.testing.assert_array_equal(
        completed[first_batch.observed_mask],
        first_batch.values[first_batch.observed_mask],
    )


def test_selector_requires_fit_and_validates_complete_candidate_groups() -> None:
    clean = np.arange(8, dtype=float)
    batch = _batch(clean, (3,))
    selector = _small_selector()
    with pytest.raises(RuntimeError, match="must be fitted"):
        selector.complete(batch)

    rows = list(_rows(batch, clean))
    rows.pop()
    with pytest.raises(ValueError, match="missing candidates"):
        selector.fit_from_rows(rows)


def test_selector_balances_represented_best_method_classes() -> None:
    pytest.importorskip("torch")
    windows = np.tile(np.arange(8, dtype=float), (6, 1))
    losses = np.ones((6, len(HYBRID_LSTM_PAPER_CANDIDATES)), dtype=float)
    best_classes = (0, 0, 0, 1, 1, 2)
    for row, candidate in enumerate(best_classes):
        losses[row, candidate] = 0.0

    selector = _small_selector().fit(windows, losses)

    assert selector.fitted_window_count == 3


def test_pipeline_runs_internal_completion_with_empty_registry_shortlist() -> None:
    pytest.importorskip("torch")
    clean = np.arange(8, dtype=float)
    batch = _batch(clean, (2, 5))
    selector = _small_selector().fit_from_rows(_rows(batch, clean))
    router = RouterBundle(
        prior=selector,
        unary=selector,
        pairwise=PairwiseRiskModel(),
        feature_names=(),
        candidate_ids=HYBRID_LSTM_PAPER_CANDIDATES,
        metadata={"selector_method": "hybrid_lstm", "requires_pseudo_candidates": False},
    )
    pipeline = HybridLSTMSequencePipeline(router=router)
    item = TimeSeriesItem(
        item_id="item",
        values=batch.values[0],
        variate_names=("x",),
        start=pd.Timestamp("2020-01-01"),
        freq="h",
    )
    plan = pipeline.prepare_route(
        item,
        batch.observed_mask[0],
        ForecastSpec(model_id="mock", mode="joint_multivariate", horizon=1),
    )
    assert plan.shortlist == ()
    assert plan.candidate_ids == ()
    result = pipeline.finish_route(plan, {})
    assert result.candidates == {}
    assert np.isfinite(result.values).all()
    assert result.routing.metadata["selection_scope"] == "fixed_univariate_window"
    assert result.routing.metadata["uses_missing_block_graph"] is False


def test_hybrid_keeps_registry_candidates_evaluation_only_when_saved() -> None:
    work = SimpleNamespace(plan=SimpleNamespace())
    pipeline = SimpleNamespace(uses_registry_candidates=False)
    evaluation_ids = frozenset({"locf"})

    assert (
        _desired_actual_mode(
            work,
            "locf",
            pipeline,
            evaluation_ids,
            save_all=True,
        )
        == "evaluation"
    )
    assert (
        _desired_actual_mode(
            work,
            "locf",
            pipeline,
            evaluation_ids,
            save_all=False,
        )
        is None
    )
