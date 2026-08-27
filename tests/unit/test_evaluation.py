from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tsfm_fais.contracts import ForecastResult
from tsfm_fais.evaluation import (
    SUMMARY_METRICS,
    _evaluate_episode,
    _imputation_metrics,
    _resolve_forecaster_artifact,
    forecast_metrics,
    parse_ids,
    summarize_evaluation,
)


def test_forecast_metrics_use_target_macro_clean_history_scales() -> None:
    context = np.column_stack(
        (
            np.arange(5, dtype=float),
            10.0 * np.arange(5, dtype=float),
        )
    )
    future = np.asarray([[5.0, 50.0], [6.0, 60.0]])
    predictions = np.asarray(
        [
            [[4.0, 40.0], [5.0, 50.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ]
    )
    result = ForecastResult(predictions, target_indices=(0, 1))

    metrics = forecast_metrics(context, future, result, seasonality=1)

    np.testing.assert_allclose(metrics["mase"], [1.0, 0.0])
    np.testing.assert_allclose(metrics["mae"], [5.5, 0.0])
    np.testing.assert_allclose(metrics["rmse"], [np.sqrt(50.5), 0.0])


def test_forecast_metrics_fall_back_to_lag_one_when_period_exceeds_context() -> None:
    context = np.asarray([[0.0], [1.0], [3.0], [6.0]])
    future = np.asarray([[7.0]])
    result = ForecastResult(np.asarray([[[5.0]]]), target_indices=(0,))

    metrics = forecast_metrics(context, future, result, seasonality=24)

    # Lag-one absolute differences are [1, 2, 3], so MASE = 2 / 2 = 1.
    np.testing.assert_allclose(metrics["mase"], [1.0])


def test_forecast_metrics_use_frozen_training_prefix_scale_when_supplied() -> None:
    context = np.asarray([[0.0], [100.0], [0.0], [100.0]])
    future = np.asarray([[2.0]])
    result = ForecastResult(np.asarray([[[0.0]]]), target_indices=(0,))

    metrics = forecast_metrics(
        context,
        future,
        result,
        seasonality=1,
        mase_scale=np.asarray([0.5]),
    )

    np.testing.assert_allclose(metrics["mase"], [4.0])


def test_forecast_metrics_ignore_unobserved_native_future_targets() -> None:
    context = np.column_stack((np.arange(4, dtype=float), np.arange(4, dtype=float)))
    future = np.asarray([[1.0, np.nan], [2.0, 20.0], [np.nan, 30.0]])
    observed = np.isfinite(future)
    result = ForecastResult(np.zeros((1, 3, 2), dtype=float), target_indices=(0, 1))

    metrics = forecast_metrics(
        context,
        future,
        result,
        seasonality=1,
        mase_scale=np.asarray([1.0, 10.0]),
        future_observed_mask=observed,
    )

    np.testing.assert_allclose(metrics["mase"], [2.0])
    np.testing.assert_allclose(metrics["mae"], [13.25])
    np.testing.assert_allclose(metrics["rmse"], [np.sqrt(326.25)])


def test_imputation_metrics_do_not_score_unknown_native_cells() -> None:
    truth = np.asarray([[1.0], [np.nan], [3.0]])
    candidate = np.asarray([[1.0], [999.0], [3.0]])
    observed = np.asarray([[True], [False], [True]])

    assert _imputation_metrics(truth, candidate, observed, np.isfinite(truth)) == (None, None)


def test_native_episode_evaluation_has_no_clean_reference_or_truth_leakage() -> None:
    clean_context = np.asarray([[1.0], [np.nan], [3.0], [4.0]])
    observed = np.isfinite(clean_context)
    assembled = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    candidate = np.asarray([[1.0], [1.5], [3.0], [4.0]])
    clean_future = np.asarray([[5.0], [np.nan]])
    archive = {
        "schema_version": np.asarray([3]),
        "values": assembled,
        "observed_mask": observed,
        "clean_context": clean_context,
        "clean_future": clean_future,
        "context_truth_mask": np.isfinite(clean_context),
        "future_observed_mask": np.isfinite(clean_future),
        "mask_protocol": np.asarray(["native_observation_mask_v1"]),
        "mask_seed": np.asarray([0]),
        "mask_realization_id": np.asarray(["native-mask"]),
        "target_missing_rate": np.asarray([0.0]),
        "global_missing_rate": np.asarray([0.25]),
        "local_missing_rate": np.asarray([0.25]),
        "mase_scale": np.asarray([1.0]),
        "mase_scale_lag": np.asarray([1]),
        "period": np.asarray([1]),
        "candidate_ids": np.asarray(["locf"]),
        "candidate_values": candidate[None, ...],
        "candidate_native_valid": np.ones((1, 4, 1), dtype=bool),
        "candidate_status": np.asarray(["success"]),
        "candidate_runtime_seconds": np.asarray([0.1]),
        "candidate_peak_memory_bytes": np.asarray([0]),
    }
    record = {
        "episode_id": "native__item__4__native__0__0",
        "dataset_id": "native",
        "family_id": "native",
        "item_id": "item",
        "forecast_origin": 4,
        "forecaster_id": "chronos2",
        "assembled_method_id": "b_fais",
    }
    config = SimpleNamespace(
        seed=7,
        experiment=SimpleNamespace(
            masking_protocol="native_only",
            training_base_mask="none",
            target_indices="all",
            horizon=2,
            context_length=4,
            forecast_num_samples=1,
        ),
    )

    class Predictor:
        def predict(self, contexts, spec):
            last = contexts[:, -1:, :][:, :, spec.target_indices]
            point = np.repeat(last, spec.horizon, axis=1)
            return ForecastResult(point, target_indices=spec.target_indices)

    rows = _evaluate_episode(record, archive, config, "chronos2", Predictor(), ())

    assert {row["method"] for row in rows} == {"b_fais", "locf", "oracle"}
    assert all(row["clean_reference_available"] is False for row in rows)
    assert all(row["degradation_vs_clean_mase"] is None for row in rows)
    assert all(row["imputation_mae"] is None for row in rows)
    assert all(np.isfinite(row["mase"]) for row in rows)


def test_parse_ids_is_strict_and_preserves_order() -> None:
    assert parse_ids("locf, linear_interp") == ("locf", "linear_interp")


def test_forecaster_artifact_mapping_resolves_one_requested_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights" / "chronos"
    checkpoint.mkdir(parents=True)
    mapping = tmp_path / "forecasters.json"
    mapping.write_text(
        json.dumps({"artifacts": {"chronos2": "weights/chronos"}}),
        encoding="utf-8",
    )

    assert _resolve_forecaster_artifact(mapping, "chronos2") == checkpoint.resolve()


def test_summary_excludes_invalid_candidate_fallback_metrics(tmp_path: Path) -> None:
    valid = {metric: 1.0 for metric in SUMMARY_METRICS}
    invalid = {metric: 999.0 for metric in SUMMARY_METRICS}
    rows = (
        {"method": "candidate", "metric_eligible": True, **valid},
        {"method": "candidate", "metric_eligible": False, **invalid},
    )
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = summarize_evaluation(
        metrics_path=source,
        output_dir=tmp_path / "summary",
        group_by=("method",),
    )

    payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    group = payload["groups"][0]
    assert group["count"] == 2
    assert group["metric_count"] == 1
    assert group["invalid_count"] == 1
    assert group["invalid_rate"] == 0.5
    assert group["mase_mean"] == 1.0
