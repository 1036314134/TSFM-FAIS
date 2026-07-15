from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tsfm_fais.contracts import ForecastResult
from tsfm_fais.evaluation import (
    SUMMARY_METRICS,
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


def test_parse_ids_is_strict_and_preserves_order() -> None:
    assert parse_ids("locf, linear_interp") == ("locf", "linear_interp")


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
