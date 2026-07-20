from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_fais.main_results import summarize_multi_forecaster


def _evaluation(root: Path, forecaster_id: str, family_id: str) -> Path:
    evaluation = root / forecaster_id
    evaluation.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    specifications = (
        ("episode-0", "independent_block", 0.1, True),
        ("episode-1", "mixed_outage", 0.2, False),
    )
    methods = (
        ("clean", "reference", 1.0, 0.0, 0.0),
        ("b_fais", "method", 2.0, 0.5, 2.0),
        ("locf", "missing_anchor", 3.0, 1.5, 3.0),
        ("linear_interp", "baseline", 1.5, 0.0, 1.0),
        ("extra", "baseline", 4.0, 2.5, 4.0),
        ("oracle", "oracle", 1.5, 0.0, 1.0),
    )
    for episode_id, mechanism, missing_rate, extra_valid in specifications:
        for method, role, mase, regret, imputation_error in methods:
            eligible = method != "extra" or extra_valid
            rows.append(
                {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "dataset_id": f"dataset-{family_id}",
                    "family_id": family_id,
                    "forecaster_id": forecaster_id,
                    "mechanism": mechanism,
                    "missing_rate": missing_rate,
                    "item_id": f"item-{episode_id}",
                    "mask_protocol": "sequence_mask_v2",
                    "mask_realization_id": f"mask-{episode_id}",
                    "contains_missing": episode_id == "episode-0",
                    "method": method,
                    "method_role": role,
                    "metric_eligible": eligible,
                    "native_valid": eligible,
                    "mase": mase if eligible else None,
                    "mae": 10.0 * mase if eligible else None,
                    "rmse": 100.0 * mase if eligible else None,
                    "imputation_mae": imputation_error if eligible else None,
                    "imputation_rmse": 2.0 * imputation_error if eligible else None,
                    "regret_mase": regret if eligible else None,
                    "degradation_vs_clean_mase": mase - 1.0 if eligible else None,
                    "runtime_seconds": {
                        "clean": 0.0,
                        "b_fais": 2.0,
                        "locf": 0.1,
                        "linear_interp": 0.2,
                        "extra": 0.3,
                        "oracle": 0.2,
                    }[method],
                    "rss_delta_bytes": 0,
                }
            )
    (evaluation / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (evaluation / "evaluation_manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    return evaluation


def _selector_evaluation(
    root: Path,
    source: Path,
    method: str,
    mase: float,
) -> Path:
    evaluation = root / method
    evaluation.mkdir(parents=True)
    source_rows = [
        json.loads(line)
        for line in (source / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows = [dict(row) for row in source_rows if row["method"] != "b_fais"]
    for b_fais in (row for row in source_rows if row["method"] == "b_fais"):
        selector = dict(b_fais)
        selector.update(
            {
                "method": method,
                "method_role": "selector_baseline",
                "mase": mase,
                "mae": 10.0 * mase,
                "rmse": 100.0 * mase,
                "imputation_mae": mase,
                "imputation_rmse": 2.0 * mase,
                "regret_mase": mase - 1.5,
                "degradation_vs_clean_mase": mase - 1.0,
                "runtime_seconds": 1.0,
            }
        )
        rows.append(selector)
    (evaluation / "episode_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (evaluation / "evaluation_manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    return evaluation


def test_multi_forecaster_summary_is_paired_grouped_and_reproducible(
    tmp_path: Path,
) -> None:
    first = _evaluation(tmp_path, "forecast-a", "family-a")
    second = _evaluation(tmp_path, "forecast-b", "family-b")

    result = summarize_multi_forecaster(
        evaluation_inputs=(first, second),
        output_dir=tmp_path / "summary-a",
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    repeated = summarize_multi_forecaster(
        evaluation_inputs=(second, first),
        output_dir=tmp_path / "summary-b",
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    payload = json.loads(
        Path(result["main_summary_json"]).read_text(encoding="utf-8")
    )
    repeated_payload = json.loads(
        Path(repeated["main_summary_json"]).read_text(encoding="utf-8")
    )
    assert payload == repeated_payload
    assert result["episode_count"] == 4
    assert result["forecaster_count"] == 2
    assert result["method_group_count"] == 54
    assert result["comparison_group_count"] == 45

    overall_b_fais = next(
        row
        for row in payload["method_summary"]
        if row["scope"] == "overall" and row["method"] == "b_fais"
    )
    assert overall_b_fais["mase_mean"] == 2.0
    assert overall_b_fais["mae_mean"] == 20.0
    assert overall_b_fais["rmse_mean"] == 200.0
    assert overall_b_fais["imputation_mae_mean"] == 2.0
    assert overall_b_fais["imputation_rmse_mean"] == 4.0
    assert overall_b_fais["average_rank_mase"] == 2.0
    assert overall_b_fais["regret_mase_mean"] == 0.5
    assert overall_b_fais["runtime_seconds_mean"] == 2.0

    overall_extra = next(
        row
        for row in payload["method_summary"]
        if row["scope"] == "overall" and row["method"] == "extra"
    )
    assert overall_extra["recorded_count"] == 4
    assert overall_extra["valid_count"] == 2
    assert overall_extra["valid_rate"] == 0.5

    locf = next(
        row
        for row in payload["comparison_summary"]
        if row["scope"] == "overall" and row["comparator"] == "locf"
    )
    assert locf["pair_count"] == 4
    assert locf["mase_mean_delta"] == -1.0
    assert locf["mase_mean_delta_ci95_low"] == -1.0
    assert locf["mase_mean_delta_ci95_high"] == -1.0
    assert locf["mase_win_rate"] == 1.0
    assert locf["imputation_mae_pair_count"] == 4
    assert locf["imputation_mae_mean_delta"] == -1.0
    assert locf["imputation_mae_mean_delta_ci95_low"] == -1.0
    assert locf["imputation_mae_mean_delta_ci95_high"] == -1.0
    assert locf["imputation_mae_win_rate"] == 1.0
    assert locf["imputation_rmse_mean_delta"] == -2.0
    assert locf["imputation_rmse_win_rate"] == 1.0
    assert locf["b_fais_average_rank_mase"] == 2.0
    assert locf["comparator_average_rank_mase"] == 3.0

    one_pair = next(
        row
        for row in payload["comparison_summary"]
        if row["scope"] == "forecaster"
        and row["group_value"] == "forecast-a"
        and row["comparator"] == "extra"
    )
    assert one_pair["pair_count"] == 1
    assert one_pair["bootstrap_replicates"] == 0
    assert one_pair["mase_mean_delta_ci95_low"] is None
    assert one_pair["imputation_mae_pair_count"] == 1
    assert one_pair["imputation_mae_mean_delta_ci95_low"] is None
    assert one_pair["small_sample"] is True

    assert Path(result["method_summary_csv"]).is_file()
    assert Path(result["comparison_summary_csv"]).is_file()
    assert Path(result["family_macro_comparison_summary_csv"]).is_file()
    primary = payload["family_macro_comparison_summary"]
    primary_locf = next(
        row
        for row in primary
        if row["scope"] == "overall"
        and row["view"] == "all_windows"
        and row["comparator"] == "locf"
    )
    assert primary_locf["family_count"] == 2
    assert primary_locf["mase_family_macro_delta"] == -1.0
    assert primary_locf["mase_family_macro_delta_ci95_low"] == -1.0
    assert primary_locf["mase_family_macro_delta_ci95_high"] == -1.0
    report = Path(result["report_markdown"]).read_text(encoding="utf-8")
    assert "does not claim statistical significance" in report
    assert "Oracle is the valid single imputer selected by minimum MASE" in report
    assert "not an imputation-error oracle" in report
    assert "not executable when the input contains missing values" in report
    assert "Overall paired imputation-error comparisons" in report


def test_multi_forecaster_summary_rejects_duplicate_episode_method_rows(
    tmp_path: Path,
) -> None:
    source = _evaluation(tmp_path, "forecast-a", "family-a")
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        (source / "episode_metrics.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate forecaster/episode/method"):
        summarize_multi_forecaster(
            evaluation_inputs=(source, duplicate),
            output_dir=tmp_path / "summary",
            bootstrap_replicates=10,
        )


def test_selector_evaluations_merge_with_b_fais_and_enter_comparisons(
    tmp_path: Path,
) -> None:
    b_fais = _evaluation(tmp_path, "forecast-a", "family-a")
    metaod = _selector_evaluation(tmp_path, b_fais, "metaod", 2.5)
    hybrid = _selector_evaluation(tmp_path, b_fais, "hybrid_lstm", 1.75)

    result = summarize_multi_forecaster(
        evaluation_inputs=(metaod, b_fais, hybrid),
        output_dir=tmp_path / "summary",
        bootstrap_replicates=10,
        bootstrap_seed=7,
    )
    payload = json.loads(
        Path(result["main_summary_json"]).read_text(encoding="utf-8")
    )

    assert payload["episode_count"] == 2
    assert payload["evaluated_selector_ids"] == ["hybrid_lstm", "metaod"]
    overall_methods = {
        row["method"]: row
        for row in payload["method_summary"]
        if row["scope"] == "overall"
    }
    assert overall_methods["metaod"]["method_role"] == "selector_baseline"
    assert overall_methods["metaod"]["recorded_count"] == 2
    comparisons = {
        row["comparator"]: row
        for row in payload["comparison_summary"]
        if row["scope"] == "overall"
    }
    assert comparisons["metaod"]["comparator_role"] == "selector_baseline"
    assert comparisons["metaod"]["pair_count"] == 2
    assert comparisons["metaod"]["mase_mean_delta"] == -0.5
    assert comparisons["hybrid_lstm"]["mase_mean_delta"] == 0.25


def test_selector_merge_rejects_conflicting_shared_rows(tmp_path: Path) -> None:
    b_fais = _evaluation(tmp_path, "forecast-a", "family-a")
    metaod = _selector_evaluation(tmp_path, b_fais, "metaod", 2.5)
    metrics_path = metaod / "episode_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    conflicting = next(
        row
        for row in rows
        if row["episode_id"] == "episode-0" and row["method"] == "locf"
    )
    conflicting["mase"] = float(conflicting["mase"]) + 0.1
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"conflicting shared .* fields: mase"):
        summarize_multi_forecaster(
            evaluation_inputs=(b_fais, metaod),
            output_dir=tmp_path / "summary",
            bootstrap_replicates=10,
        )


def test_imputation_metric_availability_does_not_remove_forecast_pairs(
    tmp_path: Path,
) -> None:
    source = _evaluation(tmp_path, "forecast-a", "family-a")
    metrics_path = source / "episode_metrics.jsonl"
    rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["episode_id"] == "episode-1" and row["method"] == "locf":
            row["imputation_mae"] = None
            row["imputation_rmse"] = None
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = summarize_multi_forecaster(
        evaluation_inputs=(source,),
        output_dir=tmp_path / "summary",
        bootstrap_replicates=10,
    )
    payload = json.loads(
        Path(result["main_summary_json"]).read_text(encoding="utf-8")
    )
    locf_method = next(
        row
        for row in payload["method_summary"]
        if row["scope"] == "overall" and row["method"] == "locf"
    )
    locf_comparison = next(
        row
        for row in payload["comparison_summary"]
        if row["scope"] == "overall" and row["comparator"] == "locf"
    )
    assert locf_method["valid_count"] == 2
    assert locf_method["mase_count"] == 2
    assert locf_method["imputation_mae_count"] == 1
    assert locf_comparison["mase_pair_count"] == 2
    assert locf_comparison["imputation_mae_pair_count"] == 1
    assert locf_comparison["imputation_mae_mean_delta_ci95_low"] is None


def test_multi_forecaster_summary_rejects_incomplete_manifest(tmp_path: Path) -> None:
    source = _evaluation(tmp_path, "forecast-a", "family-a")
    (source / "evaluation_manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "failed"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not completed"):
        summarize_multi_forecaster(
            evaluation_inputs=(source,),
            output_dir=tmp_path / "summary",
            bootstrap_replicates=10,
        )
