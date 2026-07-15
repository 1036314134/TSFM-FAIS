from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.config import load_config
from tsfm_fais.contracts import ForecastResult, ForecastSpec
from tsfm_fais.evaluation import evaluate_imputations, summarize_evaluation


class LastValueForecaster:
    def __init__(self) -> None:
        self.calls = 0
        self.specs: list[ForecastSpec] = []
        self.batch_sizes: list[int] = []

    def predict(
        self, contexts: np.ndarray, forecast_spec: ForecastSpec
    ) -> ForecastResult:
        self.calls += 1
        self.specs.append(forecast_spec)
        self.batch_sizes.append(int(contexts.shape[0]))
        targets = forecast_spec.target_indices or tuple(range(contexts.shape[2]))
        point = np.repeat(
            contexts[:, -1:, targets], forecast_spec.horizon, axis=1
        )
        return ForecastResult(point=point, target_indices=tuple(targets))


def _config():
    config = load_config("configs/smoke.yaml")
    experiment = config.experiment.model_copy(
        update={"context_length": 4, "horizon": 2, "target_indices": (0,)}
    )
    return config.model_copy(update={"experiment": experiment})


def _impute_artifact(
    root: Path,
    *,
    routing_forecaster_id: str = "chronos2",
    invalid_linear: bool = False,
) -> Path:
    artifact = root / "impute-run"
    episode_dir = artifact / "imputations" / "toy"
    episode_dir.mkdir(parents=True)
    clean = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    observed = np.ones_like(clean, dtype=bool)
    observed[-1, 0] = False
    b_fais = clean.copy()
    locf = clean.copy()
    locf[-1, 0] = 2.0
    linear = clean.copy()
    linear[-1, 0] = 3.5
    future = np.asarray([[4.0, 0.0], [4.0, 0.0]])
    native_valid = np.ones((2, *clean.shape), dtype=bool)
    if invalid_linear:
        native_valid[1, -1, 0] = False
    np.savez_compressed(
        episode_dir / "00000000.npz",
        values=b_fais,
        observed_mask=observed,
        clean_context=clean,
        clean_future=future,
        period=np.asarray([1]),
        candidate_ids=np.asarray(["locf", "linear_interp"]),
        candidate_values=np.stack((locf, linear)),
        candidate_native_valid=native_valid,
        candidate_status=np.asarray(["success", "success"]),
        candidate_runtime_seconds=np.asarray([0.01, 0.02]),
        candidate_peak_memory_bytes=np.asarray([10, 20]),
        pipeline_runtime_seconds=np.asarray([0.5]),
        pipeline_rss_delta_bytes=np.asarray([500]),
    )
    record = {
        "episode_id": "toy__item-0__8__tail_mixed__0.2__7",
        "dataset_id": "toy",
        "family_id": "toy-family",
        "forecaster_id": routing_forecaster_id,
        "item_id": "item-0",
        "forecast_origin": 8,
        "mechanism": "tail_mixed",
        "missing_rate": 0.2,
        "seed": 7,
        "file": "toy/00000000.npz",
    }
    (artifact / "routing_assignments.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return artifact


def test_mock_forecast_evaluation_resume_and_grouped_summary(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path)
    output = tmp_path / "evaluation"
    forecaster = LastValueForecaster()
    first = evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
    )

    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert first["status"] == "completed"
    assert {row["method"] for row in rows} == {
        "clean",
        "b_fais",
        "locf",
        "linear_interp",
        "oracle",
    }
    b_fais = next(row for row in rows if row["method"] == "b_fais")
    oracle = next(row for row in rows if row["method"] == "oracle")
    assert b_fais["mase"] == 1.0
    assert b_fais["relative_regret"] == -0.5
    assert b_fais["degradation_vs_clean_mase"] == 0.0
    assert b_fais["relative_degradation_vs_clean"] == 0.0
    assert b_fais["runtime_seconds"] == 0.5
    assert b_fais["rss_delta_bytes"] == 500
    assert b_fais["runtime_scope"] == "end_to_end_imputation"
    assert b_fais["metric_eligible"] is True
    assert oracle["oracle_source"] == "locf"
    assert oracle["mase"] == 2.0
    assert oracle["degradation_vs_clean_mase"] == 1.0
    assert forecaster.specs[0].num_samples == _config().experiment.forecast_num_samples

    with (output / "episode_metrics.jsonl").open("ab") as handle:
        handle.write(b'{"episode_id":')
    resumed = evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        resume=True,
        forecast_runner=forecaster,
    )
    assert resumed["rows_written"] == 0
    assert resumed["total_rows"] == 5
    assert forecaster.calls == 1

    summary = summarize_evaluation(
        metrics_path=output,
        output_dir=tmp_path / "summary",
    )
    payload = json.loads(
        Path(summary["summary_json"]).read_text(encoding="utf-8")
    )
    b_fais_group = next(
        group for group in payload["groups"] if group["method"] == "b_fais"
    )
    assert b_fais_group["count"] == 1
    assert b_fais_group["metric_count"] == 1
    assert b_fais_group["invalid_count"] == 0
    assert b_fais_group["mase_mean"] == 1.0
    assert Path(summary["summary_csv"]).is_file()


def test_evaluation_reuses_same_mode_routing_artifact(tmp_path: Path) -> None:
    artifact = _impute_artifact(
        tmp_path, routing_forecaster_id="timesfm2p5"
    )
    output = tmp_path / "evaluation"

    result = evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronosbolt",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
    )

    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert result["routing_forecaster_ids"] == ["timesfm2p5"]
    assert {row["forecaster_id"] for row in rows} == {"chronosbolt"}
    assert {row["routing_forecaster_id"] for row in rows} == {"timesfm2p5"}


def test_evaluation_rejects_cross_mode_routing_artifact(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path, routing_forecaster_id="chronos2")

    with pytest.raises(ValueError, match="incompatible"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id="chronosbolt",
            forecaster_artifact=None,
            output_dir=tmp_path / "evaluation",
            forecast_runner=LastValueForecaster(),
        )


def test_resume_rejects_changed_evaluation_signature(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path)
    output = tmp_path / "evaluation"
    evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
    )
    changed_experiment = _config().experiment.model_copy(
        update={"forecast_num_samples": 99}
    )
    changed_config = _config().model_copy(update={"experiment": changed_experiment})

    with pytest.raises(ValueError, match="resume signature"):
        evaluate_imputations(
            config=changed_config,
            impute_artifact=artifact,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=output,
            resume=True,
            forecast_runner=LastValueForecaster(),
        )


def test_invalid_candidate_is_recorded_but_not_forecast_or_summarized(
    tmp_path: Path,
) -> None:
    artifact = _impute_artifact(tmp_path, invalid_linear=True)
    output = tmp_path / "evaluation"
    forecaster = LastValueForecaster()

    evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
    )

    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    linear = next(row for row in rows if row["method"] == "linear_interp")
    oracle = next(row for row in rows if row["method"] == "oracle")
    assert forecaster.batch_sizes == [3]  # clean, B-FAIS, and valid LOCF only
    assert linear["metric_eligible"] is False
    assert linear["mase"] is None
    assert oracle["oracle_source"] == "locf"

    summary = summarize_evaluation(
        metrics_path=output,
        output_dir=tmp_path / "summary",
    )
    payload = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
    linear_group = next(
        group for group in payload["groups"] if group["method"] == "linear_interp"
    )
    assert linear_group["count"] == 1
    assert linear_group["metric_count"] == 0
    assert linear_group["mase_mean"] is None
