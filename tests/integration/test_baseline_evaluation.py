from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.config import load_config
from tsfm_fais.contracts import ForecastResult, ForecastSpec
from tsfm_fais.evaluation import evaluate_imputations
from tsfm_fais.main_results import summarize_multi_forecaster


class LastValueForecaster:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        contexts: np.ndarray,
        forecast_spec: ForecastSpec,
    ) -> ForecastResult:
        self.calls += 1
        targets = forecast_spec.target_indices or tuple(range(contexts.shape[2]))
        point = np.repeat(contexts[:, -1:, targets], forecast_spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=tuple(targets))


def _config():
    config = load_config("configs/smoke.yaml")
    experiment = config.experiment.model_copy(
        update={"context_length": 4, "horizon": 2, "target_indices": (0,)}
    )
    return config.model_copy(update={"experiment": experiment})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_integrity_ledger(artifact: Path, *, assembled_method_id: str) -> None:
    manifest_path = artifact / "imputation_manifest.json"
    assignments_path = artifact / "routing_assignments.jsonl"
    npz_path = artifact / "imputations" / "toy" / "00000000.npz"
    manifest_path.write_text(
        json.dumps({"episode_count": 1, "assembled_method_id": assembled_method_id}) + "\n",
        encoding="utf-8",
    )
    progress = {
        "status": "completed",
        "expected_episode_count": 1,
        "imputation_manifest_sha256": _sha256(manifest_path),
        "routing_assignments_sha256": _sha256(assignments_path),
        "entries": {
            "00000000": {
                "file": "toy/00000000.npz",
                "npz_sha256": _sha256(npz_path),
            }
        },
    }
    (artifact / "imputation_progress.json").write_text(
        json.dumps(progress) + "\n",
        encoding="utf-8",
    )


def _refresh_assignment_integrity(artifact: Path) -> None:
    progress_path = artifact / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["routing_assignments_sha256"] = _sha256(artifact / "routing_assignments.jsonl")
    progress_path.write_text(json.dumps(progress) + "\n", encoding="utf-8")


def _artifact(
    root: Path,
    *,
    record_method: str | None = None,
    archive_method: str | None = None,
) -> Path:
    artifact = root / "impute-run"
    episode_dir = artifact / "imputations" / "toy"
    episode_dir.mkdir(parents=True)
    clean = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    observed = np.ones_like(clean, dtype=bool)
    observed[-1, 0] = False
    assembled = clean.copy()
    locf = clean.copy()
    locf[-1, 0] = 2.0
    future = np.asarray([[4.0, 0.0], [4.0, 0.0]])
    payload = {
        "schema_version": np.asarray([3]),
        "values": assembled,
        "observed_mask": observed,
        "clean_context": clean,
        "clean_future": future,
        "mask_protocol": np.asarray(["sequence_mask_v2"]),
        "mask_seed": np.asarray([17], dtype=np.uint64),
        "mask_realization_id": np.asarray(["fixture-mask"]),
        "target_missing_rate": np.asarray([0.2]),
        "global_missing_rate": np.asarray([0.2]),
        "local_missing_rate": np.asarray([0.125]),
        "mase_scale": np.asarray([1.0, 1.0]),
        "mase_scale_lag": np.asarray([1]),
        "period": np.asarray([1]),
        "candidate_ids": np.asarray(["locf"]),
        "candidate_values": locf[None, ...],
        "candidate_native_valid": np.ones((1, *clean.shape), dtype=bool),
        "candidate_status": np.asarray(["success"]),
        "candidate_runtime_seconds": np.asarray([0.01]),
        "candidate_peak_memory_bytes": np.asarray([10]),
        "pipeline_runtime_seconds": np.asarray([0.5]),
        "pipeline_rss_delta_bytes": np.asarray([500]),
    }
    if archive_method is not None:
        payload["assembled_method_id"] = np.asarray([archive_method])
    np.savez_compressed(episode_dir / "00000000.npz", **payload)

    record = {
        "episode_id": "toy__item-0__8__mixed_outage__0.2__7",
        "dataset_id": "toy",
        "family_id": "toy-family",
        "forecaster_id": "chronos2",
        "item_id": "item-0",
        "forecast_origin": 8,
        "mechanism": "mixed_outage",
        "missing_rate": 0.2,
        "seed": 7,
        "mask_protocol": "sequence_mask_v2",
        "mask_seed": 17,
        "mask_realization_id": "fixture-mask",
        "target_missing_rate": 0.2,
        "global_missing_rate": 0.2,
        "local_missing_rate": 0.125,
        "mase_scale_lag": 1,
        "file": "toy/00000000.npz",
    }
    if record_method is not None:
        record["assembled_method_id"] = record_method
    (artifact / "routing_assignments.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    _write_integrity_ledger(
        artifact,
        assembled_method_id=record_method or archive_method or "b_fais",
    )
    return artifact


def test_dynamic_selector_method_is_evaluated_and_resumed(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path,
        record_method="metaod",
        archive_method="metaod",
    )
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
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert first["status"] == "completed"
    assert {row["method"] for row in rows} == {
        "clean",
        "metaod",
        "locf",
        "linear_interp",
        "oracle",
    }
    selector = next(row for row in rows if row["method"] == "metaod")
    assert selector["method_role"] == "selector_baseline"
    assert selector["candidate_status"] == "assembled"
    assert selector["runtime_seconds"] == 0.5
    assert not any(row["method"] == "b_fais" for row in rows)

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


def test_archive_only_selector_method_is_supported(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, archive_method="dselect_1")
    output = tmp_path / "evaluation"

    evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
    )
    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assembled = next(row for row in rows if row["method"] == "dselect_1")
    assert assembled["method_role"] == "selector_baseline"


def test_resume_rejects_repaired_selector_identity(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    output = tmp_path / "evaluation"
    evaluate_imputations(
        config=_config(),
        impute_artifact=artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
    )

    assignments = artifact / "routing_assignments.jsonl"
    record = json.loads(assignments.read_text(encoding="utf-8"))
    record["assembled_method_id"] = "metaod"
    assignments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _refresh_assignment_integrity(artifact)

    with pytest.raises(ValueError, match="resume signature"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=output,
            resume=True,
            forecast_runner=LastValueForecaster(),
        )


def test_selector_evaluation_merges_with_b_fais_main_results(tmp_path: Path) -> None:
    b_fais_artifact = _artifact(tmp_path / "b-fais")
    metaod_artifact = _artifact(
        tmp_path / "metaod",
        record_method="metaod",
        archive_method="metaod",
    )
    b_fais_output = tmp_path / "b-fais-evaluation"
    metaod_output = tmp_path / "metaod-evaluation"
    for artifact, output in (
        (b_fais_artifact, b_fais_output),
        (metaod_artifact, metaod_output),
    ):
        evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=output,
            baseline_ids=("locf",),
            forecast_runner=LastValueForecaster(),
        )

    result = summarize_multi_forecaster(
        evaluation_inputs=(b_fais_output, metaod_output),
        output_dir=tmp_path / "main-results",
        bootstrap_replicates=10,
    )
    payload = json.loads(Path(result["main_summary_json"]).read_text(encoding="utf-8"))
    assert payload["evaluated_selector_ids"] == ["metaod"]
    assert any(
        row["scope"] == "overall" and row["comparator"] == "metaod" and row["pair_count"] == 1
        for row in payload["comparison_summary"]
    )
