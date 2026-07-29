from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.config import load_config
from tsfm_fais.contracts import ForecastResult, ForecastSpec
from tsfm_fais.evaluation import evaluate_imputations, summarize_evaluation
from tsfm_fais.main_results import summarize_multi_forecaster


class LastValueForecaster:
    def __init__(self) -> None:
        self.calls = 0
        self.specs: list[ForecastSpec] = []
        self.batch_sizes: list[int] = []

    def predict(self, contexts: np.ndarray, forecast_spec: ForecastSpec) -> ForecastResult:
        self.calls += 1
        self.specs.append(forecast_spec)
        self.batch_sizes.append(int(contexts.shape[0]))
        targets = forecast_spec.target_indices or tuple(range(contexts.shape[2]))
        point = np.repeat(contexts[:, -1:, targets], forecast_spec.horizon, axis=1)
        return ForecastResult(point=point, target_indices=tuple(targets))


class BatchSensitiveForecaster(LastValueForecaster):
    def predict(self, contexts: np.ndarray, forecast_spec: ForecastSpec) -> ForecastResult:
        result = super().predict(contexts, forecast_spec)
        batch_offsets = (
            float(contexts.shape[0]) * 0.1 + np.arange(contexts.shape[0], dtype=float) * 0.01
        )
        return ForecastResult(
            point=result.point + batch_offsets[:, None, None],
            target_indices=result.target_indices,
        )


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


def _refresh_npz_integrity(artifact: Path) -> None:
    progress_path = artifact / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["entries"]["00000000"]["npz_sha256"] = _sha256(
        artifact / "imputations" / "toy" / "00000000.npz"
    )
    progress_path.write_text(json.dumps(progress) + "\n", encoding="utf-8")


def _impute_artifact(
    root: Path,
    *,
    routing_forecaster_id: str = "chronos2",
    invalid_linear: bool = False,
    selector_independent: bool = False,
    selector_method: str = "metaod",
    assembled_missing_value: float | None = None,
) -> Path:
    artifact = root / "impute-run"
    episode_dir = artifact / "imputations" / "toy"
    episode_dir.mkdir(parents=True)
    clean = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    observed = np.ones_like(clean, dtype=bool)
    observed[-1, 0] = False
    b_fais = clean.copy()
    if assembled_missing_value is not None:
        b_fais[-1, 0] = assembled_missing_value
    locf = clean.copy()
    locf[-1, 0] = 2.0
    linear = clean.copy()
    linear[-1, 0] = 3.5
    future = np.asarray([[4.0, 0.0], [4.0, 0.0]])
    native_valid = np.ones((2, *clean.shape), dtype=bool)
    if invalid_linear:
        native_valid[1, -1, 0] = False
    archive_identity = (
        {
            "forecaster_id": np.asarray(["imputation"]),
            "forecast_mode": np.asarray(["selector_independent"]),
            "forecaster_independent_selection": np.asarray([True]),
            "assembled_method_id": np.asarray([selector_method]),
        }
        if selector_independent
        else {}
    )
    np.savez_compressed(
        episode_dir / "00000000.npz",
        schema_version=np.asarray([3]),
        values=b_fais,
        observed_mask=observed,
        clean_context=clean,
        clean_future=future,
        mask_protocol=np.asarray(["sequence_mask_v2"]),
        mask_seed=np.asarray([17], dtype=np.uint64),
        mask_realization_id=np.asarray(["fixture-mask"]),
        target_missing_rate=np.asarray([0.2]),
        global_missing_rate=np.asarray([0.2]),
        local_missing_rate=np.asarray([0.125]),
        mase_scale=np.asarray([1.0, 1.0]),
        mase_scale_lag=np.asarray([1]),
        period=np.asarray([1]),
        candidate_ids=np.asarray(["locf", "linear_interp"]),
        candidate_values=np.stack((locf, linear)),
        candidate_native_valid=native_valid,
        candidate_status=np.asarray(["success", "success"]),
        candidate_runtime_seconds=np.asarray([0.01, 0.02]),
        candidate_peak_memory_bytes=np.asarray([10, 20]),
        pipeline_runtime_seconds=np.asarray([0.5]),
        pipeline_rss_delta_bytes=np.asarray([500]),
        **archive_identity,
    )
    record = {
        "episode_id": "toy__item-0__8__mixed_outage__0.2__7",
        "dataset_id": "toy",
        "family_id": "toy-family",
        "forecaster_id": routing_forecaster_id,
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
    if selector_independent:
        record.update(
            {
                "forecaster_id": "imputation",
                "forecast_mode": "selector_independent",
                "forecaster_independent_selection": True,
                "assembled_method_id": selector_method,
                "routing_metadata": {
                    "selector_method": selector_method,
                    "selection_scope": "whole_series",
                    "routing_target_protocol": "sequence_imputation_quality_v1",
                    "selector_training_target": "imputation_loss",
                    "forecaster_independent_selection": True,
                    "uses_missing_block_graph": False,
                    "requires_pseudo_candidates": False,
                },
            }
        )
    (artifact / "routing_assignments.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    _write_integrity_ledger(
        artifact,
        assembled_method_id=selector_method if selector_independent else "b_fais",
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
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
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
    metrics_path = output / "episode_metrics.jsonl"
    assert first["episode_metrics_jsonl_sha256"] == _sha256(metrics_path)
    assert first["episode_metrics_jsonl_size_bytes"] == metrics_path.stat().st_size

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
    payload = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
    b_fais_group = next(group for group in payload["groups"] if group["method"] == "b_fais")
    assert b_fais_group["count"] == 1
    assert b_fais_group["metric_count"] == 1
    assert b_fais_group["invalid_count"] == 0
    assert b_fais_group["mase_mean"] == 1.0
    assert Path(summary["summary_csv"]).is_file()


def test_evaluation_reuses_same_mode_routing_artifact(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path, routing_forecaster_id="timesfm2p5")
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
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["routing_forecaster_ids"] == ["timesfm2p5"]
    assert {row["forecaster_id"] for row in rows} == {"chronosbolt"}
    assembled = next(row for row in rows if row["method"] == "b_fais")
    shared = [row for row in rows if row["method"] != "b_fais"]
    assert assembled["routing_forecaster_id"] == "timesfm2p5"
    assert {row["routing_forecaster_id"] for row in shared} == {"chronosbolt"}
    assert {row["routing_artifact_forecaster_id"] for row in rows} == {"timesfm2p5"}


def test_evaluation_reuses_selector_independent_artifact_across_modes(
    tmp_path: Path,
) -> None:
    artifact = _impute_artifact(tmp_path, selector_independent=True)

    for forecaster_id in ("chronos2", "timesfm2p5"):
        result = evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id=forecaster_id,
            forecaster_artifact=None,
            output_dir=tmp_path / f"evaluation-{forecaster_id}",
            forecast_runner=LastValueForecaster(),
        )
        rows = [
            json.loads(line)
            for line in Path(result["episode_metrics_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        assert result["routing_forecaster_ids"] == ["imputation"]
        selector = next(row for row in rows if row["method"] == "metaod")
        shared = [row for row in rows if row["method"] != "metaod"]
        assert selector["routing_forecaster_id"] == "imputation"
        assert {row["routing_forecaster_id"] for row in shared} == {forecaster_id}
        assert {row["routing_artifact_forecaster_id"] for row in rows} == {"imputation"}

    b_fais_artifact = _impute_artifact(tmp_path / "b-fais")
    b_fais_output = tmp_path / "evaluation-b-fais-chronos2"
    evaluate_imputations(
        config=_config(),
        impute_artifact=b_fais_artifact,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=b_fais_output,
        forecast_runner=LastValueForecaster(),
    )
    with pytest.raises(
        ValueError,
        match=r"conflicting shared .* routing_artifact_forecaster_id",
    ):
        summarize_multi_forecaster(
            evaluation_inputs=(b_fais_output, tmp_path / "evaluation-chronos2"),
            output_dir=tmp_path / "main-results",
            bootstrap_replicates=10,
        )


def test_evaluation_rejects_unproven_selector_independent_identity(
    tmp_path: Path,
) -> None:
    artifact = _impute_artifact(tmp_path, selector_independent=True)
    assignments = artifact / "routing_assignments.jsonl"
    record = json.loads(assignments.read_text(encoding="utf-8"))
    record.pop("routing_metadata")
    assignments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _refresh_assignment_integrity(artifact)

    with pytest.raises(ValueError, match="strict sequence protocol"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "evaluation",
            forecast_runner=LastValueForecaster(),
        )


def test_selector_safety_fallback_is_recorded_but_excluded(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path, selector_independent=True)
    assignments = artifact / "routing_assignments.jsonl"
    record = json.loads(assignments.read_text(encoding="utf-8"))
    record["routing_metadata"]["paper_native_valid"] = False
    record["routing_metadata"]["paper_ineligibility_reason"] = "no_native_valid_method"
    assignments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _refresh_assignment_integrity(artifact)

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
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selector = next(row for row in rows if row["method"] == "metaod")
    assert selector["metric_eligible"] is False
    assert selector["candidate_status"] == "assembled_fallback"
    assert selector["ineligibility_reason"] == "no_native_valid_method"
    assert selector["mase"] is None
    assert forecaster.batch_sizes == [2]  # clean and LOCF; linear is tail-ineligible


def _shared_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = _impute_artifact(
        tmp_path / "source",
        selector_independent=True,
        selector_method="metaod",
        assembled_missing_value=2.5,
    )
    shared = tmp_path / "shared-evaluation"
    evaluate_imputations(
        config=_config(),
        impute_artifact=source,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=shared,
        forecast_runner=LastValueForecaster(),
    )
    target = _impute_artifact(
        tmp_path / "target",
        selector_independent=True,
        selector_method="dselect1",
        assembled_missing_value=1.5,
    )
    return target, shared


def test_shared_evaluation_predicts_assembled_in_aligned_batch_and_reuses_common_rows(
    tmp_path: Path,
) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    output = tmp_path / "target-evaluation"
    forecaster = LastValueForecaster()

    result = evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
        shared_evaluation_artifact=shared,
    )

    assert result["status"] == "completed"
    assert forecaster.batch_sizes == [3]
    source_rows = {
        row["method"]: row
        for row in map(
            json.loads,
            (shared / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    target_rows = {
        row["method"]: row
        for row in map(
            json.loads,
            (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    assert set(target_rows) == {"clean", "dselect1", "locf", "linear_interp", "oracle"}
    for method_id in ("clean", "locf", "linear_interp", "oracle"):
        for field in ("mase", "mae", "rmse", "oracle_source", "candidate_status"):
            assert target_rows[method_id][field] == source_rows[method_id][field]
    assert target_rows["dselect1"]["mase"] == 2.5
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    shared_signature = manifest["evaluation_signature"]["shared_evaluation"]
    assert shared_signature["evaluation_manifest_sha256"] == _sha256(
        shared / "evaluation_manifest.json"
    )
    assert shared_signature["episode_metrics_sha256"] == _sha256(shared / "episode_metrics.jsonl")


def test_shared_evaluation_matches_full_evaluation_for_batch_sensitive_forecaster(
    tmp_path: Path,
) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    shared_forecaster = BatchSensitiveForecaster()
    full_forecaster = BatchSensitiveForecaster()
    shared_output = tmp_path / "shared-target-evaluation"
    full_output = tmp_path / "full-target-evaluation"

    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=shared_output,
        forecast_runner=shared_forecaster,
        shared_evaluation_artifact=shared,
    )
    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=full_output,
        forecast_runner=full_forecaster,
    )

    assert shared_forecaster.batch_sizes == full_forecaster.batch_sizes == [3]
    shared_rows = {
        row["method"]: row
        for row in map(
            json.loads,
            (shared_output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    full_rows = {
        row["method"]: row
        for row in map(
            json.loads,
            (full_output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    for metric in ("mase", "mae", "rmse"):
        assert shared_rows["dselect1"][metric] == full_rows["dselect1"][metric]


def test_shared_evaluation_rejects_candidate_content_mismatch(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    npz_path = target / "imputations" / "toy" / "00000000.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {field: np.asarray(archive[field]) for field in archive.files}
    arrays["candidate_values"] = arrays["candidate_values"].copy()
    arrays["candidate_values"][0, -1, 0] += 0.25
    np.savez_compressed(npz_path, **arrays)
    _refresh_npz_integrity(target)

    with pytest.raises(ValueError, match="shared imputation content differs.*candidate_values"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "target-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
        )


def test_shared_evaluation_rejects_forecaster_mismatch(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)

    with pytest.raises(ValueError, match="forecaster ID differs"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="timesfm2p5",
            forecaster_artifact=None,
            output_dir=tmp_path / "target-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
        )


def test_shared_evaluation_rejects_incomplete_artifact(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    manifest_path = shared / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not completed"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "target-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
        )


def test_shared_evaluation_rejects_metrics_hash_mismatch(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    metrics_path = shared / "episode_metrics.jsonl"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 differs from its manifest"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "target-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
        )


def test_shared_evaluation_invalid_assembled_does_not_predict(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    assignments = target / "routing_assignments.jsonl"
    record = json.loads(assignments.read_text(encoding="utf-8"))
    record["routing_metadata"]["paper_native_valid"] = False
    record["routing_metadata"]["paper_ineligibility_reason"] = "invalid_hybrid_windows"
    assignments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _refresh_assignment_integrity(target)
    forecaster = LastValueForecaster()
    output = tmp_path / "target-evaluation"

    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
        shared_evaluation_artifact=shared,
    )

    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assembled = next(row for row in rows if row["method"] == "dselect1")
    assert forecaster.calls == 0
    assert assembled["metric_eligible"] is False
    assert assembled["mase"] is None
    assert assembled["ineligibility_reason"] == "invalid_hybrid_windows"


def test_shared_evaluation_resume_rejects_changed_shared_manifest(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    output = tmp_path / "target-evaluation"
    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
        shared_evaluation_artifact=shared,
    )
    manifest_path = shared / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["post_completion_note"] = "changed"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume signature"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=output,
            resume=True,
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
        )


def test_shared_reference_only_allows_b_fais_routing_identity_and_predicts_assembled(
    tmp_path: Path,
) -> None:
    _, shared = _shared_evaluation_fixture(tmp_path)
    target = _impute_artifact(
        tmp_path / "b-fais-target",
        routing_forecaster_id="chronos2",
        assembled_missing_value=1.5,
    )
    output = tmp_path / "b-fais-evaluation"
    forecaster = LastValueForecaster()

    result = evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
        shared_evaluation_artifact=shared,
        shared_reference_only=True,
    )

    assert result["status"] == "completed"
    assert forecaster.batch_sizes == [3]
    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    common = [row for row in rows if row["method"] != "b_fais"]
    assembled = next(row for row in rows if row["method"] == "b_fais")
    assert {row["routing_artifact_forecaster_id"] for row in common} == {"imputation"}
    assert assembled["routing_artifact_forecaster_id"] == "chronos2"
    assert assembled["routing_forecaster_id"] == "chronos2"
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["shared_reference_only"] is True
    assert manifest["forecast_reuse_mode"] == "shared_reference_only"
    assert manifest["evaluation_signature"]["shared_reference_only"] is True


@pytest.mark.parametrize("candidate_field", ("candidate_values", "candidate_status"))
def test_shared_reference_only_rejects_changed_candidate_content(
    tmp_path: Path,
    candidate_field: str,
) -> None:
    _, shared = _shared_evaluation_fixture(tmp_path)
    target = _impute_artifact(tmp_path / "b-fais-target", routing_forecaster_id="chronos2")
    npz_path = target / "imputations" / "toy" / "00000000.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {field: np.asarray(archive[field]) for field in archive.files}
    arrays[candidate_field] = arrays[candidate_field].copy()
    if candidate_field == "candidate_values":
        arrays[candidate_field][0, -1, 0] += 0.25
    else:
        arrays[candidate_field][0] = "fallback"
    np.savez_compressed(npz_path, **arrays)
    _refresh_npz_integrity(target)

    with pytest.raises(
        ValueError,
        match=rf"shared imputation content differs.*{candidate_field}",
    ):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "b-fais-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
            shared_reference_only=True,
        )


def test_shared_reference_only_requires_shared_evaluation(tmp_path: Path) -> None:
    target = _impute_artifact(tmp_path, routing_forecaster_id="chronos2")

    with pytest.raises(ValueError, match="requires shared_evaluation_artifact"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "b-fais-evaluation",
            forecast_runner=LastValueForecaster(),
            shared_reference_only=True,
        )


def test_shared_reference_only_invalid_b_fais_does_not_predict(tmp_path: Path) -> None:
    _, shared = _shared_evaluation_fixture(tmp_path)
    target = _impute_artifact(
        tmp_path / "b-fais-target",
        routing_forecaster_id="chronos2",
    )
    assignments = target / "routing_assignments.jsonl"
    record = json.loads(assignments.read_text(encoding="utf-8"))
    record["routing_metadata"] = {
        "paper_native_valid": False,
        "paper_ineligibility_reason": "no_valid_b_fais_route",
    }
    assignments.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _refresh_assignment_integrity(target)
    output = tmp_path / "b-fais-evaluation"
    forecaster = LastValueForecaster()

    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=forecaster,
        shared_evaluation_artifact=shared,
        shared_reference_only=True,
    )

    rows = [
        json.loads(line)
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assembled = next(row for row in rows if row["method"] == "b_fais")
    assert forecaster.calls == 0
    assert assembled["metric_eligible"] is False
    assert assembled["mase"] is None
    assert assembled["ineligibility_reason"] == "no_valid_b_fais_route"


def test_shared_reference_only_mode_is_bound_to_resume_signature(tmp_path: Path) -> None:
    target, shared = _shared_evaluation_fixture(tmp_path)
    output = tmp_path / "target-evaluation"
    evaluate_imputations(
        config=_config(),
        impute_artifact=target,
        forecaster_id="chronos2",
        forecaster_artifact=None,
        output_dir=output,
        forecast_runner=LastValueForecaster(),
        shared_evaluation_artifact=shared,
        shared_reference_only=True,
    )

    with pytest.raises(ValueError, match="resume signature"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=target,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=output,
            resume=True,
            forecast_runner=LastValueForecaster(),
            shared_evaluation_artifact=shared,
            shared_reference_only=False,
        )


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
    changed_experiment = _config().experiment.model_copy(update={"forecast_num_samples": 99})
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


def test_evaluation_rejects_npz_changed_after_imputation_completion(tmp_path: Path) -> None:
    artifact = _impute_artifact(tmp_path)
    npz_path = artifact / "imputations" / "toy" / "00000000.npz"
    with npz_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="NPZ hash differs"):
        evaluate_imputations(
            config=_config(),
            impute_artifact=artifact,
            forecaster_id="chronos2",
            forecaster_artifact=None,
            output_dir=tmp_path / "evaluation",
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
        for line in (output / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
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
    linear_group = next(group for group in payload["groups"] if group["method"] == "linear_interp")
    assert linear_group["count"] == 1
    assert linear_group["metric_count"] == 0
    assert linear_group["mase_mean"] is None
