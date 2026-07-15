from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_fais.artifacts import RunArtifactStore, validate_run_id
from tsfm_fais.cli import main
from tsfm_fais.config import load_config
from tsfm_fais.stages import (
    StageInputs,
    StagePreparationError,
    _forecaster_check,
    prepare_stage,
)


def _write_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir()
    source_files = {
        "datasets.yaml": Path("configs/data/datasets.yaml"),
        "imputers.yaml": Path("configs/imputers/pool.yaml"),
        "forecasters.yaml": Path("configs/forecasters/pool.yaml"),
        "router.yaml": Path("configs/router/block_fais.yaml"),
    }
    for name, source in source_files.items():
        (config_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    config = config_dir / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "seed: 123",
                "registries:",
                "  data_manifest: datasets.yaml",
                "  imputer_registry: imputers.yaml",
                "  forecaster_registry: forecasters.yaml",
                "  router_config: router.yaml",
                "experiment:",
                "  seeds: [1, 2]",
                "runtime:",
                "  output_root: ../artifacts",
                "  device: cpu",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _accepted_audit(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [{"dataset_id": "synthetic", "accepted": True}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_blocked_stage_writes_auditable_baseline(tmp_path, capsys):
    config = _write_config(tmp_path)
    code = main(
        [
            "run",
            "--config",
            str(config),
            "--stage",
            "fit-imputers",
            "--run-id",
            "missing-audit",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "--audit-artifact" in captured.err
    run_dir = tmp_path / "artifacts" / "missing-audit"
    assert {path.name for path in run_dir.iterdir()} == {
        "candidate_status.json",
        "resolved_config.json",
        "seeds.json",
        "software_versions.json",
        "stage_manifest.json",
    }
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["execution_started"] is False
    assert manifest["automatic_downloads"] is False
    assert any(check["name"] == "data_audit" and not check["valid"] for check in manifest["checks"])
    candidates = json.loads((run_dir / "candidate_status.json").read_text(encoding="utf-8"))
    assert len(candidates["candidates"]) == 20
    seeds = json.loads((run_dir / "seeds.json").read_text(encoding="utf-8"))
    assert seeds == {"experiment_seeds": [1, 2], "root_seed": 123, "schema_version": 1}


def test_prepared_stage_returns_success_without_starting_execution(tmp_path, capsys):
    config = _write_config(tmp_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    code = main(
        [
            "run",
            "--config",
            str(config),
            "--stage",
            "fit-imputers",
            "--run-id",
            "prepared-only",
            "--audit-artifact",
            str(audit),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "STAGE PREPARED" in captured.out
    run_dir = tmp_path / "artifacts" / "prepared-only"
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert manifest["execution_started"] is False
    assert all(check["valid"] for check in manifest["checks"])
    assert not (run_dir / "imputer_artifacts").exists()


def test_rejected_audit_blocks_stage_without_running(tmp_path, capsys):
    config = _write_config(tmp_path)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"datasets": [{"dataset_id": "bad", "accepted": False}]}),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--config",
            str(config),
            "--stage",
            "fit-imputers",
            "--run-id",
            "rejected-data",
            "--audit-artifact",
            str(audit),
        ]
    )
    assert code == 2
    assert "rejected datasets: bad" in capsys.readouterr().err
    manifest = json.loads(
        (tmp_path / "artifacts" / "rejected-data" / "stage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "blocked"


@pytest.mark.parametrize(
    ("stage", "run_id", "required_option"),
    (
        ("labels", "missing-label-inputs", "--forecaster-artifact"),
        ("train-router", "missing-router-training-inputs", "--labels-artifact"),
        ("impute", "missing-imputation-inputs", "--router-artifact"),
    ),
)
def test_each_later_stage_reports_its_required_artifacts(
    tmp_path,
    capsys,
    stage,
    run_id,
    required_option,
):
    config = _write_config(tmp_path)
    code = main(
        [
            "run",
            "--config",
            str(config),
            "--stage",
            stage,
            "--run-id",
            run_id,
        ]
    )
    assert code == 2
    assert required_option in capsys.readouterr().err
    manifest = json.loads(
        (tmp_path / "artifacts" / run_id / "stage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["stage"] == stage
    assert manifest["status"] == "blocked"
    assert manifest["execution_started"] is False


def test_artifact_store_rejects_traversal_and_overwrite(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        validate_run_id("../outside")
    store = RunArtifactStore.create(tmp_path, "safe-run")
    assert store.root == (tmp_path / "safe-run").resolve()
    with pytest.raises(FileExistsError, match="choose a new --run-id"):
        RunArtifactStore.create(tmp_path, "safe-run")
    assert RunArtifactStore.open_existing(tmp_path, "safe-run") == store


def test_resume_preparation_requires_identical_resolved_config_and_inputs(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    inputs = StageInputs(audit_artifact=audit)
    prepare_stage(
        config,
        config_path,
        "fit-imputers",
        inputs,
        run_id="resume-preparation",
    )

    resumed = prepare_stage(
        config,
        config_path,
        "fit-imputers",
        inputs,
        run_id="resume-preparation",
        resume=True,
    )
    assert resumed.resuming is True

    changed = config.model_copy(update={"seed": config.seed + 1})
    with pytest.raises(ValueError, match="resolved config differs"):
        prepare_stage(
            changed,
            config_path,
            "fit-imputers",
            inputs,
            run_id="resume-preparation",
            resume=True,
        )


def test_cli_resume_requires_execute_and_run_id(tmp_path, capsys):
    config = _write_config(tmp_path)

    code = main(
        [
            "run",
            "--config",
            str(config),
            "--stage",
            "fit-imputers",
            "--resume",
        ]
    )

    assert code == 2
    assert "--resume requires --execute" in capsys.readouterr().err


def test_explicit_execution_dispatch_updates_completed_manifest(
    tmp_path, monkeypatch
):
    import tsfm_fais.stage_execution as execution

    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    preparation = prepare_stage(
        config,
        config_path,
        "fit-imputers",
        StageInputs(audit_artifact=audit),
        run_id="execute-dispatch",
    )
    monkeypatch.setitem(
        execution._EXECUTORS,
        "fit-imputers",
        lambda preparation, config, inputs: {"synthetic": "complete"},
    )
    outputs = execution.execute_prepared_stage(
        preparation, config, StageInputs(audit_artifact=audit)
    )
    assert outputs == {"synthetic": "complete"}
    manifest = json.loads(
        (preparation.store.root / "stage_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["execution_started"] is True


def test_labels_stage_accepts_comma_separated_forecaster_ids(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    checkpoints = tmp_path / "checkpoints"
    imputers.mkdir()
    checkpoints.mkdir()

    preparation = prepare_stage(
        config,
        config_path,
        "labels",
        StageInputs(
            audit_artifact=audit,
            imputer_artifacts=imputers,
            forecaster_artifact=checkpoints,
            forecaster_id="chronos2,timesfm2p5",
        ),
        run_id="multiple-forecasters",
    )

    check = next(
        entry
        for entry in preparation.manifest["checks"]
        if entry["name"] == "forecaster_id"
    )
    assert check["valid"] is True


@pytest.mark.parametrize(
    "forecaster_ids",
    ("chronos2,chronos2", "chronos2,unknown", "chronos2,"),
)
def test_comma_separated_forecaster_ids_reject_invalid_lists(
    tmp_path, forecaster_ids
):
    config = load_config(_write_config(tmp_path))

    check = _forecaster_check(config, forecaster_ids)

    assert check["valid"] is False


def test_impute_stage_accepts_router_fold_root(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    imputers.mkdir()
    fold_root = tmp_path / "router-folds"
    first_fold = fold_root / "family-a"
    second_fold = fold_root / "family-b"
    first_fold.mkdir(parents=True)
    second_fold.mkdir(parents=True)
    (first_fold / "router_bundle.joblib").write_bytes(b"test stub")
    (second_fold / "router_bundle.joblib").write_bytes(b"test stub")
    (fold_root / "folds.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "leave_family_out",
                "folds": {
                    "family-a": str(first_fold),
                    "family-b": str(second_fold),
                },
            }
        ),
        encoding="utf-8",
    )

    preparation = prepare_stage(
        config,
        config_path,
        "impute",
        StageInputs(
            audit_artifact=audit,
            imputer_artifacts=imputers,
            router_artifact=fold_root,
            forecaster_id="chronos2",
        ),
        run_id="router-fold-root",
    )

    check = next(
        entry
        for entry in preparation.manifest["checks"]
        if entry["name"] == "router_artifact"
    )
    assert check["valid"] is True


def test_impute_stage_rejects_multiple_forecaster_ids(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    router = tmp_path / "router"
    imputers.mkdir()
    router.mkdir()
    (router / "router_bundle.joblib").write_bytes(b"test stub")

    with pytest.raises(StagePreparationError, match="exactly one forecaster ID"):
        prepare_stage(
            config,
            config_path,
            "impute",
            StageInputs(
                audit_artifact=audit,
                imputer_artifacts=imputers,
                router_artifact=router,
                forecaster_id="chronos2,timesfm2p5",
            ),
            run_id="impute-multiple-forecasters",
        )
