from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm_fais.artifacts import RunArtifactStore, validate_run_id
from tsfm_fais.cli import _apply_config_overrides, build_parser, main
from tsfm_fais.config import load_config, validate_forecaster_revision_binding
from tsfm_fais.registry_configs import validate_project_configuration
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
        (config_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
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


def _write_r2_config(root: Path) -> Path:
    config = _write_config(root)
    payload = config.read_text(encoding="utf-8")
    payload = payload.replace(
        "  seeds: [1, 2]",
        "  seeds: [1101, 1102, 1103]\n"
        "  router_seed: 4101\n"
        "  exclude_family_ids: [ett]\n"
        "  feature_policy: deployment_available",
    ).replace(
        "  output_root: ../artifacts",
        "  output_root: ../artifacts/iclr27-r2",
    )
    payload += """
protocol:
  protocol_id: iclr27-r2-test-v1
  artifact_namespace: iclr27-r2
  run_id_prefix: r2
  family_split_policy: ett_dev_non_ett_leave_family_out_v1
  development_family_ids: [ett]
  target_protocol: full_candidate_forecast_loss_v2
  active_mask_partition: train
  mask_seeds:
    train: [1101, 1102, 1103]
    development: [2101, 2102, 2103]
    confirmation: [3101, 3102, 3103]
  router_seed_roots: [4101, 4102, 4103, 4104, 4105]
  teacher_forecaster_ids: [chronos2, timesfm2p5]
  held_out_forecaster_id: sundial
  held_out_forecaster_revision: 3212e42564493f520593e5414af4367fc4b49226
"""
    config.write_text(payload, encoding="utf-8")
    return config


def test_evaluate_parser_accepts_shared_evaluation_artifact() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "config.yaml",
            "--impute-artifact",
            "impute",
            "--forecaster-id",
            "chronos2",
            "--forecaster-artifact",
            "checkpoint",
            "--output-dir",
            "evaluation",
            "--shared-evaluation-artifact",
            "shared-evaluation",
            "--shared-reference-only",
        ]
    )

    assert args.shared_evaluation_artifact == "shared-evaluation"
    assert args.shared_reference_only is True


def test_run_parser_accepts_reconstruction_label_artifact() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--stage",
            "train-router",
            "--labels-artifact",
            "forecast-labels.jsonl",
            "--reconstruction-labels-artifact",
            "reconstruction-labels.jsonl",
        ]
    )

    assert args.reconstruction_labels_artifact == "reconstruction-labels.jsonl"


def test_run_parser_accepts_episode_plan_artifact() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--stage",
            "labels",
            "--episode-plan-artifact",
            "completed-labels/labels_progress.json",
        ]
    )

    assert args.episode_plan_artifact == "completed-labels/labels_progress.json"


def test_router_seed_override_is_revalidated_against_protocol(tmp_path: Path) -> None:
    config = load_config(_write_r2_config(tmp_path))

    updated = _apply_config_overrides(config, SimpleNamespace(router_seed=4105))

    assert updated.experiment.router_seed == 4105
    with pytest.raises(ValueError, match="router_seed_roots"):
        _apply_config_overrides(config, SimpleNamespace(router_seed=9999))


def test_summarize_main_parser_accepts_primary_comparator_roles() -> None:
    args = build_parser().parse_args(
        [
            "summarize-main",
            "--input",
            "evaluation-a",
            "evaluation-b",
            "--output-dir",
            "summary",
            "--primary-comparator-role",
            "selector_baseline",
        ]
    )

    assert args.primary_comparator_role == ["selector_baseline"]


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


def _sequence_router_artifact(path: Path, *, strict: bool = True) -> Path:
    path.mkdir()
    (path / "router_bundle.joblib").write_bytes(b"test stub")
    metadata = {
        "selector_method": "metaod",
        "forecaster_independent_selection": True,
        "uses_missing_block_graph": False,
        "requires_pseudo_candidates": False,
        "routing_target_protocol": "sequence_imputation_quality_v1",
        "selector_training_target": "imputation_loss",
    }
    if not strict:
        metadata.pop("routing_target_protocol")
    (path / "manifest.json").write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    return path


def _use_sequence_router_config(config_path: Path) -> None:
    (config_path.parent / "router.yaml").write_text(
        Path("configs/router/baseline_selector_suite.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


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


def test_r2_protocol_isolates_artifacts_and_writes_auditable_metadata(tmp_path):
    config_path = _write_r2_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")

    preparation = prepare_stage(
        config,
        config_path,
        "fit-imputers",
        StageInputs(audit_artifact=audit),
    )

    assert preparation.store.root.parent == (tmp_path / "artifacts" / "iclr27-r2").resolve()
    assert preparation.store.run_id.startswith("r2-fit-imputers-")
    protocol = json.loads(
        (preparation.store.root / "experiment_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["protocol"]["held_out_forecaster_id"] == "sundial"
    assert (
        protocol["protocol"]["held_out_forecaster_revision"]
        == "3212e42564493f520593e5414af4367fc4b49226"
    )
    assert protocol["execution_binding"]["feature_policy"] == "deployment_available"
    repository = json.loads(
        (preparation.store.root / "repository_state.json").read_text(encoding="utf-8")
    )
    assert {
        "commit",
        "dirty",
        "status_entry_count",
        "status_sha256",
        "tracked_diff_sha256",
        "tracked_diff_size_bytes",
        "untracked_reproducibility_files",
        "error",
    }.issubset(repository)
    assert repository["commit"] is not None or repository["error"] is not None
    if repository["error"] is None:
        assert len(repository["status_sha256"]) == 64
        assert len(repository["tracked_diff_sha256"]) == 64
        assert repository["tracked_diff_size_bytes"] >= 0
    seeds = json.loads((preparation.store.root / "seeds.json").read_text(encoding="utf-8"))
    assert seeds["root_seed"] == 123
    assert seeds["active_router_seed"] == 4101
    assert seeds["active_mask_partition"] == "train"
    assert seeds["mask_seed_partitions"]["confirmation"] == [3101, 3102, 3103]
    assert preparation.manifest["experiment_protocol"]["protocol"]["protocol_id"] == (
        "iclr27-r2-test-v1"
    )


def test_r2_protocol_rejects_run_id_without_revision_prefix(tmp_path):
    config_path = _write_r2_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")

    with pytest.raises(ValueError, match="must start with 'r2-'"):
        prepare_stage(
            config,
            config_path,
            "fit-imputers",
            StageInputs(audit_artifact=audit),
            run_id="legacy-name",
        )
    assert not (tmp_path / "artifacts" / "iclr27-r2" / "legacy-name").exists()


def test_r2_protocol_rejects_episode_cap_below_mask_cell_count(tmp_path):
    config_path = _write_r2_config(tmp_path)
    payload = config_path.read_text(encoding="utf-8").replace(
        "  seeds: [1101, 1102, 1103]",
        "  seeds: [1101, 1102, 1103]\n  max_train_episodes_per_dataset: 10",
    )
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="mechanism-rate-seed cell"):
        load_config(config_path)


def test_r2_project_validation_rejects_target_protocol_mismatch(tmp_path):
    config_path = _write_r2_config(tmp_path)
    payload = config_path.read_text(encoding="utf-8").replace(
        "target_protocol: full_candidate_forecast_loss_v2",
        "target_protocol: single_block_counterfactual_forecast_loss_v1",
    )
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="does not match router ranker_target"):
        validate_project_configuration(load_config(config_path))


def test_r2_project_validation_rejects_unknown_family_filter(tmp_path):
    config_path = _write_r2_config(tmp_path)
    payload = config_path.read_text(encoding="utf-8").replace(
        "exclude_family_ids: [ett]",
        "exclude_family_ids: [unknown_family]",
    )
    config_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown family IDs"):
        validate_project_configuration(load_config(config_path))


def test_r2_held_out_forecaster_revision_is_bound_to_artifact_path(tmp_path):
    config = load_config(_write_r2_config(tmp_path))
    assert config.protocol is not None
    revision = config.protocol.held_out_forecaster_revision
    assert revision is not None
    expected = tmp_path / "models" / revision / "checkpoint"
    expected.mkdir(parents=True)

    validate_forecaster_revision_binding(config, "sundial", expected)
    validate_forecaster_revision_binding(config, "chronos2", tmp_path / "anywhere")

    wrong = tmp_path / "models" / "different-revision"
    wrong.mkdir(parents=True)
    with pytest.raises(ValueError, match="declared revision"):
        validate_forecaster_revision_binding(config, "sundial", wrong)


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
        (tmp_path / "artifacts" / run_id / "stage_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == stage
    assert manifest["status"] == "blocked"
    assert manifest["execution_started"] is False


def test_reconstruction_control_requires_its_second_label_artifact(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    router_path = config_path.parent / "router.yaml"
    router_payload = router_path.read_text(encoding="utf-8")
    router_payload += (
        "\nranker_target: imputation_loss\n"
        "selection_granularity: sequence\n"
        "routing_structure: independent\n"
    )
    router_path.write_text(router_payload, encoding="utf-8")
    labels = tmp_path / "forecast-labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    config = load_config(config_path)

    with pytest.raises(StagePreparationError, match="reconstruction-labels-artifact"):
        prepare_stage(
            config,
            config_path,
            "train-router",
            StageInputs(labels_artifact=labels),
            run_id="reconstruction-input-missing",
        )

    reconstruction = tmp_path / "reconstruction-labels.jsonl"
    reconstruction.write_text("{}\n", encoding="utf-8")
    preparation = prepare_stage(
        config,
        config_path,
        "train-router",
        StageInputs(
            labels_artifact=labels,
            reconstruction_labels_artifact=reconstruction,
        ),
        run_id="reconstruction-input-present",
    )
    dependency = next(
        check
        for check in preparation.manifest["checks"]
        if check["name"] == "reconstruction_labels"
    )
    assert dependency["valid"] is True
    assert dependency["required"] is True


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


def test_explicit_execution_dispatch_updates_completed_manifest(tmp_path, monkeypatch):
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
        entry for entry in preparation.manifest["checks"] if entry["name"] == "forecaster_id"
    )
    assert check["valid"] is True


@pytest.mark.parametrize(
    "forecaster_ids",
    ("chronos2,chronos2", "chronos2,unknown", "chronos2,"),
)
def test_comma_separated_forecaster_ids_reject_invalid_lists(tmp_path, forecaster_ids):
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
        entry for entry in preparation.manifest["checks"] if entry["name"] == "router_artifact"
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


def test_impute_stage_allows_sequence_selector_without_forecaster_id(tmp_path):
    config_path = _write_config(tmp_path)
    _use_sequence_router_config(config_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    imputers.mkdir()
    router = _sequence_router_artifact(tmp_path / "router")

    preparation = prepare_stage(
        config,
        config_path,
        "impute",
        StageInputs(
            audit_artifact=audit,
            imputer_artifacts=imputers,
            router_artifact=router,
        ),
        run_id="sequence-without-forecaster",
    )

    forecaster_check = next(
        check for check in preparation.manifest["checks"] if check["name"] == "forecaster_id"
    )
    assert forecaster_check["valid"] is True
    assert forecaster_check["required"] is False


def test_impute_stage_rejects_unproven_independence_without_forecaster_id(tmp_path):
    config_path = _write_config(tmp_path)
    _use_sequence_router_config(config_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    imputers.mkdir()
    router = _sequence_router_artifact(tmp_path / "router", strict=False)

    with pytest.raises(StagePreparationError, match="missing required option --forecaster-id"):
        prepare_stage(
            config,
            config_path,
            "impute",
            StageInputs(
                audit_artifact=audit,
                imputer_artifacts=imputers,
                router_artifact=router,
            ),
            run_id="unproven-sequence-without-forecaster",
        )


def test_impute_stage_rejects_forecaster_artifact_for_independent_selector(tmp_path):
    config_path = _write_config(tmp_path)
    _use_sequence_router_config(config_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    checkpoints = tmp_path / "checkpoints"
    imputers.mkdir()
    checkpoints.mkdir()
    router = _sequence_router_artifact(tmp_path / "router")

    with pytest.raises(StagePreparationError, match="do not accept --forecaster-artifact"):
        prepare_stage(
            config,
            config_path,
            "impute",
            StageInputs(
                audit_artifact=audit,
                imputer_artifacts=imputers,
                router_artifact=router,
                forecaster_artifact=checkpoints,
            ),
            run_id="independent-with-forecaster-artifact",
        )


def test_impute_stage_rejects_router_method_not_enabled_by_config(tmp_path):
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    imputers.mkdir()
    router = _sequence_router_artifact(tmp_path / "router")

    with pytest.raises(StagePreparationError, match="selector method is not enabled"):
        prepare_stage(
            config,
            config_path,
            "impute",
            StageInputs(
                audit_artifact=audit,
                imputer_artifacts=imputers,
                router_artifact=router,
                forecaster_id="chronos2",
            ),
            run_id="router-method-config-mismatch",
        )


def test_independent_sequence_router_rejects_leave_model_out_split(tmp_path):
    config_path = _write_config(tmp_path)
    _use_sequence_router_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "experiment:\n", "experiment:\n  split: leave_model_out\n"
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    audit = _accepted_audit(tmp_path / "audit.json")
    imputers = tmp_path / "imputers"
    imputers.mkdir()
    router = _sequence_router_artifact(tmp_path / "router")

    with pytest.raises(StagePreparationError, match="do not support leave_model_out"):
        prepare_stage(
            config,
            config_path,
            "impute",
            StageInputs(
                audit_artifact=audit,
                imputer_artifacts=imputers,
                router_artifact=router,
            ),
            run_id="independent-leave-model-out",
        )
