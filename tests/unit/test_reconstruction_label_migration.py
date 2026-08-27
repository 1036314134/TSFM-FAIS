from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tsfm_fais.label_resume import (
    LabelEpisodeExpectation,
    LabelProgressStore,
    canonical_sha256,
    validate_label_rows,
)


def _load_migration_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "migrate_reconstruction_label_sidecars_v001.py"
    )
    spec = importlib.util.spec_from_file_location("reconstruction_label_migration", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unary_rows(expectation: LabelEpisodeExpectation) -> list[dict]:
    rows = []
    for index, candidate_id in enumerate(expectation.candidate_ids):
        rows.append(
            {
                "episode_id": expectation.episode_id,
                "dataset_id": expectation.dataset_id,
                "family_id": expectation.family_id,
                "forecaster_id": expectation.forecaster_id,
                "group_id": (
                    f"{expectation.forecaster_id}::{expectation.episode_id}::__sequence__"
                ),
                "block_id": "__sequence__",
                "candidate_id": candidate_id,
                "label_scope": "whole_series",
                "prior_features": {"length": 8.0},
                "unary_features": {"proxy": 0.1 + index},
                "imputation_loss": 0.2 + index,
                "imputation_mae": 0.1 + index,
                "imputation_rmse": 0.3 + index,
                "imputation_reward": 1.0 / (index + 2.0),
            }
        )
    return rows


def _write_reduced_source(tmp_path: Path):
    migration = _load_migration_module()
    source = tmp_path / "reconstruction-labels-v001"
    source.mkdir()
    for name in migration.STATIC_ROOT_FILES:
        migration._atomic_write_json(source / name, {"name": name})
    selection_summary = {"selected_episode_count": 1}
    episode_id = "dataset-a__item-a__100__random_point__0.1__1101"
    unsigned_plan = {
        "dataset_id": "dataset-a",
        "selection_summary": selection_summary,
        "episode_ids": [episode_id],
    }
    plan = {**unsigned_plan, "sha256": canonical_sha256(unsigned_plan)}
    expectation = LabelEpisodeExpectation(
        artifact_index=0,
        forecaster_id="imputation",
        episode_id=episode_id,
        dataset_id="dataset-a",
        family_id="family-a",
        item_id="item-a",
        forecast_origin=100,
        sampling_cell={
            "mechanism": "random_point",
            "missing_rate": 0.1,
            "configured_seed": 1101,
        },
        dataset_plan_sha256=plan["sha256"],
        candidate_ids=("locf", "linear_interp"),
        block_ids=("__sequence__",),
    )
    unary_rows = _unary_rows(expectation)
    validated = validate_label_rows(expectation, unary_rows, ())
    sidecar = {
        "schema_version": 1,
        "expectation": expectation.to_payload(),
        "outcome": "labeled",
        "unary_rows": validated["unary_rows"],
        "pair_rows": [],
        "created_at": "2026-08-10T00:00:00Z",
    }
    sidecar_path = source / expectation.sidecar_relative_path
    migration._atomic_write_json(sidecar_path, sidecar)
    entry = {
        "artifact_index": 0,
        "expectation": expectation.to_payload(),
        "outcome": "labeled",
        "sidecar_file": expectation.sidecar_relative_path.as_posix(),
        "sidecar_sha256": migration._sha256(sidecar_path),
        "unary_rows": 2,
        "pair_rows": 0,
        "ranking_groups": 1,
        "completed_at": "2026-08-10T00:00:00Z",
    }
    progress = {
        "schema_version": 1,
        "status": "rebuilt",
        "identity": {"schema_version": 1, "protocol": "fixture"},
        "dataset_plans": {"dataset-a": plan},
        "entries": {"00000000": entry},
        "completed_count": 1,
        "resume_count": 0,
        "repair_count": 0,
        "created_at": "2026-08-10T00:00:00Z",
        "updated_at": "2026-08-10T00:00:00Z",
    }
    migration._atomic_write_json(source / "labels_progress.json", progress)
    migration._atomic_write_jsonl(source / "teacher_labels.jsonl", unary_rows)
    migration._atomic_write_jsonl(source / "pair_labels.jsonl", [])
    manifest = {
        "teacher_labels": str((source / "teacher_labels.jsonl").resolve()),
        "pair_labels": str((source / "pair_labels.jsonl").resolve()),
        "progress": str((source / "labels_progress.json").resolve()),
        "teacher_labels_sha256": migration._sha256(source / "teacher_labels.jsonl"),
        "pair_labels_sha256": migration._sha256(source / "pair_labels.jsonl"),
        "forecasters": ["imputation"],
        "routing_target_protocol": "sequence_imputation_quality_v1",
        "target_protocol": "masked_context_reconstruction_asmape_v1",
        "active_mask_partition": "train",
        "active_mask_seeds": [1101, 1102, 1103],
        "episode_count": 1,
        "expected_episode_count": 1,
        "labeled_episode_count": 1,
        "no_label_episode_count": 0,
        "unary_rows": 2,
        "pair_rows": 0,
        "ranking_groups": 1,
        "resume_count": 0,
        "repair_count": 0,
        "selected_candidates": ["locf", "linear_interp"],
        "episodes_executed_last_invocation": 1,
        "episodes_reused_last_invocation": 0,
    }
    migration._atomic_write_json(source / "labels_manifest.json", manifest)
    stage = {
        "schema_version": 1,
        "run_id": source.name,
        "stage": "labels",
        "status": "completed",
        "created_at": "2026-08-10T00:00:00Z",
        "updated_at": "2026-08-10T00:00:00Z",
        "execution_started": True,
        "outputs": manifest,
    }
    migration._atomic_write_json(source / "stage_manifest.json", stage)
    return migration, source, expectation


def _migrate_fixture(migration, source: Path, target: Path):
    return migration.migrate_label_artifact(
        source,
        target,
        expected_episode_count=1,
        expected_dataset_count=1,
        expected_family_count=1,
        expected_episodes_per_dataset=1,
        require_complete_sampling=False,
    )


def test_migration_preserves_rows_and_produces_strict_sidecars(tmp_path):
    migration, source, expectation = _write_reduced_source(tmp_path)
    target = tmp_path / "reconstruction-labels-v002"
    source_signature = migration._tree_signature(source)

    result = _migrate_fixture(migration, source, target)

    assert result["status"] == "verified"
    assert result["sidecars_validated"] == 1
    assert result["resume_count"] == 0
    assert result["repair_count"] == 0
    assert migration._tree_signature(source) == source_signature
    assert (target / "teacher_labels.jsonl").read_bytes() == (
        source / "teacher_labels.jsonl"
    ).read_bytes()
    assert (target / "pair_labels.jsonl").read_bytes() == (
        source / "pair_labels.jsonl"
    ).read_bytes()

    progress = json.loads((target / "labels_progress.json").read_text(encoding="utf-8"))
    store = LabelProgressStore(target, progress)
    validation = store.validate_episode(expectation)
    assert validation.status == "valid"
    assert validation.sidecar is not None
    for field in (
        "unary_rows_sha256",
        "pair_rows_sha256",
        "unary_row_count",
        "pair_row_count",
        "ranking_group_count",
    ):
        assert field in validation.sidecar
    entry = progress["entries"][expectation.key]
    assert "unary_rows_sha256" in entry
    assert "pair_rows_sha256" in entry
    assert progress["status"] == "rebuilt"
    assert progress["resume_count"] == progress["repair_count"] == 0
    assert not (target / "migration_failure.json").exists()

    labels_manifest = json.loads((target / "labels_manifest.json").read_text(encoding="utf-8"))
    assert labels_manifest["artifact_derivation"]["numerical_rows_changed"] is False
    assert labels_manifest["episodes_executed_last_invocation"] == 0
    assert labels_manifest["episodes_reused_last_invocation"] == 1

    migration_manifest = json.loads(
        (target / "migration_manifest.json").read_text(encoding="utf-8")
    )
    digest = migration_manifest.pop("canonical_sha256")
    assert digest == canonical_sha256(migration_manifest)


def test_invalid_source_is_rejected_before_target_creation(tmp_path):
    migration, source, _ = _write_reduced_source(tmp_path)
    target = tmp_path / "reconstruction-labels-v002"
    (source / "teacher_labels.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="teacher-label signature"):
        _migrate_fixture(migration, source, target)

    assert not target.exists()


def test_partial_failure_is_preserved_without_retry(tmp_path, monkeypatch):
    migration, source, _ = _write_reduced_source(tmp_path)
    target = tmp_path / "reconstruction-labels-v002"

    def fail_after_target_creation(*_args, **_kwargs):
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(migration, "_build_strict_progress", fail_after_target_creation)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        _migrate_fixture(migration, source, target)

    assert target.is_dir()
    failure = json.loads((target / "migration_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "injected migration failure"
