from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.artifacts import RunArtifactStore
from tsfm_fais.config import (
    AppConfig,
    ExperimentConfig,
    RegistryRef,
    RuntimeConfig,
)
from tsfm_fais.contracts import TimeSeriesItem
from tsfm_fais.stage_execution import execute_fit_imputers
from tsfm_fais.stages import StageInputs, StagePreparation


def _config(tmp_path, *, fail_fast: bool = False) -> AppConfig:
    placeholder = tmp_path / "placeholder.yaml"
    placeholder.write_text("schema_version: 1\n", encoding="utf-8")
    return AppConfig(
        seed=17,
        registries=RegistryRef(
            data_manifest=placeholder,
            imputer_registry=placeholder,
            forecaster_registry=placeholder,
            router_config=placeholder,
        ),
        experiment=ExperimentConfig(
            context_length=4,
            horizon=2,
            candidate_ids=("locf", "linear_interp", "knn_multivariate"),
            max_training_windows_per_dataset=3,
            missforest_n_jobs=3,
        ),
        runtime=RuntimeConfig(
            output_root=tmp_path / "artifacts",
            device="cpu",
            fail_fast=fail_fast,
        ),
    )


def _item(offset: float = 0.0) -> TimeSeriesItem:
    time = np.arange(24, dtype=float)
    values = np.column_stack((time + offset, np.sin(time), np.cos(time)))
    return TimeSeriesItem(
        item_id="item-0",
        values=values,
        variate_names=("trend", "sin", "cos"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )


def _preparation(config: AppConfig, *, resume: bool) -> StagePreparation:
    if resume:
        store = RunArtifactStore.open_existing(config.runtime.output_root, "fit-resume")
    else:
        store = RunArtifactStore.create(config.runtime.output_root, "fit-resume")
    return StagePreparation(
        stage="fit-imputers",
        store=store,
        manifest={},
        resuming=resume,
    )


def _patch_dataset(monkeypatch, item: TimeSeriesItem) -> None:
    dataset = SimpleNamespace(dataset_id="synthetic", period=4)
    monkeypatch.setattr(
        "tsfm_fais.stage_execution._datasets",
        lambda config, audit: iter(((dataset, (item,)),)),
    )


def test_fit_resume_validates_and_skips_loadable_candidate(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    audit = tmp_path / "audit.json"
    audit.write_text('{"accepted":true}\n', encoding="utf-8")
    inputs = StageInputs(audit_artifact=audit)
    _patch_dataset(monkeypatch, _item())

    first = execute_fit_imputers(_preparation(config, resume=False), config, inputs)
    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["resume"]["root_seed"] == 17
    assert manifest["datasets"]["synthetic"]["candidates"]["knn_multivariate"][
        "status"
    ] == "fitted"

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("a verified artifact must not be fitted again")

    monkeypatch.setattr("tsfm_fais.stage_execution.CandidateRunner.fit", unexpected_fit)
    execute_fit_imputers(_preparation(config, resume=True), config, inputs)
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = resumed["datasets"]["synthetic"]["candidates"]["knn_multivariate"]
    assert entry["attempts"] == 1
    assert "verified_at" in entry


def test_fit_resume_migrates_legacy_final_manifest(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    audit = tmp_path / "audit.json"
    audit.write_text('{"accepted":true}\n', encoding="utf-8")
    inputs = StageInputs(audit_artifact=audit)
    _patch_dataset(monkeypatch, _item())
    first = execute_fit_imputers(_preparation(config, resume=False), config, inputs)
    manifest_path = Path(first["manifest"])
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy.pop("resume")
    legacy.pop("status")
    legacy["datasets"]["synthetic"].pop("training_summary")
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("a legacy loadable artifact must be reused")

    monkeypatch.setattr("tsfm_fais.stage_execution.CandidateRunner.fit", unexpected_fit)
    execute_fit_imputers(_preparation(config, resume=True), config, inputs)

    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["legacy_manifest_migrated"] is True
    assert migrated["status"] == "completed"
    assert "training_summary" in migrated["datasets"]["synthetic"]


def test_fit_resume_rejects_changed_audit_or_training_data(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    audit = tmp_path / "audit.json"
    audit.write_text('{"accepted":true}\n', encoding="utf-8")
    inputs = StageInputs(audit_artifact=audit)
    _patch_dataset(monkeypatch, _item())
    execute_fit_imputers(_preparation(config, resume=False), config, inputs)

    audit.write_text('{"accepted":true,"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="audit, seed, resolved config"):
        execute_fit_imputers(_preparation(config, resume=True), config, inputs)

    audit.write_text('{"accepted":true}\n', encoding="utf-8")
    _patch_dataset(monkeypatch, _item(offset=0.5))
    with pytest.raises(ValueError, match="training batch summary changed"):
        execute_fit_imputers(_preparation(config, resume=True), config, inputs)


def test_fit_failure_is_atomically_recorded_before_fail_fast(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, fail_fast=True)
    audit = tmp_path / "audit.json"
    audit.write_text('{"accepted":true}\n', encoding="utf-8")
    _patch_dataset(monkeypatch, _item())

    def fail_fit(*args, **kwargs):
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr("tsfm_fais.stage_execution.CandidateRunner.fit", fail_fit)
    preparation = _preparation(config, resume=False)
    with pytest.raises(RuntimeError, match="deliberate fit failure"):
        execute_fit_imputers(
            preparation,
            config,
            StageInputs(audit_artifact=audit),
        )
    manifest = json.loads(
        (preparation.store.root / "imputer_artifacts" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    entry = manifest["datasets"]["synthetic"]["candidates"]["knn_multivariate"]
    assert entry["status"] == "failed"
    assert entry["attempts"] == 1
    assert "deliberate fit failure" in entry["reason"]
