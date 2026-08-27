from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tsfm_fais.evaluation import _impute_content_signature, _verify_imputation_npz_integrity

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_runner() -> ModuleType:
    path = REPOSITORY_ROOT / "scripts" / "run_risk_fallback_v001.py"
    spec = importlib.util.spec_from_file_location("risk_fallback_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signature(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_source(tmp_path: Path, runner: ModuleType) -> Path:
    source = tmp_path / "source"
    (source / "imputations" / "toy").mkdir(parents=True)
    (source / "assignment_records" / "toy").mkdir(parents=True)
    audit = tmp_path / "audit.json"
    imputer_manifest = tmp_path / "fit_manifest.json"
    imputer_registry = tmp_path / "imputer_pool.yaml"
    audit.write_text("{}\n", encoding="utf-8")
    imputer_manifest.write_text("{}\n", encoding="utf-8")
    imputer_registry.write_text("schema_version: 1\n", encoding="utf-8")
    candidate_specs = {
        "linear_interp": {"execution_params": {}, "spec": {}},
        "locf": {"execution_params": {}, "spec": {}},
    }
    source_files = {
        name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for name, path in runner.SOURCE_FILE_PATHS.items()
    }
    generation_identity = {
        "schema_version": 1,
        "root_seed": 1,
        "torch_device": "cpu",
        "candidate_ids": ["locf", "linear_interp"],
        "candidate_specs": candidate_specs,
        "candidate_specs_sha256": _canonical_sha256(candidate_specs),
        "episode_protocol": {},
        "audit_artifact": _signature(audit),
        "imputer_artifact_manifest": _signature(imputer_manifest),
        "imputer_registry": _signature(imputer_registry),
        "source_files": source_files,
    }
    observed = np.ones((8, 2), dtype=bool)
    observed[3:5, 0] = False
    clean = np.arange(16, dtype=float).reshape(8, 2)
    locf = clean.copy()
    locf[~observed] = 5.0
    linear = clean.copy()
    linear[~observed] = 7.0
    npz_path = source / "imputations" / "toy" / "00000000.npz"
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray([3]),
        episode_id=np.asarray(["toy__item__8__random_point__0.1__3101"]),
        dataset_id=np.asarray(["toy"]),
        family_id=np.asarray(["toy_family"]),
        item_id=np.asarray(["item"]),
        forecaster_id=np.asarray(["chronos2"]),
        assembled_method_id=np.asarray(["b_fais"]),
        forecast_mode=np.asarray(["joint_multivariate"]),
        forecaster_independent_selection=np.asarray([False]),
        values=locf,
        observed_mask=observed,
        clean_context=clean,
        clean_future=np.ones((2, 2), dtype=float),
        context_truth_mask=np.ones_like(observed),
        future_observed_mask=np.ones((2, 2), dtype=bool),
        base_observed_mask=np.ones_like(observed),
        mask_protocol=np.asarray(["sequence_mask_v2"]),
        mask_seed=np.asarray([11], dtype=np.uint64),
        mask_realization_id=np.asarray(["mask-1"]),
        target_missing_rate=np.asarray([0.1]),
        global_missing_rate=np.asarray([0.1]),
        local_missing_rate=np.asarray([0.125]),
        mase_scale=np.ones(2),
        mase_scale_lag=np.asarray([1]),
        period=np.asarray([2]),
        candidate_ids=np.asarray(["locf", "linear_interp"]),
        candidate_values=np.stack((locf, linear)),
        candidate_native_valid=np.ones((2, 8, 2), dtype=bool),
        candidate_status=np.asarray(["success", "success"]),
        candidate_runtime_seconds=np.asarray([0.1, 0.2]),
        candidate_peak_memory_bytes=np.asarray([10, 20]),
        pipeline_runtime_seconds=np.asarray(0.3),
        pipeline_rss_before_bytes=np.asarray(100),
        pipeline_rss_after_bytes=np.asarray(120),
        pipeline_rss_delta_bytes=np.asarray(20),
        pipeline_peak_memory_bytes=np.asarray(20),
    )
    assignment = {
        "schema_version": 3,
        "artifact_index": 0,
        "episode_id": "toy__item__8__random_point__0.1__3101",
        "dataset_id": "toy",
        "family_id": "toy_family",
        "item_id": "item",
        "forecaster_id": "chronos2",
        "assembled_method_id": "b_fais",
        "forecast_mode": "joint_multivariate",
        "forecaster_independent_selection": False,
        "file": "toy/00000000.npz",
        "assignment_file": "assignment_records/toy/00000000.json",
        "assignments": {"n0:d0:3-5": "locf"},
        "shortlist": ["locf"],
        "activated_candidates": ["locf"],
        "fallback_blocks": [],
        "fallback_records": {},
        "routing_metadata": {
            "evidence_blend": {"global_prior": 0.0, "proxy": 0.0, "r0": 0.0, "r1": 1.0}
        },
        "forecast_origin": 8,
        "mechanism": "random_point",
        "mask_seed": 11,
        "mask_realization_id": "mask-1",
        "episode_seed": 17,
        "seed": 3101,
        "target_missing_rate": 0.1,
        "global_missing_rate": 0.1,
        "local_missing_rate": 0.125,
        "mask_protocol": "sequence_mask_v2",
        "pipeline_runtime_seconds": 0.3,
    }
    assignment_path = source / "assignment_records" / "toy" / "00000000.json"
    _write_json(assignment_path, assignment)
    assignments_path = source / "routing_assignments.jsonl"
    assignments_path.write_text(
        json.dumps(assignment, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    identity_sha = _canonical_sha256(generation_identity)
    manifest = {
        "schema_version": 3,
        "assembled_method_id": "b_fais",
        "episode_count": 1,
        "forecaster_id": "chronos2",
        "save_all_candidate_outputs": True,
        "candidate_generation_identity": generation_identity,
        "candidate_generation_identity_sha256": identity_sha,
        "resume_identity_sha256": "1" * 64,
        "pipeline_runtime_total_seconds": 0.3,
    }
    _write_json(source / "imputation_manifest.json", manifest)
    entry = {
        "index": 0,
        "episode_id": assignment["episode_id"],
        "dataset_id": "toy",
        "family_id": "toy_family",
        "item_id": "item",
        "forecaster_id": "chronos2",
        "forecast_mode": "joint_multivariate",
        "assembled_method_id": "b_fais",
        "file": "toy/00000000.npz",
        "assignment_file": "assignment_records/toy/00000000.json",
        "candidate_ids": ["locf", "linear_interp"],
        "npz_sha256": _sha256(npz_path),
        "assignment_sha256": _sha256(assignment_path),
    }
    progress = {
        "schema_version": 1,
        "status": "completed",
        "expected_episode_count": 1,
        "completed_count": 1,
        "entries": {"00000000": entry},
        "resume_count": 0,
        "repair_count": 0,
        "imputation_manifest_sha256": _sha256(source / "imputation_manifest.json"),
        "routing_assignments_sha256": _sha256(assignments_path),
    }
    _write_json(source / "imputation_progress.json", progress)
    protocol = {
        "active_mask_partition": "confirmation",
        "protocol_id": "test",
        "target_protocol": "test",
    }
    resolved = {
        "schema_version": 1,
        "source": str(tmp_path / "config.yaml"),
        "config": {
            "protocol": protocol,
            "registries": {"router_config": str(tmp_path / "router.yaml")},
        },
    }
    _write_json(source / "resolved_config.json", resolved)
    _write_json(source / "experiment_protocol.json", {"schema_version": 1, "protocol": protocol})
    _write_json(source / "repository_state.json", {"schema_version": 1})
    _write_json(
        source / "seeds.json",
        {
            "schema_version": 1,
            "active_mask_partition": "confirmation",
            "experiment_seeds": [3101],
        },
    )
    _write_json(
        source / "stage_manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "stage": "impute",
            "run_id": "test-impute",
            "experiment_protocol": protocol,
            "outputs": {
                "episode_count": 1,
                "forecaster_id": "chronos2",
                "candidate_generation_identity_sha256": identity_sha,
                "resume_identity_sha256": "1" * 64,
            },
        },
    )
    return source


def _signed_robust_selection(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "protocol_id": "test",
        "selection": {"selected_candidate_id": "linear_interp"},
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    _write_json(path, payload)
    return path


def _signed_threshold_selection(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "protocol_id": "test",
        "robust_candidate_id": "linear_interp",
        "selection": {"selected_threshold": {"kind": "always", "value": None}},
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    _write_json(path, payload)
    return path


def test_score_and_derive_preserve_and_switch_only_allowed_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    source = _build_source(tmp_path, runner)
    robust = _signed_robust_selection(tmp_path / "robust_selection.json")
    monkeypatch.setattr(runner, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "load_dataset_imputer_artifacts",
        lambda *args, **kwargs: ({}, np.zeros(2), np.eye(2), {}),
    )
    score_root = tmp_path / "score"
    score_result = runner._score(
        SimpleNamespace(
            source_imputation=str(source),
            lineage_mode="normal",
            robust_selection=str(robust),
            imputer_artifacts=str(tmp_path / "fit"),
            output_dir=str(score_root),
            target_indices=(0, 1),
        )
    )
    assert score_result["status"] == "verified"
    source_npz = source / "imputations" / "toy" / "00000000.npz"
    score_npz = score_root / "imputations" / "toy" / "00000000.npz"
    assert source_npz.read_bytes() == score_npz.read_bytes()
    score_signature, score_hashes = _impute_content_signature(
        score_root, score_root / "routing_assignments.jsonl"
    )
    _verify_imputation_npz_integrity(score_root / "imputations", score_hashes)
    assert score_signature["assembled_method_id"] == "b_fais"

    threshold = _signed_threshold_selection(tmp_path / "threshold_selection.json")
    derived_root = tmp_path / "derived"
    derived_result = runner._derive(
        SimpleNamespace(
            score_artifact=str(score_root),
            threshold_selection=str(threshold),
            output_dir=str(derived_root),
        )
    )
    assert derived_result["status"] == "verified"
    assert derived_result["switched_count"] == 1
    derived_signature, derived_hashes = _impute_content_signature(
        derived_root, derived_root / "routing_assignments.jsonl"
    )
    _verify_imputation_npz_integrity(derived_root / "imputations", derived_hashes)
    assert derived_signature["assembled_method_id"] == "b_fais"
    with (
        np.load(score_npz, allow_pickle=False) as before,
        np.load(derived_root / "imputations" / "toy" / "00000000.npz", allow_pickle=False) as after,
    ):
        changed = {
            name
            for name in before.files
            if not (
                np.array_equal(before[name], after[name], equal_nan=True)
                if before[name].dtype.kind in "fc"
                else np.array_equal(before[name], after[name])
            )
        }
        assert changed == {"values", "pipeline_runtime_seconds"}
        missing = ~before["observed_mask"]
        robust_index = list(before["candidate_ids"]).index("linear_interp")
        assert np.array_equal(
            after["values"][missing], before["candidate_values"][robust_index][missing]
        )
