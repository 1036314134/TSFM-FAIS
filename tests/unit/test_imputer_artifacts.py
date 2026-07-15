from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

import tsfm_fais.imputers.artifacts as artifact_module
from tsfm_fais.contracts import CandidateStatus, SeriesBatch
from tsfm_fais.imputers import (
    CandidateRunner,
    DatasetImputerArtifactStore,
    KNNMultivariateImputer,
    MissForestImputer,
)


def _write_store(
    root: Path,
    candidates: dict[str, dict[str, object]],
) -> Path:
    directory = root / "toy"
    directory.mkdir(parents=True)
    np.savez(
        directory / "training_statistics.npz",
        medians=np.asarray([1.0, 2.0, 3.0]),
        correlation=np.eye(3),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "toy": {
                        "statistics": "training_statistics.npz",
                        "candidates": candidates,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def _batch(*, missing: bool = False) -> SeriesBatch:
    values = np.arange(36, dtype=float).reshape(1, 12, 3)
    mask = np.ones_like(values, dtype=bool)
    if missing:
        mask[0, 4:7, 1] = False
    return SeriesBatch(values, mask)


def test_candidate_filter_is_applied_before_deserialization(tmp_path, monkeypatch):
    _write_store(
        tmp_path,
        {
            "knn_multivariate": {
                "status": "fitted",
                "serializer": "must-not-be-inspected",
                "path": "absent.bin",
            },
            "mice": {
                "status": "fitted",
                "serializer": "joblib",
                "path": "mice.joblib",
            },
            "missforest": {
                "status": "failed",
                "reason": "synthetic fit failure",
            },
        },
    )
    calls = []

    def fake_load(path):
        calls.append(Path(path).name)
        return {"loaded": Path(path).name}

    monkeypatch.setattr(artifact_module.joblib, "load", fake_load)
    store = DatasetImputerArtifactStore(tmp_path, "toy")
    result = store.load_artifacts(("mice",))

    assert calls == ["mice.joblib"]
    assert result.requested_ids == ("mice",)
    assert result.attempted_ids == ("mice",)
    assert result.artifacts == {"mice": {"loaded": "mice.joblib"}}
    assert not result.failures


def test_filtered_loader_propagates_manifest_and_deserialization_failures(tmp_path):
    _write_store(
        tmp_path,
        {
            "mice": {
                "status": "fitted",
                "serializer": "joblib",
                "path": "missing.joblib",
            },
            "missforest": {
                "status": "failed",
                "reason": "synthetic fit failure",
            },
        },
    )
    store = DatasetImputerArtifactStore(tmp_path, "toy")
    result = store.load_artifacts(("mice", "missforest"))

    assert result.attempted_ids == ("mice",)
    assert "FileNotFoundError" in result.failures["mice"]
    assert result.failures["missforest"] == "synthetic fit failure"

    candidate = CandidateRunner().run_many(
        ("mice",),
        _batch(missing=True),
        artifact_failures={"mice": result.failures["mice"]},
    )["mice"]
    assert candidate.status is CandidateStatus.FAILED
    assert candidate.metadata["artifact_load_failure"] == result.failures["mice"]
    assert "artifact load failed" in (candidate.failure_reason or "")


def test_filtered_deserialization_preserves_candidate_output(tmp_path):
    training = _batch()
    artifact = KNNMultivariateImputer(n_neighbors=2).fit(training, {})
    directory = _write_store(
        tmp_path,
        {
            "knn_multivariate": {
                "status": "fitted",
                "serializer": "joblib",
                "path": "knn.joblib",
            }
        },
    )
    joblib.dump(artifact, directory / "knn.joblib")
    loaded = DatasetImputerArtifactStore(tmp_path, "toy").load_artifacts(
        ("knn_multivariate",)
    )
    evaluation = _batch(missing=True)

    direct = CandidateRunner().run("knn_multivariate", evaluation, artifact, seed=9)
    filtered = CandidateRunner().run(
        "knn_multivariate",
        evaluation,
        loaded.artifacts["knn_multivariate"],
        seed=9,
    )

    assert direct.status is filtered.status
    np.testing.assert_allclose(direct.values, filtered.values)
    np.testing.assert_array_equal(direct.native_valid_mask, filtered.native_valid_mask)


def test_missforest_uses_read_only_mmap_without_modifying_artifact(tmp_path):
    training = _batch()
    artifact = MissForestImputer(
        n_estimators=2,
        max_iter=1,
        n_jobs=1,
    ).fit(training, {})
    directory = _write_store(
        tmp_path,
        {
            "missforest": {
                "status": "fitted",
                "serializer": "joblib",
                "path": "missforest.joblib",
            }
        },
    )
    path = directory / "missforest.joblib"
    joblib.dump(artifact, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = DatasetImputerArtifactStore(tmp_path, "toy").load_artifacts(
        ("missforest",)
    )
    result = CandidateRunner().run(
        "missforest",
        _batch(missing=True),
        loaded.artifacts["missforest"],
        seed=7,
        params={"n_estimators": 2, "max_iter": 1, "n_jobs": 1},
    )
    after = hashlib.sha256(path.read_bytes()).hexdigest()

    assert loaded.load_modes == {"missforest": "joblib_mmap_r"}
    assert result.status is CandidateStatus.SUCCESS
    assert before == after
