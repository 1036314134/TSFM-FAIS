from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import tsfm_fais.label_resume as label_resume
from tsfm_fais.label_resume import (
    LabelEpisodeExpectation,
    LabelProgressStore,
    LabelResumeError,
    build_label_resume_identity,
    validate_label_rows,
)


def _identity(tmp_path: Path) -> dict:
    audit = tmp_path / "audit.json"
    audit.write_text('{"accepted": true}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"datasets": {}}', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model": "mock"}', encoding="utf-8")
    (checkpoint / "model.bin").write_bytes(b"weights")
    registry = tmp_path / "forecasters.yaml"
    registry.write_text("forecasters: [mock]\n", encoding="utf-8")
    return build_label_resume_identity(
        resolved_config={"seed": 17, "experiment": {"context_length": 48}},
        audit_artifact=audit,
        imputer_manifest=manifest,
        checkpoint=checkpoint,
        forecaster_id="mock",
        forecaster_mode="joint_multivariate",
        forecaster_spec={"id": "mock", "max_context": 48},
        selected_candidates=("locf", "linear_interp"),
        source_artifacts={"forecaster_registry": registry},
    )


def test_atomic_progress_write_retries_transient_reader_lock(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "labels_progress.json"
    original_replace = Path.replace
    attempts = 0

    def transient_lock(path, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows sharing violation")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", transient_lock)
    monkeypatch.setattr(label_resume, "sleep", lambda _delay: None)

    label_resume._atomic_write_json(target, {"status": "running"})

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running"}


def _expectation(index: int, plan_sha: str) -> LabelEpisodeExpectation:
    return LabelEpisodeExpectation(
        artifact_index=index,
        forecaster_id="mock",
        episode_id=f"episode-{index}",
        dataset_id="dataset-a",
        family_id="family-a",
        item_id="item-a",
        forecast_origin=100 + index,
        sampling_cell={
            "mechanism": "independent_block",
            "missing_rate": 0.2,
            "configured_seed": 17,
        },
        dataset_plan_sha256=plan_sha,
        candidate_ids=("locf", "linear_interp"),
        block_ids=("b0", "b1"),
    )


def _unary_rows(expectation: LabelEpisodeExpectation) -> list[dict]:
    rows = []
    for block_index, block_id in enumerate(expectation.block_ids):
        for candidate_index, candidate_id in enumerate(expectation.candidate_ids):
            clean = 1.0
            forecast = 1.1 + 0.1 * block_index + 0.01 * candidate_index
            rows.append(
                {
                    "episode_id": expectation.episode_id,
                    "dataset_id": expectation.dataset_id,
                    "family_id": expectation.family_id,
                    "forecaster_id": expectation.forecaster_id,
                    "group_id": (
                        f"{expectation.forecaster_id}::{expectation.episode_id}::"
                        f"{block_id}"
                    ),
                    "block_id": block_id,
                    "candidate_id": candidate_id,
                    "prior_features": {"length": 4.0},
                    "unary_features": {"proxy": 0.25},
                    "forecast_loss": forecast,
                    "clean_loss": clean,
                    "anchor_loss": 1.2,
                    "degradation": forecast - clean,
                }
            )
    return rows


def _pair_rows(expectation: LabelEpisodeExpectation) -> list[dict]:
    return [
        {
            "episode_id": expectation.episode_id,
            "dataset_id": expectation.dataset_id,
            "family_id": expectation.family_id,
            "forecaster_id": expectation.forecaster_id,
            "left_block": "b0",
            "right_block": "b1",
            "left_candidate": "locf",
            "right_candidate": "linear_interp",
            "features": {"gap": 2.0},
            "interaction": -0.05,
        }
    ]


def _store(tmp_path: Path) -> tuple[LabelProgressStore, str]:
    store = LabelProgressStore.create(tmp_path / "run", _identity(tmp_path))
    plan_sha = store.register_dataset_plan(
        "dataset-a",
        {"selected_episode_count": 2},
        ("episode-0", "episode-1"),
    )
    return store, plan_sha


def test_identity_binds_checkpoint_and_all_declared_sources(tmp_path):
    first = _identity(tmp_path)
    assert first["checkpoint"]["kind"] == "directory"
    assert len(first["checkpoint"]["files"]) == 2

    (tmp_path / "checkpoint" / "model.bin").write_bytes(b"changed")
    second = build_label_resume_identity(
        resolved_config={"seed": 17, "experiment": {"context_length": 48}},
        audit_artifact=tmp_path / "audit.json",
        imputer_manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "checkpoint",
        forecaster_id="mock",
        forecaster_mode="joint_multivariate",
        forecaster_spec={"id": "mock", "max_context": 48},
        selected_candidates=("locf", "linear_interp"),
        source_artifacts={"forecaster_registry": tmp_path / "forecasters.yaml"},
    )

    assert first["checkpoint"]["tree_sha256"] != second["checkpoint"]["tree_sha256"]


def test_progress_open_requires_exact_identity_and_consistent_counts(tmp_path):
    identity = _identity(tmp_path)
    store = LabelProgressStore.create(tmp_path / "run", identity)
    reopened = LabelProgressStore.open_existing(tmp_path / "run", identity)
    assert reopened.payload["resume_count"] == 1

    changed = copy.deepcopy(identity)
    changed["forecaster"]["mode"] = "independent_univariate"
    with pytest.raises(LabelResumeError, match="identity changed"):
        LabelProgressStore.open_existing(tmp_path / "run", changed)

    payload = json.loads(store.progress_path.read_text(encoding="utf-8"))
    payload["completed_count"] = 1
    store.progress_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabelResumeError, match="completed_count"):
        LabelProgressStore.open_existing(tmp_path / "run", identity)


def test_progress_open_rejects_corrupt_dataset_plan_signature(tmp_path):
    store, _ = _store(tmp_path)
    payload = json.loads(store.progress_path.read_text(encoding="utf-8"))
    payload["dataset_plans"]["dataset-a"]["sha256"] = "0" * 64
    store.progress_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabelResumeError, match="plan .* signature is invalid"):
        LabelProgressStore.open_existing(store.root, store.payload["identity"])


def test_progress_open_rejects_malformed_entry_without_raw_key_error(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    store.commit_episode(expectation, _unary_rows(expectation), _pair_rows(expectation))
    payload = json.loads(store.progress_path.read_text(encoding="utf-8"))
    del payload["entries"][expectation.key]["expectation"]
    store.progress_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabelResumeError, match="expectation is invalid"):
        LabelProgressStore.open_existing(store.root, store.payload["identity"])


def test_commit_and_validate_episode_sidecar(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    entry = store.commit_episode(
        expectation,
        _unary_rows(expectation),
        _pair_rows(expectation),
        artifact_loading_delta={"load_call_count": 2},
    )

    assert entry["unary_rows"] == 4
    assert entry["pair_rows"] == 1
    assert entry["ranking_groups"] == 2
    validation = store.validate_episode(expectation)
    assert validation.status == "valid"
    assert store.payload["completed_count"] == 1


def test_no_label_episode_is_a_committed_transaction(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    store.commit_episode(expectation, (), (), outcome="no_labels")

    validation = store.validate_episode(expectation)
    assert validation.status == "valid"
    assert validation.sidecar is not None
    assert validation.sidecar["outcome"] == "no_labels"


def test_orphan_sidecar_is_not_a_committed_episode(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    orphan = store.root / expectation.sidecar_relative_path
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}", encoding="utf-8")

    validation = store.validate_episode(expectation)

    assert validation.status == "missing"


def test_interruption_after_sidecar_before_progress_leaves_orphan(
    tmp_path, monkeypatch
):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    original = label_resume._atomic_write_json
    interrupted = False

    def fail_progress(path, payload):
        nonlocal interrupted
        if (
            not interrupted
            and path.name == "labels_progress.json"
            and payload.get("completed_count") == 1
        ):
            interrupted = True
            raise OSError("injected interruption")
        return original(path, payload)

    monkeypatch.setattr(label_resume, "_atomic_write_json", fail_progress)
    with pytest.raises(OSError, match="injected interruption"):
        store.commit_episode(
            expectation,
            _unary_rows(expectation),
            _pair_rows(expectation),
        )
    assert (store.root / expectation.sidecar_relative_path).is_file()

    monkeypatch.setattr(label_resume, "_atomic_write_json", original)
    reopened = LabelProgressStore.open_existing(store.root, store.payload["identity"])
    assert reopened.validate_episode(expectation).status == "missing"


def test_corrupt_sidecar_is_invalid_and_can_be_replaced(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    store.commit_episode(expectation, _unary_rows(expectation), _pair_rows(expectation))
    sidecar = store.root / expectation.sidecar_relative_path
    sidecar.write_text("{", encoding="utf-8")

    assert store.validate_episode(expectation).status == "invalid"
    store.commit_episode(
        expectation,
        _unary_rows(expectation),
        _pair_rows(expectation),
        replace=True,
    )
    assert store.validate_episode(expectation).status == "valid"
    assert store.payload["repair_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda unary, _pairs: unary.__setitem__(1, copy.deepcopy(unary[0])), "duplicate"),
        (
            lambda unary, _pairs: unary[0].__setitem__("group_id", "wrong"),
            "group_id",
        ),
        (
            lambda unary, _pairs: unary[0]["prior_features"].__setitem__(
                "bad", np.nan
            ),
            "finite",
        ),
        (
            lambda unary, _pairs: unary[0].__setitem__("degradation", 9.0),
            "degradation",
        ),
        (
            lambda _unary, pairs: pairs[0].__setitem__(
                "left_candidate", "unknown"
            ),
            "unexpected endpoint",
        ),
    ],
)
def test_strict_row_validation_rejects_invalid_labels(
    tmp_path, mutation, message
):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    unary = _unary_rows(expectation)
    pairs = _pair_rows(expectation)
    mutation(unary, pairs)

    with pytest.raises(ValueError, match=message):
        validate_label_rows(expectation, unary, pairs)


def test_pair_endpoint_requires_matching_unary_row(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    unary = [
        row
        for row in _unary_rows(expectation)
        if not (row["block_id"] == "b0" and row["candidate_id"] == "locf")
    ]

    with pytest.raises(ValueError, match="endpoint has no unary"):
        validate_label_rows(expectation, unary, _pair_rows(expectation))


def test_inner_hash_and_row_validation_survive_updated_outer_hash(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    store.commit_episode(expectation, _unary_rows(expectation), _pair_rows(expectation))
    sidecar_path = store.root / expectation.sidecar_relative_path
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["unary_rows"][0]["degradation"] = 123.0
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    entry = store.payload["entries"][expectation.key]
    entry["sidecar_sha256"] = label_resume._file_sha256(sidecar_path)

    validation = store.validate_episode(expectation)

    assert validation.status == "invalid"
    assert "degradation" in str(validation.reason)


def test_sidecar_outcome_must_match_progress_entry(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(0, plan_sha)
    store.commit_episode(expectation, _unary_rows(expectation), _pair_rows(expectation))
    store.payload["entries"][expectation.key]["outcome"] = "no_labels"

    validation = store.validate_episode(expectation)

    assert validation.status == "invalid"
    assert "outcomes differ" in str(validation.reason)


def test_rebuild_outputs_uses_numeric_index_order_and_reports_counts(tmp_path):
    store, plan_sha = _store(tmp_path)
    second = _expectation(1, plan_sha)
    first = _expectation(0, plan_sha)
    store.commit_episode(second, _unary_rows(second), _pair_rows(second))
    store.commit_episode(first, _unary_rows(first), _pair_rows(first))

    summary = store.rebuild_outputs(
        store.root / "teacher_labels.jsonl",
        store.root / "pair_labels.jsonl",
        expected_episode_count=2,
    )

    teacher_rows = [
        json.loads(line)
        for line in (store.root / "teacher_labels.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    pair_rows = [
        json.loads(line)
        for line in (store.root / "pair_labels.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert teacher_rows[0]["episode_id"] == "episode-0"
    assert teacher_rows[-1]["episode_id"] == "episode-1"
    assert pair_rows[0]["episode_id"] == "episode-0"
    assert summary["episode_count"] == 2
    assert summary["unary_rows"] == 8
    assert summary["pair_rows"] == 2
    assert summary["ranking_groups"] == 4
    assert summary["forecasters"] == ["mock"]
    assert store.payload["status"] == "rebuilt"


def test_rebuild_rejects_missing_or_extra_progress_indices(tmp_path):
    store, plan_sha = _store(tmp_path)
    expectation = _expectation(1, plan_sha)
    store.commit_episode(
        expectation, _unary_rows(expectation), _pair_rows(expectation)
    )

    with pytest.raises(LabelResumeError, match="deterministic episode index"):
        store.rebuild_outputs(
            store.root / "teacher_labels.jsonl",
            store.root / "pair_labels.jsonl",
            expected_episode_count=2,
        )


def test_dataset_plan_change_is_rejected(tmp_path):
    store, _ = _store(tmp_path)

    with pytest.raises(LabelResumeError, match="dataset plan changed"):
        store.register_dataset_plan(
            "dataset-a",
            {"selected_episode_count": 1},
            ("episode-0",),
        )
