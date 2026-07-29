from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tsfm_fais.stage_execution as stage_execution
from tsfm_fais.cli import build_parser
from tsfm_fais.config import AppConfig, load_config
from tsfm_fais.contracts import TimeSeriesItem
from tsfm_fais.imputers import CandidateRunner
from tsfm_fais.pipeline import BlockwiseFAIS as RealBlockwiseFAIS
from tsfm_fais.routing.models import PairwiseRiskModel, RouterBundle
from tsfm_fais.stage_execution import execute_prepared_stage
from tsfm_fais.stages import StageInputs, prepare_stage


def _write_config(tmp_path: Path) -> tuple[Path, AppConfig]:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    sources = {
        "datasets.yaml": Path("configs/data/datasets.yaml"),
        "imputers.yaml": Path("configs/imputers/pool.yaml"),
        "forecasters.yaml": Path("configs/forecasters/pool.yaml"),
        "router.yaml": Path("configs/router/block_fais.yaml"),
    }
    for name, source in sources.items():
        (config_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config_path = config_dir / "resume.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "seed: 17",
                "registries:",
                "  data_manifest: datasets.yaml",
                "  imputer_registry: imputers.yaml",
                "  forecaster_registry: forecasters.yaml",
                "  router_config: router.yaml",
                "experiment:",
                "  split: rolling_origin",
                "  context_length: 4",
                "  horizon: 2",
                "  target_indices: [0, 1]",
                "  missing_mechanisms: [independent_block]",
                "  missing_rates: [0.25]",
                "  seeds: [7, 8]",
                "  candidate_ids: [locf, linear_interp]",
                "  max_items_per_dataset: 1",
                "  max_eval_origins_per_item: 1",
                "  save_all_candidate_outputs: true",
                "runtime:",
                "  output_root: ../artifacts",
                "  device: cpu",
                "  fail_fast: true",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, load_config(config_path)


def _item() -> TimeSeriesItem:
    time = np.arange(24, dtype=float)
    return TimeSeriesItem(
        item_id="item-0",
        values=np.column_stack((time, np.sin(time), np.cos(time))),
        variate_names=("trend", "sin", "cos"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )


class _FakePipeline:
    calls = 0

    def __init__(self, *, imputer_registry, **kwargs):
        self.imputer_registry = imputer_registry
        self.candidate_runner = CandidateRunner(imputer_registry)
        self.shortlist_size = 2
        self.fallback_internal = ("linear_interp", "locf", "train_median")
        self.fallback_tail = ("locf", "train_median")
        router = kwargs.get("router")
        self.selector_method = (
            "b_fais" if router is None else str(router.metadata.get("selector_method", "b_fais"))
        )
        self._pipeline = RealBlockwiseFAIS(
            imputer_registry=imputer_registry,
            imputer_artifacts={},
            training_medians=kwargs.get("training_medians"),
            training_correlation=kwargs.get("training_correlation"),
        )
        self._pipeline.shortlist_size = self.shortlist_size

    def prepare_route(self, *args, **kwargs):
        return self._pipeline.prepare_route(*args, **kwargs)

    def finish_route(self, *args, **kwargs):
        type(self).calls += 1
        result = self._pipeline.finish_route(*args, **kwargs)
        result.routing.metadata["selector_method"] = self.selector_method
        return result


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


class _AlternatingNeuralSelector:
    def __init__(self) -> None:
        self.candidate_ids = ("kalman_local_trend", "kalman_ar")
        self.round_count = 0

    def select(self, features, valid_candidate_ids):
        np.testing.assert_allclose(np.asarray(features, dtype=float).shape, (1,))
        valid = tuple(valid_candidate_ids)
        selected = self.candidate_ids[self.round_count % len(self.candidate_ids)]
        return selected if selected in valid else valid[0]

    def score(self, features):
        np.testing.assert_allclose(np.asarray(features, dtype=float).shape, (1,))
        return np.asarray((1.0, 0.0), dtype=float)

    def observe(self, features, candidate_id, reward):
        np.testing.assert_allclose(np.asarray(features, dtype=float).shape, (1,))
        assert candidate_id == self.candidate_ids[self.round_count % len(self.candidate_ids)]
        assert 0.0 <= float(reward) <= 1.0
        self.round_count += 1


def _neural_router() -> RouterBundle:
    selector = _AlternatingNeuralSelector()
    return RouterBundle(
        prior=selector,
        unary=selector,
        pairwise=PairwiseRiskModel(),
        feature_names=("missing_fraction",),
        candidate_ids=selector.candidate_ids,
        metadata={
            "split": "rolling_origin",
            "selector_method": "neuralucb",
            "requires_pseudo_candidates": False,
            "routing_target_protocol": "sequence_imputation_quality_v1",
            "selector_training_target": "imputation_loss",
            "forecaster_independent_selection": True,
            "uses_missing_block_graph": False,
            "beta": 0.0,
            "cost_weight": 0.0,
            "forecast_consensus": {"mode": "disabled", "candidates": []},
        },
    )


def _setup(tmp_path: Path, monkeypatch):
    config_path, config = _write_config(tmp_path)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "dataset_id": "synthetic",
                        "accepted": True,
                        "content_sha256": "stable",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    imputer_artifacts = tmp_path / "imputer-artifacts"
    imputer_artifacts.mkdir()
    dataset_artifacts = imputer_artifacts / "synthetic"
    dataset_artifacts.mkdir()
    with (dataset_artifacts / "training_statistics.npz").open("wb") as handle:
        np.savez_compressed(handle, medians=np.zeros(3), correlation=np.eye(3))
    (imputer_artifacts / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "datasets": {
                    "synthetic": {
                        "statistics": "training_statistics.npz",
                        "candidates": {
                            "locf": {"status": "stateless"},
                            "linear_interp": {"status": "stateless"},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    router = tmp_path / "router"
    router.mkdir()
    (router / "router_bundle.joblib").write_bytes(b"stable-router")
    (router / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "split": "rolling_origin"}),
        encoding="utf-8",
    )
    inputs = StageInputs(
        audit_artifact=audit,
        imputer_artifacts=imputer_artifacts,
        router_artifact=router,
        forecaster_id="chronos2",
    )
    dataset = SimpleNamespace(
        dataset_id="synthetic",
        family_id="synthetic-family",
        period=4,
    )
    item = _item()
    monkeypatch.setattr(
        stage_execution,
        "_datasets",
        lambda _config, _audit: iter(((dataset, (item,)),)),
    )
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: SimpleNamespace(metadata={"split": "rolling_origin"}),
    )
    monkeypatch.setattr(stage_execution, "BlockwiseFAIS", _FakePipeline)
    monkeypatch.setattr(stage_execution, "_resident_memory_bytes", lambda: 1024)
    monkeypatch.setattr(stage_execution, "perf_counter", _Clock())
    _FakePipeline.calls = 0
    return config_path, config, inputs


def _setup_neural(tmp_path: Path, monkeypatch):
    config_path, _, inputs = _setup(tmp_path, monkeypatch)
    (config_path.parent / "router.yaml").write_text(
        Path("configs/router/baseline_selector_suite.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("context_length: 4", "context_length: 12")
    text = text.replace(
        "missing_mechanisms: [independent_block]", "missing_mechanisms: [random_point]"
    )
    text = text.replace("seeds: [7, 8]", "seeds: [7, 8, 9]")
    text = text.replace(
        "candidate_ids: [locf, linear_interp]",
        "candidate_ids: [locf, linear_interp, kalman_local_trend, kalman_ar]",
    )
    config_path.write_text(text, encoding="utf-8")
    return config_path, load_config(config_path), inputs


def _run(
    config_path: Path,
    config: AppConfig,
    inputs: StageInputs,
    *,
    resume: bool,
):
    preparation = prepare_stage(
        config,
        config_path,
        "impute",
        inputs,
        run_id="resume-impute",
        resume=resume,
    )
    return execute_prepared_stage(preparation, config, inputs), preparation.store.root


def _archive_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def test_run_cli_accepts_impute_resume_flags():
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--stage",
            "impute",
            "--run-id",
            "resume-impute",
            "--execute",
            "--resume",
        ]
    )

    assert args.stage == "impute"
    assert args.execute is True
    assert args.resume is True


def test_impute_resume_strictly_skips_valid_committed_episodes(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    first, root = _run(config_path, config, inputs, resume=False)
    assert first["episode_count"] == 2
    assert first["episodes_executed"] == 2
    assert _FakePipeline.calls == 2
    files = sorted((root / "imputations" / "synthetic").glob("*.npz"))
    before = [_archive_payload(path) for path in files]
    assignments_before = (root / "routing_assignments.jsonl").read_text(encoding="utf-8")

    _FakePipeline.calls = 0
    monkeypatch.setattr(
        stage_execution,
        "DatasetImputerArtifactStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("valid resumed episodes must not load imputer artifacts")
        ),
    )
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episode_count"] == 2
    assert resumed["episodes_executed"] == 0
    assert resumed["episodes_reused"] == 2
    assert _FakePipeline.calls == 0
    for expected, path in zip(before, files, strict=True):
        actual = _archive_payload(path)
        assert actual.keys() == expected.keys()
        assert all(np.array_equal(actual[key], expected[key]) for key in actual)
    assert (root / "routing_assignments.jsonl").read_text(encoding="utf-8") == assignments_before
    progress = json.loads((root / "imputation_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["completed_count"] == progress["expected_episode_count"] == 2


def test_impute_resume_recomputes_pair_left_before_progress_commit(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    original_write_json = stage_execution._write_json
    failed = False

    def fail_first_progress_commit(path, payload):
        nonlocal failed
        if (
            not failed
            and path.name == "imputation_progress.json"
            and payload.get("completed_count") == 1
        ):
            failed = True
            raise OSError("injected progress interruption")
        return original_write_json(path, payload)

    monkeypatch.setattr(stage_execution, "_write_json", fail_first_progress_commit)
    preparation = prepare_stage(
        config,
        config_path,
        "impute",
        inputs,
        run_id="resume-impute",
    )
    with pytest.raises(OSError, match="injected progress interruption"):
        execute_prepared_stage(preparation, config, inputs)
    root = preparation.store.root
    assert len(list((root / "imputations").rglob("*.npz"))) == 1
    progress = json.loads((root / "imputation_progress.json").read_text(encoding="utf-8"))
    assert progress["completed_count"] == 0

    monkeypatch.setattr(stage_execution, "_write_json", original_write_json)
    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 2
    assert resumed["episodes_reused"] == 0
    assert _FakePipeline.calls == 2


@pytest.mark.parametrize("orphan_kind", ["npz", "assignment"])
def test_impute_resume_does_not_accept_uncommitted_half_output(tmp_path, monkeypatch, orphan_kind):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _, root = _run(config_path, config, inputs, resume=False)
    progress_path = root / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    removed = progress["entries"].pop("00000001")
    progress["completed_count"] = 1
    progress["status"] = "running"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    target = (
        root / "imputations" / removed["file"]
        if orphan_kind == "npz"
        else root / removed["assignment_file"]
    )
    target.write_bytes(b"truncated")

    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 1
    assert resumed["episodes_reused"] == 1
    assert _FakePipeline.calls == 1
    assert len((root / "routing_assignments.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with np.load(root / "imputations" / removed["file"], allow_pickle=False) as archive:
        assert int(np.asarray(archive["schema_version"]).reshape(-1)[0]) == 3
    json.loads((root / removed["assignment_file"]).read_text(encoding="utf-8"))


def test_impute_resume_repairs_corrupted_committed_file(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _, root = _run(config_path, config, inputs, resume=False)
    progress = json.loads((root / "imputation_progress.json").read_text(encoding="utf-8"))
    damaged = progress["entries"]["00000001"]
    (root / "imputations" / damaged["file"]).write_bytes(b"half-npz")

    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 1
    assert resumed["episodes_reused"] == 1
    assert resumed["repaired_episode_count"] == 1
    assert _FakePipeline.calls == 1
    repaired_progress = json.loads((root / "imputation_progress.json").read_text(encoding="utf-8"))
    assert "hash mismatch" in repaired_progress["entries"]["00000001"]["repaired_reason"]


def test_impute_resume_revalidates_finiteness_after_hash_match(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _, root = _run(config_path, config, inputs, resume=False)
    progress_path = root / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["entries"]["00000000"]
    target = root / "imputations" / entry["file"]
    payload = _archive_payload(target)
    payload["candidate_values"][0, 0, 0] = np.nan
    with target.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    entry["npz_sha256"] = stage_execution._file_sha256(target)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 1
    assert resumed["episodes_reused"] == 1
    assert _FakePipeline.calls == 1
    repaired_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "non-finite" in repaired_progress["entries"]["00000000"]["repaired_reason"]


def test_impute_resume_revalidates_candidate_ids_after_hash_match(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _, root = _run(config_path, config, inputs, resume=False)
    progress_path = root / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["entries"]["00000000"]
    assignment_path = root / entry["assignment_file"]
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment["candidate_ids"] = ["locf"]
    assignment_path.write_text(json.dumps(assignment), encoding="utf-8")
    entry["candidate_ids"] = ["locf"]
    entry["assignment_sha256"] = stage_execution._file_sha256(assignment_path)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 1
    assert resumed["episodes_reused"] == 1
    assert _FakePipeline.calls == 1
    repaired_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "candidate IDs differ" in repaired_progress["entries"]["00000000"]["repaired_reason"]


def test_impute_resume_repairs_assembled_method_mismatch(tmp_path, monkeypatch):
    from tsfm_fais.routing import sequence_pipeline

    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    (config_path.parent / "router.yaml").write_text(
        Path("configs/router/baseline_selector_suite.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr(sequence_pipeline, "WholeSeriesSelectorFAIS", _FakePipeline)
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: SimpleNamespace(
            metadata={"split": "rolling_origin", "selector_method": "metaod"}
        ),
    )
    _, root = _run(config_path, config, inputs, resume=False)
    progress_path = root / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = progress["entries"]["00000000"]
    assignment_path = root / entry["assignment_file"]
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment["assembled_method_id"] = "b_fais"
    assignment_path.write_text(json.dumps(assignment), encoding="utf-8")
    entry["assembled_method_id"] = "b_fais"
    entry["assignment_sha256"] = stage_execution._file_sha256(assignment_path)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    _FakePipeline.calls = 0
    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_executed"] == 1
    assert resumed["episodes_reused"] == 1
    assert _FakePipeline.calls == 1
    repaired_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert (
        "assembled method ID differs" in repaired_progress["entries"]["00000000"]["repaired_reason"]
    )


def test_impute_execution_rejects_router_method_not_enabled_by_config(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: SimpleNamespace(
            metadata={"split": "rolling_origin", "selector_method": "metaod"}
        ),
    )

    with pytest.raises(ValueError, match="not enabled by the runtime router config"):
        _run(config_path, config, inputs, resume=False)


def test_neuralucb_resume_replays_prefix_before_recomputing_suffix(tmp_path, monkeypatch):
    from tsfm_fais.routing import sequence_pipeline

    reference_path, reference_config, reference_inputs = _setup_neural(
        tmp_path / "reference", monkeypatch
    )
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: _neural_router(),
    )
    reference, reference_root = _run(
        reference_path,
        reference_config,
        reference_inputs,
        resume=False,
    )
    assert reference["episodes_executed"] == 3

    resume_path, resume_config, resume_inputs = _setup_neural(tmp_path / "resume", monkeypatch)
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: _neural_router(),
    )
    real_pipeline = sequence_pipeline.WholeSeriesSelectorFAIS

    class _InterruptSecondEpisode(real_pipeline):
        calls = 0

        def finish_route(self, *args, **kwargs):
            type(self).calls += 1
            if type(self).calls == 2:
                raise OSError("injected NeuralUCB interruption")
            return super().finish_route(*args, **kwargs)

    monkeypatch.setattr(
        sequence_pipeline,
        "WholeSeriesSelectorFAIS",
        _InterruptSecondEpisode,
    )
    preparation = prepare_stage(
        resume_config,
        resume_path,
        "impute",
        resume_inputs,
        run_id="resume-impute",
    )
    with pytest.raises(OSError, match="injected NeuralUCB interruption"):
        execute_prepared_stage(preparation, resume_config, resume_inputs)
    interrupted_progress = json.loads(
        (preparation.store.root / "imputation_progress.json").read_text(encoding="utf-8")
    )
    assert interrupted_progress["completed_count"] == 1

    monkeypatch.setattr(
        sequence_pipeline,
        "WholeSeriesSelectorFAIS",
        real_pipeline,
    )
    resumed, resumed_root = _run(
        resume_path,
        resume_config,
        resume_inputs,
        resume=True,
    )
    assert resumed["episodes_reused"] == 1
    assert resumed["episodes_executed"] == 2

    def selected_actions(root: Path):
        records = [
            json.loads(line)
            for line in (root / "routing_assignments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        return [
            (
                record["assignments"],
                record["routing_metadata"]["selected_candidate"],
            )
            for record in records
        ]

    assert selected_actions(resumed_root) == selected_actions(reference_root)
    assert [selected for _, selected in selected_actions(resumed_root)] == [
        "kalman_local_trend",
        "kalman_ar",
        "kalman_local_trend",
    ]


def test_neuralucb_resume_recomputes_valid_records_after_a_broken_prefix(tmp_path, monkeypatch):
    config_path, config, inputs = _setup_neural(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: _neural_router(),
    )
    _, root = _run(config_path, config, inputs, resume=False)
    original_records = [
        json.loads(line)
        for line in (root / "routing_assignments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    progress_path = root / "imputation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    damaged = progress["entries"]["00000001"]
    (root / "imputations" / damaged["file"]).write_bytes(b"broken")

    resumed, _ = _run(config_path, config, inputs, resume=True)

    assert resumed["episodes_reused"] == 1
    assert resumed["episodes_executed"] == 2
    repaired_records = [
        json.loads(line)
        for line in (root / "routing_assignments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["assignments"] for record in repaired_records] == [
        record["assignments"] for record in original_records
    ]
    assert [record["routing_metadata"]["selected_candidate"] for record in repaired_records] == [
        record["routing_metadata"]["selected_candidate"] for record in original_records
    ]
    repaired_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert repaired_progress["entries"]["00000002"]["repaired_reason"] == (
        "NeuralUCB resume recomputes the non-prefix suffix"
    )


def test_sequence_impute_without_forecaster_writes_independent_identity(tmp_path, monkeypatch):
    config_path, config, inputs = _setup_neural(tmp_path, monkeypatch)
    router = _neural_router()
    router_manifest = inputs.router_artifact / "manifest.json"
    router_manifest.write_text(
        json.dumps({"schema_version": 1, "metadata": router.metadata}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stage_execution.RouterBundle,
        "load",
        lambda _path: _neural_router(),
    )

    independent_inputs = replace(inputs, forecaster_id=None)
    summary, root = _run(
        config_path,
        config,
        independent_inputs,
        resume=False,
    )

    assert summary["forecaster_id"] == "imputation"
    assert summary["forecast_mode"] == "selector_independent"
    assert summary["forecaster_independent_selection"] is True
    assignment = json.loads(
        (root / "routing_assignments.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert assignment["forecaster_id"] == "imputation"
    assert assignment["forecast_mode"] == "selector_independent"
    assert assignment["forecaster_independent_selection"] is True
    with np.load(root / "imputations" / assignment["file"], allow_pickle=False) as archive:
        assert str(archive["forecaster_id"][0]) == "imputation"
        assert str(archive["forecast_mode"][0]) == "selector_independent"
        assert bool(archive["forecaster_independent_selection"][0]) is True

    resumed, _ = _run(
        config_path,
        config,
        independent_inputs,
        resume=True,
    )
    assert resumed["episodes_reused"] == summary["episode_count"]
    assert resumed["episodes_executed"] == 0


@pytest.mark.parametrize("changed_upstream", ["audit", "imputer", "router", "forecaster_registry"])
def test_impute_resume_rejects_changed_upstream_content(tmp_path, monkeypatch, changed_upstream):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _run(config_path, config, inputs, resume=False)
    if changed_upstream == "audit":
        assert inputs.audit_artifact is not None
        payload = json.loads(inputs.audit_artifact.read_text(encoding="utf-8"))
        payload["revision"] = 2
        inputs.audit_artifact.write_text(json.dumps(payload), encoding="utf-8")
    elif changed_upstream == "imputer":
        assert inputs.imputer_artifacts is not None
        (inputs.imputer_artifacts / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "status": "completed", "revision": 2}),
            encoding="utf-8",
        )
    elif changed_upstream == "router":
        assert inputs.router_artifact is not None
        (inputs.router_artifact / "router_bundle.joblib").write_bytes(b"changed-router")
    else:
        registry = config.registries.forecaster_registry
        registry.write_text(
            registry.read_text(encoding="utf-8") + "\n# signature change\n",
            encoding="utf-8",
        )

    preparation = prepare_stage(
        config,
        config_path,
        "impute",
        inputs,
        run_id="resume-impute",
        resume=True,
    )
    with pytest.raises(ValueError, match="signature changed"):
        execute_prepared_stage(preparation, config, inputs)


def test_impute_resume_rejects_changed_resolved_config(tmp_path, monkeypatch):
    config_path, config, inputs = _setup(tmp_path, monkeypatch)
    _run(config_path, config, inputs, resume=False)
    changed = config.model_copy(update={"seed": config.seed + 1})

    with pytest.raises(ValueError, match="resolved config differs"):
        prepare_stage(
            changed,
            config_path,
            "impute",
            inputs,
            run_id="resume-impute",
            resume=True,
        )
