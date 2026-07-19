from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tsfm_fais.cli as cli
import tsfm_fais.stage_execution as stage_execution
from tsfm_fais.config import AppConfig, load_config
from tsfm_fais.contracts import (
    CandidateResult,
    CandidateStatus,
    MissingBlock,
    SeriesBatch,
)
from tsfm_fais.data import Episode
from tsfm_fais.routing.graph import BlockEdge, BlockGraph
from tsfm_fais.routing.teacher import TeacherLabel
from tsfm_fais.stages import StageInputs, StagePreparation, prepare_stage


def _write_config(root: Path) -> tuple[Path, AppConfig]:
    config_dir = root / "configs"
    config_dir.mkdir()
    sources = {
        "datasets.yaml": Path("configs/data/datasets.yaml"),
        "imputers.yaml": Path("configs/imputers/pool.yaml"),
        "forecasters.yaml": Path("configs/forecasters/pool.yaml"),
        "router.yaml": Path("configs/router/block_fais.yaml"),
    }
    for name, source in sources.items():
        (config_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    config_path = config_dir / "config.yaml"
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
                "  context_length: 4",
                "  horizon: 2",
                "  missing_mechanisms: [independent_block]",
                "  missing_rates: [0.25]",
                "  seeds: [7]",
                "  candidate_ids: [locf, linear_interp]",
                "  max_teacher_blocks_per_episode: 2",
                "  max_teacher_candidates_per_episode: 2",
                "  max_pair_labels_per_episode: 1",
                "  forecast_num_samples: 1",
                "runtime:",
                "  output_root: ../artifacts",
                "  device: cpu",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, load_config(config_path)


def _stage_inputs(root: Path, forecaster_id: str = "chronos2") -> StageInputs:
    audit = root / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [{"dataset_id": "toy", "accepted": True}],
            }
        ),
        encoding="utf-8",
    )
    imputer_artifacts = root / "imputer-artifacts"
    imputer_artifacts.mkdir()
    (imputer_artifacts / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed"}),
        encoding="utf-8",
    )
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"local mock checkpoint")
    return StageInputs(
        audit_artifact=audit,
        imputer_artifacts=imputer_artifacts,
        forecaster_artifact=checkpoint,
        forecaster_id=forecaster_id,
    )


def _episode(index: int, *, with_blocks: bool = True) -> Episode:
    clean = np.asarray(
        (
            (1.0 + index, 11.0 + index),
            (2.0 + index, 12.0 + index),
            (3.0 + index, 13.0 + index),
            (4.0 + index, 14.0 + index),
        )
    )
    observed = np.ones_like(clean, dtype=bool)
    blocks: tuple[MissingBlock, ...] = ()
    if with_blocks:
        observed[1, :] = False
        blocks = (
            MissingBlock(f"b{index}-left", 0, 0, 1, 2, "independent_block"),
            MissingBlock(f"b{index}-right", 0, 1, 1, 2, "independent_block"),
        )
    context = SeriesBatch(
        clean[None, ...],
        observed[None, ...],
        item_ids=(f"item-{index}",),
    )
    return Episode(
        dataset_id="toy",
        item_id=f"item-{index}",
        forecast_origin=8 + index,
        context=context,
        clean_context=clean,
        clean_future=np.asarray(
            ((5.0 + index, 15.0 + index), (6.0 + index, 16.0 + index))
        ),
        blocks=blocks,
        seed=100 + index,
    )


def _completed_candidate(
    candidate_id: str,
    batch: SeriesBatch,
    fill: float,
) -> CandidateResult:
    values = np.where(batch.observed_mask, batch.values, fill)
    return CandidateResult(
        imputer_id=candidate_id,
        values=values,
        native_valid_mask=np.ones(batch.shape, dtype=bool),
        status=CandidateStatus.SUCCESS,
        runtime_seconds=0.01,
    )


class _FakeTeacher:
    def __init__(self, _forecast: Any, seasonality: int = 1) -> None:
        self.seasonality = seasonality

    def unary_labels_batched(
        self,
        episode_id: str,
        _clean_context: np.ndarray,
        _clean_future: np.ndarray,
        _anchor: np.ndarray,
        blocks: tuple[MissingBlock, ...],
        candidates: dict[str, CandidateResult],
        _spec: Any,
        *,
        candidate_filter: Any,
    ) -> tuple[list[TeacherLabel], float, float]:
        clean_loss = 1.0
        labels: list[TeacherLabel] = []
        for block_index, block in enumerate(blocks):
            for candidate_index, (candidate_id, result) in enumerate(candidates.items()):
                if not candidate_filter(block, candidate_id, result):
                    continue
                loss = clean_loss + 0.1 * (block_index + 1) + 0.01 * candidate_index
                labels.append(
                    TeacherLabel(
                        episode_id=episode_id,
                        block_id=block.block_id,
                        candidate_id=candidate_id,
                        forecast_loss=loss,
                        clean_loss=clean_loss,
                        degradation=loss - clean_loss,
                    )
                )
        return labels, clean_loss, 1.5

    def candidate_losses_batched(
        self,
        _clean_context: np.ndarray,
        _clean_future: np.ndarray,
        candidates: dict[str, CandidateResult],
        _spec: Any,
    ) -> dict[str, float]:
        return {
            candidate_id: 1.25 + 0.01 * index
            for index, candidate_id in enumerate(candidates)
        }

    def pair_interactions_batched(
        self,
        _future: np.ndarray,
        _anchor: np.ndarray,
        requests: tuple[Any, ...],
        _spec: Any,
        *,
        anchor_loss: float,
        scale_context: np.ndarray,
    ) -> list[float]:
        assert anchor_loss == 1.5
        assert scale_context.ndim == 3
        return [0.25] * len(requests)


class _FakePipeline:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def _pseudo_batch(
        self,
        batch: SeriesBatch,
        _seed: int,
        *,
        max_blocks: int,
        target_blocks: tuple[Any, ...] = (),
        priority_channels: tuple[int, ...] = (),
    ) -> SeriesBatch:
        assert max_blocks >= 1
        assert target_blocks
        assert priority_channels == (0, 1)
        return batch


class _Harness:
    def __init__(self, episodes: tuple[tuple[str, Episode], ...]) -> None:
        self.episodes = episodes
        self.candidate_calls = 0
        self.forecaster_loads = 0
        self.artifact_store_loads = 0
        self.artifact_manager_creations = 0
        self.fail_candidate_call: int | None = None

    def reset_invocation(self, *, fail_candidate_call: int | None = None) -> None:
        self.candidate_calls = 0
        self.forecaster_loads = 0
        self.artifact_store_loads = 0
        self.artifact_manager_creations = 0
        self.fail_candidate_call = fail_candidate_call

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = self
        dataset = SimpleNamespace(dataset_id="toy", family_id="toy-family", period=1)

        def datasets(_config: AppConfig, _audit_path: Path) -> Iterator[tuple[Any, tuple[Any, ...]]]:
            yield dataset, ()

        def episode_iter(
            _config: AppConfig,
            _dataset: Any,
            _items: Any,
            *,
            partition: str,
            selection_summary: dict[str, Any],
        ) -> Iterator[tuple[str, Episode]]:
            assert partition == "train"
            selection_summary.update(
                {
                    "partition": "train",
                    "cap_per_dataset": None,
                    "eligible_episode_count": len(harness.episodes),
                    "selected_episode_count": len(harness.episodes),
                    "truncated": False,
                }
            )
            yield from harness.episodes

        class ArtifactStore:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                harness.artifact_store_loads += 1

            def load_statistics(self) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(2), np.eye(2)

        class ArtifactManager:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                harness.artifact_manager_creations += 1
                self.load_call_count = 0

            def candidate_pool(self, _allowed_devices: tuple[str, ...]) -> tuple[str, ...]:
                return ("locf", "linear_interp")

            def close(self) -> None:
                pass

            def audit(self) -> dict[str, Any]:
                return {
                    "artifact_request_count": 2 * self.load_call_count,
                    "load_call_count": self.load_call_count,
                    "deserialization_attempt_count": self.load_call_count,
                    "load_success_count": self.load_call_count,
                    "load_failure_count": 0,
                    "cache_store_count": 0,
                    "cache_hit_count": 0,
                    "failure_cache_store_count": 0,
                    "cached_failure_hit_count": 0,
                    "evict_count": 0,
                    "deep_evict_count": 0,
                    "deep_cleanup_count": 0,
                    "dataset_cache_evict_count": 0,
                    "dataset_cache_cleanup_count": 0,
                    "load_seconds": 0.01 * self.load_call_count,
                    "load_mode_counts": {"mock": self.load_call_count},
                    "max_deep_load_batch": 0,
                    "by_candidate": {},
                }

        def preflight(*_args: Any, **_kwargs: Any) -> object:
            harness.forecaster_loads += 1
            return object()

        def candidate_pairs(
            manager: ArtifactManager,
            _runner: Any,
            candidate_ids: tuple[str, ...],
            real_batch: SeriesBatch,
            pseudo_batch: SeriesBatch,
            **_kwargs: Any,
        ) -> tuple[dict[str, CandidateResult], dict[str, CandidateResult]]:
            harness.candidate_calls += 1
            if harness.fail_candidate_call == harness.candidate_calls:
                raise RuntimeError("injected label interruption")
            manager.load_call_count += 1
            real = {
                candidate_id: _completed_candidate(
                    candidate_id, real_batch, float(index + 1)
                )
                for index, candidate_id in enumerate(candidate_ids)
            }
            pseudo = {
                candidate_id: _completed_candidate(
                    candidate_id, pseudo_batch, float(index + 1)
                )
                for index, candidate_id in enumerate(candidate_ids)
            }
            return real, pseudo

        def graph(blocks: tuple[MissingBlock, ...], _correlation: Any) -> BlockGraph:
            edges = (
                (BlockEdge(blocks[0].block_id, blocks[1].block_id, 0.75, "overlap"),)
                if len(blocks) == 2
                else ()
            )
            return BlockGraph(tuple(blocks), edges)

        def pair_requests(
            edges: tuple[BlockEdge, ...],
            _eligible: Any,
            candidate_ids: tuple[str, ...],
            _seed: int,
            *,
            limit: int | None,
        ) -> tuple[tuple[BlockEdge, str, str], ...]:
            if not edges or not candidate_ids or limit == 0:
                return ()
            return ((edges[0], candidate_ids[0], candidate_ids[-1]),)

        monkeypatch.setattr(stage_execution, "_datasets", datasets)
        monkeypatch.setattr(stage_execution, "_episode_iter", episode_iter)
        monkeypatch.setattr(stage_execution, "DatasetImputerArtifactStore", ArtifactStore)
        monkeypatch.setattr(stage_execution, "_LabelArtifactManager", ArtifactManager)
        monkeypatch.setattr(stage_execution, "_preflight_forecaster", preflight)
        monkeypatch.setattr(stage_execution, "_run_label_candidate_pairs", candidate_pairs)
        monkeypatch.setattr(stage_execution, "BlockwiseFAIS", _FakePipeline)
        monkeypatch.setattr(stage_execution, "TeacherBuilder", _FakeTeacher)
        monkeypatch.setattr(stage_execution, "build_block_graph", graph)
        monkeypatch.setattr(stage_execution, "_pair_label_requests", pair_requests)
        monkeypatch.setattr(
            stage_execution, "block_features", lambda *_args: {"block_length": 1.0}
        )
        monkeypatch.setattr(
            stage_execution,
            "candidate_features",
            lambda spec, _forecast: {"candidate_cost_tier": float(spec.cost_tier)},
        )
        monkeypatch.setattr(
            stage_execution,
            "proxy_features",
            lambda *_args, **_kwargs: {"proxy_mae": 0.0},
        )
        monkeypatch.setattr(
            stage_execution,
            "pair_features",
            lambda *_args, edge_weight, **_kwargs: {"edge_weight": float(edge_weight)},
        )


def _prepare(
    config_path: Path,
    config: AppConfig,
    inputs: StageInputs,
    *,
    run_id: str,
    resume: bool,
) -> StagePreparation:
    return prepare_stage(
        config,
        config_path,
        "labels",
        inputs,
        run_id=run_id,
        resume=resume,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_labels_resume_cli_and_preparation_require_one_forecaster(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, config = _write_config(tmp_path)
    single = _stage_inputs(tmp_path)
    initial = _prepare(
        config_path,
        config,
        single,
        run_id="single-label-resume",
        resume=False,
    )
    reopened = _prepare(
        config_path,
        config,
        single,
        run_id="single-label-resume",
        resume=True,
    )
    assert initial.resuming is False
    assert reopened.resuming is True

    multi = StageInputs(
        audit_artifact=single.audit_artifact,
        imputer_artifacts=single.imputer_artifacts,
        forecaster_artifact=single.forecaster_artifact,
        forecaster_id="chronos2,timesfm2p5",
    )
    _prepare(
        config_path,
        config,
        multi,
        run_id="multi-label-initial",
        resume=False,
    )
    with pytest.raises(ValueError, match="exactly one forecaster"):
        _prepare(
            config_path,
            config,
            multi,
            run_id="multi-label-initial",
            resume=True,
        )

    code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--stage",
            "labels",
            "--run-id",
            "multi-label-initial",
            "--audit-artifact",
            str(single.audit_artifact),
            "--imputer-artifacts",
            str(single.imputer_artifacts),
            "--forecaster-artifact",
            str(single.forecaster_artifact),
            "--forecaster-id",
            "chronos2,timesfm2p5",
            "--execute",
            "--resume",
        ]
    )
    assert code == 2
    assert "exactly one forecaster" in capsys.readouterr().err


def test_labels_resume_executes_only_pending_and_invalid_then_skips_all_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _write_config(tmp_path)
    inputs = _stage_inputs(tmp_path)
    harness = _Harness(
        (
            ("toy__item-0__8__independent_block__0.25__7", _episode(0)),
            ("toy__item-1__9__independent_block__0.25__7", _episode(1)),
            (
                "toy__item-2__10__independent_block__0.25__7",
                _episode(2, with_blocks=False),
            ),
        )
    )
    harness.install(monkeypatch)

    preparation = _prepare(
        config_path,
        config,
        inputs,
        run_id="resume-labels",
        resume=False,
    )
    harness.reset_invocation(fail_candidate_call=2)
    with pytest.raises(RuntimeError, match="injected label interruption"):
        stage_execution.execute_labels(preparation, config, inputs)

    progress_path = preparation.store.root / "labels_progress.json"
    interrupted = json.loads(progress_path.read_text(encoding="utf-8"))
    assert interrupted["completed_count"] == 1
    assert interrupted["resume_count"] == 0

    resumed_preparation = _prepare(
        config_path,
        config,
        inputs,
        run_id="resume-labels",
        resume=True,
    )
    harness.reset_invocation()
    resumed = stage_execution.execute_labels(resumed_preparation, config, inputs)

    assert harness.forecaster_loads == 1
    assert harness.artifact_store_loads == 1
    assert harness.artifact_manager_creations == 1
    assert harness.candidate_calls == 1
    assert resumed["expected_episode_count"] == 3
    assert resumed["episodes_executed_last_invocation"] == 2
    assert resumed["episodes_reused_last_invocation"] == 1
    assert resumed["resume_count"] == 1
    assert resumed["no_label_episode_count"] == 1
    assert resumed["unary_rows"] == 8
    assert resumed["pair_rows"] == 2
    assert resumed["ranking_groups"] == 4

    root = preparation.store.root
    teacher_rows = _jsonl(root / "teacher_labels.jsonl")
    pair_rows = _jsonl(root / "pair_labels.jsonl")
    assert len(teacher_rows) == resumed["unary_rows"]
    assert len(pair_rows) == resumed["pair_rows"]
    assert set(teacher_rows[0]) >= {
        "episode_id",
        "dataset_id",
        "family_id",
        "forecaster_id",
        "group_id",
        "block_id",
        "candidate_id",
        "prior_features",
        "unary_features",
        "forecast_loss",
        "clean_loss",
        "anchor_loss",
        "degradation",
    }
    assert set(pair_rows[0]) >= {
        "episode_id",
        "dataset_id",
        "family_id",
        "forecaster_id",
        "left_block",
        "right_block",
        "left_candidate",
        "right_candidate",
        "features",
        "interaction",
    }
    manifest = json.loads((root / "labels_manifest.json").read_text(encoding="utf-8"))
    assert manifest == resumed
    assert len(manifest["teacher_labels_sha256"]) == 64
    assert len(manifest["pair_labels_sha256"]) == 64
    assert manifest["origin_partition"] == "train"
    assert manifest["episode_count"] == 3
    assert manifest["unique_episode_count"] == 3
    assert manifest["episode_sampling"]["selected_episode_count"] == 3
    assert manifest["episode_sampling"]["episode_execution_count"] == 3
    assert manifest["artifact_loading"]["load_call_count"] == 2
    assert manifest["artifact_loading"]["datasets"]["toy"]["forecasters"][
        "chronos2"
    ]["load_call_count"] == 2
    completed_loading = manifest["artifact_loading"]

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    no_label_entries = [
        entry for entry in progress["entries"].values() if entry["outcome"] == "no_labels"
    ]
    assert len(no_label_entries) == 1
    no_label_sidecar = json.loads(
        (root / no_label_entries[0]["sidecar_file"]).read_text(encoding="utf-8")
    )
    assert no_label_sidecar["unary_rows"] == []
    assert no_label_sidecar["pair_rows"] == []

    damaged_entry = progress["entries"]["00000000"]
    damaged_path = root / damaged_entry["sidecar_file"]
    damaged_path.write_text(
        damaged_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    repair_preparation = _prepare(
        config_path,
        config,
        inputs,
        run_id="resume-labels",
        resume=True,
    )
    harness.reset_invocation()
    repaired = stage_execution.execute_labels(repair_preparation, config, inputs)
    assert harness.forecaster_loads == 1
    assert harness.artifact_store_loads == 1
    assert harness.artifact_manager_creations == 1
    assert harness.candidate_calls == 1
    assert repaired["episodes_executed_last_invocation"] == 1
    assert repaired["episodes_reused_last_invocation"] == 2
    assert repaired["artifact_loading"] == completed_loading

    complete_preparation = _prepare(
        config_path,
        config,
        inputs,
        run_id="resume-labels",
        resume=True,
    )
    harness.reset_invocation(fail_candidate_call=1)
    complete = stage_execution.execute_labels(complete_preparation, config, inputs)

    assert harness.forecaster_loads == 0
    assert harness.artifact_store_loads == 0
    assert harness.artifact_manager_creations == 0
    assert harness.candidate_calls == 0
    assert complete["episodes_executed_last_invocation"] == 0
    assert complete["episodes_reused_last_invocation"] == 3
    assert complete["artifact_loading"] == completed_loading
    assert _jsonl(root / "teacher_labels.jsonl") == teacher_rows
    assert _jsonl(root / "pair_labels.jsonl") == pair_rows
