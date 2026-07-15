from __future__ import annotations

import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tsfm_fais.stage_execution as stage_execution
from tsfm_fais.config import load_config
from tsfm_fais.contracts import BudgetSpec, SeriesBatch, TimeSeriesItem
from tsfm_fais.data import load_manifest
from tsfm_fais.imputers import (
    DEFAULT_REGISTRY,
    ArtifactLoadResult,
    CandidateRunner,
    ImputerRegistry,
    failed_candidate_result,
)
from tsfm_fais.stage_execution import (
    _allowed_devices,
    _artifact_loading_manifest,
    _episode_iter,
    _execution_metadata,
    _fit_candidate_params,
    _fit_region_end,
    _forecast_spec,
    _forecaster_artifacts,
    _LabelArtifactManager,
    _pair_label_requests,
    _preflight_forecaster,
    _pypots_params,
    _run_label_candidate_pairs,
    _selected_candidate_ids,
    _supplement_candidate_outputs,
    _torch_device,
    _training_batch,
)
from tsfm_fais.stages import StageInputs


def _config(tmp_path: Path, *experiment_lines: str):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    files = {
        "datasets.yaml": Path("configs/data/datasets.yaml"),
        "imputers.yaml": Path("configs/imputers/pool.yaml"),
        "forecasters.yaml": Path("configs/forecasters/pool.yaml"),
        "router.yaml": Path("configs/router/block_fais.yaml"),
    }
    for name, source in files.items():
        (config_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    path = config_dir / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
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
                *experiment_lines,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(path)


def _item(length: int = 24) -> TimeSeriesItem:
    time = np.arange(length, dtype=float)
    values = np.column_stack((time, np.sin(time), np.cos(time)))
    return TimeSeriesItem(
        item_id="item-0",
        values=values,
        variate_names=("trend", "sin", "cos"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )


def _item_with_id(item_id: str, length: int = 24) -> TimeSeriesItem:
    item = _item(length)
    return TimeSeriesItem(
        item_id=item_id,
        values=item.values,
        variate_names=item.variate_names,
        start=item.start,
        freq=item.freq,
        timestamps=item.timestamps,
        metadata=item.metadata,
    )


def _origins(
    config,
    partition: str,
    item: TimeSeriesItem | None = None,
) -> tuple[int, ...]:
    episodes = _episode_iter(
        config,
        SimpleNamespace(dataset_id="synthetic"),
        [item or _item()],
        partition=partition,
    )
    return tuple(episode.forecast_origin for _, episode in episodes)


def _updated_config(config, **experiment_updates):
    experiment = config.experiment.model_copy(update=experiment_updates)
    return config.model_copy(update={"experiment": experiment})


def test_episode_partitions_are_chronological_disjoint_and_complete(tmp_path):
    config = _config(tmp_path)

    training = _origins(config, "train")
    evaluation = _origins(config, "eval")
    combined = _origins(config, "all")

    assert training
    assert evaluation
    assert set(training).isdisjoint(evaluation)
    assert training + evaluation == combined
    assert max(training) < min(evaluation)


def test_training_batch_uses_only_rolling_windows_before_episode_origins(tmp_path):
    config = _config(tmp_path)
    item = _item(80)
    batch = _training_batch(
        [item],
        config.experiment.context_length,
        config.experiment.horizon,
    )
    fit_end = _fit_region_end(
        len(item.values),
        config.experiment.context_length,
        config.experiment.horizon,
    )
    first_origin = _origins(config, "all", item)[0]

    assert batch.shape[0] > 1
    assert all(
        int(item_id.rsplit("@", 1)[1]) + batch.shape[1] <= fit_end for item_id in batch.item_ids
    )
    assert fit_end + config.experiment.context_length == first_origin


def test_training_window_cap_covers_full_fit_range(tmp_path):
    config = _config(tmp_path)
    item = _item(80)

    uncapped = _training_batch(
        [item],
        config.experiment.context_length,
        config.experiment.horizon,
    )
    capped = _training_batch(
        [item],
        config.experiment.context_length,
        config.experiment.horizon,
        max_windows=3,
    )

    assert capped.shape[0] == 3
    assert capped.item_ids[0] == uncapped.item_ids[0]
    assert capped.item_ids[-1] == uncapped.item_ids[-1]


def test_origin_caps_cover_each_chronological_partition(tmp_path):
    config = _config(tmp_path)
    item = _item(80)
    capped_config = _updated_config(
        config,
        max_train_origins_per_item=2,
        max_eval_origins_per_item=2,
    )

    full_train = _origins(config, "train", item)
    full_eval = _origins(config, "eval", item)
    capped_train = _origins(capped_config, "train", item)
    capped_eval = _origins(capped_config, "eval", item)

    assert capped_train == (full_train[0], full_train[-1])
    assert capped_eval == (full_eval[0], full_eval[-1])


def test_dataset_episode_cap_is_deterministic_balanced_and_applied_before_build(
    tmp_path, monkeypatch
):
    config = _updated_config(
        _config(tmp_path),
        missing_mechanisms=(
            "random_point",
            "independent_block",
            "synchronous_block",
        ),
        missing_rates=(0.1, 0.2),
        seeds=(11, 22, 33),
        max_eval_origins_per_item=2,
        max_eval_episodes_per_dataset=12,
    )
    dataset = SimpleNamespace(dataset_id="synthetic")
    items = [_item_with_id("item-a", 80), _item_with_id("item-b", 80)]
    calls = []
    original_build_episode = stage_execution.build_episode

    def recording_build_episode(*args, **kwargs):
        calls.append((args, kwargs))
        return original_build_episode(*args, **kwargs)

    monkeypatch.setattr(stage_execution, "build_episode", recording_build_episode)
    first_summary = {}
    first = list(
        _episode_iter(
            config,
            dataset,
            items,
            partition="eval",
            selection_summary=first_summary,
        )
    )
    second_summary = {}
    second = list(
        _episode_iter(
            config,
            dataset,
            items,
            partition="eval",
            selection_summary=second_summary,
        )
    )

    assert len(first) == len(second) == 12
    assert len(calls) == 24
    assert [episode_id for episode_id, _ in first] == [
        episode_id for episode_id, _ in second
    ]
    assert first_summary == second_summary
    assert first_summary["cap_per_dataset"] == 12
    assert first_summary["eligible_episode_count"] == 72
    assert first_summary["selected_episode_count"] == 12
    strata = first_summary["coverage"]["mechanism_rate"]
    assert strata["eligible_level_count"] == 6
    assert strata["selected_level_count"] == 6
    assert set(strata["selected_counts"].values()) == {2}
    for field in ("item", "seed"):
        coverage = first_summary["coverage"][field]
        assert coverage["selected_count_max"] - coverage["selected_count_min"] <= 1


def test_episode_cap_preserves_episode_ids_seeds_and_future_isolation(tmp_path):
    base = _updated_config(
        _config(tmp_path),
        missing_mechanisms=("independent_block", "tail_mixed"),
        missing_rates=(0.1, 0.2),
        seeds=(11, 22),
        max_eval_origins_per_item=2,
    )
    capped = _updated_config(base, max_eval_episodes_per_dataset=5)
    dataset = SimpleNamespace(dataset_id="synthetic")
    item = _item_with_id("item-a", 80)

    uncapped = dict(_episode_iter(base, dataset, [item], partition="eval"))
    selected = dict(_episode_iter(capped, dataset, [item], partition="eval"))

    assert len(selected) == 5
    assert set(selected).issubset(uncapped)
    assert all(selected[key].seed == uncapped[key].seed for key in selected)

    changed_values = item.values.copy()
    changed_values[-8:] += 100_000
    changed_item = TimeSeriesItem(
        item_id=item.item_id,
        values=changed_values,
        variate_names=item.variate_names,
        start=item.start,
        freq=item.freq,
        timestamps=item.timestamps,
        metadata=item.metadata,
    )
    changed = dict(_episode_iter(capped, dataset, [changed_item], partition="eval"))
    assert tuple(changed) == tuple(selected)
    assert all(changed[key].seed == selected[key].seed for key in selected)
    assert all(
        np.array_equal(
            changed[key].context.observed_mask,
            selected[key].context.observed_mask,
        )
        for key in selected
    )


def test_episode_cap_smaller_than_grid_balances_mechanism_and_rate_marginals(tmp_path):
    config = _updated_config(
        _config(tmp_path),
        missing_mechanisms=(
            "random_point",
            "independent_block",
            "synchronous_block",
        ),
        missing_rates=(0.1, 0.2),
        seeds=(11, 22),
        max_eval_origins_per_item=2,
        max_eval_episodes_per_dataset=4,
    )
    summary = {}
    episodes = list(
        _episode_iter(
            config,
            SimpleNamespace(dataset_id="synthetic"),
            [_item_with_id("item-a", 80), _item_with_id("item-b", 80)],
            partition="eval",
            selection_summary=summary,
        )
    )

    assert len(episodes) == 4
    assert summary["coverage"]["mechanism"]["selected_count_max"] <= 2
    assert summary["coverage"]["rate"]["selected_counts"] == {"0.1": 2, "0.2": 2}


def test_main_configs_apply_dataset_episode_caps_without_changing_pilot_or_smoke():
    main = load_config(Path("configs/main.yaml"))
    main_eval = load_config(Path("configs/main_eval.yaml"))
    pilot = load_config(Path("configs/pilot.yaml"))
    smoke = load_config(Path("configs/smoke.yaml"))
    manifest = load_manifest(main.registries.data_manifest)

    assert main.experiment.max_train_episodes_per_dataset == 12
    assert main.experiment.max_eval_episodes_per_dataset is None
    assert main_eval.experiment.max_train_episodes_per_dataset is None
    assert main_eval.experiment.max_eval_episodes_per_dataset == 24
    assert pilot.experiment.max_train_episodes_per_dataset is None
    assert pilot.experiment.max_eval_episodes_per_dataset is None
    assert smoke.experiment.max_train_episodes_per_dataset is None
    assert smoke.experiment.max_eval_episodes_per_dataset is None
    assert main.experiment.csdi_num_samples == 5
    assert main_eval.experiment.csdi_num_samples == 5
    assert pilot.experiment.csdi_num_samples == 3
    assert smoke.experiment.csdi_num_samples == 20
    assert len([dataset for dataset in manifest.datasets if dataset.enabled]) == 32


def test_dataset_episode_caps_reject_zero(tmp_path):
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        _config(tmp_path / "train", "  max_train_episodes_per_dataset: 0")
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        _config(tmp_path / "eval", "  max_eval_episodes_per_dataset: 0")
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        _config(tmp_path / "csdi", "  csdi_num_samples: 0")


class _FakeArtifactStore:
    def __init__(self, *, failures=()):
        self.calls = []
        self.failures = set(failures)

    def status(self, candidate_id):
        return "fitted"

    def load_artifacts(self, candidate_ids, *, adapter_params=None):
        requested = tuple(candidate_ids)
        self.calls.append((requested, dict(adapter_params or {})))
        failures = {
            candidate_id: "synthetic load failure"
            for candidate_id in requested
            if candidate_id in self.failures
        }
        artifacts = {
            candidate_id: object()
            for candidate_id in requested
            if candidate_id not in failures
        }
        return ArtifactLoadResult(
            artifacts=artifacts,
            failures=failures,
            requested_ids=requested,
            attempted_ids=requested,
            loaded_ids=tuple(artifacts),
            load_seconds=0.25,
        )


def test_label_artifact_manager_caches_structured_candidates_once(tmp_path):
    config = _config(tmp_path)
    store = _FakeArtifactStore()
    manager = _LabelArtifactManager(store, DEFAULT_REGISTRY, config)

    pool = manager.candidate_pool(("cpu", "gpu"))
    assert "missforest" in pool
    assert "knn_multivariate" in pool
    assert store.calls == []
    first, first_failures, first_ephemeral = manager.acquire(
        ("missforest", "knn_multivariate")
    )
    manager.release(first, first_ephemeral)
    second, second_failures, second_ephemeral = manager.acquire(
        ("missforest", "knn_multivariate")
    )
    manager.release(second, second_ephemeral)
    manager.close()
    audit = manager.audit()

    assert [call[0] for call in store.calls] == [
        ("missforest", "knn_multivariate"),
    ]
    assert not first_failures
    assert not second_failures
    assert audit["by_candidate"]["missforest"]["deserialization_attempt_count"] == 1
    assert audit["by_candidate"]["missforest"]["cache_hit_count"] == 1
    assert audit["by_candidate"]["missforest"]["dataset_cache_evict_count"] == 1
    assert audit["by_candidate"]["knn_multivariate"]["deserialization_attempt_count"] == 1


def test_label_artifact_manager_caches_structured_load_failure(tmp_path):
    manager = _LabelArtifactManager(
        _FakeArtifactStore(failures=("missforest",)),
        DEFAULT_REGISTRY,
        _config(tmp_path),
    )

    _, first, _ = manager.acquire(("missforest",))
    _, second, _ = manager.acquire(("missforest",))
    audit = manager.audit()

    assert first == second == {"missforest": "synthetic load failure"}
    assert audit["by_candidate"]["missforest"]["deserialization_attempt_count"] == 1
    assert audit["by_candidate"]["missforest"]["cached_failure_hit_count"] == 1


def test_label_artifact_manager_close_releases_memmap_and_records_gc(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    manager = _LabelArtifactManager(_FakeArtifactStore(), DEFAULT_REGISTRY, config)
    path = tmp_path / "mapped.bin"
    path.write_bytes(np.arange(16, dtype=np.float64).tobytes())
    mapped = np.memmap(path, dtype=np.float64, mode="r", shape=(16,))
    reference = weakref.ref(mapped)
    manager._cache["missforest"] = mapped
    del mapped
    native_collect = stage_execution.gc.collect
    calls = []

    def recording_collect():
        calls.append(True)
        return native_collect()

    monkeypatch.setattr(stage_execution.gc, "collect", recording_collect)
    manager.close()
    audit = manager.audit()

    assert manager._cache == {}
    assert reference() is None
    assert calls == [True]
    assert audit["dataset_cache_evict_count"] == 1
    assert audit["dataset_cache_cleanup_count"] == 1
    reopened = np.memmap(path, dtype=np.float64, mode="r", shape=(16,))
    np.testing.assert_array_equal(reopened, np.arange(16, dtype=np.float64))


def test_label_deep_candidates_load_and_release_one_at_a_time(tmp_path, monkeypatch):
    config = _config(tmp_path)
    store = _FakeArtifactStore()
    manager = _LabelArtifactManager(store, DEFAULT_REGISTRY, config)
    cache_clears = []
    monkeypatch.setattr(
        "tsfm_fais.stage_execution._empty_cuda_cache",
        lambda device: cache_clears.append(device),
    )
    batch = SeriesBatch(
        np.arange(24, dtype=float).reshape(1, 8, 3),
        np.ones((1, 8, 3), dtype=bool),
    )
    pseudo = SeriesBatch(batch.values.copy(), batch.observed_mask.copy())
    calls = []

    class RecordingRunner:
        def run_many(self, candidate_ids, current_batch, artifacts, **kwargs):
            candidate_id = candidate_ids[0]
            calls.append(
                (
                    candidate_id,
                    "real" if current_batch is batch else "pseudo",
                    candidate_id in artifacts,
                )
            )
            return {
                candidate_id: failed_candidate_result(
                    candidate_id,
                    current_batch,
                    "synthetic result",
                )
            }

    candidate_ids = ("locf", "saits", "gpvae")
    real, proxies = _run_label_candidate_pairs(
        manager,
        RecordingRunner(),
        candidate_ids,
        batch,
        pseudo,
        seed=7,
        params={},
        budget=BudgetSpec(max_candidates=3),
    )
    audit = manager.audit()
    manifest = _artifact_loading_manifest({"toy": {"mock": audit}})

    assert [call[0] for call in store.calls] == [("saits",), ("gpvae",)]
    assert calls == [
        ("locf", "real", False),
        ("locf", "pseudo", False),
        ("saits", "real", True),
        ("saits", "pseudo", True),
        ("gpvae", "real", True),
        ("gpvae", "pseudo", True),
    ]
    assert tuple(real) == candidate_ids
    assert tuple(proxies) == candidate_ids
    assert audit["max_deep_load_batch"] == 1
    assert audit["deep_evict_count"] == 2
    assert len(cache_clears) == 2
    assert manifest["dataset_count"] == 1
    assert manifest["max_deep_load_batch"] == 1
    assert manifest["datasets"]["toy"]["deep_cleanup_count"] == 2


def test_runtime_device_resolution_keeps_cpu_candidates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr("tsfm_fais.stage_execution._cuda_available", lambda: True)

    assert _torch_device(config) == "cuda"
    assert _allowed_devices(config) == ("cpu", "gpu")

    cpu_runtime = config.runtime.model_copy(update={"device": "cpu"})
    cpu_config = config.model_copy(update={"runtime": cpu_runtime})
    assert _torch_device(cpu_config) == "cpu"
    assert _allowed_devices(cpu_config) == ("cpu",)


def test_pypots_fit_params_use_configured_scale_and_device(tmp_path, monkeypatch):
    config = _updated_config(
        _config(tmp_path),
        deep_imputer_epochs=2,
        deep_imputer_batch_size=3,
        csdi_num_samples=5,
    )
    monkeypatch.setattr("tsfm_fais.stage_execution._cuda_available", lambda: True)

    params = _pypots_params(config, DEFAULT_REGISTRY.get_spec("brits"))

    assert params == {
        "epochs": 2,
        "batch_size": 3,
        "num_samples": 5,
        "device": "cuda",
    }


def test_missforest_fit_params_use_strict_configured_parallelism(tmp_path):
    config = _updated_config(_config(tmp_path), missforest_n_jobs=3)

    params = _fit_candidate_params(config, DEFAULT_REGISTRY.get_spec("missforest"))

    assert params == {"n_jobs": 3}


def test_forecast_spec_and_manifest_include_runtime_controls(tmp_path, monkeypatch):
    config = _updated_config(_config(tmp_path), forecast_num_samples=7)
    monkeypatch.setattr("tsfm_fais.stage_execution._cuda_available", lambda: False)

    spec = _forecast_spec(config, "sundial", dimensions=3)
    metadata = _execution_metadata(config)

    assert spec.num_samples == 7
    assert metadata["sampling_limits"]["forecast_num_samples"] == 7
    assert metadata["sampling_limits"]["missforest_n_jobs"] == 1
    assert metadata["sampling_limits"]["csdi_num_samples"] == 20
    assert metadata["sampling_limits"]["candidate_ids"] == list(_selected_candidate_ids(config))
    assert metadata["device_resolution"]["torch_device"] == "cpu"


def test_preflight_forecaster_passes_device_and_conservative_batch(tmp_path):
    calls = {}

    class Adapter:
        def _ensure_backend(self):
            calls["loaded"] = True

    class Registry:
        def build(self, model_id, **kwargs):
            calls["model_id"] = model_id
            calls.update(kwargs)
            return Adapter()

    adapter = _preflight_forecaster(
        Registry(),
        "chronos2",
        tmp_path / "checkpoint",
        device="cuda",
        batch_size=8,
    )

    assert isinstance(adapter, Adapter)
    assert calls == {
        "model_id": "chronos2",
        "model_name": str(tmp_path / "checkpoint"),
        "device": "cuda",
        "batch_size": 8,
        "loaded": True,
    }


def test_candidate_selection_rejects_duplicates_and_unknown_ids(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        _config(
            tmp_path / "duplicate",
            "  candidate_ids: [locf, linear_interp, locf]",
        )
    with pytest.raises(ValueError, match="unknown"):
        _config(
            tmp_path / "unknown",
            "  candidate_ids: [locf, linear_interp, unpublished_method]",
        )


def test_save_all_candidate_outputs_reuses_existing_results(tmp_path):
    config = _updated_config(_config(tmp_path), save_all_candidate_outputs=True)
    registry = ImputerRegistry(
        DEFAULT_REGISTRY.get_spec(candidate_id) for candidate_id in ("locf", "linear_interp")
    )
    values = _item(8).values[None, ...]
    mask = np.ones_like(values, dtype=bool)
    mask[0, 3, 0] = False
    batch = SeriesBatch(values, mask)
    locf = CandidateRunner(registry).run("locf", batch)
    candidates = {"locf": locf}

    added = _supplement_candidate_outputs(
        config,
        registry,
        {},
        batch,
        candidates,
        seed=5,
    )

    assert added == ("linear_interp",)
    assert candidates["locf"] is locf
    assert tuple(candidates) == ("locf", "linear_interp")


def test_forecaster_artifacts_resolve_model_named_directories(tmp_path):
    root = tmp_path / "checkpoints"
    chronos = root / "chronos2"
    timesfm = root / "timesfm2p5"
    chronos.mkdir(parents=True)
    timesfm.mkdir()

    resolved = _forecaster_artifacts(
        StageInputs(
            forecaster_id="chronos2,timesfm2p5",
            forecaster_artifact=root,
        )
    )

    assert resolved == (("chronos2", chronos), ("timesfm2p5", timesfm))


def test_forecaster_artifacts_resolve_json_paths_relative_to_mapping(tmp_path):
    chronos = tmp_path / "weights" / "chronos"
    timesfm = tmp_path / "weights" / "timesfm"
    chronos.mkdir(parents=True)
    timesfm.mkdir()
    mapping = tmp_path / "forecasters.json"
    mapping.write_text(
        json.dumps(
            {
                "artifacts": {
                    "chronos2": "weights/chronos",
                    "timesfm2p5": "weights/timesfm",
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = _forecaster_artifacts(
        StageInputs(
            forecaster_id="chronos2,timesfm2p5",
            forecaster_artifact=mapping,
        )
    )

    assert resolved == (
        ("chronos2", chronos.resolve()),
        ("timesfm2p5", timesfm.resolve()),
    )


def test_forecaster_artifacts_report_missing_model_path(tmp_path):
    root = tmp_path / "checkpoints"
    (root / "chronos2").mkdir(parents=True)

    with pytest.raises(ValueError, match="timesfm2p5"):
        _forecaster_artifacts(
            StageInputs(
                forecaster_id="chronos2,timesfm2p5",
                forecaster_artifact=root,
            )
        )


def test_pair_label_sampling_is_deterministic_and_covers_candidate_roles():
    edges = (
        SimpleNamespace(left="b0", right="b1"),
        SimpleNamespace(left="b1", right="b2"),
    )
    candidates = tuple(f"c{index}" for index in range(20))
    eligible = {block_id: candidates for block_id in ("b0", "b1", "b2")}

    first = _pair_label_requests(edges, eligible, candidates, seed=19, limit=64)
    second = _pair_label_requests(edges, eligible, candidates, seed=19, limit=64)

    assert first == second
    assert len(first) == 64
    assert {left for _, left, _ in first} == set(candidates)
    assert {right for _, _, right in first} == set(candidates)
    assert {(edge.left, edge.right) for edge, _, _ in first} == {
        ("b0", "b1"),
        ("b1", "b2"),
    }
