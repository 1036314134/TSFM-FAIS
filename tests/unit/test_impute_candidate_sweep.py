from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tsfm_fais.config import AppConfig, load_config
from tsfm_fais.contracts import (
    CandidateResult,
    CandidateStatus,
    ForecastSpec,
    ImputerSpec,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY, ImputerRegistry
from tsfm_fais.imputers.artifacts import ArtifactLoadResult
from tsfm_fais.pipeline import RoutePlan
from tsfm_fais.routing.blocks import build_block_graph, detect_missing_blocks
from tsfm_fais.stage_execution import (
    _budgeted_route_results,
    _ImputeArtifactManager,
    _ImputeEpisodeWork,
    _prepare_impute_work,
    _run_impute_candidate_sweep,
)


def _config(*, save_all: bool) -> AppConfig:
    base = load_config("configs/smoke.yaml")
    experiment = base.experiment.model_copy(
        update={
            "save_all_candidate_outputs": save_all,
            "csdi_num_samples": 5,
        }
    )
    return base.model_copy(update={"experiment": experiment})


def _fitted_spec(candidate_id: str) -> ImputerSpec:
    return ImputerSpec(
        imputer_id=candidate_id,
        family="test",
        mode="joint_multivariate",
        factory="tsfm_fais.imputers.classical:LOCFImputer",
        fit_scope="dataset",
    )


class _Store:
    def __init__(
        self,
        candidate_ids: tuple[str, ...],
        *,
        load_failures: dict[str, str] | None = None,
    ) -> None:
        self.candidate_ids = candidate_ids
        self.load_failures = dict(load_failures or {})
        self.calls: list[tuple[str, ...]] = []

    def status(self, candidate_id: str) -> str | None:
        return "fitted" if candidate_id in self.candidate_ids else None

    def load_artifacts(self, candidate_ids, *, adapter_params=None):
        del adapter_params
        requested = tuple(candidate_ids)
        self.calls.append(requested)
        artifacts = {
            candidate_id: object()
            for candidate_id in requested
            if candidate_id not in self.load_failures
        }
        failures = {
            candidate_id: self.load_failures[candidate_id]
            for candidate_id in requested
            if candidate_id in self.load_failures
        }
        return ArtifactLoadResult(
            artifacts=artifacts,
            failures=failures,
            requested_ids=requested,
            attempted_ids=requested,
            loaded_ids=tuple(artifacts),
            load_seconds=0.01 * len(requested),
            load_modes={candidate_id: "test" for candidate_id in artifacts},
        )


class _Runner:
    def __init__(
        self,
        *,
        inference_failures: set[tuple[str, int, str]] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.inference_failures = set(inference_failures or set())

    def run_many(
        self,
        imputer_ids,
        batch,
        artifacts=None,
        *,
        seed=0,
        params=None,
        artifact_failures=None,
        budget=None,
    ):
        identifiers = tuple(imputer_ids)
        assert len(identifiers) == 1
        assert batch.shape[0] == 1
        candidate_id = identifiers[0]
        role = str(batch.metadata.get("role", "actual"))
        self.calls.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "role": role,
                "batch_size": batch.shape[0],
                "artifact_ids": tuple((artifacts or {}).keys()),
                "params": dict(params or {}),
                "artifact_failures": dict(artifact_failures or {}),
                "budget": budget,
            }
        )
        values = batch.values.copy()
        values[~batch.observed_mask] = float(len(self.calls))
        native = np.ones(batch.shape, dtype=bool)
        status = CandidateStatus.SUCCESS
        failure_reason = None
        artifact_failure = (artifact_failures or {}).get(candidate_id)
        if artifact_failure is not None:
            status = CandidateStatus.FAILED
            failure_reason = f"artifact load failed: {artifact_failure}"
            native[~batch.observed_mask] = False
        elif (candidate_id, seed, role) in self.inference_failures:
            status = CandidateStatus.FAILED
            failure_reason = "synthetic inference failure"
            native[~batch.observed_mask] = False
        return {
            candidate_id: CandidateResult(
                candidate_id,
                values,
                native,
                runtime_seconds=0.25,
                status=status,
                failure_reason=failure_reason,
            )
        }


class _Pipeline:
    def __init__(self, registry: ImputerRegistry, runner: _Runner) -> None:
        self.imputer_registry = registry
        self.candidate_runner = runner
        self.shortlist_size = 2
        self.fallback_internal: tuple[str, ...] = ()
        self.fallback_tail: tuple[str, ...] = ()
        self.prepare_calls: list[tuple[str, ...]] = []

    def prepare_route(
        self,
        item,
        observed_mask,
        forecast_spec,
        budget,
        *,
        seed,
        available_artifact_ids,
        artifact_load_failures,
    ):
        available = tuple(
            candidate_id
            for candidate_id in self.imputer_registry.ids
            if candidate_id in set(available_artifact_ids)
        )
        self.prepare_calls.append(available)
        mask = np.asarray(observed_mask, dtype=bool)
        batch = SeriesBatch(
            item.values[None, ...],
            mask[None, ...],
            item_ids=(item.item_id,),
            metadata={"role": "actual"},
        )
        blocks = detect_missing_blocks(batch.observed_mask)
        graph = build_block_graph(blocks, np.eye(batch.shape[2]))
        shortlist = available[: min(self.shortlist_size, budget.max_candidates)]
        pseudo_mask = batch.observed_mask.copy()
        pseudo_mask[0, 0, 0] = False
        pseudo = SeriesBatch(
            batch.values.copy(),
            pseudo_mask,
            item_ids=batch.item_ids,
            metadata={"role": "pseudo"},
        )
        return RoutePlan(
            batch=batch,
            blocks=blocks,
            graph=graph,
            forecast_spec=forecast_spec,
            budget=budget,
            seed=seed,
            period=4,
            correlation_source="training",
            candidate_ids=available,
            shortlist=shortlist,
            costs={candidate_id: 1.0 for candidate_id in available},
            prior_unary={},
            pseudo_batch=pseudo,
            training_medians=np.zeros(batch.shape[2]),
            artifact_load_failures=dict(artifact_load_failures),
            available_artifact_ids=frozenset(available),
            allow_fallback_execution=False,
        )


def _work(index: int, seed: int) -> _ImputeEpisodeWork:
    time = np.arange(12, dtype=float)
    values = np.column_stack((time, np.sin(time), np.cos(time)))
    mask = np.ones_like(values, dtype=bool)
    mask[4:7, 1] = False
    batch = SeriesBatch(
        values[None, ...],
        mask[None, ...],
        item_ids=(f"item-{index}",),
    )
    item = TimeSeriesItem(
        item_id=f"item-{index}",
        values=values,
        variate_names=("a", "b", "c"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
        metadata={"period": 4},
    )
    episode = SimpleNamespace(context=batch, seed=seed)
    return _ImputeEpisodeWork(
        index=index,
        episode_id=f"episode-{index}",
        episode=episode,
        item=item,
        spec=ForecastSpec(
            "mock",
            "independent_univariate",
            horizon=2,
            target_indices=(0,),
        ),
        relative=Path(f"episode-{index}.npz"),
        relative_assignment=Path(f"episode-{index}.json"),
        entry_key=f"{index:08d}",
        invalid_reason=None,
        pipeline_rss_before=0,
        mase_scale=np.ones(3, dtype=float),
        mase_scale_lag=1,
    )


def _prepare_all(
    works: list[_ImputeEpisodeWork],
    pipeline: _Pipeline,
    available: set[str],
    failures: dict[str, str],
) -> None:
    for work in works:
        _prepare_impute_work(
            work,
            pipeline,  # type: ignore[arg-type]
            available,
            failures,
            ("cpu",),
        )


def test_candidate_major_sweep_loads_once_and_keeps_roles_episode_local() -> None:
    registry = ImputerRegistry(_fitted_spec(value) for value in ("fit_a", "fit_b", "fit_c"))
    config = _config(save_all=True)
    store = _Store(registry.ids)
    manager = _ImputeArtifactManager(store, registry, config)  # type: ignore[arg-type]
    runner = _Runner(inference_failures={("fit_b", 22, "actual")})
    pipeline = _Pipeline(registry, runner)
    works = [_work(0, 11), _work(1, 22)]
    available = manager.declared_available(registry.ids)
    failures = manager.declared_failures(registry.ids)
    _prepare_all(works, pipeline, available, failures)

    audit = _run_impute_candidate_sweep(
        works,
        pipeline,  # type: ignore[arg-type]
        manager,
        config,
        ("cpu",),
        available,
        failures,
    )

    assert store.calls == [("fit_a",), ("fit_b",), ("fit_c",)]
    assert audit["load_counts"] == {"fit_a": 1, "fit_b": 1, "fit_c": 1}
    assert audit["max_active_leases"] == 1
    assert audit["repair_reload_count"] == 0
    assert all(call["batch_size"] == 1 for call in runner.calls)
    assert {(call["seed"], call["role"]) for call in runner.calls} == {
        (11, "actual"),
        (11, "pseudo"),
        (22, "actual"),
        (22, "pseudo"),
    }
    assert [call["role"] for call in runner.calls if call["candidate_id"] == "fit_c"] == [
        "actual",
        "actual",
    ]
    assert all(set(work.raw_actual) == set(registry.ids) for work in works)
    assert all(set(work.raw_pseudo) == {"fit_a", "fit_b"} for work in works)
    assert works[1].raw_actual["fit_b"].status is CandidateStatus.FAILED
    budgeted, _ = _budgeted_route_results(works[1], registry)
    assert budgeted["fit_b"].status is CandidateStatus.FAILED


def test_artifact_load_failure_rebuilds_plan_and_fills_new_shortlist() -> None:
    registry = ImputerRegistry(_fitted_spec(value) for value in ("fit_a", "fit_b", "fit_c"))
    config = _config(save_all=False)
    store = _Store(registry.ids, load_failures={"fit_a": "synthetic load failure"})
    manager = _ImputeArtifactManager(store, registry, config)  # type: ignore[arg-type]
    runner = _Runner()
    pipeline = _Pipeline(registry, runner)
    works = [_work(0, 31), _work(1, 32)]
    available = manager.declared_available(registry.ids)
    failures = manager.declared_failures(registry.ids)
    _prepare_all(works, pipeline, available, failures)

    audit = _run_impute_candidate_sweep(
        works,
        pipeline,  # type: ignore[arg-type]
        manager,
        config,
        ("cpu",),
        available,
        failures,
    )

    assert available == {"fit_b", "fit_c"}
    assert failures == {"fit_a": "synthetic load failure"}
    assert all(work.plan is not None for work in works)
    assert all(work.plan.shortlist == ("fit_b", "fit_c") for work in works if work.plan)
    assert all(set(work.raw_actual) == {"fit_b", "fit_c"} for work in works)
    assert all(set(work.raw_pseudo) == {"fit_b", "fit_c"} for work in works)
    assert store.calls == [("fit_a",), ("fit_b",), ("fit_c",)]
    assert audit["load_counts"] == {"fit_a": 1, "fit_b": 1, "fit_c": 1}
    assert audit["max_active_leases"] == 1
    assert audit["repair_reload_count"] == 0


def test_csdi_routing_and_pseudo_calls_receive_configured_sampling_params() -> None:
    registry = ImputerRegistry((DEFAULT_REGISTRY.get_spec("csdi"),))
    config = _config(save_all=False)
    store = _Store(("csdi",))
    manager = _ImputeArtifactManager(store, registry, config)  # type: ignore[arg-type]
    runner = _Runner()
    pipeline = _Pipeline(registry, runner)
    works = [_work(0, 41)]
    available = manager.declared_available(registry.ids)
    failures = manager.declared_failures(registry.ids)
    _prepare_all(works, pipeline, available, failures)

    _run_impute_candidate_sweep(
        works,
        pipeline,  # type: ignore[arg-type]
        manager,
        config,
        ("cpu",),
        available,
        failures,
    )

    assert [call["role"] for call in runner.calls] == ["actual", "pseudo"]
    for call in runner.calls:
        params = call["params"]
        assert isinstance(params, dict)
        assert params["csdi"]["num_samples"] == 5
        assert params["csdi"]["device"] == "cpu"
