"""Indexed and filtered loading of frozen dataset-level imputer artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from .registry import DEFAULT_REGISTRY, ImputerRegistry


@dataclass(frozen=True)
class ArtifactLoadResult:
    """Result and audit data for one filtered deserialization request."""

    artifacts: dict[str, Any]
    failures: dict[str, str]
    requested_ids: tuple[str, ...]
    attempted_ids: tuple[str, ...]
    loaded_ids: tuple[str, ...]
    load_seconds: float
    load_modes: dict[str, str] = field(default_factory=dict)


class DatasetImputerArtifactStore:
    """Manifest index for one dataset, without eager artifact deserialization."""

    def __init__(
        self,
        root: str | Path,
        dataset_id: str,
        registry: ImputerRegistry | None = None,
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.root = Path(root).resolve()
        self.dataset_id = dataset_id
        manifest_path = self.root / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = payload.get("datasets") if isinstance(payload, dict) else None
        if not isinstance(datasets, dict) or dataset_id not in datasets:
            raise KeyError(f"imputer artifact manifest has no dataset {dataset_id!r}")
        dataset = datasets[dataset_id]
        if not isinstance(dataset, dict):
            raise ValueError(f"imputer artifact entry for {dataset_id!r} must be a mapping")
        candidates = dataset.get("candidates")
        if not isinstance(candidates, dict):
            raise ValueError(f"imputer artifact entry for {dataset_id!r} has no candidates map")
        statistics = dataset.get("statistics")
        if not isinstance(statistics, str) or not statistics:
            raise ValueError(f"imputer artifact entry for {dataset_id!r} has no statistics path")
        self.directory = (self.root / dataset_id).resolve()
        self.statistics_path = self._resolved_path(statistics)
        self._candidates: dict[str, Mapping[str, Any]] = {}
        for candidate_id, entry in candidates.items():
            if not isinstance(candidate_id, str) or not isinstance(entry, dict):
                raise ValueError(f"invalid candidate entry in dataset {dataset_id!r}")
            self._candidates[candidate_id] = entry

    def _resolved_path(self, relative: str) -> Path:
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError(f"artifact path escapes dataset directory: {relative!r}")
        return path

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._candidates)

    def status(self, candidate_id: str) -> str | None:
        entry = self._candidates.get(candidate_id)
        if entry is None:
            return None
        status = entry.get("status")
        return str(status) if status is not None else None

    def fitted_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            candidate_id
            for candidate_id in self._candidates
            if self.status(candidate_id) == "fitted"
        )

    def load_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.statistics_path) as statistics:
            medians = np.asarray(statistics["medians"], dtype=float).copy()
            correlation = np.asarray(statistics["correlation"], dtype=float).copy()
        return medians, correlation

    @staticmethod
    def _entry_failure(entry: Mapping[str, Any]) -> str:
        reason = entry.get("reason") or entry.get("missing_dependencies") or entry.get("status")
        return str(reason)

    def load_artifacts(
        self,
        candidate_ids: Sequence[str] | None = None,
        *,
        adapter_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ArtifactLoadResult:
        """Deserialize only requested fitted candidates.

        Candidate filtering is completed before availability checks, adapter
        construction, path resolution, or calls to ``joblib.load``.
        """

        requested = tuple(
            dict.fromkeys(self.candidate_ids if candidate_ids is None else candidate_ids)
        )
        params = adapter_params or {}
        artifacts: dict[str, Any] = {}
        failures: dict[str, str] = {}
        attempted: list[str] = []
        load_modes: dict[str, str] = {}
        started = perf_counter()
        for candidate_id in requested:
            entry = self._candidates.get(candidate_id)
            if entry is None:
                failures[candidate_id] = "candidate is absent from the artifact manifest"
                continue
            status = entry.get("status")
            if status != "fitted":
                if status in {"failed", "unavailable"}:
                    failures[candidate_id] = self._entry_failure(entry)
                continue
            if candidate_id not in self.registry:
                failures[candidate_id] = "candidate is absent from the runtime registry"
                continue
            availability = self.registry.availability(candidate_id)
            if not availability.available:
                failures[candidate_id] = (
                    availability.reason
                    or "missing dependencies: " + ", ".join(availability.missing)
                )
                continue
            path_value = entry.get("path")
            serializer = entry.get("serializer")
            if not isinstance(path_value, str) or not path_value:
                failures[candidate_id] = "fitted artifact has no path"
                continue
            try:
                path = self._resolved_path(path_value)
                attempted.append(candidate_id)
                if serializer == "joblib":
                    if candidate_id == "missforest":
                        load_modes[candidate_id] = "joblib_mmap_r"
                        artifacts[candidate_id] = joblib.load(path, mmap_mode="r")
                    else:
                        load_modes[candidate_id] = "joblib_standard"
                        artifacts[candidate_id] = joblib.load(path)
                elif serializer == "adapter":
                    load_modes[candidate_id] = "adapter_native"
                    adapter = self.registry.create(
                        candidate_id,
                        **dict(params.get(candidate_id, {})),
                    )
                    load_artifact = getattr(adapter, "load_artifact", None)
                    if not callable(load_artifact):
                        raise TypeError(f"candidate {candidate_id!r} has no artifact loader")
                    artifacts[candidate_id] = load_artifact(path)
                else:
                    raise ValueError(
                        f"unknown artifact serializer for {candidate_id!r}: {serializer!r}"
                    )
            except Exception as error:
                failures[candidate_id] = f"{type(error).__name__}: {error}"
        return ArtifactLoadResult(
            artifacts=artifacts,
            failures=failures,
            requested_ids=requested,
            attempted_ids=tuple(attempted),
            loaded_ids=tuple(artifacts),
            load_seconds=perf_counter() - started,
            load_modes=load_modes,
        )


def load_dataset_imputer_artifacts(
    root: str | Path,
    dataset_id: str,
    registry: ImputerRegistry | None = None,
    *,
    candidate_ids: Sequence[str] | None = None,
    adapter_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, str]]:
    """Load statistics and a filtered set of fitted candidates for one dataset."""

    store = DatasetImputerArtifactStore(root, dataset_id, registry)
    medians, correlation = store.load_statistics()
    result = store.load_artifacts(candidate_ids, adapter_params=adapter_params)
    return result.artifacts, medians, correlation, result.failures


__all__ = [
    "ArtifactLoadResult",
    "DatasetImputerArtifactStore",
    "load_dataset_imputer_artifacts",
]
