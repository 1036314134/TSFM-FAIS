"""Loading of frozen dataset-level imputer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .registry import DEFAULT_REGISTRY, ImputerRegistry


def load_dataset_imputer_artifacts(
    root: str | Path,
    dataset_id: str,
    registry: ImputerRegistry | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, str]]:
    """Load fitted candidates and training statistics for one dataset."""

    registry = registry or DEFAULT_REGISTRY
    source = Path(root).resolve()
    manifest_path = source / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict) or dataset_id not in datasets:
        raise KeyError(f"imputer artifact manifest has no dataset {dataset_id!r}")
    dataset = datasets[dataset_id]
    directory = source / dataset_id
    with np.load(directory / dataset["statistics"]) as statistics:
        medians = np.asarray(statistics["medians"], dtype=float).copy()
        correlation = np.asarray(statistics["correlation"], dtype=float).copy()

    artifacts: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for candidate_id, entry in dataset["candidates"].items():
        if entry["status"] != "fitted":
            if entry["status"] in {"failed", "unavailable"}:
                failures[candidate_id] = str(
                    entry.get("reason")
                    or entry.get("missing_dependencies")
                    or entry["status"]
                )
            continue
        availability = registry.availability(candidate_id)
        if not availability.available:
            failures[candidate_id] = (
                "missing dependencies: " + ", ".join(availability.missing)
            )
            continue
        path = directory / entry["path"]
        try:
            if entry["serializer"] == "joblib":
                artifacts[candidate_id] = joblib.load(path)
            elif entry["serializer"] == "adapter":
                adapter = registry.create(candidate_id)
                load_artifact = getattr(adapter, "load_artifact", None)
                if not callable(load_artifact):
                    raise TypeError(
                        f"candidate {candidate_id!r} has no artifact loader"
                    )
                artifacts[candidate_id] = load_artifact(path)
            else:
                raise ValueError(
                    f"unknown artifact serializer for {candidate_id!r}: "
                    f"{entry['serializer']!r}"
                )
        except Exception as error:
            failures[candidate_id] = f"{type(error).__name__}: {error}"
    return artifacts, medians, correlation, failures


__all__ = ["load_dataset_imputer_artifacts"]
