"""Small, auditable run-artifact primitives used by the CLI stage scheduler."""

from __future__ import annotations

import json
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from tsfm_fais.config import AppConfig
from tsfm_fais.imputers import DEFAULT_REGISTRY, ImputerRegistry

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_DISTRIBUTIONS = (
    "tsfm-fais",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pyarrow",
    "lightgbm",
    "pydantic",
    "PyYAML",
    "joblib",
    "psutil",
    "tqdm",
    "torch",
    "pypots",
    "chronos-forecasting",
    "timesfm",
    "tirex-ts",
    "transformers",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id(stage: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}-{stamp}-{uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-' (maximum 128 characters)"
        )
    return run_id


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def software_version_payload() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for distribution in _DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "schema_version": 1,
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }


def candidate_status_payload(
    registry: ImputerRegistry = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    candidates = []
    for spec in registry.specs():
        availability = registry.availability(spec.imputer_id)
        candidates.append(
            {
                "id": spec.imputer_id,
                "family": spec.family,
                "mode": spec.mode,
                "device": spec.device,
                "cost_tier": spec.cost_tier,
                "optional_extra": spec.optional_extra,
                "dependencies": list(spec.dependencies),
                "available": availability.available,
                "missing_dependencies": list(availability.missing),
            }
        )
    return {
        "schema_version": 1,
        "captured_at": utc_now(),
        "candidates": candidates,
    }


@dataclass(frozen=True)
class RunArtifactStore:
    """A new, non-overwriting directory for one requested CLI stage."""

    run_id: str
    root: Path

    @classmethod
    def create(cls, output_root: str | Path, run_id: str) -> RunArtifactStore:
        safe_id = validate_run_id(run_id)
        base = Path(output_root).resolve()
        root = base / safe_id
        base.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"run artifact directory already exists; choose a new --run-id: {root}"
            ) from error
        return cls(run_id=safe_id, root=root)

    @classmethod
    def open_existing(cls, output_root: str | Path, run_id: str) -> RunArtifactStore:
        """Open one existing run directory without creating or overwriting files."""

        safe_id = validate_run_id(run_id)
        root = Path(output_root).resolve() / safe_id
        if not root.is_dir():
            raise FileNotFoundError(f"run artifact directory does not exist: {root}")
        return cls(run_id=safe_id, root=root)

    def write(self, name: str, payload: Mapping[str, Any]) -> Path:
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("artifact name must be one JSON filename")
        return _write_json(self.root / name, payload)

    def write_baseline(
        self,
        config: AppConfig,
        config_source: str | Path,
        registry: ImputerRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.write(
            "resolved_config.json",
            {
                "schema_version": 1,
                "source": str(Path(config_source).resolve()),
                "config": config.model_dump(mode="json"),
            },
        )
        self.write("software_versions.json", software_version_payload())
        self.write(
            "seeds.json",
            {
                "schema_version": 1,
                "root_seed": config.seed,
                "experiment_seeds": list(config.experiment.seeds),
            },
        )
        self.write("candidate_status.json", candidate_status_payload(registry))


__all__ = [
    "RunArtifactStore",
    "candidate_status_payload",
    "make_run_id",
    "software_version_payload",
    "utc_now",
    "validate_run_id",
]
