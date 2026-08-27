"""Small, auditable run-artifact primitives used by the CLI stage scheduler."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
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


def make_run_id(stage: str, prefix: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = stage if prefix is None else f"{validate_run_id(prefix)}-{stage}"
    return f"{stem}-{stamp}-{uuid4().hex[:8]}"


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


def revision_protocol_payload(config: AppConfig) -> dict[str, Any] | None:
    protocol = config.protocol
    if protocol is None:
        return None
    return {
        "schema_version": 1,
        "protocol": protocol.model_dump(mode="json"),
        "execution_binding": {
            "split": config.experiment.split,
            "include_family_ids": (
                config.experiment.include_family_ids
                if config.experiment.include_family_ids == "all"
                else list(config.experiment.include_family_ids)
            ),
            "exclude_family_ids": list(config.experiment.exclude_family_ids),
            "feature_policy": config.experiment.feature_policy,
            "active_mask_seeds": list(config.experiment.seeds),
            "active_router_seed": config.experiment.router_seed,
            "output_root": str(config.runtime.output_root.resolve()),
        },
    }


def _find_repository_root(config_source: str | Path) -> Path | None:
    starts = (Path(config_source).resolve().parent, Path.cwd().resolve())
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in checked:
                continue
            checked.add(directory)
            if (directory / ".git").exists():
                return directory
    return None


def repository_state_payload(config_source: str | Path) -> dict[str, Any]:
    """Capture read-only Git provenance without assuming Git is available."""

    root = _find_repository_root(config_source)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "repository_root": None if root is None else str(root),
        "commit": None,
        "dirty": None,
        "status_entry_count": None,
        "status_sha256": None,
        "tracked_diff_sha256": None,
        "tracked_diff_size_bytes": None,
        "untracked_reproducibility_files": [],
        "error": None,
    }
    if root is None:
        payload["error"] = "repository root containing .git was not found"
        return payload
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_output = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        status_lines = tuple(
            line
            for line in status_output.splitlines()
            if line
        )
        tracked_diff = subprocess.run(
            (
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "src",
                "configs",
                "scripts",
                "tests",
                "pyproject.toml",
            ),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            (
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "src",
                "configs",
                "scripts",
                "tests",
            ),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_files: list[dict[str, Any]] = []
        for raw_path in filter(None, untracked_output.split(b"\0")):
            relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                continue
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            untracked_files.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    except (OSError, subprocess.CalledProcessError) as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        return payload
    payload.update(
        {
            "commit": commit or None,
            "dirty": bool(status_lines),
            "status_entry_count": len(status_lines),
            "status_sha256": hashlib.sha256(status_output.encode()).hexdigest(),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "tracked_diff_size_bytes": len(tracked_diff),
            "untracked_reproducibility_files": untracked_files,
        }
    )
    return payload


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
        protocol_payload = revision_protocol_payload(config)
        if protocol_payload is not None:
            self.write("experiment_protocol.json", protocol_payload)
            self.write("repository_state.json", repository_state_payload(config_source))
        seed_payload: dict[str, Any] = {
            "schema_version": 1,
            "root_seed": config.seed,
            "experiment_seeds": list(config.experiment.seeds),
        }
        if config.protocol is not None:
            seed_payload.update(
                {
                    "active_mask_partition": config.protocol.active_mask_partition,
                    "mask_seed_partitions": config.protocol.mask_seeds.model_dump(mode="json"),
                    "router_seed_roots": list(config.protocol.router_seed_roots),
                    "active_router_seed": config.experiment.router_seed,
                }
            )
        self.write(
            "seeds.json",
            seed_payload,
        )
        self.write("candidate_status.json", candidate_status_payload(registry))


__all__ = [
    "RunArtifactStore",
    "candidate_status_payload",
    "make_run_id",
    "revision_protocol_payload",
    "repository_state_payload",
    "software_version_payload",
    "utc_now",
    "validate_run_id",
]
