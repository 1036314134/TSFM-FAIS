"""Preparation-only experiment stages with durable dependency diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from tsfm_fais.artifacts import RunArtifactStore, make_run_id, utc_now
from tsfm_fais.config import AppConfig

StageName = Literal["fit-imputers", "labels", "train-router", "impute"]


class StagePreparationError(RuntimeError):
    """Raised after dependency failures have been written to the run manifest."""


@dataclass(frozen=True)
class StageInputs:
    audit_artifact: Path | None = None
    imputer_artifacts: Path | None = None
    labels_artifact: Path | None = None
    router_artifact: Path | None = None
    forecaster_artifact: Path | None = None
    forecaster_id: str | None = None

    def to_manifest(self) -> dict[str, str | None]:
        return {
            "audit_artifact": _resolved_or_none(self.audit_artifact),
            "imputer_artifacts": _resolved_or_none(self.imputer_artifacts),
            "labels_artifact": _resolved_or_none(self.labels_artifact),
            "router_artifact": _resolved_or_none(self.router_artifact),
            "forecaster_artifact": _resolved_or_none(self.forecaster_artifact),
            "forecaster_id": self.forecaster_id,
        }


@dataclass
class StagePreparation:
    stage: StageName
    store: RunArtifactStore
    manifest: dict[str, Any]

    def mark_prepared(self, reason: str) -> Path:
        self.manifest["status"] = "prepared"
        self.manifest["execution_started"] = False
        self.manifest["message"] = reason
        self.manifest["updated_at"] = utc_now()
        return self.store.write("stage_manifest.json", self.manifest)


_CONFIG_REQUIREMENTS: Mapping[StageName, tuple[str, ...]] = {
    "fit-imputers": ("data_manifest", "imputer_registry"),
    "labels": ("data_manifest", "imputer_registry", "forecaster_registry"),
    "train-router": ("router_config",),
    "impute": ("data_manifest", "imputer_registry", "router_config", "forecaster_registry"),
}


def _resolved_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def _check(
    name: str,
    path: Path | None,
    *,
    required: bool,
    kind: Literal["file", "directory", "any"] = "any",
    option: str,
) -> dict[str, Any]:
    resolved = path.resolve() if path is not None else None
    if resolved is None:
        exists = False
        valid_kind = False
    else:
        exists = resolved.exists()
        valid_kind = exists and (
            kind == "any"
            or (kind == "file" and resolved.is_file())
            or (kind == "directory" and resolved.is_dir())
        )
    valid = (not required and path is None) or bool(valid_kind)
    if path is None:
        message = f"missing required option {option}" if required else "not provided"
    elif not exists:
        message = f"path does not exist: {resolved}"
    elif not valid_kind:
        message = f"expected a {kind}: {resolved}"
    else:
        message = "ok"
    return {
        "name": name,
        "option": option,
        "path": None if resolved is None else str(resolved),
        "required": required,
        "kind": kind,
        "exists": exists,
        "valid": valid,
        "message": message,
    }


def _audit_semantics(check: dict[str, Any]) -> None:
    if not check["valid"]:
        return
    path = Path(check["path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        check["valid"] = False
        check["message"] = f"invalid audit JSON: {type(error).__name__}: {error}"
        return
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list) or not datasets:
        check["valid"] = False
        check["message"] = "audit JSON must contain a non-empty datasets list"
        return
    rejected = [
        str(entry.get("dataset_id", "<unknown>"))
        for entry in datasets
        if not isinstance(entry, dict) or entry.get("accepted") is not True
    ]
    if rejected:
        check["valid"] = False
        check["message"] = "audit contains rejected datasets: " + ", ".join(rejected)


def _router_semantics(check: dict[str, Any]) -> None:
    if not check["valid"]:
        return
    path = Path(check["path"])
    bundle = path / "router_bundle.joblib" if path.is_dir() else path
    if bundle.is_file():
        check["resolved_bundle"] = str(bundle.resolve())
        return
    folds_path = path / "folds.json" if path.is_dir() else None
    if folds_path is None or not folds_path.is_file():
        check["valid"] = False
        check["message"] = f"router bundle or fold manifest not found: {path}"
        return
    try:
        payload = json.loads(folds_path.read_text(encoding="utf-8"))
        split = payload.get("split") if isinstance(payload, dict) else None
        folds = payload.get("folds") if isinstance(payload, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        check["valid"] = False
        check["message"] = f"invalid router fold manifest: {type(error).__name__}: {error}"
        return
    if split not in {"leave_dataset_out", "leave_model_out"} or not isinstance(
        folds, dict
    ) or not folds:
        check["valid"] = False
        check["message"] = "router fold manifest has an invalid split or empty folds"
        return
    resolved: dict[str, str] = {}
    for held_out, raw_target in folds.items():
        target = Path(str(raw_target))
        if not target.is_absolute():
            target = (folds_path.parent / target).resolve()
        artifact = target / "router_bundle.joblib" if target.is_dir() else target
        if not artifact.is_file():
            check["valid"] = False
            check["message"] = f"router fold {held_out!r} not found: {artifact}"
            return
        resolved[str(held_out)] = str(target)
    check["resolved_folds"] = resolved
    check["fold_split"] = split


def parse_forecaster_ids(value: str | None) -> tuple[str, ...]:
    """Parse the CLI's comma-separated model selection without silent omissions."""

    if value is None:
        return ()
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("forecaster IDs must be non-empty comma-separated values")
    if len(set(parts)) != len(parts):
        raise ValueError("forecaster IDs must be unique")
    return parts


def _forecaster_check(
    config: AppConfig,
    forecaster_id: str | None,
    *,
    allow_multiple: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "forecaster_id",
        "option": "--forecaster-id",
        "path": None,
        "required": True,
        "kind": "registry_id",
        "exists": forecaster_id is not None,
        "valid": False,
        "message": "missing required option --forecaster-id",
        "value": forecaster_id,
    }
    if forecaster_id is None:
        return result
    try:
        requested = parse_forecaster_ids(forecaster_id)
    except ValueError as error:
        result["message"] = str(error)
        return result
    if not allow_multiple and len(requested) != 1:
        result["message"] = "this stage requires exactly one forecaster ID"
        return result
    try:
        payload = yaml.safe_load(config.registries.forecaster_registry.read_text(encoding="utf-8"))
        entries = payload.get("forecasters", []) if isinstance(payload, dict) else []
        ids = {
            entry["id"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        result["message"] = f"cannot read forecaster registry: {type(error).__name__}: {error}"
        return result
    from tsfm_fais.forecasting import default_forecast_registry

    runtime_ids = {entry.model_id for entry in default_forecast_registry().specs()}
    missing = tuple(
        model_id
        for model_id in requested
        if model_id not in ids or model_id not in runtime_ids
    )
    result["valid"] = not missing
    result["values"] = list(requested)
    result["message"] = (
        "ok"
        if result["valid"]
        else (
            f"forecasters {list(missing)!r} are not executable; available: "
            f"{sorted(ids & runtime_ids)}"
        )
    )
    return result


def _input_checks(
    stage: StageName,
    config: AppConfig,
    inputs: StageInputs,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if stage in {"fit-imputers", "labels", "impute"}:
        audit = _check(
            "data_audit",
            inputs.audit_artifact,
            required=True,
            kind="file",
            option="--audit-artifact",
        )
        _audit_semantics(audit)
        checks.append(audit)
    if stage in {"labels", "impute"}:
        checks.append(
            _check(
                "imputer_artifacts",
                inputs.imputer_artifacts,
                required=True,
                kind="directory",
                option="--imputer-artifacts",
            )
        )
        checks.append(
            _forecaster_check(
                config,
                inputs.forecaster_id,
                allow_multiple=stage == "labels",
            )
        )
    if stage == "labels":
        checks.append(
            _check(
                "forecaster_artifact",
                inputs.forecaster_artifact,
                required=True,
                kind="any",
                option="--forecaster-artifact",
            )
        )
    if stage == "train-router":
        checks.append(
            _check(
                "teacher_labels",
                inputs.labels_artifact,
                required=True,
                kind="file",
                option="--labels-artifact",
            )
        )
    if stage == "impute":
        router = _check(
            "router_artifact",
            inputs.router_artifact,
            required=True,
            kind="any",
            option="--router-artifact",
        )
        _router_semantics(router)
        checks.append(router)
    return checks


def prepare_stage(
    config: AppConfig,
    config_source: str | Path,
    stage: StageName,
    inputs: StageInputs,
    *,
    run_id: str | None = None,
) -> StagePreparation:
    """Create audit artifacts and validate dependencies without running an experiment."""

    store = RunArtifactStore.create(
        config.runtime.output_root,
        run_id or make_run_id(stage),
    )
    store.write_baseline(config, config_source)
    checks: list[dict[str, Any]] = []
    for field in _CONFIG_REQUIREMENTS[stage]:
        path = getattr(config.registries, field)
        checks.append(
            _check(
                f"config.{field}",
                path,
                required=True,
                kind="file",
                option=f"config.registries.{field}",
            )
        )
    checks.extend(_input_checks(stage, config, inputs))
    invalid = [check for check in checks if not check["valid"]]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": store.run_id,
        "stage": stage,
        "status": "blocked" if invalid else "prepared",
        "created_at": utc_now(),
        "execution_started": False,
        "automatic_downloads": False,
        "inputs": inputs.to_manifest(),
        "checks": checks,
        "message": (
            "; ".join(check["message"] for check in invalid)
            if invalid
            else "all declared dependencies are available; no experiment has been executed"
        ),
    }
    manifest_path = store.write("stage_manifest.json", manifest)
    if invalid:
        details = "; ".join(
            f"{check['name']}: {check['message']}" for check in invalid
        )
        raise StagePreparationError(
            f"stage {stage!r} is blocked: {details}. Audit manifest: {manifest_path}"
        )
    return StagePreparation(stage=stage, store=store, manifest=manifest)


def finish_preparation(preparation: StagePreparation) -> Path:
    reason = (
        f"stage {preparation.stage!r} dependencies are valid; auditable inputs are prepared "
        "and experiment execution remains disabled for this code-only run"
    )
    return preparation.mark_prepared(reason)


__all__ = [
    "StageInputs",
    "StageName",
    "StagePreparation",
    "StagePreparationError",
    "parse_forecaster_ids",
    "prepare_stage",
    "finish_preparation",
]
