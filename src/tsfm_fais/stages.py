"""Preparation-only experiment stages with durable dependency diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from tsfm_fais.artifacts import (
    RunArtifactStore,
    make_run_id,
    revision_protocol_payload,
    utc_now,
    validate_run_id,
)
from tsfm_fais.config import AppConfig
from tsfm_fais.routing.sequence_protocol import (
    is_independent_sequence_router_metadata,
    normalize_configured_selector_method,
)

StageName = Literal["fit-imputers", "labels", "train-router", "impute"]


class StagePreparationError(RuntimeError):
    """Raised after dependency failures have been written to the run manifest."""


@dataclass(frozen=True)
class StageInputs:
    audit_artifact: Path | None = None
    imputer_artifacts: Path | None = None
    labels_artifact: Path | None = None
    reconstruction_labels_artifact: Path | None = None
    episode_plan_artifact: Path | None = None
    router_artifact: Path | None = None
    forecaster_artifact: Path | None = None
    candidate_source_impute_artifact: Path | None = None
    forecaster_id: str | None = None

    def to_manifest(self) -> dict[str, str | None]:
        return {
            "audit_artifact": _resolved_or_none(self.audit_artifact),
            "imputer_artifacts": _resolved_or_none(self.imputer_artifacts),
            "labels_artifact": _resolved_or_none(self.labels_artifact),
            "reconstruction_labels_artifact": _resolved_or_none(
                self.reconstruction_labels_artifact
            ),
            "episode_plan_artifact": _resolved_or_none(self.episode_plan_artifact),
            "router_artifact": _resolved_or_none(self.router_artifact),
            "forecaster_artifact": _resolved_or_none(self.forecaster_artifact),
            "candidate_source_impute_artifact": _resolved_or_none(
                self.candidate_source_impute_artifact
            ),
            "forecaster_id": self.forecaster_id,
        }


@dataclass
class StagePreparation:
    stage: StageName
    store: RunArtifactStore
    manifest: dict[str, Any]
    resuming: bool = False

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


def _configured_router_methods(config: AppConfig) -> tuple[str, ...]:
    return _configured_router(config).selector_methods


def _configured_router(config: AppConfig):
    try:
        payload = yaml.safe_load(config.registries.router_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read router config: {type(error).__name__}: {error}") from error
    from tsfm_fais.registry_configs import RouterConfig

    return RouterConfig.model_validate(payload)


def _labels_require_forecaster(config: AppConfig) -> bool:
    """Return whether the configured labels are forecast-aware B-FAIS labels."""

    return "block_fais" in _configured_router_methods(config)


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
    if (
        split not in {"leave_family_out", "leave_model_out"}
        or not isinstance(folds, dict)
        or not folds
    ):
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


def _router_manifest_metadata(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read metadata from a direct router or every declared router fold."""

    resolved = path.resolve()
    manifest_paths: tuple[Path, ...]
    if resolved.is_file():
        manifest_paths = (resolved.with_name("manifest.json"),)
    elif (resolved / "router_bundle.joblib").is_file():
        manifest_paths = (resolved / "manifest.json",)
    else:
        folds_path = resolved / "folds.json"
        try:
            payload = json.loads(folds_path.read_text(encoding="utf-8"))
            folds = payload.get("folds") if isinstance(payload, Mapping) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ()
        if not isinstance(folds, Mapping) or not folds:
            return ()
        paths: list[Path] = []
        for raw_target in folds.values():
            target = Path(str(raw_target))
            if not target.is_absolute():
                target = (folds_path.parent / target).resolve()
            paths.append(
                target / "manifest.json" if target.is_dir() else target.with_name("manifest.json")
            )
        manifest_paths = tuple(paths)

    metadata: list[Mapping[str, Any]] = []
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ()
        entry = payload.get("metadata") if isinstance(payload, Mapping) else None
        if not isinstance(entry, Mapping):
            return ()
        metadata.append(entry)
    return tuple(metadata)


def _router_is_forecaster_independent(path: Path | None) -> bool:
    if path is None:
        return False

    metadata = _router_manifest_metadata(path)
    return bool(metadata) and all(
        is_independent_sequence_router_metadata(entry) for entry in metadata
    )


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
        model_id for model_id in requested if model_id not in ids or model_id not in runtime_ids
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
        if stage == "labels" and _labels_require_forecaster(config):
            checks.append(
                _forecaster_check(
                    config,
                    inputs.forecaster_id,
                    allow_multiple=stage == "labels",
                )
            )
    if stage == "labels":
        episode_plan = _check(
            "episode_plan",
            inputs.episode_plan_artifact,
            required=False,
            kind="file",
            option="--episode-plan-artifact",
        )
        if inputs.episode_plan_artifact is not None and _labels_require_forecaster(config):
            episode_plan["valid"] = False
            episode_plan["message"] = (
                "an episode plan can be bound only to forecaster-independent sequence labels"
            )
        checks.append(episode_plan)
        checks.append(
            _check(
                "forecaster_artifact",
                inputs.forecaster_artifact,
                required=_labels_require_forecaster(config),
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
        router_config = _configured_router(config)
        checks.append(
            _check(
                "reconstruction_labels",
                inputs.reconstruction_labels_artifact,
                required=(
                    "block_fais" in router_config.selector_methods
                    and router_config.ranker_target == "imputation_loss"
                ),
                kind="file",
                option="--reconstruction-labels-artifact",
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
        router_metadata = (
            _router_manifest_metadata(inputs.router_artifact)
            if router["valid"] and inputs.router_artifact is not None
            else ()
        )
        artifact_methods = {
            normalize_configured_selector_method(entry["selector_method"])
            for entry in router_metadata
            if "selector_method" in entry
        }
        if len(artifact_methods) > 1:
            router["valid"] = False
            router["message"] = "router folds contain different selector methods"
        elif artifact_methods:
            configured_methods = set(_configured_router_methods(config))
            if not artifact_methods.issubset(configured_methods):
                router["valid"] = False
                router["message"] = (
                    "router artifact selector method is not enabled by "
                    "config.registries.router_config"
                )
            router["selector_methods"] = sorted(artifact_methods)
        forecaster_independent = bool(
            router["valid"] and _router_is_forecaster_independent(inputs.router_artifact)
        )
        router["forecaster_independent_selection"] = forecaster_independent
        checks.append(router)
        if forecaster_independent and config.experiment.split == "leave_model_out":
            checks.append(
                {
                    "name": "router_protocol",
                    "option": "config.experiment.split",
                    "path": None,
                    "required": True,
                    "kind": "protocol",
                    "exists": True,
                    "valid": False,
                    "message": (
                        "forecaster-independent sequence selectors do not support "
                        "leave_model_out routing folds"
                    ),
                }
            )
        if forecaster_independent and inputs.forecaster_id is None:
            checks.append(
                {
                    "name": "forecaster_id",
                    "option": "--forecaster-id",
                    "path": None,
                    "required": False,
                    "kind": "registry_id",
                    "exists": False,
                    "valid": True,
                    "message": "not required for a forecaster-independent sequence selector",
                    "value": None,
                }
            )
        else:
            forecaster = _forecaster_check(config, inputs.forecaster_id)
            if forecaster_independent:
                forecaster["required"] = False
                if forecaster["valid"]:
                    forecaster["message"] = (
                        "accepted for compatibility; the imputation artifact uses the "
                        "stable independent identity"
                    )
            checks.append(forecaster)
        checks.append(
            _check(
                "candidate_source_impute_artifact",
                inputs.candidate_source_impute_artifact,
                required=False,
                kind="directory",
                option="--candidate-source-impute-artifact",
            )
        )
        forecaster_artifact = _check(
            "forecaster_artifact",
            inputs.forecaster_artifact,
            required=False,
            kind="any",
            option="--forecaster-artifact",
        )
        if forecaster_independent and inputs.forecaster_artifact is not None:
            forecaster_artifact["valid"] = False
            forecaster_artifact["message"] = (
                "forecaster-independent sequence selectors do not accept "
                "--forecaster-artifact during imputation"
            )
        checks.append(forecaster_artifact)
    return checks


def prepare_stage(
    config: AppConfig,
    config_source: str | Path,
    stage: StageName,
    inputs: StageInputs,
    *,
    run_id: str | None = None,
    resume: bool = False,
) -> StagePreparation:
    """Create audit artifacts and validate dependencies without running an experiment."""

    protocol = config.protocol
    if protocol is not None and run_id is not None:
        safe_run_id = validate_run_id(run_id)
        expected_prefix = f"{protocol.run_id_prefix}-"
        if not safe_run_id.startswith(expected_prefix):
            raise ValueError(
                f"revision protocol run_id must start with {expected_prefix!r}: {safe_run_id!r}"
            )

    if resume:
        if stage not in {"fit-imputers", "labels", "impute"}:
            raise ValueError(
                "--resume is currently supported only for fit-imputers, labels, and impute"
            )
        if (
            stage == "labels"
            and _labels_require_forecaster(config)
            and len(parse_forecaster_ids(inputs.forecaster_id)) != 1
        ):
            raise ValueError("labels resume requires exactly one forecaster ID")
        if run_id is None:
            raise ValueError("--resume requires an explicit --run-id")
        store = RunArtifactStore.open_existing(config.runtime.output_root, run_id)
        resolved_path = store.root / "resolved_config.json"
        stage_path = store.root / "stage_manifest.json"
        try:
            resolved_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
            manifest = json.loads(stage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot resume run with invalid baseline metadata: {type(error).__name__}: {error}"
            ) from error
        stored_config = resolved_payload.get("config")
        if not isinstance(stored_config, dict):
            raise ValueError("cannot resume: resolved_config.json has no config mapping")
        try:
            normalized_stored = AppConfig.model_validate(stored_config).model_dump(mode="json")
        except ValueError as error:
            raise ValueError(
                f"cannot resume: stored resolved config is invalid: {error}"
            ) from error
        if normalized_stored != config.model_dump(mode="json"):
            raise ValueError("cannot resume: resolved config differs from the original run")
        if not isinstance(manifest, dict) or manifest.get("stage") != stage:
            raise ValueError(f"cannot resume: stage manifest does not describe {stage}")
        if manifest.get("run_id") != store.run_id:
            raise ValueError("cannot resume: stage manifest run_id mismatch")
        if manifest.get("inputs") != inputs.to_manifest():
            raise ValueError("cannot resume: declared stage inputs differ from the original run")
    else:
        store = RunArtifactStore.create(
            config.runtime.output_root,
            run_id
            or make_run_id(
                stage,
                prefix=None if protocol is None else protocol.run_id_prefix,
            ),
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
    if resume:
        manifest = dict(manifest)
        manifest["checks"] = checks
        manifest["resume_requested"] = True
        manifest["message"] = "resume inputs validated; execution has not restarted"
        manifest_path = stage_path
    else:
        manifest = {
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
        protocol_payload = revision_protocol_payload(config)
        if protocol_payload is not None:
            manifest["experiment_protocol"] = protocol_payload
            manifest["protocol_artifact"] = str((store.root / "experiment_protocol.json").resolve())
        manifest_path = store.write("stage_manifest.json", manifest)
    if invalid:
        details = "; ".join(f"{check['name']}: {check['message']}" for check in invalid)
        raise StagePreparationError(
            f"stage {stage!r} is blocked: {details}. Audit manifest: {manifest_path}"
        )
    return StagePreparation(
        stage=stage,
        store=store,
        manifest=manifest,
        resuming=resume,
    )


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
