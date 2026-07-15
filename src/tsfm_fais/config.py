"""Strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryRef(StrictModel):
    data_manifest: Path
    imputer_registry: Path
    forecaster_registry: Path
    router_config: Path


class ExperimentConfig(StrictModel):
    split: Literal["leave_dataset_out", "leave_model_out", "rolling_origin"] = "leave_dataset_out"
    context_length: int = Field(default=512, ge=2)
    horizon: int = Field(default=96, ge=1)
    target_indices: tuple[int, ...] | Literal["all"] = "all"
    missing_mechanisms: tuple[
        Literal[
            "random_point",
            "independent_block",
            "synchronous_block",
            "staggered_correlated",
            "value_dependent",
            "tail_mixed",
        ],
        ...,
    ] = (
        "independent_block",
        "tail_mixed",
        "synchronous_block",
    )
    missing_rates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    candidate_ids: tuple[str, ...] | Literal["all"] = "all"
    deep_imputer_epochs: int = Field(default=10, ge=1)
    deep_imputer_batch_size: int = Field(default=32, ge=1)
    csdi_num_samples: int = Field(default=20, ge=1)
    missforest_n_jobs: int = Field(default=1, ge=1)
    max_items_per_dataset: int | None = Field(default=None, ge=1)
    max_training_windows_per_dataset: int | None = Field(default=None, ge=1)
    max_train_origins_per_item: int | None = Field(default=None, ge=1)
    max_eval_origins_per_item: int | None = Field(default=None, ge=1)
    max_train_episodes_per_dataset: int | None = Field(default=None, ge=1)
    max_eval_episodes_per_dataset: int | None = Field(default=None, ge=1)
    max_teacher_blocks_per_episode: int | None = Field(default=8, ge=1)
    max_teacher_candidates_per_episode: int | None = Field(default=8, ge=2)
    max_pair_labels_per_episode: int = Field(default=8, ge=1)
    forecast_num_samples: int = Field(default=20, ge=1)
    save_all_candidate_outputs: bool = False

    @model_validator(mode="after")
    def validate_rates_and_targets(self) -> ExperimentConfig:
        if not self.missing_mechanisms:
            raise ValueError("missing_mechanisms cannot be empty")
        if len(set(self.missing_mechanisms)) != len(self.missing_mechanisms):
            raise ValueError("missing_mechanisms must be unique")
        if not self.missing_rates:
            raise ValueError("missing_rates cannot be empty")
        if any(not 0 < rate <= 0.5 for rate in self.missing_rates):
            raise ValueError("missing_rates must be in (0, 0.5]")
        if len(set(self.missing_rates)) != len(self.missing_rates):
            raise ValueError("missing_rates must be unique")
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.candidate_ids != "all":
            if not self.candidate_ids:
                raise ValueError("candidate_ids cannot be empty")
            if any(not candidate_id.strip() for candidate_id in self.candidate_ids):
                raise ValueError("candidate_ids entries cannot be empty")
            if len(set(self.candidate_ids)) != len(self.candidate_ids):
                raise ValueError("candidate_ids must be unique")
            missing_baselines = {
                "locf",
                "linear_interp",
            }.difference(self.candidate_ids)
            if missing_baselines:
                raise ValueError(
                    "candidate_ids must include routing baselines: "
                    + ", ".join(sorted(missing_baselines))
                )
        if self.target_indices != "all":
            if not self.target_indices:
                raise ValueError("target_indices cannot be empty")
            if any(index < 0 for index in self.target_indices):
                raise ValueError("target_indices must be non-negative")
            if len(set(self.target_indices)) != len(self.target_indices):
                raise ValueError("target_indices must be unique")
        return self


class RuntimeConfig(StrictModel):
    output_root: Path = Path("artifacts")
    device: Literal["auto", "cpu", "gpu"] = "auto"
    fail_fast: bool = False


class AppConfig(StrictModel):
    schema_version: Literal[1] = 1
    seed: int = 20260710
    registries: RegistryRef
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def _resolve_path(value: Path, base: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    return payload


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    config = AppConfig.model_validate(load_yaml(config_path))
    base = config_path.parent
    refs = config.registries.model_copy(
        update={
            field: _resolve_path(getattr(config.registries, field), base)
            for field in (
                "data_manifest",
                "imputer_registry",
                "forecaster_registry",
                "router_config",
            )
        }
    )
    runtime = config.runtime.model_copy(
        update={"output_root": _resolve_path(config.runtime.output_root, base)}
    )
    resolved = config.model_copy(update={"registries": refs, "runtime": runtime})
    requested = resolved.experiment.candidate_ids
    if requested != "all" and refs.imputer_registry.is_file():
        registry_payload = load_yaml(refs.imputer_registry)
        entries = registry_payload.get("imputers", [])
        available = {
            str(entry["id"])
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
        unknown = tuple(candidate_id for candidate_id in requested if candidate_id not in available)
        if unknown:
            raise ValueError("candidate_ids contains unknown candidates: " + ", ".join(unknown))
    return resolved
