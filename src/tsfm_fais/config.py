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


FeaturePolicy = Literal["legacy", "deployment_available", "identity_free"]
MaskSeedPartition = Literal["train", "development", "confirmation"]
TargetProtocol = Literal[
    "full_candidate_forecast_loss_v2",
    "single_block_counterfactual_forecast_loss_v1",
    "coherence_adjusted_marginal_v1",
    "masked_context_reconstruction_asmape_v1",
]


class MaskSeedPartitions(StrictModel):
    train: tuple[int, ...]
    development: tuple[int, ...]
    confirmation: tuple[int, ...]

    @model_validator(mode="after")
    def validate_partitions(self) -> MaskSeedPartitions:
        partitions = {
            "train": self.train,
            "development": self.development,
            "confirmation": self.confirmation,
        }
        for name, values in partitions.items():
            if not values:
                raise ValueError(f"mask seed partition {name!r} cannot be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"mask seed partition {name!r} must be unique")
        seen: dict[int, str] = {}
        for name, values in partitions.items():
            for value in values:
                previous = seen.setdefault(value, name)
                if previous != name:
                    raise ValueError(
                        "mask seed partitions must be disjoint: "
                        f"seed {value} occurs in {previous!r} and {name!r}"
                    )
        return self

    def active(self, partition: MaskSeedPartition) -> tuple[int, ...]:
        return getattr(self, partition)


class RevisionProtocolConfig(StrictModel):
    """Auditable identity for a revision experiment family.

    The executable split, feature filtering, and active mask seeds remain in
    ``ExperimentConfig``.  This record binds those settings to an isolated
    artifact namespace and records the seeds and forecaster transfer contract.
    """

    protocol_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    artifact_namespace: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    run_id_prefix: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
    family_split_policy: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    development_family_ids: tuple[str, ...] = ()
    target_protocol: TargetProtocol
    active_mask_partition: MaskSeedPartition
    mask_seeds: MaskSeedPartitions
    router_seed_roots: tuple[int, ...]
    teacher_forecaster_ids: tuple[str, ...] = ()
    held_out_forecaster_id: str | None = None
    held_out_forecaster_revision: str | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> RevisionProtocolConfig:
        for field, values in (
            ("development_family_ids", self.development_family_ids),
            ("router_seed_roots", self.router_seed_roots),
            ("teacher_forecaster_ids", self.teacher_forecaster_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{field} must be unique")
        if not self.router_seed_roots:
            raise ValueError("router_seed_roots cannot be empty")
        if self.held_out_forecaster_id is not None:
            if self.held_out_forecaster_id in self.teacher_forecaster_ids:
                raise ValueError("held-out forecaster cannot be a teacher forecaster")
            if not self.held_out_forecaster_revision:
                raise ValueError(
                    "held_out_forecaster_revision is required for a held-out forecaster"
                )
        elif self.held_out_forecaster_revision is not None:
            raise ValueError("held_out_forecaster_revision requires held_out_forecaster_id")
        return self


class ExperimentConfig(StrictModel):
    split: Literal["leave_family_out", "leave_model_out", "rolling_origin"] = "leave_family_out"
    context_length: int = Field(default=96, ge=2)
    horizon: int = Field(default=96, ge=1)
    forecast_stride: int | None = Field(default=None, ge=1)
    fit_prefix_fraction: float = Field(default=0.2, gt=0, lt=1)
    training_window_stride: int = Field(default=24, ge=1)
    masking_protocol: Literal["synthetic", "native_only"] = "synthetic"
    training_base_mask: Literal["none", "no_complete_window"] = "none"
    training_base_missing_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    min_future_target_observed_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    missing_block_lengths: tuple[int, ...] = (6, 12, 24, 48)
    target_indices: tuple[int, ...] | Literal["all"] = "all"
    missing_mechanisms: tuple[
        Literal[
            "random_point",
            "independent_block",
            "synchronous_block",
            "staggered_correlated",
            "value_dependent",
            "mixed_outage",
        ],
        ...,
    ] = (
        "independent_block",
        "mixed_outage",
        "synchronous_block",
    )
    missing_rates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    router_seed: int | None = None
    include_family_ids: tuple[str, ...] | Literal["all"] = "all"
    exclude_family_ids: tuple[str, ...] = ()
    feature_policy: FeaturePolicy = "legacy"
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
    forecast_batch_size: int = Field(default=128, ge=1)
    save_all_candidate_outputs: bool = False

    @model_validator(mode="after")
    def validate_rates_and_targets(self) -> ExperimentConfig:
        if self.training_base_mask == "none" and self.training_base_missing_rate != 0.0:
            raise ValueError(
                "training_base_missing_rate must be zero when training_base_mask is 'none'"
            )
        if (
            self.training_base_mask == "no_complete_window"
            and self.training_base_missing_rate <= 0.0
        ):
            raise ValueError(
                "no_complete_window training requires a positive training_base_missing_rate"
            )
        if not self.missing_mechanisms:
            raise ValueError("missing_mechanisms cannot be empty")
        if len(set(self.missing_mechanisms)) != len(self.missing_mechanisms):
            raise ValueError("missing_mechanisms must be unique")
        if not self.missing_block_lengths or any(
            length < 1 for length in self.missing_block_lengths
        ):
            raise ValueError("missing_block_lengths must contain positive integers")
        if len(set(self.missing_block_lengths)) != len(self.missing_block_lengths):
            raise ValueError("missing_block_lengths must be unique")
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
        if self.include_family_ids != "all":
            if not self.include_family_ids:
                raise ValueError("include_family_ids cannot be empty")
            if any(not value.strip() for value in self.include_family_ids):
                raise ValueError("include_family_ids entries cannot be empty")
            if len(set(self.include_family_ids)) != len(self.include_family_ids):
                raise ValueError("include_family_ids must be unique")
        if any(not value.strip() for value in self.exclude_family_ids):
            raise ValueError("exclude_family_ids entries cannot be empty")
        if len(set(self.exclude_family_ids)) != len(self.exclude_family_ids):
            raise ValueError("exclude_family_ids must be unique")
        if self.include_family_ids != "all":
            overlap = set(self.include_family_ids).intersection(self.exclude_family_ids)
            if overlap:
                raise ValueError(
                    "family IDs cannot be both included and excluded: " + ", ".join(sorted(overlap))
                )
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
    protocol: RevisionProtocolConfig | None = None

    @model_validator(mode="after")
    def validate_revision_protocol(self) -> AppConfig:
        protocol = self.protocol
        if protocol is None:
            return self
        if self.runtime.output_root.name != protocol.artifact_namespace:
            raise ValueError(
                "runtime.output_root must end in protocol.artifact_namespace: "
                f"{protocol.artifact_namespace!r}"
            )
        active_mask_seeds = protocol.mask_seeds.active(protocol.active_mask_partition)
        if tuple(self.experiment.seeds) != tuple(active_mask_seeds):
            raise ValueError("experiment.seeds must equal the active protocol mask seed partition")
        if self.experiment.router_seed is None:
            raise ValueError("revision protocols require experiment.router_seed")
        if self.experiment.router_seed not in protocol.router_seed_roots:
            raise ValueError("experiment.router_seed must be one of protocol.router_seed_roots")
        expected_mask_cells = (
            len(self.experiment.missing_mechanisms)
            * len(self.experiment.missing_rates)
            * len(active_mask_seeds)
        )
        episode_cap = (
            self.experiment.max_train_episodes_per_dataset
            if protocol.active_mask_partition == "train"
            else self.experiment.max_eval_episodes_per_dataset
        )
        if episode_cap is not None and episode_cap < expected_mask_cells:
            raise ValueError(
                "revision protocol episode cap must cover every configured mechanism-rate-seed cell"
            )
        return self


def validate_forecaster_revision_binding(
    config: AppConfig,
    forecaster_id: str,
    artifact: str | Path | None,
) -> None:
    """Bind a held-out forecaster run to the revision declared by R2."""

    protocol = config.protocol
    if protocol is None or forecaster_id != protocol.held_out_forecaster_id:
        return
    if artifact is None:
        raise ValueError(
            f"held-out forecaster {forecaster_id!r} requires a local revision-bound artifact"
        )
    revision = protocol.held_out_forecaster_revision
    if revision is None:  # Protected by RevisionProtocolConfig validation.
        raise ValueError("held-out forecaster revision is missing")
    resolved = Path(artifact).resolve()
    if revision.casefold() not in {part.casefold() for part in resolved.parts}:
        raise ValueError(
            f"held-out forecaster artifact is not under declared revision {revision!r}: {resolved}"
        )


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
