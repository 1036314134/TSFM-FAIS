"""Strict schemas and cross-file checks for project registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_fais.config import AppConfig, load_yaml


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImputerEntry(_StrictModel):
    id: str
    family: str
    mode: Literal["per_channel", "joint_multivariate"]
    fit_scope: Literal["none", "dataset", "online"]
    supports_tail: bool
    requires_period: bool
    stochastic: bool
    device: Literal["cpu", "gpu", "any"]
    cost_tier: int = Field(ge=1)
    optional_extra: str | None = None
    factory: str | None = None
    dependencies: tuple[str, ...] = ()
    default_params: dict[str, Any] = Field(default_factory=dict)


class ImputerPoolConfig(_StrictModel):
    schema_version: Literal[1] = 1
    imputers: tuple[ImputerEntry, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> "ImputerPoolConfig":
        ids = [entry.id for entry in self.imputers]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("imputer IDs must be non-empty and unique")
        return self


class ForecasterEntry(_StrictModel):
    id: str
    mode: Literal["joint_multivariate", "independent_univariate"]
    model_name: str
    max_context: int = Field(ge=2)
    output_type: Literal["point", "quantile", "sample"]
    optional_extra: str
    factory: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)


class ForecasterPoolConfig(_StrictModel):
    schema_version: Literal[1] = 1
    forecasters: tuple[ForecasterEntry, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> "ForecasterPoolConfig":
        ids = [entry.id for entry in self.forecasters]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("forecaster IDs must be non-empty and unique")
        return self


class FallbackConfig(_StrictModel):
    internal: tuple[str, ...]
    tail: tuple[str, ...]

    @model_validator(mode="after")
    def validate_sequences(self) -> "FallbackConfig":
        if not self.internal or not self.tail:
            raise ValueError("fallback sequences cannot be empty")
        if len(set(self.internal)) != len(self.internal) or len(set(self.tail)) != len(
            self.tail
        ):
            raise ValueError("fallback sequences must not contain duplicates")
        return self


class RouterConfig(_StrictModel):
    schema_version: Literal[1] = 1
    shortlist_size: int = Field(ge=1)
    forced_candidates: tuple[str, ...]
    pseudo_blocks: int = Field(ge=1, le=8)
    beam_width: int = Field(ge=1)
    beta: float = Field(default=1.0, ge=0)
    cost_weight: float = Field(default=0.05, ge=0)
    beta_grid: tuple[float, ...]
    cost_weight_grid: tuple[float, ...]
    prior_model: Literal["lightgbm_lambdarank"]
    unary_model: Literal["lightgbm_lambdarank"]
    pairwise_model: Literal["lightgbm_huber"]
    fallback: FallbackConfig

    @model_validator(mode="after")
    def validate_values(self) -> "RouterConfig":
        for name, values in (
            ("forced_candidates", self.forced_candidates),
            ("beta_grid", self.beta_grid),
            ("cost_weight_grid", self.cost_weight_grid),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
        if any(value < 0 for value in (*self.beta_grid, *self.cost_weight_grid)):
            raise ValueError("router weights must be non-negative")
        if self.beta not in self.beta_grid or self.cost_weight not in self.cost_weight_grid:
            raise ValueError("selected router weights must be members of their grids")
        if len(self.forced_candidates) > self.shortlist_size:
            raise ValueError("forced_candidates cannot exceed shortlist_size")
        return self


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


def validate_project_configuration(config: AppConfig) -> dict[str, Any]:
    """Validate every referenced registry without reading dataset values."""

    for label, path in (
        ("data manifest", config.registries.data_manifest),
        ("imputer registry", config.registries.imputer_registry),
        ("forecaster registry", config.registries.forecaster_registry),
        ("router config", config.registries.router_config),
    ):
        _require_file(path, label)

    # Imported lazily because the data catalog itself reuses config.load_yaml.
    from tsfm_fais.data.catalog import load_manifest

    manifest = load_manifest(config.registries.data_manifest)
    imputers = ImputerPoolConfig.model_validate(
        load_yaml(config.registries.imputer_registry)
    )
    forecasters = ForecasterPoolConfig.model_validate(
        load_yaml(config.registries.forecaster_registry)
    )
    router = RouterConfig.model_validate(load_yaml(config.registries.router_config))

    from tsfm_fais.forecasting import default_forecast_registry
    from tsfm_fais.imputers import DEFAULT_REGISTRY

    runtime_imputers = {entry.imputer_id: entry for entry in DEFAULT_REGISTRY.specs()}
    declared_imputers = {entry.id: entry for entry in imputers.imputers}
    if set(declared_imputers) != set(runtime_imputers):
        raise ValueError(
            "imputer YAML IDs do not match the executable registry: "
            f"declared_only={sorted(set(declared_imputers) - set(runtime_imputers))}, "
            f"runtime_only={sorted(set(runtime_imputers) - set(declared_imputers))}"
        )
    imputer_fields = (
        "family",
        "mode",
        "fit_scope",
        "supports_tail",
        "requires_period",
        "stochastic",
        "device",
        "cost_tier",
        "optional_extra",
    )
    for imputer_id, declared in declared_imputers.items():
        runtime = runtime_imputers[imputer_id]
        mismatched = [
            field
            for field in imputer_fields
            if getattr(declared, field) != getattr(runtime, field)
        ]
        if mismatched:
            raise ValueError(
                f"imputer {imputer_id!r} metadata differs from runtime fields: {mismatched}"
            )

    runtime_forecasters = {
        entry.model_id: entry for entry in default_forecast_registry().specs()
    }
    declared_forecasters = {entry.id: entry for entry in forecasters.forecasters}
    if set(declared_forecasters) != set(runtime_forecasters):
        raise ValueError(
            "forecaster YAML IDs do not match the executable registry: "
            f"declared_only={sorted(set(declared_forecasters) - set(runtime_forecasters))}, "
            f"runtime_only={sorted(set(runtime_forecasters) - set(declared_forecasters))}"
        )
    forecaster_fields = (
        "mode",
        "model_name",
        "max_context",
        "output_type",
        "optional_extra",
    )
    for forecaster_id, declared_forecaster in declared_forecasters.items():
        runtime_forecaster = runtime_forecasters[forecaster_id]
        mismatched = [
            field
            for field in forecaster_fields
            if getattr(declared_forecaster, field)
            != getattr(runtime_forecaster, field)
        ]
        if mismatched:
            raise ValueError(
                f"forecaster {forecaster_id!r} metadata differs from runtime fields: "
                f"{mismatched}"
            )

    imputer_ids = set(declared_imputers)
    missing_forced = set(router.forced_candidates) - imputer_ids
    if missing_forced:
        raise ValueError(
            f"router forced candidates are not registered: {sorted(missing_forced)}"
        )
    fallback_ids = set(router.fallback.internal) | set(router.fallback.tail)
    allowed_fallbacks = imputer_ids | {"train_median"}
    if unknown := fallback_ids - allowed_fallbacks:
        raise ValueError(f"router fallback candidates are not registered: {sorted(unknown)}")
    if config.experiment.target_indices != "all":
        for dataset in manifest.datasets:
            if not dataset.enabled or dataset.expected_num_variates is None:
                continue
            invalid = [
                index
                for index in config.experiment.target_indices
                if index >= dataset.expected_num_variates
            ]
            if invalid:
                raise ValueError(
                    f"target indices {invalid} are invalid for dataset {dataset.dataset_id} "
                    f"with D={dataset.expected_num_variates}"
                )
    return {
        "datasets": manifest,
        "imputers": imputers,
        "forecasters": forecasters,
        "router": router,
    }


__all__ = [
    "ForecasterPoolConfig",
    "ImputerPoolConfig",
    "RouterConfig",
    "validate_project_configuration",
]
