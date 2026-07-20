"""Strict schemas and cross-file checks for project registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
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
    max_fit_variates: int | None = Field(default=None, ge=2)
    optional_extra: str | None = None
    factory: str | None = None
    dependencies: tuple[str, ...] = ()
    default_params: dict[str, Any] = Field(default_factory=dict)


class ImputerPoolConfig(_StrictModel):
    schema_version: Literal[1] = 1
    imputers: tuple[ImputerEntry, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> ImputerPoolConfig:
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
    def validate_ids(self) -> ForecasterPoolConfig:
        ids = [entry.id for entry in self.forecasters]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("forecaster IDs must be non-empty and unique")
        return self


class FallbackConfig(_StrictModel):
    internal: tuple[str, ...]
    tail: tuple[str, ...]

    @model_validator(mode="after")
    def validate_sequences(self) -> FallbackConfig:
        if not self.internal or not self.tail:
            raise ValueError("fallback sequences cannot be empty")
        if len(set(self.internal)) != len(self.internal) or len(set(self.tail)) != len(self.tail):
            raise ValueError("fallback sequences must not contain duplicates")
        return self


class EvidenceBlendWeights(_StrictModel):
    """Convex inference weights selected on a declared tuning family."""

    r0: float = Field(ge=0, le=1)
    r1: float = Field(ge=0, le=1)
    proxy: float = Field(ge=0, le=1)
    global_prior: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> EvidenceBlendWeights:
        total = self.r0 + self.r1 + self.proxy + self.global_prior
        if abs(total - 1.0) > 1e-9:
            raise ValueError("evidence blend weights must sum to one")
        return self


class ForecastConsensusConfig(_StrictModel):
    """Optional label-free downstream forecast agreement evidence."""

    mode: Literal[
        "disabled",
        "medoid",
        "historical_backtest",
        "value_median",
        "value_topk_mean",
        "router_risk",
        "proxy_min",
    ] = "disabled"
    candidates: tuple[str, ...] = ()
    ensemble_top_k: int = Field(default=2, ge=2)
    ensemble_third_relative_gap: float | None = Field(default=None, ge=0.0)
    ensemble_proxy_weight_power: float = Field(default=0.0, ge=0.0)
    pseudo_weight_calibration: Literal["disabled", "convex_l2"] = "disabled"
    pseudo_weight_prior_strength: float = Field(default=8.0, ge=0.0)
    pseudo_weight_min_points: int = Field(default=4, ge=1)
    proxy_blend_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    proxy_blend_weight_by_model: dict[str, float] = Field(default_factory=dict)
    proxy_blend_min_relative_margin: float = Field(default=0.0, ge=0.0)
    proxy_blend_min_relative_margin_by_model: dict[str, float] = Field(default_factory=dict)
    candidate_shrinkage_id: str | None = None
    candidate_shrinkage_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_shrinkage_fallback_id: str | None = None
    candidate_shrinkage_fallback_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    dataset_prior_candidates: int = Field(default=0, ge=0)
    dataset_prior_candidates_by_model: dict[str, int] = Field(default_factory=dict)
    prior_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_weight_by_model: dict[str, float] = Field(default_factory=dict)
    prior_override_max_medoid_penalty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    prior_override_max_medoid_penalty_by_model: dict[str, float] = Field(default_factory=dict)
    prior_override_min_margin: float = Field(default=0.0, ge=0.0)
    prior_override_min_margin_by_model: dict[str, float] = Field(default_factory=dict)
    anchor_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    anchor_weight_by_model: dict[str, float] = Field(default_factory=dict)
    anchor_max_period_ratio: float | None = Field(default=None, gt=0.0)
    anchor_max_period_ratio_by_model: dict[str, float] = Field(default_factory=dict)
    anchor_period_exceeded_mode: Literal["disabled", "medoid"] = "disabled"
    selection_granularity: Literal["episode", "target"] = "episode"
    selection_granularity_by_model: dict[str, Literal["episode", "target"]] = Field(
        default_factory=dict
    )
    context_mode: Literal[
        "native",
        "targets_only",
        "targets_with_correlates",
    ] = "native"
    max_context_variates: int = Field(default=8, ge=2)
    validation_length: int = Field(default=24, ge=1)
    min_observed_per_target: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_values(self) -> ForecastConsensusConfig:
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("forecast consensus candidates must be unique")
        if any(
            not model_id or count < 0
            for model_id, count in self.dataset_prior_candidates_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific candidate counts are invalid")
        if any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in self.prior_weight_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific prior weights are invalid")
        if any(
            not model_id or not 0.0 <= penalty <= 1.0
            for model_id, penalty in self.prior_override_max_medoid_penalty_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific override penalties are invalid")
        if any(
            not model_id or not np.isfinite(margin) or margin < 0.0
            for model_id, margin in self.prior_override_min_margin_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific override margins are invalid")
        if any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in self.anchor_weight_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific anchor weights are invalid")
        if any(
            not model_id or not 0.0 <= weight <= 1.0
            for model_id, weight in self.proxy_blend_weight_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific proxy blend weights are invalid")
        if any(
            not model_id or not np.isfinite(margin) or margin < 0.0
            for model_id, margin in self.proxy_blend_min_relative_margin_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific proxy margins are invalid")
        if any(
            not model_id or not np.isfinite(ratio) or ratio <= 0.0
            for model_id, ratio in self.anchor_max_period_ratio_by_model.items()
        ):
            raise ValueError("forecast consensus model-specific period ratios are invalid")
        if any(not model_id for model_id in self.selection_granularity_by_model):
            raise ValueError("forecast consensus granularity model IDs are invalid")
        if self.mode in {
            "medoid",
            "historical_backtest",
            "value_median",
            "value_topk_mean",
            "router_risk",
            "proxy_min",
        } and (len(self.candidates) + self.dataset_prior_candidates < 2):
            raise ValueError("forecast consensus selection requires at least two candidates")
        if self.mode == "disabled" and (
            self.candidates
            or self.dataset_prior_candidates
            or self.dataset_prior_candidates_by_model
            or self.prior_weight
            or self.prior_weight_by_model
            or self.prior_override_max_medoid_penalty is not None
            or self.prior_override_max_medoid_penalty_by_model
            or self.prior_override_min_margin
            or self.prior_override_min_margin_by_model
            or self.anchor_weight != 1.0
            or self.anchor_weight_by_model
            or self.anchor_max_period_ratio is not None
            or self.anchor_max_period_ratio_by_model
            or self.selection_granularity != "episode"
            or self.selection_granularity_by_model
            or self.ensemble_top_k != 2
            or self.ensemble_third_relative_gap is not None
            or self.ensemble_proxy_weight_power
            or self.pseudo_weight_calibration != "disabled"
            or self.proxy_blend_weight
            or self.proxy_blend_weight_by_model
            or self.proxy_blend_min_relative_margin
            or self.proxy_blend_min_relative_margin_by_model
            or self.candidate_shrinkage_id is not None
            or self.candidate_shrinkage_weight
            or self.candidate_shrinkage_fallback_id is not None
            or self.candidate_shrinkage_fallback_weight
        ):
            raise ValueError("disabled forecast consensus cannot configure candidates")
        if (self.candidate_shrinkage_id is None) != (self.candidate_shrinkage_weight == 0.0):
            raise ValueError("candidate shrinkage requires both a candidate ID and positive weight")
        if (
            self.candidate_shrinkage_id is not None
            and self.candidate_shrinkage_id not in self.candidates
        ):
            raise ValueError(
                "candidate shrinkage ID must be included in forecast consensus candidates"
            )
        if (self.candidate_shrinkage_fallback_id is None) != (
            self.candidate_shrinkage_fallback_weight == 0.0
        ):
            raise ValueError("candidate shrinkage fallback requires both an ID and positive weight")
        if self.candidate_shrinkage_fallback_id is not None:
            if self.candidate_shrinkage_id is None:
                raise ValueError("candidate shrinkage fallback requires primary shrinkage")
            if self.candidate_shrinkage_fallback_id == self.candidate_shrinkage_id:
                raise ValueError("candidate shrinkage fallback must differ from primary shrinkage")
            if self.candidate_shrinkage_fallback_id not in self.candidates:
                raise ValueError(
                    "candidate shrinkage fallback ID must be included in forecast "
                    "consensus candidates"
                )
        if self.ensemble_third_relative_gap is not None and (
            self.mode != "value_topk_mean" or self.ensemble_top_k != 2
        ):
            raise ValueError("third-candidate gap gating requires value_topk_mean with top-k two")
        if self.ensemble_proxy_weight_power and self.mode != "value_topk_mean":
            raise ValueError("proxy-weighted candidate aggregation requires value_topk_mean")
        proxy_blend_configured = bool(
            self.proxy_blend_weight or any(self.proxy_blend_weight_by_model.values())
        )
        calibrated_top_two = bool(
            self.mode == "value_topk_mean"
            and self.ensemble_top_k == 2
            and self.ensemble_third_relative_gap is None
        )
        if self.pseudo_weight_calibration != "disabled" and not (
            proxy_blend_configured or calibrated_top_two
        ):
            raise ValueError(
                "pseudo-weight calibration requires a proxy blend or fixed top-two mean"
            )
        if self.pseudo_weight_calibration != "disabled" and self.ensemble_proxy_weight_power:
            raise ValueError(
                "pseudo-weight calibration and inverse-error weighting are mutually exclusive"
            )
        if self.mode == "historical_backtest" and self.context_mode != "native":
            raise ValueError("historical forecast consensus requires native context")
        if self.anchor_period_exceeded_mode == "medoid" and (
            self.mode not in {"router_risk", "proxy_min"}
            or (self.anchor_max_period_ratio is None and not self.anchor_max_period_ratio_by_model)
        ):
            raise ValueError("period-exceeded medoid requires a gated signal mode")
        if (
            self.selection_granularity == "target"
            or "target" in self.selection_granularity_by_model.values()
        ) and self.mode not in {"medoid", "value_topk_mean"}:
            raise ValueError(
                "target-level forecast consensus requires medoid or value_topk_mean mode"
            )
        return self


def _validate_selector_param_value(value: Any, path: str) -> None:
    """Keep serialized selector parameters deterministic and JSON-compatible."""

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, entry in enumerate(value):
            _validate_selector_param_value(entry, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, entry in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            _validate_selector_param_value(entry, f"{path}.{key}")
        return
    raise ValueError(f"{path} must be JSON-compatible")


class RouterConfig(_StrictModel):
    schema_version: Literal[1] = 1
    selector_methods: tuple[str, ...] = ("block_fais",)
    selector_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    shortlist_size: int = Field(ge=1)
    forced_candidates: tuple[str, ...]
    pseudo_blocks: int = Field(ge=1, le=8)
    beam_width: int = Field(ge=1)
    beta: float = Field(default=1.0, ge=0)
    cost_weight: float = Field(default=0.0, ge=0)
    beta_grid: tuple[float, ...]
    cost_weight_grid: tuple[float, ...]
    prior_model: Literal["lightgbm_lambdarank"]
    unary_model: Literal["lightgbm_lambdarank"]
    pairwise_model: Literal["lightgbm_huber"]
    ranker_target: Literal[
        "forecast_loss",
        "full_candidate_loss",
        "routing_target",
    ] = "full_candidate_loss"
    evidence_tuning_family: str | None = None
    evidence_blend: dict[str, EvidenceBlendWeights] = Field(default_factory=dict)
    candidate_prior_min_support: int = Field(default=4, ge=1)
    candidate_prior_recent_origin_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    candidate_switch_penalty: float = Field(default=0.0, ge=0)
    forecast_consensus: ForecastConsensusConfig = Field(default_factory=ForecastConsensusConfig)
    fallback: FallbackConfig

    @model_validator(mode="after")
    def validate_values(self) -> RouterConfig:
        from tsfm_fais.routing.baselines import (
            BASELINE_SELECTOR_METHODS,
            BASELINE_SELECTOR_PARAM_NAMES,
        )

        supported_selectors = {"block_fais", *BASELINE_SELECTOR_METHODS}
        if not self.selector_methods or len(set(self.selector_methods)) != len(
            self.selector_methods
        ):
            raise ValueError("selector_methods must be non-empty and unique")
        unknown_selectors = set(self.selector_methods).difference(supported_selectors)
        if unknown_selectors:
            raise ValueError(
                "unsupported selector methods: " + ", ".join(sorted(unknown_selectors))
            )
        unknown_param_methods = set(self.selector_params).difference(supported_selectors)
        if unknown_param_methods:
            raise ValueError(
                "selector_params contains unsupported methods: "
                + ", ".join(sorted(unknown_param_methods))
            )
        unselected_param_methods = set(self.selector_params).difference(self.selector_methods)
        if unselected_param_methods:
            raise ValueError(
                "selector_params contains unselected methods: "
                + ", ".join(sorted(unselected_param_methods))
            )
        for method, params in self.selector_params.items():
            allowed_params = BASELINE_SELECTOR_PARAM_NAMES.get(method, frozenset())
            unknown_params = set(params).difference(allowed_params)
            if unknown_params:
                raise ValueError(
                    f"selector_params.{method} contains unsupported parameters: "
                    + ", ".join(sorted(unknown_params))
                )
            for name, value in params.items():
                if not name.strip():
                    raise ValueError(f"selector_params.{method} contains an empty key")
                _validate_selector_param_value(value, f"selector_params.{method}.{name}")
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
        # ``forecast_consensus.candidates`` is an eligibility pool.  The
        # pipeline intersects it with the already budgeted shortlist before
        # making forecasts, so only forced candidates reserve shortlist slots.
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
    imputers = ImputerPoolConfig.model_validate(load_yaml(config.registries.imputer_registry))
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
        "max_fit_variates",
        "optional_extra",
    )
    for imputer_id, declared in declared_imputers.items():
        runtime = runtime_imputers[imputer_id]
        mismatched = [
            field for field in imputer_fields if getattr(declared, field) != getattr(runtime, field)
        ]
        if mismatched:
            raise ValueError(
                f"imputer {imputer_id!r} metadata differs from runtime fields: {mismatched}"
            )

    runtime_forecasters = {entry.model_id: entry for entry in default_forecast_registry().specs()}
    declared_forecasters = {entry.id: entry for entry in forecasters.forecasters}
    if set(declared_forecasters) != set(runtime_forecasters):
        raise ValueError(
            "forecaster YAML IDs do not match the executable registry: "
            f"declared_only={sorted(set(declared_forecasters) - set(runtime_forecasters))}, "
            f"runtime_only={sorted(set(runtime_forecasters) - set(declared_forecasters))}"
        )
    unknown_blend_models = set(router.evidence_blend) - set(runtime_forecasters)
    if unknown_blend_models:
        raise ValueError(
            f"router evidence blend models are not registered: {sorted(unknown_blend_models)}"
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
            if getattr(declared_forecaster, field) != getattr(runtime_forecaster, field)
        ]
        if mismatched:
            raise ValueError(
                f"forecaster {forecaster_id!r} metadata differs from runtime fields: {mismatched}"
            )

    imputer_ids = set(declared_imputers)
    missing_forced = set(router.forced_candidates) - imputer_ids
    if missing_forced:
        raise ValueError(f"router forced candidates are not registered: {sorted(missing_forced)}")
    missing_consensus = set(router.forecast_consensus.candidates) - imputer_ids
    if missing_consensus:
        raise ValueError(
            f"router forecast-consensus candidates are not registered: {sorted(missing_consensus)}"
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
