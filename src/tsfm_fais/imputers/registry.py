"""Extensible registry for the fixed default candidate pool."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from tsfm_fais.contracts import ImputerSpec

from .base import ImputerProtocol
from .pypots import TRMF_FROZEN_PROTOCOL_BLOCKER, TRMF_FROZEN_PROTOCOL_REASON


@dataclass(frozen=True)
class DependencyAvailability:
    available: bool
    missing: tuple[str, ...] = ()
    reason: str | None = None


def _spec(
    imputer_id: str,
    family: str,
    mode: str,
    factory: str,
    *,
    fit_scope: str = "none",
    supports_tail: bool = True,
    requires_period: bool = False,
    stochastic: bool = False,
    device: str = "cpu",
    cost_tier: int = 1,
    optional_extra: str | None = None,
    dependencies: tuple[str, ...] = ("numpy",),
    default_params: dict[str, Any] | None = None,
) -> ImputerSpec:
    return ImputerSpec(
        imputer_id=imputer_id,
        family=family,
        mode=mode,  # type: ignore[arg-type]
        factory=factory,
        source="project",
        fit_scope=fit_scope,  # type: ignore[arg-type]
        supports_tail=supports_tail,
        requires_period=requires_period,
        stochastic=stochastic,
        device=device,  # type: ignore[arg-type]
        cost_tier=cost_tier,
        optional_extra=optional_extra,
        dependencies=dependencies,
        default_params=default_params or {},
    )


IMPUTER_SPECS: tuple[ImputerSpec, ...] = (
    _spec(
        "locf",
        "persistence",
        "per_channel",
        "tsfm_fais.imputers.classical:LOCFImputer",
    ),
    _spec(
        "linear_interp",
        "interpolation",
        "per_channel",
        "tsfm_fais.imputers.classical:LinearInterpolationImputer",
        supports_tail=False,
    ),
    _spec(
        "seasonal_lag",
        "seasonal",
        "per_channel",
        "tsfm_fais.imputers.classical:SeasonalLagImputer",
        requires_period=True,
        default_params={"period": 24},
    ),
    _spec(
        "kalman_local_trend",
        "state_space",
        "per_channel",
        "tsfm_fais.imputers.classical:KalmanLocalTrendImputer",
        cost_tier=2,
    ),
    _spec(
        "kalman_ar",
        "state_space",
        "per_channel",
        "tsfm_fais.imputers.classical:KalmanARImputer",
        cost_tier=2,
        default_params={"max_lag": 3},
    ),
    _spec(
        "stl_kalman",
        "decomposition",
        "per_channel",
        "tsfm_fais.imputers.classical:STLKalmanImputer",
        requires_period=True,
        cost_tier=2,
        dependencies=("numpy", "statsmodels"),
        default_params={"period": 24},
    ),
    _spec(
        "gp_rbf",
        "gaussian_process",
        "per_channel",
        "tsfm_fais.imputers.classical:GPRBFImputer",
        cost_tier=3,
        default_params={"max_train_points": 512, "noise": 0.0001},
    ),
    _spec(
        "knn_multivariate",
        "neighbor",
        "joint_multivariate",
        "tsfm_fais.imputers.structured:KNNMultivariateImputer",
        fit_scope="dataset",
        cost_tier=2,
        dependencies=("numpy", "sklearn"),
        default_params={"n_neighbors": 5, "weights": "distance"},
    ),
    _spec(
        "mice",
        "chained_regression",
        "joint_multivariate",
        "tsfm_fais.imputers.structured:MICEImputer",
        fit_scope="dataset",
        cost_tier=2,
        stochastic=True,
        dependencies=("numpy", "sklearn"),
        default_params={"max_iter": 10, "random_state": 0},
    ),
    _spec(
        "missforest",
        "forest",
        "joint_multivariate",
        "tsfm_fais.imputers.structured:MissForestImputer",
        fit_scope="dataset",
        cost_tier=3,
        stochastic=True,
        dependencies=("numpy", "sklearn"),
        default_params={"n_estimators": 100, "max_iter": 10, "random_state": 0},
    ),
    _spec(
        "softimpute",
        "low_rank",
        "joint_multivariate",
        "tsfm_fais.imputers.structured:SoftImputeImputer",
        fit_scope="dataset",
        cost_tier=2,
        dependencies=("numpy", "scipy"),
        default_params={"max_iter": 100, "tolerance": 1e-5},
    ),
    _spec(
        "trmf",
        "low_rank_temporal",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:TRMFImputer",
        fit_scope="dataset",
        cost_tier=3,
        stochastic=True,
        device="cpu",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "brits",
        "recurrent",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:BRITSImputer",
        fit_scope="dataset",
        cost_tier=3,
        stochastic=True,
        device="any",
        supports_tail=False,
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "gpvae",
        "probabilistic_latent",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:GPVAEImputer",
        fit_scope="dataset",
        cost_tier=4,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "saits",
        "attention",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:SAITSImputer",
        fit_scope="dataset",
        cost_tier=3,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "csdi",
        "diffusion",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:CSDIImputer",
        fit_scope="dataset",
        cost_tier=5,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32, "num_samples": 20},
    ),
    _spec(
        "imputeformer",
        "spatiotemporal_attention",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:ImputeFormerImputer",
        fit_scope="dataset",
        cost_tier=4,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "helix",
        "cross_dimensional",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:HELIXImputer",
        fit_scope="dataset",
        cost_tier=4,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "timemixerpp",
        "multiscale",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:TimeMixerPPImputer",
        fit_scope="dataset",
        cost_tier=4,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
    _spec(
        "totem",
        "tokenized",
        "joint_multivariate",
        "tsfm_fais.imputers.pypots:TOTEMImputer",
        fit_scope="dataset",
        cost_tier=4,
        stochastic=True,
        device="any",
        optional_extra="deep-imputers",
        dependencies=("pypots", "torch"),
        default_params={"epochs": 10, "batch_size": 32},
    ),
)

IMPUTER_IDS: tuple[str, ...] = tuple(spec.imputer_id for spec in IMPUTER_SPECS)


class ImputerRegistry:
    def __init__(self, specs: Iterable[ImputerSpec] = ()) -> None:
        self._specs: dict[str, ImputerSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ImputerSpec, *, replace: bool = False) -> None:
        if spec.imputer_id in self._specs and not replace:
            raise ValueError(f"duplicate imputer ID: {spec.imputer_id}")
        self._specs[spec.imputer_id] = spec

    def get(self, imputer_id: str) -> ImputerSpec:
        try:
            return self._specs[imputer_id]
        except KeyError as error:
            raise KeyError(
                f"unknown imputer {imputer_id!r}; available: {', '.join(self.ids)}"
            ) from error

    get_spec = get

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ImputerSpec, ...]:
        return tuple(self._specs.values())

    def availability(self, imputer_id: str) -> DependencyAvailability:
        spec = self.get(imputer_id)
        missing: list[str] = []
        for dependency in spec.dependencies:
            try:
                found = importlib.util.find_spec(dependency) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                found = False
            if not found:
                missing.append(dependency)
        if imputer_id == "trmf":
            # Keep a blocker token in ``missing`` because existing stage
            # manifests persist that field when skipping unavailable methods.
            # The prefix distinguishes this protocol incompatibility from a
            # package that can be installed.
            missing.append(TRMF_FROZEN_PROTOCOL_BLOCKER)
            return DependencyAvailability(
                False,
                tuple(missing),
                TRMF_FROZEN_PROTOCOL_REASON,
            )
        return DependencyAvailability(not missing, tuple(missing))

    def create(self, imputer_id: str, **overrides: Any) -> ImputerProtocol:
        spec = self.get(imputer_id)
        module_name, separator, attribute_path = spec.factory.partition(":")
        if not separator or not module_name or not attribute_path:
            raise ValueError(f"invalid imputer factory: {spec.factory}")
        module = importlib.import_module(module_name)
        factory: Any = module
        for attribute in attribute_path.split("."):
            factory = getattr(factory, attribute)
        params = {**dict(spec.default_params), **overrides}
        imputer = factory(**params)
        if not isinstance(imputer, ImputerProtocol):
            raise TypeError(f"factory {spec.factory} did not return an ImputerProtocol")
        if imputer.imputer_id != imputer_id:
            raise ValueError(
                f"factory {spec.factory} returned candidate {imputer.imputer_id!r}"
            )
        return imputer

    def __contains__(self, imputer_id: object) -> bool:
        return imputer_id in self._specs

    def __iter__(self) -> Iterator[ImputerSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


DEFAULT_REGISTRY = ImputerRegistry(IMPUTER_SPECS)
DEFAULT_IMPUTER_REGISTRY = DEFAULT_REGISTRY


def get_imputer_spec(imputer_id: str) -> ImputerSpec:
    return DEFAULT_REGISTRY.get(imputer_id)


def list_imputer_specs() -> tuple[ImputerSpec, ...]:
    return DEFAULT_REGISTRY.specs()


def create_imputer(imputer_id: str, **overrides: Any) -> ImputerProtocol:
    return DEFAULT_REGISTRY.create(imputer_id, **overrides)


def dependency_status(imputer_id: str) -> DependencyAvailability:
    return DEFAULT_REGISTRY.availability(imputer_id)


__all__ = [
    "DEFAULT_IMPUTER_REGISTRY",
    "DEFAULT_REGISTRY",
    "DependencyAvailability",
    "IMPUTER_IDS",
    "IMPUTER_SPECS",
    "ImputerRegistry",
    "create_imputer",
    "dependency_status",
    "get_imputer_spec",
    "list_imputer_specs",
]
