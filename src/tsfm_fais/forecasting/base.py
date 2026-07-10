"""Native forecast adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastAdapterSpec:
    model_id: str
    mode: Literal["joint_multivariate", "independent_univariate"]
    factory: str
    model_name: str
    optional_extra: str
    max_context: int
    device: str = "any"
    output_type: Literal["point", "quantile", "sample"] = "quantile"
    default_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeForecast:
    point: np.ndarray
    quantiles: np.ndarray | None = None
    samples: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastAdapter(Protocol):
    spec: ForecastAdapterSpec

    def predict_native(
        self,
        contexts: np.ndarray,
        horizon: int,
        quantile_levels: Sequence[float],
        num_samples: int,
    ) -> NativeForecast: ...


@dataclass(frozen=True)
class ForecastCapabilities:
    """Capabilities exposed by optional SDK-specific adapters."""

    modes: frozenset[str]
    max_context: int


class LazyForecastAdapter:
    """Shared validation and lazy backend loading for model adapters."""

    model_id = "unknown"
    capabilities = ForecastCapabilities(frozenset(), 0)

    def __init__(self, *, backend: Any | None = None) -> None:
        self._backend = backend

    @property
    def backend(self) -> Any | None:
        return self._backend

    @backend.setter
    def backend(self, value: Any | None) -> None:
        self._backend = value

    def _load_backend(self) -> Any:  # pragma: no cover - implemented by adapters
        raise NotImplementedError

    def _ensure_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend

    def predict(self, contexts: np.ndarray, spec: Any):
        values = np.asarray(contexts, dtype=float)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3:
            raise ValueError("contexts must have shape [N,L,D]")
        if not np.isfinite(values).all():
            raise ValueError("forecast context must be complete and finite")
        if spec.mode not in self.capabilities.modes:
            raise ValueError(f"{self.model_id} does not support mode {spec.mode}")
        if self.capabilities.max_context > 0:
            values = values[:, -self.capabilities.max_context :, :]
        targets = spec.target_indices or tuple(range(values.shape[2]))
        if any(target < 0 or target >= values.shape[2] for target in targets):
            raise ValueError("target index is outside the context dimensions")
        return self._predict(values, spec, tuple(targets))

    def _predict(self, contexts: np.ndarray, spec: Any, targets: tuple[int, ...]):
        raise NotImplementedError
