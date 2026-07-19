"""Capability-aware TSFM adapter registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Literal, cast

from .base import ForecastAdapter, ForecastAdapterSpec


def _import_factory(path: str) -> Callable[..., ForecastAdapter]:
    module_name, _, attribute = path.rpartition(":")
    if not module_name or not attribute:
        raise ValueError(f"factory must use module:attribute syntax: {path}")
    return getattr(importlib.import_module(module_name), attribute)


class ForecastRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ForecastAdapterSpec] = {}

    def register(self, spec: ForecastAdapterSpec) -> None:
        if spec.model_id in self._specs:
            raise ValueError(f"duplicate forecast model id: {spec.model_id}")
        self._specs[spec.model_id] = spec

    def specs(self) -> tuple[ForecastAdapterSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))

    def get(self, model_id: str) -> ForecastAdapterSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown forecast model: {model_id}") from exc

    def build(self, model_id: str, **overrides: Any) -> ForecastAdapter:
        spec = self.get(model_id)
        params = dict(spec.default_params)
        params.update(overrides)
        params.setdefault("model_name", spec.model_name)
        adapter = _import_factory(spec.factory)(**params)
        adapter.spec = spec
        return adapter


def default_forecast_registry() -> ForecastRegistry:
    registry = ForecastRegistry()
    module = "tsfm_fais.forecasting.adapters"
    registry.register(
        ForecastAdapterSpec(
            model_id="chronos2",
            mode="joint_multivariate",
            factory=f"{module}:Chronos2Adapter",
            model_name="amazon/chronos-2",
            optional_extra="forecast-chronos",
            max_context=8192,
        )
    )
    for model_id, factory, model_name, extra, max_context, output_type in (
        ("timesfm2p5", "TimesFM2p5Adapter", "google/timesfm-2.5-200m-pytorch", "forecast-timesfm", 4096, "quantile"),
        ("chronosbolt", "ChronosBoltAdapter", "amazon/chronos-bolt-base", "forecast-chronos", 2048, "quantile"),
        ("sundial", "SundialAdapter", "thuml/sundial-base-128m", "forecast-sundial", 2880, "sample"),
        ("tirex", "TiRexAdapter", "NX-AI/TiRex", "forecast-tirex", 2048, "quantile"),
    ):
        registry.register(
            ForecastAdapterSpec(
                model_id=model_id,
                mode="independent_univariate",
                factory=f"{module}:{factory}",
                model_name=model_name,
                optional_extra=extra,
                max_context=max_context,
                output_type=cast(
                    Literal["point", "quantile", "sample"], output_type
                ),
            )
        )
    return registry
