"""Mode expansion and canonical forecast shape normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsfm_fais.contracts import ForecastResult, ForecastSpec

from .base import NativeForecast
from .registry import ForecastRegistry


class ForecastRunner:
    def __init__(
        self,
        registry: ForecastRegistry,
        adapters: dict[str, Any] | None = None,
    ):
        self.registry = registry
        self.adapters = adapters or {}

    def _adapter(self, model_id: str) -> Any:
        if model_id not in self.adapters:
            self.adapters[model_id] = self.registry.build(model_id)
        return self.adapters[model_id]

    def predict(self, contexts: np.ndarray, forecast_spec: ForecastSpec) -> ForecastResult:
        values = np.asarray(contexts, dtype=float)
        if values.ndim != 3:
            raise ValueError("contexts must have shape [N,L,D]")
        if not np.isfinite(values).all():
            raise ValueError("forecast adapters require a complete finite context")
        n, length, dimensions = values.shape
        if forecast_spec.context_length is not None:
            values = values[:, -forecast_spec.context_length :, :]
        targets = forecast_spec.target_indices or tuple(range(dimensions))
        if any(target < 0 or target >= dimensions for target in targets):
            raise ValueError("target index is outside the context dimensions")
        adapter = self._adapter(forecast_spec.model_id)
        adapter_spec = self.registry.get(forecast_spec.model_id)
        if adapter_spec.mode != forecast_spec.mode:
            raise ValueError(
                f"model {forecast_spec.model_id} supports {adapter_spec.mode}, requested {forecast_spec.mode}"
            )
        if forecast_spec.mode == "joint_multivariate" and hasattr(adapter, "predict"):
            return adapter.predict(values, forecast_spec)
        if forecast_spec.mode == "joint_multivariate":
            native = adapter.predict_native(
                values,
                forecast_spec.horizon,
                forecast_spec.quantile_levels,
                forecast_spec.num_samples,
            )
            point = np.asarray(native.point, dtype=float)
            if point.shape != (n, forecast_spec.horizon, dimensions):
                raise ValueError(f"joint adapter returned unexpected point shape {point.shape}")
            quantiles = None if native.quantiles is None else np.asarray(native.quantiles)[:, :, targets, :]
            samples = None if native.samples is None else np.asarray(native.samples)[:, :, :, targets]
            return ForecastResult(
                point=point[:, :, targets],
                target_indices=tuple(targets),
                quantiles=quantiles,
                samples=samples,
                metadata=native.metadata,
            )
        flattened = np.stack(
            [values[batch_index, :, target] for batch_index in range(n) for target in targets]
        )
        if hasattr(adapter, "predict_native"):
            native = adapter.predict_native(
                flattened,
                forecast_spec.horizon,
                forecast_spec.quantile_levels,
                forecast_spec.num_samples,
            )
        else:
            local_spec = type(forecast_spec)(
                model_id=forecast_spec.model_id,
                mode="independent_univariate",
                horizon=forecast_spec.horizon,
                context_length=forecast_spec.context_length,
                target_indices=(0,),
                quantile_levels=forecast_spec.quantile_levels,
                num_samples=forecast_spec.num_samples,
            )
            result = adapter.predict(flattened[:, :, None], local_spec)
            point = result.point[:, :, 0]
            quantiles = None if result.quantiles is None else result.quantiles[:, :, 0, :]
            samples = None if result.samples is None else result.samples[:, :, :, 0]
            native = NativeForecast(
                point=point,
                quantiles=quantiles,
                samples=samples,
                metadata=result.metadata,
            )
        point = np.asarray(native.point, dtype=float)
        expected = (n * len(targets), forecast_spec.horizon)
        if point.shape != expected:
            raise ValueError(f"univariate adapter returned {point.shape}, expected {expected}")
        point = point.reshape(n, len(targets), forecast_spec.horizon).transpose(0, 2, 1)
        quantiles = None
        if native.quantiles is not None:
            q = np.asarray(native.quantiles)
            quantiles = q.reshape(n, len(targets), forecast_spec.horizon, q.shape[-1]).transpose(0, 2, 1, 3)
        samples = None
        if native.samples is not None:
            s = np.asarray(native.samples)
            samples = s.reshape(n, len(targets), s.shape[1], forecast_spec.horizon).transpose(0, 2, 3, 1)
        return ForecastResult(
            point=point,
            target_indices=tuple(targets),
            quantiles=quantiles,
            samples=samples,
            metadata=native.metadata,
        )
