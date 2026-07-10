"""Lazy Chronos-2 and Chronos-Bolt adapters."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from tsfm_fais.contracts import ForecastResult, ForecastSpec

from ..base import ForecastCapabilities, LazyForecastAdapter
from ._utils import normalize_quantiles, point_from_quantiles, stack_payload


class Chronos2Adapter(LazyForecastAdapter):
    """Chronos-2 adapter supporting native multivariate forecasting."""

    model_id = "chronos2"
    capabilities = ForecastCapabilities(
        modes=frozenset({"joint_multivariate", "independent_univariate"}),
        max_context=8192,
    )

    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        *,
        batch_size: int = 32,
        device: str = "cpu",
        torch_dtype: str | None = None,
        predict_batches_jointly: bool = False,
        backend: Any | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.torch_dtype = torch_dtype
        self.predict_batches_jointly = bool(predict_batches_jointly)

    def _load_backend(self) -> Any:
        try:
            chronos = importlib.import_module("chronos")
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 requires the 'forecast-chronos' optional dependency"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device_map"] = self.device
        if self.torch_dtype:
            torch = importlib.import_module("torch")
            dtype = getattr(torch, self.torch_dtype, None)
            if dtype is None:
                raise ValueError(f"unsupported torch dtype: {self.torch_dtype}")
            kwargs["torch_dtype"] = dtype
        pipeline = chronos.BaseChronosPipeline.from_pretrained(self.model_name, **kwargs)
        expected = getattr(chronos, "Chronos2Pipeline", None)
        if expected is not None and not isinstance(pipeline, expected):
            raise TypeError(f"checkpoint {self.model_name!r} is not a Chronos-2 pipeline")
        return pipeline

    def _predict(
        self,
        contexts: np.ndarray,
        spec: ForecastSpec,
        targets: tuple[int, ...],
    ) -> ForecastResult:
        backend = self._ensure_backend()
        inputs = [
            {
                "target": row[:, 0]
                if row.shape[1] == 1
                else row.T
            }
            for row in contexts.astype(np.float32, copy=False)
        ]
        raw = backend.predict_quantiles(
            inputs=inputs,
            prediction_length=spec.horizon,
            batch_size=self.batch_size,
            quantile_levels=list(spec.quantile_levels),
            predict_batches_jointly=self.predict_batches_jointly,
        )
        raw_quantiles = raw[0] if isinstance(raw, tuple) else raw
        all_quantiles = normalize_quantiles(
            raw_quantiles,
            contexts.shape[0],
            spec.horizon,
            contexts.shape[2],
            len(spec.quantile_levels),
        )
        quantiles = all_quantiles[:, :, targets, :]
        return ForecastResult(
            point=point_from_quantiles(quantiles, spec.quantile_levels),
            target_indices=targets,
            quantiles=quantiles,
            metadata={"checkpoint": self.model_name},
        )


class ChronosBoltAdapter(LazyForecastAdapter):
    """Chronos-Bolt adapter for independent univariate requests."""

    model_id = "chronosbolt"
    capabilities = ForecastCapabilities(
        modes=frozenset({"independent_univariate"}),
        max_context=2048,
    )

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-base",
        *,
        batch_size: int = 32,
        device: str = "cpu",
        torch_dtype: str | None = None,
        backend: Any | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.torch_dtype = torch_dtype

    def _load_backend(self) -> Any:
        try:
            chronos = importlib.import_module("chronos")
        except ImportError as exc:
            raise ImportError(
                "Chronos-Bolt requires the 'forecast-chronos' optional dependency"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device_map"] = self.device
        if self.torch_dtype:
            torch = importlib.import_module("torch")
            dtype = getattr(torch, self.torch_dtype, None)
            if dtype is None:
                raise ValueError(f"unsupported torch dtype: {self.torch_dtype}")
            kwargs["torch_dtype"] = dtype
        return chronos.BaseChronosPipeline.from_pretrained(self.model_name, **kwargs)

    def _predict(
        self,
        contexts: np.ndarray,
        spec: ForecastSpec,
        targets: tuple[int, ...],
    ) -> ForecastResult:
        if contexts.shape[2] != 1 or targets != (0,):
            raise ValueError("Chronos-Bolt accepts one channel per request")
        backend = self._ensure_backend()
        try:
            torch = importlib.import_module("torch")
        except ImportError:  # permits dependency-free injected test backends
            inputs = [row[:, 0].astype(np.float32, copy=False) for row in contexts]
        else:
            inputs = [
                torch.as_tensor(row[:, 0], dtype=torch.float32)
                for row in contexts
            ]
        raw = backend.predict_quantiles(
            inputs=inputs,
            prediction_length=spec.horizon,
            quantile_levels=list(spec.quantile_levels),
        )
        raw_quantiles = raw[0] if isinstance(raw, tuple) else raw
        # Chronos-Bolt exposes the three-dimensional layout as [N,Q,H].
        # Normalize this explicitly because H == Q is otherwise ambiguous.
        raw_array = stack_payload(raw_quantiles)
        if raw_array.ndim == 3:
            raw_quantiles = raw_array.transpose(0, 2, 1)[:, :, None, :]
        quantiles = normalize_quantiles(
            raw_quantiles,
            contexts.shape[0],
            spec.horizon,
            1,
            len(spec.quantile_levels),
        )
        return ForecastResult(
            point=point_from_quantiles(quantiles, spec.quantile_levels),
            target_indices=(0,),
            quantiles=quantiles,
            metadata={"checkpoint": self.model_name},
        )
