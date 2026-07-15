"""Lazy TimesFM 2.5 adapter."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from tsfm_fais.contracts import ForecastResult, ForecastSpec

from ..base import ForecastCapabilities, LazyForecastAdapter
from ._utils import normalize_quantiles, point_from_quantiles, select_quantiles, to_numpy


class TimesFM2p5Adapter(LazyForecastAdapter):
    """TimesFM 2.5 adapter for independent univariate requests."""

    model_id = "timesfm2p5"
    capabilities = ForecastCapabilities(
        modes=frozenset({"independent_univariate"}),
        max_context=4096,
    )
    native_quantile_levels = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        *,
        batch_size: int = 128,
        device: str = "cpu",
        torch_compile: bool = False,
        backend: Any | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.torch_compile = bool(torch_compile)
        self._configs: Any | None = None
        self._compiled_key: tuple[int, int] | None = None

    def _load_backend(self) -> Any:
        try:
            timesfm = importlib.import_module("timesfm")
            self._configs = importlib.import_module("timesfm.configs")
        except ImportError as exc:
            raise ImportError(
                "TimesFM 2.5 requires the 'forecast-timesfm' optional dependency"
            ) from exc
        model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_cls is None:
            raise ImportError("installed timesfm package does not expose TimesFM 2.5")
        try:
            return model_cls.from_pretrained(
                self.model_name,
                torch_compile=self.torch_compile,
                local_files_only=True,
            )
        except TypeError as error:
            # timesfm 2.0 uses an older huggingface_hub mixin whose
            # ``from_pretrained`` forwards the hub-only ``proxies`` argument to
            # the model constructor.  Its public backend loader is otherwise
            # compatible and correctly handles an already downloaded snapshot.
            if "proxies" not in str(error) or not hasattr(model_cls, "_from_pretrained"):
                raise
            return model_cls._from_pretrained(
                model_id=self.model_name,
                revision=None,
                cache_dir=None,
                force_download=False,
                local_files_only=True,
                token=None,
                torch_compile=self.torch_compile,
            )

    def _maybe_compile(self, backend: Any, context_length: int, horizon: int) -> None:
        if not hasattr(backend, "compile") or self._configs is None:
            return
        patch_size = getattr(getattr(backend, "model", None), "p", None)
        compiled_context = context_length
        if isinstance(patch_size, int) and patch_size > 0:
            compiled_context = (
                (context_length + patch_size - 1) // patch_size
            ) * patch_size
        key = (compiled_context, horizon)
        if key == self._compiled_key:
            return
        backend.compile(
            forecast_config=self._configs.ForecastConfig(
                max_context=compiled_context,
                max_horizon=horizon,
                infer_is_positive=True,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                per_core_batch_size=max(1, self.batch_size),
            )
        )
        self._compiled_key = key

    def _predict(
        self,
        contexts: np.ndarray,
        spec: ForecastSpec,
        targets: tuple[int, ...],
    ) -> ForecastResult:
        if contexts.shape[2] != 1 or targets != (0,):
            raise ValueError("TimesFM 2.5 accepts one channel per request")
        backend = self._ensure_backend()
        self._maybe_compile(backend, contexts.shape[1], spec.horizon)
        raw = backend.forecast(
            horizon=spec.horizon,
            inputs=[row[:, 0].astype(np.float32, copy=False) for row in contexts],
        )

        if isinstance(raw, dict):
            point_raw = raw.get("point")
            quantile_raw = raw.get("quantiles")
            if quantile_raw is None:
                raise ValueError("TimesFM backend did not return quantiles")
            quantiles = normalize_quantiles(
                quantile_raw,
                contexts.shape[0],
                spec.horizon,
                1,
                len(spec.quantile_levels),
            )
        else:
            if not isinstance(raw, tuple) or len(raw) < 2:
                raise ValueError("TimesFM backend must return (point, full_predictions)")
            point_raw, full_raw = raw[:2]
            full = to_numpy(full_raw)
            if full.ndim != 3 or full.shape[:2] != (contexts.shape[0], spec.horizon):
                raise ValueError(f"unexpected TimesFM full prediction shape: {full.shape}")
            if full.shape[2] == len(self.native_quantile_levels) + 1:
                full = full[:, :, 1:]
            native = normalize_quantiles(
                full,
                contexts.shape[0],
                spec.horizon,
                1,
                len(self.native_quantile_levels),
            )
            quantiles = select_quantiles(
                native, self.native_quantile_levels, spec.quantile_levels
            )

        if point_raw is None:
            point = point_from_quantiles(quantiles, spec.quantile_levels)
        else:
            point_arr = to_numpy(point_raw)
            if point_arr.shape == (contexts.shape[0], spec.horizon):
                point = point_arr[:, :, None]
            elif point_arr.shape == (contexts.shape[0], spec.horizon, 1):
                point = point_arr
            else:
                raise ValueError(f"unexpected TimesFM point prediction shape: {point_arr.shape}")

        return ForecastResult(
            point=point,
            target_indices=(0,),
            quantiles=quantiles,
            metadata={"checkpoint": self.model_name},
        )
