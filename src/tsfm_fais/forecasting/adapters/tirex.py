"""Lazy TiRex adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_fais.contracts import ForecastResult, ForecastSpec

from ..base import ForecastCapabilities, LazyForecastAdapter
from ._utils import normalize_quantiles, point_from_quantiles, select_quantiles


class TiRexAdapter(LazyForecastAdapter):
    """TiRex adapter for independent univariate requests."""

    model_id = "tirex"
    capabilities = ForecastCapabilities(
        modes=frozenset({"independent_univariate"}),
        max_context=2048,
        supports_missing_context=True,
    )
    native_quantile_levels = tuple(float(value) for value in np.arange(0.1, 1.0, 0.1))

    def __init__(
        self,
        model_name: str = "NX-AI/TiRex",
        *,
        device: str = "cpu",
        batch_size: int = 1,
        backend_name: str = "torch",
        backend: Any | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.backend_name = backend_name

    def _load_backend(self) -> Any:
        try:
            module = importlib.import_module("tirex")
        except ImportError as exc:
            raise ImportError("TiRex requires the 'forecast-tirex' optional dependency") from exc
        load_model = getattr(module, "load_model", None)
        if load_model is None:
            raise ImportError("installed tirex package does not expose load_model")
        local_path = Path(self.model_name)
        if local_path.exists():
            checkpoint = local_path / "model.ckpt" if local_path.is_dir() else local_path
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"TiRex checkpoint model.ckpt is missing under {local_path}"
                )
            base = getattr(module, "base", None)
            registry = getattr(getattr(base, "PretrainedModel", None), "REGISTRY", {})
            model_class = registry.get("TiRex")
            if model_class is None:
                raise ImportError("installed tirex package does not register the TiRex model")
            return model_class.from_pretrained(
                str(checkpoint),
                backend=self.backend_name,
                device=self.device,
                compile=False,
            )
        kwargs: dict[str, Any] = {"backend": self.backend_name}
        if self.device:
            kwargs["device"] = self.device
        return load_model(self.model_name, **kwargs)

    def _predict(
        self,
        contexts: np.ndarray,
        spec: ForecastSpec,
        targets: tuple[int, ...],
    ) -> ForecastResult:
        if contexts.shape[2] != 1 or targets != (0,):
            raise ValueError("TiRex accepts one channel per request")
        try:
            torch = importlib.import_module("torch")
        except ImportError:  # permits dependency-free injected test backends
            context = contexts[:, :, 0].astype(np.float32, copy=False)
            output_type = "numpy"
        else:
            context = torch.as_tensor(
                contexts[:, :, 0], dtype=torch.float32, device=self.device
            )
            output_type = "torch"
        raw = self._ensure_backend().forecast(
            context=context,
            prediction_length=spec.horizon,
            output_type=output_type,
            batch_size=min(self.batch_size, contexts.shape[0]),
        )
        raw_quantiles = raw[0] if isinstance(raw, tuple) else raw
        try:
            quantiles = normalize_quantiles(
                raw_quantiles,
                contexts.shape[0],
                spec.horizon,
                1,
                len(spec.quantile_levels),
            )
        except ValueError:
            native = normalize_quantiles(
                raw_quantiles,
                contexts.shape[0],
                spec.horizon,
                1,
                len(self.native_quantile_levels),
            )
            quantiles = select_quantiles(
                native, self.native_quantile_levels, spec.quantile_levels
            )
        return ForecastResult(
            point=point_from_quantiles(quantiles, spec.quantile_levels),
            target_indices=(0,),
            quantiles=quantiles,
            metadata={"checkpoint": self.model_name},
        )
