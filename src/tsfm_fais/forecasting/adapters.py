"""Lazy adapters for the five primary TSFM predictors.

The adapters never repair missing inputs. Imputation is exclusively the
responsibility of the upstream multivariate imputation pipeline.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import ForecastAdapterSpec, NativeForecast


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _permute_to(array: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    if array.ndim != len(target):
        raise ValueError(f"cannot normalize rank-{array.ndim} output to {target}")
    for permutation in itertools.permutations(range(array.ndim)):
        shape = tuple(array.shape[index] for index in permutation)
        if all(expected == -1 or observed == expected for observed, expected in zip(shape, target)):
            return np.transpose(array, permutation)
    raise ValueError(f"cannot normalize output shape {array.shape} to {target}")


@dataclass
class _LazyAdapter:
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    max_context: int = 2048
    backend: Any = None
    spec: ForecastAdapterSpec | None = None

    def _contexts(self, contexts: np.ndarray) -> np.ndarray:
        values = np.asarray(contexts, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("TSFM context contains missing or non-finite values")
        if self.max_context > 0:
            values = values[:, -self.max_context :]
        return values


class Chronos2Adapter(_LazyAdapter):
    def _load(self):
        if self.backend is None:
            try:
                from chronos import BaseChronosPipeline, Chronos2Pipeline
            except ImportError as exc:
                raise ImportError("Chronos-2 requires `pip install .[forecast-chronos]`") from exc
            self.backend = BaseChronosPipeline.from_pretrained(self.model_name, device_map=self.device)
            if not isinstance(self.backend, Chronos2Pipeline):
                raise TypeError("checkpoint is not a Chronos2Pipeline")
        return self.backend

    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = self._contexts(contexts)
        n, _, dimensions = values.shape
        quantiles, mean = self._load().predict_quantiles(
            inputs=[{"target": item} for item in values],
            prediction_length=horizon,
            batch_size=self.batch_size,
            quantile_levels=list(quantile_levels),
            predict_batches_jointly=False,
        )
        raw = np.stack([_to_numpy(item) for item in quantiles])
        q = _permute_to(raw, (n, horizon, dimensions, len(quantile_levels)))
        point = _to_numpy(mean)
        try:
            point = _permute_to(point, (n, horizon, dimensions))
        except ValueError:
            point = q[..., int(np.argmin(np.abs(np.asarray(quantile_levels) - 0.5)))]
        return NativeForecast(point=point, quantiles=q)


class ChronosBoltAdapter(_LazyAdapter):
    def _load(self):
        if self.backend is None:
            try:
                from chronos import BaseChronosPipeline, ChronosBoltPipeline
            except ImportError as exc:
                raise ImportError("Chronos-Bolt requires `pip install .[forecast-chronos]`") from exc
            self.backend = BaseChronosPipeline.from_pretrained(self.model_name, device_map=self.device)
            if not isinstance(self.backend, ChronosBoltPipeline):
                raise TypeError("checkpoint is not a ChronosBoltPipeline")
        return self.backend

    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = self._contexts(contexts)
        quantiles, mean = self._load().predict_quantiles(
            inputs=[row for row in values],
            prediction_length=horizon,
            quantile_levels=list(quantile_levels),
        )
        raw = np.stack([_to_numpy(item) for item in quantiles])
        q = _permute_to(raw, (len(values), horizon, len(quantile_levels)))
        point = _to_numpy(mean)
        try:
            point = _permute_to(point, (len(values), horizon))
        except ValueError:
            point = q[..., int(np.argmin(np.abs(np.asarray(quantile_levels) - 0.5)))]
        return NativeForecast(point=point, quantiles=q)


class TiRexAdapter(_LazyAdapter):
    def _load(self):
        if self.backend is None:
            try:
                from tirex import load_model
            except ImportError as exc:
                raise ImportError("TiRex requires `pip install .[forecast-tirex]`") from exc
            self.backend = load_model(self.model_name, backend="torch", device=self.device)
        return self.backend

    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = self._contexts(contexts)
        quantiles, mean = self._load().forecast(
            context=values,
            prediction_length=horizon,
            output_type="numpy",
            batch_size=self.batch_size,
        )
        q = _permute_to(_to_numpy(quantiles), (len(values), horizon, len(quantile_levels)))
        point = _to_numpy(mean)
        try:
            point = _permute_to(point, (len(values), horizon))
        except ValueError:
            point = q[..., len(quantile_levels) // 2]
        return NativeForecast(point=point, quantiles=q)


class TimesFM2p5Adapter(_LazyAdapter):
    def _load(self):
        if self.backend is None:
            try:
                import timesfm
            except ImportError as exc:
                raise ImportError("TimesFM requires `pip install .[forecast-timesfm]`") from exc
            self.backend = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_name, torch_compile=False
            )
        return self.backend

    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = self._contexts(contexts)
        model = self._load()
        if hasattr(model, "compile"):
            try:
                from timesfm import configs

                model.compile(
                    forecast_config=configs.ForecastConfig(
                        max_context=values.shape[1],
                        max_horizon=horizon,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        return_backcast=False,
                    )
                )
            except (ImportError, TypeError):
                pass
        point, full = model.forecast(horizon=horizon, inputs=[row for row in values])
        point_arr = _permute_to(_to_numpy(point)[..., :horizon], (len(values), horizon))
        full_arr = _to_numpy(full)
        quantiles = None
        if full_arr.ndim == 3 and full_arr.shape[-1] > 1:
            available = full_arr[:, :horizon, 1:]
            indices = [min(available.shape[-1] - 1, max(0, int(round(level * 10)) - 1)) for level in quantile_levels]
            quantiles = available[..., indices]
        return NativeForecast(point=point_arr, quantiles=quantiles)


class SundialAdapter(_LazyAdapter):
    def _load(self):
        if self.backend is None:
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError("Sundial requires `pip install .[forecast-sundial]`") from exc
            self.backend = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
        return self.backend

    def predict_native(self, contexts, horizon, quantile_levels, num_samples):
        values = self._contexts(contexts)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Sundial requires PyTorch") from exc
        inputs = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        model = self._load()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=horizon,
                num_samples=num_samples,
                revin=True,
            )
        samples = _to_numpy(outputs)
        samples = _permute_to(samples, (len(values), num_samples, horizon))
        point = np.median(samples, axis=1)
        quantiles = np.quantile(samples, quantile_levels, axis=1).transpose(1, 2, 0)
        return NativeForecast(point=point, quantiles=quantiles, samples=samples)
