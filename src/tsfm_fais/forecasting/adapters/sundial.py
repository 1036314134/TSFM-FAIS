"""Lazy Sundial adapter."""

from __future__ import annotations

import importlib
import types
from typing import Any

import numpy as np

from tsfm_fais.contracts import ForecastResult, ForecastSpec

from ..base import ForecastCapabilities, LazyForecastAdapter
from ._utils import normalize_samples, quantiles_from_samples, to_numpy


class SundialAdapter(LazyForecastAdapter):
    """Sample-producing Sundial adapter for independent univariate requests."""

    model_id = "sundial"
    capabilities = ForecastCapabilities(
        modes=frozenset({"independent_univariate"}),
        max_context=2880,
    )

    def __init__(
        self,
        model_name: str = "thuml/sundial-base-128m",
        *,
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = True,
        backend: Any | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.local_files_only = bool(local_files_only)

    @staticmethod
    def _patch_transformers_cache(transformers: Any) -> None:
        cache_utils = importlib.import_module("transformers.cache_utils")
        dynamic_cache = cache_utils.DynamicCache
        if not hasattr(dynamic_cache, "seen_tokens"):
            dynamic_cache.seen_tokens = property(lambda cache: cache.get_seq_length())
        if not hasattr(dynamic_cache, "get_max_length"):
            if hasattr(dynamic_cache, "get_max_cache_shape"):
                dynamic_cache.get_max_length = lambda cache: cache.get_max_cache_shape()
            else:
                dynamic_cache.get_max_length = lambda cache: None
        if not hasattr(dynamic_cache, "get_usable_length"):
            dynamic_cache.get_usable_length = (
                lambda cache, new_seq_length=None, layer_idx=0: cache.get_seq_length(
                    layer_idx
                )
            )

    @staticmethod
    def _patch_model_forward(model: Any) -> None:
        """Adapt Sundial remote code to recent Transformers cache dimensions."""

        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "forward"):
            return
        original_forward = inner.forward

        def patched_forward(
            inner_self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: Any = None,
            output_attentions: Any = None,
            output_hidden_states: Any = None,
            return_dict: Any = None,
            **kwargs: Any,
        ) -> Any:
            token_width = max(1, int(getattr(inner_self.config, "input_token_len", 1)))
            if inputs_embeds is not None:
                token_length = int(inputs_embeds.shape[1])
                batch_size = int(inputs_embeds.shape[0])
                device = inputs_embeds.device
            elif input_ids is not None:
                token_length = int(input_ids.shape[1] // token_width)
                batch_size = int(input_ids.shape[0])
                device = input_ids.device
            else:
                token_length = batch_size = 0
                device = None
            past_length = 0
            if past_key_values is not None:
                if hasattr(past_key_values, "get_seq_length"):
                    try:
                        past_length = int(past_key_values.get_seq_length())
                    except Exception:
                        past_length = 0
                elif isinstance(past_key_values, (list, tuple)) and past_key_values:
                    try:
                        past_length = int(past_key_values[0][0].shape[2])
                    except Exception:
                        past_length = 0
            if device is not None and token_length > 0:
                torch = importlib.import_module("torch")
                total_length = past_length + token_length
                if (
                    attention_mask is None
                    or attention_mask.ndim != 2
                    or tuple(attention_mask.shape) != (batch_size, total_length)
                ):
                    attention_mask = torch.ones(
                        batch_size, total_length, dtype=torch.long, device=device
                    )
                if (
                    position_ids is None
                    or position_ids.ndim != 2
                    or tuple(position_ids.shape) != (batch_size, token_length)
                ):
                    position_ids = (
                        torch.arange(
                            past_length,
                            past_length + token_length,
                            dtype=torch.long,
                            device=device,
                        )
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                        .contiguous()
                    )
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        inner.forward = types.MethodType(patched_forward, inner)
        if not hasattr(model, "_extract_past_from_model_output"):
            model._extract_past_from_model_output = types.MethodType(
                lambda _self, outputs, standardize_cache_format=False: getattr(
                    outputs, "past_key_values", None
                ),
                model,
            )

    def _load_backend(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "Sundial requires the 'forecast-sundial' optional dependency"
            ) from exc
        self._patch_transformers_cache(transformers)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._patch_model_forward(model)
        return model

    def _direct_generate(self, backend: Any, inputs: np.ndarray, spec: ForecastSpec) -> Any:
        torch = importlib.import_module("torch")
        tensor = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        token_width = max(1, int(getattr(backend.config, "input_token_len", 1)))
        remainder = int(tensor.shape[-1] % token_width)
        if remainder:
            tensor = tensor[..., remainder:]
        if tensor.shape[-1] < token_width:
            raise ValueError("Sundial context is shorter than one input token")
        if not hasattr(backend, "_greedy_search"):
            with torch.no_grad():
                return backend.generate(
                    tensor,
                    max_new_tokens=spec.horizon,
                    num_samples=spec.num_samples,
                    revin=True,
                )

        generation = importlib.import_module("transformers.generation")
        means = tensor.mean(dim=-1, keepdim=True)
        standard_deviation = tensor.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        normalized = (tensor - means) / standard_deviation
        stopping = generation.StoppingCriteriaList(
            [
                generation.MaxLengthCriteria(
                    max_length=int(normalized.shape[1]) + spec.horizon
                )
            ]
        )
        attention_mask = torch.ones(
            normalized.shape[0],
            normalized.shape[1] // token_width,
            dtype=torch.long,
            device=normalized.device,
        )
        with torch.no_grad():
            output = backend._greedy_search(
                normalized,
                stopping_criteria=stopping,
                attention_mask=attention_mask,
                num_samples=spec.num_samples,
                revin=False,
            )
        return output * standard_deviation.unsqueeze(1) + means.unsqueeze(1)

    def _call_backend(self, backend: Any, contexts: np.ndarray, spec: ForecastSpec) -> Any:
        inputs = contexts[:, :, 0].astype(np.float32, copy=False)
        if hasattr(backend, "forecast"):
            return backend.forecast(
                inputs=inputs,
                prediction_length=spec.horizon,
                num_samples=spec.num_samples,
            )
        return self._direct_generate(backend, inputs, spec)

    def _predict(
        self,
        contexts: np.ndarray,
        spec: ForecastSpec,
        targets: tuple[int, ...],
    ) -> ForecastResult:
        if contexts.shape[2] != 1 or targets != (0,):
            raise ValueError("Sundial accepts one channel per request")
        raw = self._call_backend(self._ensure_backend(), contexts, spec)
        if isinstance(raw, dict):
            raw = raw.get("samples")
        if isinstance(raw, tuple):
            raw = raw[0]
        samples = normalize_samples(
            to_numpy(raw),
            contexts.shape[0],
            spec.num_samples,
            spec.horizon,
            1,
        )
        quantiles = quantiles_from_samples(samples, spec.quantile_levels)
        return ForecastResult(
            point=np.mean(samples, axis=1),
            target_indices=(0,),
            quantiles=quantiles,
            samples=samples,
            metadata={"checkpoint": self.model_name},
        )
