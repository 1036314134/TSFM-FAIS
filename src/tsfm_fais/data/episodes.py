"""Leakage-safe forecasting episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsfm_fais.contracts import MissingBlock, SeriesBatch, TimeSeriesItem

from .masking import MaskingSpec, inject_missing, stable_seed


@dataclass(frozen=True)
class Episode:
    dataset_id: str
    item_id: str
    forecast_origin: int
    context: SeriesBatch
    clean_context: np.ndarray
    clean_future: np.ndarray
    blocks: tuple[MissingBlock, ...]
    seed: int


def rolling_origins(
    length: int,
    context_length: int,
    horizon: int,
    stride: int | None = None,
) -> tuple[int, ...]:
    if context_length < 2 or horizon < 1:
        raise ValueError("invalid context length or horizon")
    if stride is not None and stride < 1:
        raise ValueError("stride must be positive")
    step = horizon if stride is None else stride
    return tuple(range(context_length, length - horizon + 1, step))


def build_episode(
    item: TimeSeriesItem,
    dataset_id: str,
    forecast_origin: int,
    context_length: int,
    horizon: int,
    masking: MaskingSpec,
    repetition: int = 0,
) -> Episode:
    if forecast_origin < context_length or forecast_origin + horizon > len(item.values):
        raise ValueError("forecast origin cannot provide the requested context and future")
    clean_context = item.values[forecast_origin - context_length : forecast_origin].copy()
    clean_future = item.values[forecast_origin : forecast_origin + horizon].copy()
    if not np.isfinite(clean_context).all() or not np.isfinite(clean_future).all():
        raise ValueError("episodes require source-complete context and future")
    seed = stable_seed(dataset_id, item.item_id, forecast_origin, masking, repetition)
    masked, observed_mask, blocks = inject_missing(clean_context, masking, seed)
    batch = SeriesBatch(
        values=masked[None, ...],
        observed_mask=observed_mask[None, ...],
        item_ids=(item.item_id,),
        metadata={
            "dataset_id": dataset_id,
            "period": item.metadata.get("period"),
            "forecast_origin": forecast_origin,
        },
    )
    return Episode(
        dataset_id=dataset_id,
        item_id=item.item_id,
        forecast_origin=forecast_origin,
        context=batch,
        clean_context=clean_context,
        clean_future=clean_future,
        blocks=blocks,
        seed=seed,
    )
