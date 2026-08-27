"""Leakage-safe rolling episodes sliced from a sequence-level mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsfm_fais.contracts import MissingBlock, SeriesBatch, TimeSeriesItem

from .masking import MaskedSeries, extract_missing_blocks, stable_seed


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
    mask_seed: int = 0
    mask_realization_id: str = "unspecified"
    mechanism: str = "unknown"
    target_missing_rate: float = 0.0
    global_missing_rate: float = 0.0
    local_missing_rate: float = 0.0
    context_truth_mask: np.ndarray | None = None
    future_observed_mask: np.ndarray | None = None
    base_observed_mask: np.ndarray | None = None


def fit_prefix_end(
    length: int,
    context_length: int,
    horizon: int,
    fraction: float = 0.2,
) -> int:
    """Return the end of the historical calibration prefix."""

    if context_length < 2 or horizon < 1 or not 0 < fraction < 1:
        raise ValueError("invalid fit-prefix parameters")
    if length < context_length + horizon:
        raise ValueError("series cannot provide one context and forecast horizon")
    return min(
        length - horizon,
        max(context_length, int(np.floor(fraction * length))),
    )


def rolling_origins(
    length: int,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    *,
    start: int | None = None,
) -> tuple[int, ...]:
    if context_length < 2 or horizon < 1:
        raise ValueError("invalid context length or horizon")
    if stride is not None and stride < 1:
        raise ValueError("stride must be positive")
    first = context_length if start is None else max(context_length, int(start))
    step = horizon if stride is None else stride
    return tuple(range(first, length - horizon + 1, step))


def build_episode(
    item: TimeSeriesItem,
    dataset_id: str,
    masked_series: MaskedSeries,
    forecast_origin: int,
    context_length: int,
    horizon: int,
) -> Episode:
    """Slice a rolling episode without generating or changing missingness."""

    if masked_series.values.shape != item.values.shape:
        raise ValueError("masked series and source item shapes differ")
    if forecast_origin < context_length or forecast_origin + horizon > len(item.values):
        raise ValueError("forecast origin cannot provide the requested context and future")
    start = forecast_origin - context_length
    clean_context = item.values[start:forecast_origin].copy()
    clean_future = item.values[forecast_origin : forecast_origin + horizon].copy()
    context_truth_mask = np.isfinite(clean_context)
    future_observed_mask = np.isfinite(clean_future)
    if not context_truth_mask.any() or not future_observed_mask.any():
        raise ValueError("episodes require observed source context and future values")
    masked = masked_series.values[start:forecast_origin].copy()
    observed = masked_series.observed_mask[start:forecast_origin].copy()
    base_observed = np.asarray(masked_series.base_observed_mask, dtype=bool)[
        start:forecast_origin
    ].copy()
    if np.any(observed & ~context_truth_mask):
        raise ValueError("episode mask exposes an unavailable source value")
    if not np.array_equal(masked[observed], clean_context[observed]):
        raise ValueError("masked series observed values differ from the source item")
    blocks = extract_missing_blocks(observed, masked_series.spec.mechanism)
    local_rate = float((~observed).mean())
    episode_seed = stable_seed(
        dataset_id,
        item.item_id,
        masked_series.realization_id,
        forecast_origin,
        "episode",
    )
    batch = SeriesBatch(
        values=masked[None, ...],
        observed_mask=observed[None, ...],
        item_ids=(item.item_id,),
        metadata={
            "dataset_id": dataset_id,
            "period": item.metadata.get("period"),
            "forecast_origin": forecast_origin,
            "mask_protocol": masked_series.metadata["protocol"],
            "mask_realization_id": masked_series.realization_id,
            "missing_mechanism": masked_series.spec.mechanism,
            "target_missing_rate": masked_series.spec.missing_rate,
            "global_missing_rate": masked_series.metadata["realized_missing_rate"],
            "local_missing_rate": local_rate,
            "base_missing_rate": float((~base_observed).mean()),
            "context_truth_fraction": float(context_truth_mask.mean()),
            "future_observed_fraction": float(future_observed_mask.mean()),
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
        seed=episode_seed,
        mask_seed=masked_series.seed,
        mask_realization_id=masked_series.realization_id,
        mechanism=masked_series.spec.mechanism,
        target_missing_rate=masked_series.spec.missing_rate,
        global_missing_rate=float(masked_series.metadata["realized_missing_rate"]),
        local_missing_rate=local_rate,
        context_truth_mask=context_truth_mask,
        future_observed_mask=future_observed_mask,
        base_observed_mask=base_observed,
    )
