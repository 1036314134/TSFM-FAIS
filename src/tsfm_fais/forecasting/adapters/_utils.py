"""Shape normalization shared by optional forecast-model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def stack_payload(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and value and not hasattr(value, "shape"):
        return np.stack([to_numpy(item) for item in value], axis=0)
    return to_numpy(value)


def normalize_quantiles(
    raw: Any,
    batch_size: int,
    horizon: int,
    n_targets: int,
    n_quantiles: int,
) -> np.ndarray:
    """Normalize common SDK layouts to ``[N,H,K,Q]``."""

    arr = stack_payload(raw)
    # Keep these checks ordered. A mapping keyed by shape silently overwrites
    # entries when horizon, target, and quantile dimensions happen to match.
    if arr.shape == (batch_size, horizon, n_targets, n_quantiles):
        return arr
    if arr.shape == (batch_size, n_quantiles, horizon, n_targets):
        return arr.transpose(0, 2, 3, 1)
    if arr.shape == (batch_size, n_targets, horizon, n_quantiles):
        return arr.transpose(0, 2, 1, 3)
    if arr.shape == (batch_size, n_targets, n_quantiles, horizon):
        return arr.transpose(0, 3, 1, 2)
    if arr.shape == (batch_size, n_quantiles, n_targets, horizon):
        return arr.transpose(0, 3, 2, 1)

    if n_targets == 1:
        if arr.shape == (batch_size, horizon, n_quantiles):
            return arr[:, :, None, :]
        if arr.shape == (batch_size, n_quantiles, horizon):
            return arr.transpose(0, 2, 1)[:, :, None, :]
        if n_quantiles == 1 and arr.shape == (batch_size, horizon):
            return arr[:, :, None, None]

    raise ValueError(
        "cannot normalize quantile output with shape "
        f"{arr.shape} to [N={batch_size},H={horizon},K={n_targets},Q={n_quantiles}]"
    )


def normalize_samples(
    raw: Any,
    batch_size: int,
    n_samples: int,
    horizon: int,
    n_targets: int = 1,
) -> np.ndarray:
    """Normalize common sample layouts to ``[N,S,H,K]``."""

    arr = stack_payload(raw)
    if arr.shape == (batch_size, n_samples, horizon, n_targets):
        return arr
    if arr.shape == (batch_size, horizon, n_targets, n_samples):
        return arr.transpose(0, 3, 1, 2)
    if n_targets == 1 and arr.shape == (batch_size, n_samples, horizon):
        return arr[:, :, :, None]
    if n_targets == 1 and arr.shape == (batch_size, horizon, n_samples):
        return arr.transpose(0, 2, 1)[:, :, :, None]
    raise ValueError(
        "cannot normalize sample output with shape "
        f"{arr.shape} to [N={batch_size},S={n_samples},H={horizon},K={n_targets}]"
    )


def point_from_quantiles(quantiles: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    levels_arr = np.asarray(levels, dtype=float)
    median = np.flatnonzero(np.isclose(levels_arr, 0.5))
    if len(median):
        return quantiles[..., int(median[0])]
    return np.mean(quantiles, axis=-1)


def quantiles_from_samples(samples: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    values = np.quantile(samples, np.asarray(levels, dtype=float), axis=1)
    return values.transpose(1, 2, 3, 0)


def select_quantiles(
    quantiles: np.ndarray,
    native_levels: Sequence[float],
    requested_levels: Sequence[float],
) -> np.ndarray:
    native = np.asarray(native_levels, dtype=float)
    indices: list[int] = []
    for level in requested_levels:
        matches = np.flatnonzero(np.isclose(native, level))
        if not len(matches):
            raise ValueError(
                f"requested quantile {level} is unavailable; native levels={tuple(native_levels)}"
            )
        indices.append(int(matches[0]))
    return quantiles[..., indices]
