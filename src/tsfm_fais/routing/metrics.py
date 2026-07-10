"""Scale-aware losses used by the TSFM teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tsfm_fais.contracts import CandidateResult, MissingBlock


def _masked_errors(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    truth_arr = np.asarray(truth, dtype=float)
    prediction_arr = np.asarray(prediction, dtype=float)
    if truth_arr.shape != prediction_arr.shape:
        raise ValueError("truth and prediction must have the same shape")
    selected = np.ones(truth_arr.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != truth_arr.shape:
        raise ValueError("mask must have the same shape as truth")
    selected &= np.isfinite(truth_arr) & np.isfinite(prediction_arr)
    if not np.any(selected):
        raise ValueError("metric mask does not select any finite values")
    return prediction_arr[selected] - truth_arr[selected]


def masked_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    return float(np.mean(np.abs(_masked_errors(truth, prediction, mask))))


def masked_mse(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    error = _masked_errors(truth, prediction, mask)
    return float(np.mean(error**2))


def masked_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    return float(np.sqrt(masked_mse(truth, prediction, mask)))


def forecast_degradation(candidate_loss: float, clean_loss: float) -> float:
    """Signed loss increase relative to forecasting from clean context."""

    return float(candidate_loss) - float(clean_loss)


def block_candidate_losses(
    truth: np.ndarray,
    blocks: Sequence[MissingBlock],
    candidates: Mapping[str, CandidateResult | np.ndarray],
    *,
    metric: str = "mae",
) -> dict[tuple[str, str], float]:
    """Evaluate every candidate only on each block's missing positions."""

    metric_functions = {
        "mae": masked_mae,
        "mse": masked_mse,
        "rmse": masked_rmse,
    }
    if metric not in metric_functions:
        raise ValueError("metric must be mae, mse, or rmse")
    truth_arr = np.asarray(truth, dtype=float)
    result: dict[tuple[str, str], float] = {}
    for block in blocks:
        selector = (block.batch_index, slice(block.start, block.end), block.channel)
        for candidate_id, candidate in candidates.items():
            values = candidate.values if isinstance(candidate, CandidateResult) else candidate
            values_arr = np.asarray(values, dtype=float)
            if values_arr.shape != truth_arr.shape:
                raise ValueError(f"candidate {candidate_id} does not match truth shape")
            result[(block.block_id, candidate_id)] = metric_functions[metric](
                truth_arr[selector], values_arr[selector], None
            )
    return result


def routing_regret(
    selected_losses: Sequence[float] | np.ndarray,
    oracle_losses: Sequence[float] | np.ndarray,
) -> float:
    selected = np.asarray(selected_losses, dtype=float)
    oracle = np.asarray(oracle_losses, dtype=float)
    if selected.shape != oracle.shape or selected.size == 0:
        raise ValueError("selected_losses and oracle_losses must have the same non-empty shape")
    if not np.all(np.isfinite(selected)) or not np.all(np.isfinite(oracle)):
        raise ValueError("routing losses must be finite")
    return float(np.mean(selected - oracle))


def top_k_hit(
    scores: Mapping[tuple[str, str], float],
    optimal: Mapping[str, str],
    k: int = 1,
) -> float:
    """Fraction of blocks whose oracle candidate appears in the k lowest scores."""

    if k < 1:
        raise ValueError("k must be positive")
    if not optimal:
        raise ValueError("optimal assignments cannot be empty")
    hits = 0
    for block_id, candidate_id in optimal.items():
        ranked = sorted(
            ((float(value), candidate) for (block, candidate), value in scores.items() if block == block_id),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked:
            raise ValueError(f"scores do not contain block {block_id}")
        hits += candidate_id in {candidate for _, candidate in ranked[:k]}
    return float(hits / len(optimal))


def mase(
    truth: np.ndarray,
    prediction: np.ndarray,
    history: np.ndarray,
    seasonality: int = 1,
    epsilon: float = 1e-8,
) -> float:
    truth_arr = np.asarray(truth, dtype=float)
    prediction_arr = np.asarray(prediction, dtype=float)
    history_arr = np.asarray(history, dtype=float)
    if truth_arr.shape != prediction_arr.shape:
        raise ValueError("truth and prediction must have the same shape")
    if history_arr.ndim == 0:
        raise ValueError("history must include a time dimension")
    time_length = history_arr.shape[-1] if history_arr.ndim > 1 else history_arr.shape[0]
    lag = max(1, min(int(seasonality), max(1, time_length - 1)))
    naive = np.abs(history_arr[..., lag:] - history_arr[..., :-lag])
    scale = float(np.mean(naive)) if naive.size else 0.0
    return float(np.mean(np.abs(truth_arr - prediction_arr)) / max(scale, epsilon))


def normalized_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    history: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    truth_arr = np.asarray(truth, dtype=float)
    prediction_arr = np.asarray(prediction, dtype=float)
    scale = float(np.std(np.asarray(history, dtype=float)))
    return float(np.sqrt(np.mean((truth_arr - prediction_arr) ** 2)) / max(scale, epsilon))
