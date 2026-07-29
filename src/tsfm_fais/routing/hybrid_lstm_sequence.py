"""Paper-scoped HybridLSTM imputation-method recommendation.

HybridLSTM is a univariate, fixed-window method.  It recommends one of its
own classical imputation techniques for each fixed sub-series and combines the
completed sub-series before the repository forecaster is called.  This module
therefore does not consume :class:`~tsfm_fais.contracts.MissingBlock` objects or
repository candidate identifiers.

The 2025 paper reports experiments at window sizes 48, 512, and 1024.  The
default here is 48 because it is the paper's all-series setting.  The ten
classical candidates below are the candidates used by the author's open thesis
precursor.  Pix2Pix was an additional candidate only in its 1024-point study,
so it is recorded separately and is not silently approximated at 48 points.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from scipy.interpolate import (
    Akima1DInterpolator,
    BarycentricInterpolator,
    interp1d,
    make_interp_spline,
)

from tsfm_fais.contracts import (
    BudgetSpec,
    CandidateResult,
    ForecastSpec,
    RoutingResult,
    SeriesBatch,
    TimeSeriesItem,
)
from tsfm_fais.imputers.base import deterministic_safe_values
from tsfm_fais.pipeline import BlockwiseFAIS, FAISResult, RoutePlan
from tsfm_fais.routing.blocks import detect_missing_blocks
from tsfm_fais.routing.graph import BlockGraph

HYBRID_LSTM_DEFAULT_WINDOW_SIZE: Final = 48
HYBRID_LSTM_FORMAL_WINDOW_SIZES: Final = (48, 512, 1024)
HYBRID_LSTM_PAPER_CANDIDATES: Final = (
    "mean",
    "median",
    "linear",
    "cubic",
    "akima",
    "polynomial_5",
    "spline_5",
    "moving_mean_3",
    "backfill",
    "forward_fill",
)
HYBRID_LSTM_INTERNAL_CANDIDATES: Final = HYBRID_LSTM_PAPER_CANDIDATES
HYBRID_LSTM_THESIS_CANDIDATES: Final = (*HYBRID_LSTM_PAPER_CANDIDATES, "pix2pix")
HYBRID_LSTM_SELECTOR_ID: Final = "hybrid_lstm"

_VALUE_PREFIX = "hybrid_value_"
_OBSERVED_PREFIX = "hybrid_observed_"
_PADDING_PREFIX = "hybrid_padding_"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "HybridLSTM training requires `pip install -e .[selector-baselines]` "
            "or `pip install torch`"
        ) from error
    return torch


def _configure_torch(torch: Any, seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(max(1, int(threads)))
    except RuntimeError:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):  # pragma: no cover - old/unsupported torch
        pass


def _validate_window_size(window_size: int) -> int:
    if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 2:
        raise ValueError("window_size must be an integer of at least two")
    return int(window_size)


def _fixed_window_slices(length: int, window_size: int) -> tuple[tuple[int, int], ...]:
    """Return fixed, end-aligned windows that cover the complete sequence.

    The thesis aligns a sufficiently long remainder to the sequence end.  At
    inference we also end-align a shorter remainder because ``complete`` must
    return a fully imputed context.  Main-experiment contexts of length 96 are
    exactly divisible by the paper's default window size and do not use this
    edge rule.
    """

    if length < 1:
        raise ValueError("a sequence must contain at least one timestep")
    if length <= window_size:
        return ((0, length),)
    windows = [
        (start, start + window_size) for start in range(0, length - window_size + 1, window_size)
    ]
    if windows[-1][1] < length:
        final = (length - window_size, length)
        if final != windows[-1]:
            windows.append(final)
    return tuple(windows)


def _padded_window(
    values: np.ndarray,
    observed_mask: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    observed = np.asarray(observed_mask, dtype=bool)
    if raw.ndim != 1 or observed.shape != raw.shape or len(raw) > window_size:
        raise ValueError("a univariate window and its mask must fit window_size")
    if np.any(~np.isfinite(raw[observed])):
        raise ValueError("observed window values must be finite")
    padded = np.full(window_size, np.nan, dtype=float)
    padded_observed = np.zeros(window_size, dtype=bool)
    padding = np.zeros(window_size, dtype=bool)
    padded[: len(raw)] = raw
    padded_observed[: len(raw)] = observed
    padding[: len(raw)] = True
    return padded, padded_observed, padding


def _encode_windows(values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Encode paper inputs as one raw-value channel with zero at missing positions."""

    windows = np.asarray(values, dtype=float)
    observed = np.asarray(observed_mask, dtype=bool)
    if windows.ndim != 2 or observed.shape != windows.shape:
        raise ValueError("windows and observed_mask must have shape [M,W]")
    if np.any(~np.isfinite(windows[observed])):
        raise ValueError("observed window values must be finite")
    # The published method feeds the stored windows directly into the network
    # and does not add a mask channel or normalize data values.  A finite zero
    # is the tensor representation of an absent CSV/NaN entry; the mask remains
    # available outside the network for labeling and native-validity checks.
    encoded = np.where(observed, windows, 0.0)
    return encoded[:, None, :]


def _window_prior_features(
    values: np.ndarray,
    observed_mask: np.ndarray,
    *,
    padding_mask: np.ndarray,
    window_size: int,
    start: int,
    end: int,
    sequence_length: int,
    channel: int,
    channel_count: int,
    batch_index: int,
    batch_size: int,
    period: int | None,
) -> dict[str, float]:
    encoded = _encode_windows(values[None, :], observed_mask[None, :])[0, 0]
    features: dict[str, float] = {}
    for index in range(window_size):
        features[f"{_VALUE_PREFIX}{index:03d}"] = float(encoded[index])
        features[f"{_OBSERVED_PREFIX}{index:03d}"] = float(observed_mask[index])
        features[f"{_PADDING_PREFIX}{index:03d}"] = float(padding_mask[index])
    valid = padding_mask.astype(bool)
    visible = values[observed_mask]
    features.update(
        {
            "missing_fraction": float(np.mean(~observed_mask[valid])) if np.any(valid) else 1.0,
            "valid_length_ratio": float(np.mean(valid)),
            "window_start_ratio": float(start / max(sequence_length, 1)),
            "window_end_ratio": float(end / max(sequence_length, 1)),
            "channel_ratio": float(channel / max(channel_count - 1, 1)),
            "batch_index_ratio": float(batch_index / max(batch_size - 1, 1)),
            "period_ratio": float((period or 0) / max(window_size, 1)),
            "observed_center": float(np.mean(visible)) if visible.size else 0.0,
            "observed_scale": float(np.std(visible)) if visible.size else 0.0,
        }
    )
    return features


def _forward_fill(values: np.ndarray) -> np.ndarray:
    output = np.array(values, dtype=float, copy=True)
    last = np.nan
    for index in range(len(output)):
        if np.isfinite(output[index]):
            last = output[index]
        elif np.isfinite(last):
            output[index] = last
    return output


def _backward_fill(values: np.ndarray) -> np.ndarray:
    output = np.array(values, dtype=float, copy=True)
    following = np.nan
    for index in range(len(output) - 1, -1, -1):
        if np.isfinite(output[index]):
            following = output[index]
        elif np.isfinite(following):
            output[index] = following
    return output


def _moving_mean_three(values: np.ndarray) -> np.ndarray:
    output = np.array(values, dtype=float, copy=True)
    for index in range(len(output)):
        if np.isfinite(output[index]):
            continue
        history = output[max(0, index - 3) : index]
        history = history[np.isfinite(history)]
        if history.size:
            output[index] = float(np.mean(history))
    return output


def _lagrange_degree_five(
    known_x: np.ndarray,
    known_y: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Evaluate local fifth-degree Lagrange interpolants at missing positions."""

    x_values = np.asarray(known_x, dtype=float)
    y_values = np.asarray(known_y, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    if x_values.ndim != 1 or y_values.shape != x_values.shape or x_values.size < 6:
        raise ValueError("fifth-degree Lagrange interpolation needs six known values")
    output = np.empty_like(target_values, dtype=float)
    for index, target in enumerate(target_values):
        # Six nearest observations define one degree-five Lagrange polynomial.
        nearest = np.lexsort((x_values, np.abs(x_values - target)))[:6]
        order = np.argsort(x_values[nearest])
        anchors = nearest[order]
        model = BarycentricInterpolator(x_values[anchors], y_values[anchors])
        output[index] = float(model(target))
    return output


def _paper_impute_1d(candidate_id: str, values: np.ndarray) -> tuple[np.ndarray, bool]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1:
        raise ValueError("paper candidates require one-dimensional windows")
    observed = np.isfinite(raw)
    missing = ~observed
    output = np.array(raw, copy=True)
    if not np.any(missing):
        return output, True
    positions = np.arange(len(raw), dtype=float)
    known_x = positions[observed]
    known_y = raw[observed]

    try:
        if candidate_id == "mean":
            if known_y.size:
                output[missing] = float(np.mean(known_y))
        elif candidate_id == "median":
            if known_y.size:
                output[missing] = float(np.median(known_y))
        elif candidate_id == "linear":
            if known_y.size >= 2:
                model = interp1d(
                    known_x,
                    known_y,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                output[missing] = model(positions[missing])
        elif candidate_id == "cubic":
            if known_y.size >= 4:
                model = interp1d(
                    known_x,
                    known_y,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )
                output[missing] = model(positions[missing])
        elif candidate_id == "akima":
            if known_y.size >= 2:
                model = Akima1DInterpolator(known_x, known_y, extrapolate=True)
                output[missing] = model(positions[missing])
        elif candidate_id == "polynomial_5":
            if known_y.size >= 6:
                output[missing] = _lagrange_degree_five(
                    known_x,
                    known_y,
                    positions[missing],
                )
        elif candidate_id == "spline_5":
            if known_y.size >= 6:
                model = make_interp_spline(known_x, known_y, k=5)
                output[missing] = model(positions[missing])
        elif candidate_id == "moving_mean_3":
            output = _moving_mean_three(raw)
        elif candidate_id == "backfill":
            output = _backward_fill(raw)
        elif candidate_id == "forward_fill":
            output = _forward_fill(raw)
        else:
            raise ValueError(f"unknown HybridLSTM paper candidate: {candidate_id}")
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        output = np.array(raw, copy=True)

    output[observed] = raw[observed]
    native_valid = bool(np.all(np.isfinite(output[missing])))
    return output, native_valid


def _asmape(clean: np.ndarray, imputed: np.ndarray) -> float:
    truth = np.asarray(clean, dtype=float)
    prediction = np.asarray(imputed, dtype=float)
    denominator = np.abs(truth) + np.abs(prediction)
    terms = np.zeros_like(denominator, dtype=float)
    nonzero = denominator > 0.0
    terms[nonzero] = np.abs(truth[nonzero] - prediction[nonzero]) / denominator[nonzero]
    return float(np.mean(terms))


def _imputation_metrics(
    clean: np.ndarray,
    imputed: np.ndarray,
    observed_mask: np.ndarray,
    native_valid: bool,
) -> tuple[float, float, float, float]:
    hidden = ~np.asarray(observed_mask, dtype=bool)
    if native_valid and np.all(np.isfinite(imputed)):
        loss = float(np.clip(_asmape(clean, imputed), 0.0, 1.0))
        if np.any(hidden):
            residual = np.asarray(imputed)[hidden] - np.asarray(clean)[hidden]
            mae = float(np.mean(np.abs(residual)))
            rmse = float(np.sqrt(np.mean(np.square(residual))))
        else:
            mae = 0.0
            rmse = 0.0
        return loss, mae, rmse, float(np.clip(1.0 - loss, 0.0, 1.0))
    scale = max(
        float(np.ptp(clean)) if len(clean) else 0.0,
        float(np.std(clean)) if len(clean) else 0.0,
        1.0,
    )
    return 1.0, scale, scale, 0.0


def _clean_batch(clean_context: np.ndarray, batch: SeriesBatch) -> np.ndarray:
    clean = np.asarray(clean_context, dtype=float)
    if clean.ndim == 2 and batch.shape[0] == 1:
        clean = clean[None, ...]
    if clean.shape != batch.shape:
        raise ValueError("clean_context must have shape [L,D] or match batch.shape")
    if not np.all(np.isfinite(clean)):
        raise ValueError("clean_context must be finite")
    if not np.allclose(
        clean[batch.observed_mask],
        batch.values[batch.observed_mask],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("clean_context must agree with observed batch values")
    return clean


def hybrid_window_label_rows(
    batch: SeriesBatch,
    clean_context: np.ndarray,
    *,
    episode_id: str,
    dataset_id: str,
    family_id: str,
    item_id: str,
    forecast_origin: int,
    period: int | None,
    window_size: int = HYBRID_LSTM_DEFAULT_WINDOW_SIZE,
) -> tuple[dict[str, Any], ...]:
    """Build paper-scoped labels for every univariate fixed window and method.

    ``imputation_loss`` is the thesis ASMAPE over the whole window (without the
    optional factor of two).  MAE and RMSE are evaluated on hidden positions.
    Invalid native outputs receive the bounded worst ASMAPE and are marked so
    :meth:`HybridLSTMSequenceSelector.fit_from_rows` can apply the paper's
    all-techniques-valid filtering.
    """

    if not isinstance(batch, SeriesBatch):
        raise TypeError("batch must be a SeriesBatch")
    size = _validate_window_size(window_size)
    if any(not str(value).strip() for value in (episode_id, dataset_id, family_id, item_id)):
        raise ValueError("episode_id, dataset_id, family_id, and item_id must be non-empty")
    if isinstance(forecast_origin, bool) or int(forecast_origin) != forecast_origin:
        raise ValueError("forecast_origin must be an integer")
    if period is not None and (
        isinstance(period, bool) or int(period) != period or int(period) < 1
    ):
        raise ValueError("period must be a positive integer or None")
    known_period = None if period is None else int(period)
    clean = _clean_batch(clean_context, batch)
    rows: list[dict[str, Any]] = []
    windows = _fixed_window_slices(batch.shape[1], size)
    for batch_index in range(batch.shape[0]):
        for channel in range(batch.shape[2]):
            for window_index, (start, end) in enumerate(windows):
                raw = batch.values[batch_index, start:end, channel]
                observed = batch.observed_mask[batch_index, start:end, channel]
                if bool(np.asarray(observed, dtype=bool).all()):
                    # Complete windows need no imputation-method decision.
                    continue
                padded, padded_observed, padding = _padded_window(raw, observed, size)
                prior = _window_prior_features(
                    padded,
                    padded_observed,
                    padding_mask=padding,
                    window_size=size,
                    start=start,
                    end=end,
                    sequence_length=batch.shape[1],
                    channel=channel,
                    channel_count=batch.shape[2],
                    batch_index=batch_index,
                    batch_size=batch.shape[0],
                    period=known_period,
                )
                block_id = f"n{batch_index}:d{channel}:w{window_index}:{start}-{end}"
                group_id = f"imputation::{episode_id}::{block_id}"
                clean_window = clean[batch_index, start:end, channel]
                raw_for_imputation = np.where(observed, raw, np.nan)
                for candidate_id in HYBRID_LSTM_PAPER_CANDIDATES:
                    imputed, native_valid = _paper_impute_1d(candidate_id, raw_for_imputation)
                    loss, mae, rmse, reward = _imputation_metrics(
                        clean_window,
                        imputed,
                        observed,
                        native_valid,
                    )
                    rows.append(
                        {
                            "label_scope": "hybrid_window",
                            "episode_id": str(episode_id),
                            "dataset_id": str(dataset_id),
                            "family_id": str(family_id),
                            "item_id": str(item_id),
                            "forecast_origin": int(forecast_origin),
                            "forecaster_id": "imputation",
                            "group_id": group_id,
                            "block_id": block_id,
                            "candidate_id": candidate_id,
                            "batch_index": batch_index,
                            "channel": channel,
                            "window_index": window_index,
                            "window_start": start,
                            "window_end": end,
                            "window_size": size,
                            "period": known_period,
                            # The network input belongs to the window group, so the
                            # first candidate row stores it once for compact JSONL.
                            "prior_features": (
                                dict(prior)
                                if candidate_id == HYBRID_LSTM_PAPER_CANDIDATES[0]
                                else {}
                            ),
                            "unary_features": {},
                            "imputation_loss": loss,
                            "imputation_mae": mae,
                            "imputation_rmse": rmse,
                            "imputation_reward": reward,
                            "native_valid": native_valid,
                        }
                    )
    return tuple(rows)


def _build_network(
    torch: Any,
    *,
    window_size: int,
    candidate_count: int,
    filters: int,
    kernel_size: int,
    hidden_size: int,
    dense_size: int,
    dropout: float,
) -> Any:
    strides = (2, 4, 2, 4, 2, 4)

    class HybridNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_channels = 1
            self.convolutions = torch.nn.ModuleList()
            self.normalizations = torch.nn.ModuleList()
            length = window_size
            for stride in strides:
                self.convolutions.append(
                    torch.nn.Conv1d(
                        input_channels,
                        filters,
                        kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                    )
                )
                self.normalizations.append(torch.nn.BatchNorm1d(filters))
                input_channels = filters
                length = int(math.ceil(length / stride))
            self.static_dense = torch.nn.Linear(length * filters, dense_size)
            self.bilstm = torch.nn.LSTM(
                1,
                hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.recurrent_dropout = torch.nn.Dropout(dropout)
            self.lstm1 = torch.nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
            self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.recurrent_normalization = torch.nn.BatchNorm1d(hidden_size)
            self.output = torch.nn.Linear(
                dense_size + window_size * hidden_size,
                candidate_count,
            )

        @staticmethod
        def _normalize(values: Any, normalization: Any) -> Any:
            if values.shape[0] * values.shape[2] > 1 or not normalization.training:
                return normalization(values)
            return torch.nn.functional.batch_norm(
                values,
                normalization.running_mean,
                normalization.running_var,
                normalization.weight,
                normalization.bias,
                training=False,
                momentum=0.0,
                eps=normalization.eps,
            )

        def forward(self, values: Any) -> Any:
            static = values
            for convolution, normalization in zip(
                self.convolutions,
                self.normalizations,
                strict=True,
            ):
                static = convolution(static)
                static = torch.relu(self._normalize(static, normalization))
            static = torch.relu(self.static_dense(static.flatten(start_dim=1)))
            recurrent = values.transpose(1, 2)
            recurrent, _ = self.bilstm(recurrent)
            recurrent = self.recurrent_dropout(recurrent)
            recurrent, _ = self.lstm1(recurrent)
            recurrent, _ = self.lstm2(recurrent)
            recurrent = self.recurrent_normalization(recurrent.transpose(1, 2))
            recurrent = recurrent.flatten(start_dim=1)
            return self.output(torch.cat((static, recurrent), dim=1))

    return HybridNetwork()


@dataclass
class HybridLSTMSequenceSelector:
    """Train and apply the fixed-window HybridLSTM recommendation baseline.

    The architecture follows the thesis description: six Conv1D-BN-ReLU
    blocks in the static branch and BiLSTM-LSTM-LSTM-BN in the temporal
    branch.  The formal paper's hybrid objective is implemented as equal-weight
    multi-class cross entropy plus multi-label binary cross entropy by default.
    The published preview does not expose a different loss coefficient, so both
    weights remain explicit constructor parameters.
    """

    window_size: int = HYBRID_LSTM_DEFAULT_WINDOW_SIZE
    epochs: int = 800
    batch_size: int = 64
    learning_rate: float = 1e-5
    filters: int = 112
    kernel_size: int = 7
    hidden_size: int = 32
    dense_size: int = 32
    dropout: float = 0.5
    l1_strength: float = 0.01
    multiclass_weight: float = 1.0
    multilabel_weight: float = 1.0
    multilabel_threshold: float = 0.02
    balance_classes: bool = True
    seed: int = 0
    torch_threads: int = 1
    params: Mapping[str, Any] | None = field(default=None, repr=False)
    candidate_ids: tuple[str, ...] = field(
        default=HYBRID_LSTM_PAPER_CANDIDATES,
        init=False,
    )
    _state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    fitted_window_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.params is not None:
            supported = {
                "window_size",
                "epochs",
                "batch_size",
                "learning_rate",
                "filters",
                "kernel_size",
                "hidden_size",
                "dense_size",
                "dropout",
                "l1_strength",
                "multiclass_weight",
                "multilabel_weight",
                "multilabel_threshold",
                "balance_classes",
                "torch_threads",
            }
            unknown = set(self.params) - supported
            if unknown:
                raise ValueError("unsupported HybridLSTM parameters: " + ", ".join(sorted(unknown)))
            for name, value in self.params.items():
                setattr(self, name, value)
        self.window_size = _validate_window_size(self.window_size)
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.filters < 1 or self.hidden_size < 1 or self.dense_size < 1:
            raise ValueError("network dimensions must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.l1_strength < 0.0:
            raise ValueError("l1_strength cannot be negative")
        if self.multiclass_weight < 0.0 or self.multilabel_weight < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.multiclass_weight + self.multilabel_weight <= 0.0:
            raise ValueError("at least one loss weight must be positive")
        if not 0.0 <= self.multilabel_threshold <= 1.0:
            raise ValueError("multilabel_threshold must lie in [0,1]")
        if not isinstance(self.balance_classes, bool):
            raise ValueError("balance_classes must be boolean")

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
    ) -> HybridLSTMSequenceSelector:
        """Construct from the selector-parameter mapping stored in configuration."""

        return cls(params=dict(params or {}), seed=int(seed))

    def _new_network(self, torch: Any) -> Any:
        return _build_network(
            torch,
            window_size=self.window_size,
            candidate_count=len(self.candidate_ids),
            filters=self.filters,
            kernel_size=self.kernel_size,
            hidden_size=self.hidden_size,
            dense_size=self.dense_size,
            dropout=self.dropout,
        )

    def fit(
        self,
        windows: np.ndarray,
        losses: np.ndarray,
        *,
        observed_mask: np.ndarray | None = None,
        native_valid: np.ndarray | None = None,
    ) -> HybridLSTMSequenceSelector:
        """Fit from fixed univariate windows and per-paper-candidate ASMAPE."""

        raw = np.asarray(windows, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != self.window_size or not len(raw):
            raise ValueError("windows must have non-empty shape [M,window_size]")
        observed = (
            np.isfinite(raw) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
        )
        if observed.shape != raw.shape:
            raise ValueError("observed_mask must match windows")
        loss_matrix = np.asarray(losses, dtype=float)
        expected = (len(raw), len(self.candidate_ids))
        if loss_matrix.shape != expected:
            raise ValueError(f"losses must have shape {expected}")
        validity = (
            np.isfinite(loss_matrix)
            if native_valid is None
            else np.asarray(native_valid, dtype=bool)
        )
        if validity.shape != loss_matrix.shape:
            raise ValueError("native_valid must match losses")
        usable = np.all(validity & np.isfinite(loss_matrix), axis=1)
        if not np.any(usable):
            raise ValueError("HybridLSTM needs at least one all-candidates-valid window")
        raw = raw[usable]
        observed = observed[usable]
        loss_matrix = loss_matrix[usable]
        inputs = _encode_windows(raw, observed).astype(np.float32)
        targets = np.argmin(loss_matrix, axis=1).astype(np.int64)
        multi_targets = (loss_matrix < self.multilabel_threshold).astype(np.float32)
        empty = np.sum(multi_targets, axis=1) == 0
        multi_targets[empty, targets[empty]] = 1.0
        if self.balance_classes:
            class_indices = [
                np.flatnonzero(targets == index) for index in range(len(self.candidate_ids))
            ]
            represented = [indices for indices in class_indices if len(indices)]
            if represented:
                target_count = min(len(indices) for indices in represented)
                rng = np.random.default_rng(int(self.seed))
                balanced = np.concatenate(
                    [rng.permutation(indices)[:target_count] for indices in represented]
                )
                balanced = rng.permutation(balanced)
                inputs = inputs[balanced]
                targets = targets[balanced]
                multi_targets = multi_targets[balanced]

        torch = _require_torch()
        _configure_torch(torch, int(self.seed), int(self.torch_threads))
        network = self._new_network(torch)
        optimizer = torch.optim.Adam(network.parameters(), lr=float(self.learning_rate))
        input_tensor = torch.as_tensor(inputs, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
        multi_tensor = torch.as_tensor(multi_targets, dtype=torch.float32)
        generator = torch.Generator().manual_seed(int(self.seed))
        network.train()
        for _ in range(int(self.epochs)):
            order = torch.randperm(len(inputs), generator=generator)
            for start in range(0, len(inputs), int(self.batch_size)):
                indices = order[start : start + int(self.batch_size)]
                logits = network(input_tensor[indices])
                objective = torch.zeros((), dtype=logits.dtype)
                if self.multiclass_weight:
                    objective = objective + float(self.multiclass_weight) * (
                        torch.nn.functional.cross_entropy(logits, target_tensor[indices])
                    )
                if self.multilabel_weight:
                    objective = objective + float(self.multilabel_weight) * (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits,
                            multi_tensor[indices],
                        )
                    )
                if self.l1_strength:
                    l1 = sum(convolution.weight.abs().sum() for convolution in network.convolutions)
                    objective = objective + float(self.l1_strength) * l1
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
        self._state = {
            name: value.detach().cpu().numpy().copy()
            for name, value in network.state_dict().items()
        }
        self.fitted_window_count = int(len(inputs))
        return self

    def fit_from_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> HybridLSTMSequenceSelector:
        """Fit directly from :func:`hybrid_window_label_rows` output."""

        grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if row.get("label_scope") != "hybrid_window":
                continue
            candidate_id = str(row.get("candidate_id", ""))
            if candidate_id not in self.candidate_ids:
                raise ValueError(f"unexpected HybridLSTM candidate: {candidate_id}")
            group_id = str(row.get("group_id", ""))
            if not group_id:
                raise ValueError("HybridLSTM rows require group_id")
            group = grouped.setdefault(group_id, {})
            if candidate_id in group:
                raise ValueError(f"duplicate candidate {candidate_id} in {group_id}")
            group[candidate_id] = row
        if not grouped:
            raise ValueError("no hybrid_window rows were provided")

        windows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        losses: list[list[float]] = []
        validity: list[list[bool]] = []
        expected_candidates = set(self.candidate_ids)
        for group_id, group in grouped.items():
            if set(group) != expected_candidates:
                missing = sorted(expected_candidates - set(group))
                raise ValueError(f"{group_id} is missing candidates: {missing}")
            first = group[self.candidate_ids[0]]
            prior = first.get("prior_features")
            if not isinstance(prior, Mapping):
                raise ValueError(f"{group_id} prior_features must be a mapping")
            try:
                window = np.asarray(
                    [
                        float(prior[f"{_VALUE_PREFIX}{index:03d}"])
                        for index in range(self.window_size)
                    ],
                    dtype=float,
                )
                mask = np.asarray(
                    [
                        bool(prior[f"{_OBSERVED_PREFIX}{index:03d}"])
                        for index in range(self.window_size)
                    ],
                    dtype=bool,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{group_id} has an invalid fixed-window encoding") from error
            windows.append(window)
            masks.append(mask)
            losses.append(
                [
                    float(group[candidate_id]["imputation_loss"])
                    for candidate_id in self.candidate_ids
                ]
            )
            validity.append(
                [
                    bool(group[candidate_id].get("native_valid", True))
                    for candidate_id in self.candidate_ids
                ]
            )
        return self.fit(
            np.stack(windows, axis=0),
            np.asarray(losses, dtype=float),
            observed_mask=np.stack(masks, axis=0),
            native_valid=np.asarray(validity, dtype=bool),
        )

    def predict_logits(
        self,
        windows: np.ndarray,
        *,
        observed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one recommendation logit per fixed window and paper candidate."""

        if self._state is None:
            raise RuntimeError("HybridLSTMSequenceSelector must be fitted before prediction")
        raw = np.asarray(windows, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != self.window_size:
            raise ValueError("windows must have shape [M,window_size]")
        observed = (
            np.isfinite(raw) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
        )
        if observed.shape != raw.shape:
            raise ValueError("observed_mask must match windows")
        inputs = _encode_windows(raw, observed).astype(np.float32)
        torch = _require_torch()
        network = self._new_network(torch)
        state = {
            name: torch.as_tensor(value, dtype=network.state_dict()[name].dtype)
            for name, value in self._state.items()
        }
        network.load_state_dict(state)
        network.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            tensor = torch.as_tensor(inputs, dtype=torch.float32)
            for start in range(0, len(inputs), int(self.batch_size)):
                logits = network(tensor[start : start + int(self.batch_size)])
                outputs.append(logits.detach().cpu().numpy().astype(float))
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, len(self.candidate_ids)), dtype=float)
        )

    def _complete_with_details(
        self,
        batch: SeriesBatch,
    ) -> tuple[np.ndarray, dict[str, str], tuple[str, ...]]:
        if not isinstance(batch, SeriesBatch):
            raise TypeError("batch must be a SeriesBatch")
        if self._state is None:
            raise RuntimeError("HybridLSTMSequenceSelector must be fitted before completion")
        records: list[tuple[str, int, int, int, int, tuple[np.ndarray, ...], np.ndarray]] = []
        model_windows: list[np.ndarray] = []
        model_masks: list[np.ndarray] = []
        windows = _fixed_window_slices(batch.shape[1], self.window_size)
        for batch_index in range(batch.shape[0]):
            for channel in range(batch.shape[2]):
                for window_index, (start, end) in enumerate(windows):
                    raw = batch.values[batch_index, start:end, channel]
                    observed = batch.observed_mask[batch_index, start:end, channel]
                    if bool(np.asarray(observed, dtype=bool).all()):
                        continue
                    padded, padded_observed, _ = _padded_window(
                        raw,
                        observed,
                        self.window_size,
                    )
                    candidate_outputs: list[np.ndarray] = []
                    validity_flags: list[bool] = []
                    raw_for_imputation = np.where(observed, raw, np.nan)
                    for candidate_id in self.candidate_ids:
                        imputed, valid = _paper_impute_1d(candidate_id, raw_for_imputation)
                        candidate_outputs.append(imputed)
                        validity_flags.append(valid)
                    records.append(
                        (
                            f"n{batch_index}:d{channel}:w{window_index}:{start}-{end}",
                            batch_index,
                            channel,
                            start,
                            end,
                            tuple(candidate_outputs),
                            np.asarray(validity_flags, dtype=bool),
                        )
                    )
                    model_windows.append(padded)
                    model_masks.append(padded_observed)
        logits = self.predict_logits(
            np.stack(model_windows, axis=0),
            observed_mask=np.stack(model_masks, axis=0),
        )
        output = np.array(batch.values, dtype=float, copy=True)
        window_assignments: dict[str, str] = {}
        fallback_windows: list[str] = []
        for row_logits, record in zip(logits, records, strict=True):
            window_id, batch_index, channel, start, end, candidates, native_valid_array = record
            if not np.any(native_valid_array):
                window_assignments[window_id] = "deterministic_safe_values"
                fallback_windows.append(window_id)
                continue
            eligible_logits = np.where(native_valid_array, row_logits, -np.inf)
            selected = int(np.argmax(eligible_logits))
            window_assignments[window_id] = self.candidate_ids[selected]
            observed = batch.observed_mask[batch_index, start:end, channel]
            selected_values = candidates[selected]
            target = output[batch_index, start:end, channel]
            target[~observed] = selected_values[~observed]
            output[batch_index, start:end, channel] = target
        invalid = ~np.isfinite(output)
        if np.any(invalid):
            safe = deterministic_safe_values(batch)
            output[invalid] = safe[invalid]
        output[batch.observed_mask] = batch.values[batch.observed_mask]
        if not np.all(np.isfinite(output)):
            raise RuntimeError("HybridLSTM failed to produce a finite completed sequence")
        return output, window_assignments, tuple(fallback_windows)

    def complete(self, batch: SeriesBatch) -> np.ndarray:
        """Complete every sequence using fixed-window paper recommendations."""

        values, _, _ = self._complete_with_details(batch)
        return values


class HybridLSTMSequencePipeline(BlockwiseFAIS):
    """Repository pipeline adapter for paper-native fixed-window completion."""

    uses_registry_candidates = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.router is None or not isinstance(
            self.router.prior,
            HybridLSTMSequenceSelector,
        ):
            raise ValueError(
                "HybridLSTM sequence routing requires a fitted HybridLSTMSequenceSelector"
            )
        self.selector = self.router.prior
        self.requires_pseudo_candidates = False
        self.selector_method = HYBRID_LSTM_SELECTOR_ID
        self.fallback_internal = ()
        self.fallback_tail = ()

    def prepare_route(
        self,
        item: TimeSeriesItem,
        observed_mask: np.ndarray,
        forecast_spec: ForecastSpec,
        budget: BudgetSpec | None = None,
        *,
        seed: int = 20260710,
        available_artifact_ids: Iterable[str] | None = None,
        artifact_load_failures: Mapping[str, str] | None = None,
    ) -> RoutePlan:
        """Build an execution plan with no repository candidate sweep."""

        del available_artifact_ids, artifact_load_failures
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != item.values.shape:
            raise ValueError("observed_mask must match item.values")
        batch = SeriesBatch(
            values=item.values[None, ...],
            observed_mask=mask[None, ...],
            item_ids=(item.item_id,),
            metadata=item.metadata,
        )
        blocks = detect_missing_blocks(batch.observed_mask)
        resolved_budget = budget or BudgetSpec(max_candidates=1)
        period = item.metadata.get("period")
        return RoutePlan(
            batch=batch,
            blocks=blocks,
            graph=BlockGraph(blocks=blocks, edges=()),
            forecast_spec=forecast_spec,
            budget=resolved_budget,
            seed=int(seed),
            period=None if period is None else int(period),
            correlation_source="none",
            candidate_ids=(),
            shortlist=(),
            costs={},
            prior_unary={},
            pseudo_batch=None,
            training_medians=item.metadata.get("training_medians", self.training_medians),
            artifact_load_failures={},
            available_artifact_ids=frozenset(),
            allow_fallback_execution=False,
        )

    def finish_route(
        self,
        plan: RoutePlan,
        candidates: Mapping[str, CandidateResult],
        pseudo_candidates: Mapping[str, CandidateResult] | None = None,
        *,
        backtest_candidates: Mapping[str, CandidateResult] | None = None,
        fallback_candidates: Mapping[str, CandidateResult] | None = None,
    ) -> FAISResult:
        """Complete internally, then return the standard repository result."""

        del candidates, pseudo_candidates, backtest_candidates, fallback_candidates
        mask = np.asarray(plan.batch.observed_mask[0], dtype=bool)
        if plan.is_noop:
            values = np.asarray(plan.batch.values[0], dtype=float).copy()
            window_assignments: dict[str, str] = {}
            fallback_windows: tuple[str, ...] = ()
            solver = "noop"
        else:
            completed, window_assignments, fallback_windows = self.selector._complete_with_details(
                plan.batch
            )
            values = completed[0]
            solver = "hybrid_lstm_fixed_window"
        block_assignments = {block.block_id: HYBRID_LSTM_SELECTOR_ID for block in plan.blocks}
        selected_methods = tuple(sorted(set(window_assignments.values())))
        routing = RoutingResult(
            assignments=block_assignments,
            shortlist=(),
            total_energy=0.0,
            activated_candidates=selected_methods,
            fallback_blocks=tuple(block.block_id for block in plan.blocks if fallback_windows),
            metadata={
                "selector_method": HYBRID_LSTM_SELECTOR_ID,
                "selector_implementation": HYBRID_LSTM_SELECTOR_ID,
                "selection_scope": "fixed_univariate_window",
                "selection_unit": "paper_fixed_window",
                "selection_count": len(window_assignments),
                "window_size": self.selector.window_size,
                "window_assignments": window_assignments,
                "fallback_windows": list(fallback_windows),
                "paper_candidates": list(self.selector.candidate_ids),
                "routing_target_protocol": "sequence_imputation_quality_v1",
                "selector_training_target": "imputation_loss",
                "forecaster_independent_selection": True,
                "uses_missing_block_graph": False,
                "requires_pseudo_candidates": False,
                "solver": solver,
                "paper_native_valid": not fallback_windows,
                "paper_ineligibility_reason": (
                    None if not fallback_windows else "no_native_valid_method_for_fixed_window"
                ),
            },
        )
        return FAISResult(
            values=values,
            routing=routing,
            candidates={},
            observed_mask=mask,
            metadata={
                "hybrid_window_assignments": window_assignments,
                "hybrid_fallback_windows": list(fallback_windows),
            },
        )


__all__ = [
    "HYBRID_LSTM_DEFAULT_WINDOW_SIZE",
    "HYBRID_LSTM_FORMAL_WINDOW_SIZES",
    "HYBRID_LSTM_INTERNAL_CANDIDATES",
    "HYBRID_LSTM_PAPER_CANDIDATES",
    "HYBRID_LSTM_THESIS_CANDIDATES",
    "HybridLSTMSequencePipeline",
    "HybridLSTMSequenceSelector",
    "hybrid_window_label_rows",
]
