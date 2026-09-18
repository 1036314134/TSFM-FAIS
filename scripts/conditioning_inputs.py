"""Leak-resistant NumPy input helpers for a proposed Chronos-2 experiment.

This file DOES NOT load a forecasting checkpoint, run an experiment, or demonstrate
accuracy gains. It prepares one episode in time-by-variable format for a local
adapter that must be checked against the installed Chronos-2 version.

Run the input-contract tests with: python conditioning_inputs.py
"""
from __future__ import annotations

from dataclasses import dataclass
import unittest
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ConditionalInputs:
    context: NDArray[np.float64]                  # [D, L-B]
    context_mask: NDArray[np.float32]
    future_covariates: NDArray[np.float64]        # [D, B+H]
    future_covariates_mask: NDArray[np.float32]
    group_ids: NDArray[np.int64]                 # One multivariate episode only.
    repair_span: int
    actual_horizon: int


@dataclass(frozen=True)
class RepairedContext:
    values: NDArray[np.float64]                  # [L, D]
    original_observed: NDArray[np.bool_]
    model_observed: NDArray[np.bool_]


def _sanitize(history: ArrayLike, observed: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(history, dtype=np.float64)
    raw_mask = np.asarray(observed)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("history must have nonempty shape [time, variables]")
    if raw_mask.shape != x.shape:
        raise ValueError("observed must have exactly the history shape")
    if not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise ValueError("observed must contain only Boolean or binary values")
    mask = raw_mask.astype(bool, copy=True)
    if not np.isfinite(x[mask]).all():
        raise ValueError("originally observed values must be finite")
    # Scrub hidden values BEFORE normalization, not only in a later model mask.
    clean = np.where(mask, x, np.nan)
    return clean, mask


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def build_inputs(
    history: ArrayLike,
    observed: ArrayLike,
    *,
    repair_span: int = 96,
    actual_horizon: int = 0,
    condition_on_recent: bool = True,
) -> ConditionalInputs:
    """Move the last B historical positions into a partially known query interval.

    The function accepts NO actual future values. The final H positions are always
    NaN with mask zero. For the primary repair call use B=96,H=0. The direct
    conditioning control uses B=96,H=96 and evaluates only output [B:B+H].
    Setting condition_on_recent=False constructs the unconditioned repair control.
    B=0,H>0 is useful for testing equivalence with ordinary native forecasting.
    """
    clean, mask = _sanitize(history, observed)
    b = _integer(repair_span, "repair_span")
    h = _integer(actual_horizon, "actual_horizon")
    length, dimensions = clean.shape
    if b >= length or b + h == 0:
        raise ValueError("retain a nonempty prefix and request at least one output")
    if not isinstance(condition_on_recent, (bool, np.bool_)):
        raise ValueError("condition_on_recent must be Boolean")
    split = length - b
    future = np.full((dimensions, b + h), np.nan, dtype=np.float64)
    known = np.zeros((dimensions, b + h), dtype=np.float32)
    if b and condition_on_recent:
        future[:, :b] = clean[split:].T
        known[:, :b] = mask[split:].T.astype(np.float32)
    return ConditionalInputs(
        context=clean[:split].T.copy(),
        context_mask=mask[:split].T.astype(np.float32),
        future_covariates=future,
        future_covariates_mask=known,
        group_ids=np.zeros(dimensions, dtype=np.int64),
        repair_span=b,
        actual_horizon=h,
    )


def median_output_slice(
    quantile_predictions: ArrayLike,
    quantile_levels: ArrayLike,
    *,
    start: int,
    length: int,
) -> NDArray[np.float64]:
    """Convert documented [D,Q,T] model output to a [length,D] median slice.

    Read quantile_levels from the installed model; do not assume a fixed index.
    This function neither converts device tensors nor validates a real checkpoint.
    """
    prediction = np.asarray(quantile_predictions, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    first = _integer(start, "start")
    count = _integer(length, "length", minimum=1)
    if prediction.ndim != 3 or levels.ndim != 1 or prediction.shape[1] != len(levels):
        raise ValueError("expected predictions [D,Q,T] and corresponding quantile levels")
    if not np.isfinite(levels).all():
        raise ValueError("quantile levels must be finite")
    index = np.flatnonzero(np.isclose(levels, 0.5, rtol=0.0, atol=1e-7))
    if len(index) != 1 or first + count > prediction.shape[-1]:
        raise ValueError("require one median quantile and a valid output interval")
    return prediction[:, int(index[0]), first:first + count].T.copy()


def merge_recent_repair(
    history: ArrayLike,
    observed: ArrayLike,
    predicted_recent: ArrayLike,
) -> RepairedContext:
    """Replace ONLY originally missing entries in the recent B historical steps.

    Earlier missing entries remain NaN. Original observations are retained exactly
    as numerical values. The second model call must consume the inserted values:
    use model_observed (or infer its mask from NaN), NOT original_observed.
    Keep original_observed separately for evaluation and provenance.
    """
    clean, mask = _sanitize(history, observed)
    replacement = np.asarray(predicted_recent, dtype=np.float64)
    if (replacement.ndim != 2 or replacement.shape[1] != clean.shape[1]
            or not 0 < replacement.shape[0] < clean.shape[0]):
        raise ValueError("predicted_recent must have shape [B,D], with 0<B<L")
    b = replacement.shape[0]
    needs_fill = ~mask[-b:]
    if not np.isfinite(replacement[needs_fill]).all():
        raise ValueError("missing positions require finite predictions; apply registered fallback")
    clean[-b:] = np.where(mask[-b:], clean[-b:], replacement)
    return RepairedContext(clean, mask.copy(), np.isfinite(clean))


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(16, dtype=float).reshape(8, 2)
        self.mask = np.ones_like(self.x, dtype=bool)
        self.mask[1, 0] = False
        self.mask[5, 1] = False

    def test_shapes_and_original_channel_group(self):
        p = build_inputs(self.x, self.mask, repair_span=4, actual_horizon=3)
        self.assertEqual(p.context.shape, (2, 4))
        self.assertEqual(p.future_covariates.shape, (2, 7))
        np.testing.assert_array_equal(p.group_ids, [0, 0])

    def test_actual_future_is_unknown(self):
        p = build_inputs(self.x, self.mask, repair_span=4, actual_horizon=3)
        self.assertTrue(np.isnan(p.future_covariates[:, 4:]).all())
        self.assertTrue((p.future_covariates_mask[:, 4:] == 0).all())

    def test_hidden_value_changes_do_not_change_inputs(self):
        altered = self.x.copy()
        altered[~self.mask] = [1e200, -1e200]
        a = build_inputs(self.x, self.mask, repair_span=4, actual_horizon=3)
        b = build_inputs(altered, self.mask, repair_span=4, actual_horizon=3)
        for field in ("context", "context_mask", "future_covariates", "future_covariates_mask"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))

    def test_unconditioned_control_keeps_prefix(self):
        a = build_inputs(self.x, self.mask, repair_span=4)
        b = build_inputs(self.x, self.mask, repair_span=4, condition_on_recent=False)
        np.testing.assert_array_equal(a.context, b.context)
        self.assertTrue(np.isnan(b.future_covariates).all())
        self.assertTrue((b.future_covariates_mask == 0).all())

    def test_recent_index_alignment(self):
        p = build_inputs(self.x, self.mask, repair_span=4)
        self.assertEqual(p.future_covariates[0, 0], self.x[4, 0])
        self.assertEqual(p.future_covariates[0, 3], self.x[7, 0])
        self.assertTrue(np.isnan(p.future_covariates[1, 1]))

    def test_no_shift_packing(self):
        p = build_inputs(self.x, self.mask, repair_span=0, actual_horizon=3)
        self.assertEqual(p.context.shape, (2, 8))
        self.assertTrue(np.isnan(p.future_covariates).all())

    def test_univariate_packing(self):
        p = build_inputs(self.x[:, :1], self.mask[:, :1], repair_span=4)
        self.assertEqual(p.context.shape, (1, 4))
        self.assertEqual(p.future_covariates.shape, (1, 4))

    def test_observation_protection_and_model_mask(self):
        result = merge_recent_repair(self.x, self.mask, np.full((4, 2), 99.0))
        np.testing.assert_array_equal(result.values[self.mask], self.x[self.mask])
        self.assertTrue(np.isnan(result.values[1, 0]))
        self.assertEqual(result.values[5, 1], 99.0)
        self.assertFalse(result.original_observed[5, 1])
        self.assertTrue(result.model_observed[5, 1])

    def test_quantile_axis_and_future_slice(self):
        pred = np.arange(2 * 3 * 7, dtype=float).reshape(2, 3, 7)
        actual = median_output_slice(pred, [0.1, 0.5, 0.9], start=4, length=3)
        np.testing.assert_array_equal(actual, pred[:, 1, 4:7].T)

    def test_invalid_inputs_fail_explicitly(self):
        for kw in ({"repair_span": 8}, {"repair_span": -1}, {"repair_span": 2.5},
                   {"repair_span": 0, "actual_horizon": 0}):
            with self.assertRaises(ValueError):
                build_inputs(self.x, self.mask, **kw)
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_inputs(bad, self.mask, repair_span=4)
        with self.assertRaises(ValueError):
            build_inputs(self.x, np.full(self.mask.shape, 2), repair_span=4)
        with self.assertRaises(ValueError):
            merge_recent_repair(self.x, self.mask, np.full((4, 2), np.nan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
