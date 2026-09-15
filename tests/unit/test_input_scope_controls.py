import numpy as np
import pytest

from tsfm_fais.forecasting.input_scope_controls import crossed_inputs


def test_crossed_histories_change_only_the_requested_columns_and_preserve_observations():
    context = np.array([[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan]])
    base = np.where(np.isfinite(context), context, 3.0)
    changed = np.where(np.isfinite(context), context, 9.0)
    result = crossed_inputs(context, base, changed, [2, 0])
    np.testing.assert_array_equal(result[1, :, [2, 0]], changed[:, [2, 0]].T)
    np.testing.assert_array_equal(result[1, :, 1], base[:, 1])
    np.testing.assert_array_equal(result[2, :, [2, 0]], base[:, [2, 0]].T)
    np.testing.assert_array_equal(result[2, :, 1], changed[:, 1])
    for history in result:
        np.testing.assert_array_equal(history[np.isfinite(context)], context[np.isfinite(context)])
    corrupted = changed.copy()
    corrupted[0, 0] = 99
    with pytest.raises(AssertionError):
        crossed_inputs(context, base, corrupted, [0, 2])


def test_all_target_columns_leave_no_covariate_intervention():
    context = np.array([[np.nan, 2.0], [3.0, np.nan]])
    base, changed = np.nan_to_num(context, nan=0.0), np.nan_to_num(context, nan=5.0)
    result = crossed_inputs(context, base, changed, [0, 1])
    np.testing.assert_array_equal(result[0], result[2])
    np.testing.assert_array_equal(result[1], result[3])
