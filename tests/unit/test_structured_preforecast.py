import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.preforecast import STATIC_FEATURES
from tsfm_fais.routing.structured_preforecast import (
    COVARIATE_FEATURES,
    EXTRA_FEATURES,
    input_change_features,
    structured_inputs,
    target_correlations,
)


def test_joint_features_detect_other_variable_repairs_and_independent_features_do_not():
    target = np.arange(12, dtype=float)
    context = np.column_stack([target, target * 2, target[::-1]])
    context[3:9, 1] = np.nan
    reference = np.nan_to_num(context)
    changed = reference.copy()
    changed[3:9, 1] = [2, 7, 3, 8, 4, 9]
    first = input_change_features(context, reference, reference, [0], joint=True)
    second = input_change_features(context, changed, reference, [0], joint=True)
    assert any(first[key] != second[key] for key in COVARIATE_FEATURES)
    assert input_change_features(
        context, reference, reference, [0], joint=False
    ) == input_change_features(context, changed, reference, [0], joint=False)


def test_other_variable_permutation_preserves_joint_descriptors():
    rng = np.random.default_rng(6101)
    context = rng.normal(size=(24, 5))
    context[6:13, 2:] = np.nan
    reference = np.nan_to_num(context)
    completed = np.where(np.isfinite(context), context, rng.normal(size=context.shape))
    expected = input_change_features(context, completed, reference, [0, 1], joint=True)
    order = [0, 1, 4, 2, 3]
    actual = input_change_features(
        context[:, order], completed[:, order], reference[:, order], [0, 1], joint=True
    )
    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), rtol=1e-12, atol=1e-12
    )


def test_correlation_availability_and_no_forecast_feature_entry():
    values = np.column_stack([np.arange(12), np.ones(12), np.arange(12) * 2]).astype(float)
    correlation, available = target_correlations(values, [0])
    assert not available[0, 1]
    assert correlation[0, 2] == pytest.approx(1)
    values[:6, 2] = np.nan
    assert not target_correlations(values, [0])[1][0, 2]
    frame = pd.DataFrame(
        [
            {
                **dict.fromkeys((*STATIC_FEATURES, *EXTRA_FEATURES), 0.0),
                "episode_id": "e",
                "candidate_id": "a",
                "response.future": 99,
                "mae": 1,
            }
        ]
    )
    result = structured_inputs(frame)
    assert "mae" not in result and "response.future" not in result
    with pytest.raises(ValueError):
        structured_inputs(frame.assign(**{"static.unknown": 1}))
