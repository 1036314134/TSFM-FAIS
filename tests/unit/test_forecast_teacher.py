import numpy as np
import pytest

from tsfm_fais.routing.forecast_teacher import forecast_reference_costs
from tsfm_fais.routing.preforecast import consensus_projection_costs


def test_explicit_teacher_changes_the_projection_target():
    predictions = np.array([[[[0.0], [0.0]], [[1.0], [1.0]], [[4.0], [4.0]]]])
    clean_reference = np.array([[[3.0], [5.0]]])
    np.testing.assert_allclose(
        forecast_reference_costs(predictions, clean_reference)[0, :, 0], [17, 10, 1]
    )
    assert forecast_reference_costs(predictions, clean_reference)[0, :, 0].argmin() == 2
    assert consensus_projection_costs(predictions)[0, :, 0].argmin() == 1


def test_median_reference_matches_the_existing_cost_contract():
    predictions = np.random.default_rng(6101).normal(size=(3, 7, 12, 2))
    np.testing.assert_array_equal(
        forecast_reference_costs(predictions, np.median(predictions, axis=1)),
        consensus_projection_costs(predictions),
    )
    with pytest.raises(ValueError):
        forecast_reference_costs(predictions, np.zeros((3, 12, 1)))
