import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_r6_teacher_transfer import loss_differences  # noqa: E402

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)


def test_teacher_improvement_can_reverse_on_the_actual_future():
    point, reference = np.zeros((1, 3, 2)), np.ones((1, 3, 2))
    real, proxy, cross = loss_differences(point, reference, point, 2 * reference)
    np.testing.assert_array_equal(real, [3.0])
    np.testing.assert_array_equal(proxy, [-1.0])
    np.testing.assert_array_equal(cross, [4.0])
    np.testing.assert_array_equal(real, proxy + cross)


def test_teacher_equal_to_truth_has_no_cross_term():
    rng = np.random.default_rng(9107)
    point, reference, truth = rng.normal(size=(3, 2, 5, 2))
    real, proxy, cross = loss_differences(point, reference, truth, truth)
    np.testing.assert_array_equal(real, proxy)
    np.testing.assert_array_equal(cross, np.zeros(2))


def test_convex_teacher_optimum_is_not_a_bound_for_coordinate_medians():
    points = np.eye(3)[[0, 0, 0, 1, 1, 2, 2]][None]
    teacher = np.zeros((1, 3))
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, teacher)["raw_projection"]
    weights, _, _ = simplex_quadratic_weights(gram, alignment)
    convex = compose_forecasts(points, weights)
    np.testing.assert_allclose((convex**2).mean(), 1 / 9, atol=1e-12)
    np.testing.assert_array_equal(np.median(points, axis=1), teacher)
