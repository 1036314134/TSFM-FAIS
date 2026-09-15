import numpy as np
import pytest

from tsfm_fais.forecasting.completion_curve import nearest_curve_points


def test_affine_vector_curve_reaches_intermediate_output():
    grid = np.linspace(-1, 1, 11)
    response = np.stack([2 * grid + 1, -3 * grid + 2], axis=1)
    target = np.array([1.3, 1.55])
    result = nearest_curve_points(grid, response, target)
    assert result["interpolated_input"] == pytest.approx(0.15)
    assert result["polyline_mse_approximation"] < 1e-28
    assert result["grid_mse"] > 0


def test_parabola_grid_near_exact_population_minimum():
    grid = np.linspace(-1, 1, 1001)
    response = np.stack([grid, grid**2], axis=1)
    result = nearest_curve_points(grid, response, np.array([0, 1 / 3]))
    assert result["grid_input"] == 0
    assert result["grid_mse"] == pytest.approx(1 / 18)
    assert abs(result["polyline_mse_approximation"] - 1 / 18) < 1e-6


def test_constant_curve_and_invalid_inputs():
    result = nearest_curve_points([0, 1, 2], np.ones((3, 2)), [1, 1])
    assert result["grid_mse"] == result["polyline_mse_approximation"] == 0
    with pytest.raises(ValueError):
        nearest_curve_points([0, 0], np.ones((2, 2)), [1, 1])
    with pytest.raises(ValueError):
        nearest_curve_points([0, 1], [[1, float("nan")], [1, 2]], [1, 1])
