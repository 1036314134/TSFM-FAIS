from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def history_baseline():
    path = Path(__file__).resolve().parents[2] / "scripts/analyze_utility_history_baseline.py"
    spec = importlib.util.spec_from_file_location("utility_history_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_uses_declared_anchor_until_feedback_arrives(history_baseline):
    assert history_baseline.historical_choice([], ["a", "b"], "b", "mean") == "b"
    history = [np.array([1.0, 2.0]), np.array([4.0, 1.0])]
    assert history_baseline.historical_choice(history, ["a", "b"], "a", "last") == "b"
    assert history_baseline.historical_choice(history, ["a", "b"], "a", "mean") == "b"
    assert history_baseline.historical_choice([np.ones(2)], ["a", "b"], "b", "last") == "b"


def test_probe_ignores_unobserved_future_cells_and_weights_targets_equally(history_baseline):
    truth = np.array([[10.0, np.nan], [20.0, 1.0], [np.nan, 3.0]])
    point = np.array([[[11.0, 999.0], [22.0, 2.0], [-999.0, 5.0]]])
    scale = np.array([10.0, 1.0])
    expected = (0.15 + 1.5) / 2
    first = history_baseline.observed_probe_losses(point, truth, scale, minimum=2)
    point[0, 0, 1], point[0, 2, 0] = -1e9, 1e9
    second = history_baseline.observed_probe_losses(point, truth, scale, minimum=2)
    np.testing.assert_allclose(first, [expected])
    np.testing.assert_array_equal(first, second)


def test_probe_with_insufficient_observations_does_not_supply_feedback(history_baseline):
    truth = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert history_baseline.observed_probe_losses(np.zeros((2, 2, 2)), truth, np.ones(2), 2) is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_forecast_is_not_treated_as_zero_error(history_baseline, invalid):
    point = np.zeros((2, 2, 2))
    point[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        history_baseline.observed_probe_losses(point, np.ones((2, 2)), np.ones(2), 1)
