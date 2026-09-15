from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def analysis():
    path = Path(__file__).parents[2] / "scripts/analyze_downstream_accuracy.py"
    spec = importlib.util.spec_from_file_location("accuracy_selection_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mse_ensemble_fits_complementary_past_errors(analysis):
    errors = np.array([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
    weights, _ = analysis.mse_ensemble_weights(errors)
    np.testing.assert_allclose(weights, [1 / 3, 2 / 3], atol=1e-5)
    assert abs(weights.sum() - 1) < 1e-10 and weights.min() >= 0
    assert np.mean((errors @ weights) ** 2) < 1e-10


def test_selection_ignores_current_outcome_and_keeps_reference_on_tie(analysis):
    frame = pd.DataFrame(
        {
            "episode_id": ["a", "a"],
            "candidate_id": ["locf", "other"],
            "mae": [1.0, 100.0],
            "mse": [1.0, 10000.0],
        }
    )
    first = analysis.choose_scores(frame, [0.0, 0.0])
    second = analysis.choose_scores(frame.assign(mae=[100, 1], mse=[10000, 1]), [0.0, 0.0])
    assert first.candidate_id.tolist() == second.candidate_id.tolist() == ["locf"]


def test_target_decisions_keep_distinct_targets_and_original_origin(analysis):
    frame = pd.DataFrame(
        {
            "episode_id": ["a", "a"],
            "target_slot": [0, 1],
            "mae": [1.0, 2.0],
            "mse": [2.0, 8.0],
            "origin_id": ["history_a"] * 2,
        }
    )
    transformed = analysis.attach_objective(frame, "joint", {"mae": 2.0, "mse": 4.0})
    assert transformed.episode_id.nunique() == 2
    assert transformed.origin_id.nunique() == 1
    np.testing.assert_allclose(transformed.loss, [0.5, 1.5])
