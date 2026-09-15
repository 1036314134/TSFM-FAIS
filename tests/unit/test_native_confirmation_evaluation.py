import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def evaluation(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "native_confirmation_evaluation", scripts / "run_native_confirmation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forecast_selection_keeps_joint_and_independent_target_constraints(evaluation):
    points = np.array([[[1.0, 10.0]], [[2.0, 20.0]]])
    np.testing.assert_array_equal(
        evaluation.chosen_prediction(points, ["a", "b"], ["b"], joint=True), [[2, 20]]
    )
    np.testing.assert_array_equal(
        evaluation.chosen_prediction(points, ["a", "b"], ["a", "b"], joint=False), [[1, 20]]
    )
    with pytest.raises(ValueError, match="one candidate"):
        evaluation.chosen_prediction(points, ["a", "b"], ["a", "b"], joint=True)


def test_hierarchy_balances_series_and_preserves_failed_windows(evaluation):
    frame = pd.DataFrame(
        {
            "model_id": ["m"] * 102,
            "method": ["p"] * 102,
            "family_id": ["a"] * 101 + ["b"],
            "dataset_id": ["a1"] * 101 + ["b1"],
            "item_id": ["i0"] * 100 + ["i1", "i0"],
        }
    )
    for metric in ("mae", "mse", "raw_mae", "raw_mse"):
        frame[metric] = [0.0] * 100 + [10.0, 20.0]
    _, _, result = evaluation.hierarchical_metrics(frame)
    assert result.mae.item() == 12.5
    frame.loc[0, ["mae", "mse", "raw_mae", "raw_mse"]] = np.nan
    _, _, result = evaluation.hierarchical_metrics(frame)
    assert result.mae.isna().all()
