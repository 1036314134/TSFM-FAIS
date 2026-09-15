import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def source_fit():
    path = Path(__file__).resolve().parents[2] / "scripts/fit_confirmation_selectors.py"
    spec = importlib.util.spec_from_file_location("confirmation_source_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_controls_respect_joint_and_independent_target_choices(source_fit):
    points = np.array([[[[0.0, 10.0]], [[10.0, 0.0]], [[5.0, 5.0]]]])
    truth = np.zeros((1, 1, 2))
    metadata = pd.DataFrame({"family_id": ["f"], "dataset_id": ["d"]})
    actions = ["locf", "a", "b"]
    joint = source_fit.fixed_source_controls(points, truth, truth, metadata, actions, joint=True)
    independent = source_fit.fixed_source_controls(
        points, truth, truth, metadata, actions, joint=False
    )
    assert joint["fixed1_joint"] == ["b", "b"]
    assert independent["fixed1_joint"] == ["locf", "a"]


def test_source_macro_balances_families_and_datasets_instead_of_row_counts(source_fit):
    metadata = pd.DataFrame(
        {"family_id": ["a"] * 101 + ["b"], "dataset_id": ["a1"] * 100 + ["a2", "b1"]}
    )
    values = np.array([[10.0]] * 100 + [[0.0], [0.0]])
    np.testing.assert_allclose(source_fit.source_macro(values, metadata), [2.5])
