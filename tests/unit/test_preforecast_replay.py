import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.preforecast_replay import assemble_selected_context, unique_forecaster_inputs
from tsfm_fais.routing.structured_preforecast import COVARIATE_FEATURES, TARGET_FEATURES


def context_bank():
    context = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 4.0], [2.0, np.nan, 5.0]])
    first = np.where(np.isnan(context), 10.0, context)
    second = np.where(np.isnan(context), 20.0, context)
    return context, np.stack([first, second]), ["locf", "other"]


def test_joint_empty_covariate_uses_the_complete_locf_fallback():
    context, bank, names = context_bank()
    output = assemble_selected_context(context, bank, names, ["guarded_direct"], [0, 2], joint=True)
    np.testing.assert_array_equal(output, bank[0])
    with pytest.raises(ValueError, match="one complete"):
        assemble_selected_context(context, bank, names, ["locf", "other"], [0, 2], joint=True)


def test_independent_targets_keep_native_mask_and_only_fill_the_empty_target():
    context, bank, names = context_bank()
    output = assemble_selected_context(
        context, bank, names, ["guarded_direct", "guarded_direct"], [0, 1], joint=False
    )
    np.testing.assert_array_equal(output[:, 0], context[:, 0])
    np.testing.assert_array_equal(output[:, 1], bank[0, :, 1])
    output = assemble_selected_context(context, bank, names, ["other", "locf"], [0, 1], joint=False)
    np.testing.assert_array_equal(output[:, 0], bank[1, :, 0])
    np.testing.assert_array_equal(output[:, 1], bank[0, :, 1])


def test_deduplication_respects_model_input_scope_and_nan_positions():
    first = np.array([[1.0, np.nan], [2.0, 3.0]])
    second = first.copy()
    second[1, 1] = 9.0
    _, independent = unique_forecaster_inputs([first, second, first.copy()], [0], joint=False)
    _, joint = unique_forecaster_inputs([first, second, first.copy()], [0], joint=True)
    np.testing.assert_array_equal(independent, [0, 0, 0])
    np.testing.assert_array_equal(joint, [0, 1, 0])


@pytest.fixture
def replay_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "student_replay", scripts / "replay_preforecast_student.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structured_replay_maps_target_slots_to_actual_columns(replay_module):
    context = np.tile(np.array([1.0, 7.0, 10.0]), (12, 1))
    context[[3, 9], :] = np.nan
    first = np.where(np.isfinite(context), context, np.array([1.0, 7.0, 10.0]))
    second = np.where(np.isfinite(context), context, np.array([1.0, 70.0, 100.0]))
    frame = pd.DataFrame({"candidate_id": ["other", "other"], "target_slot": [0, 1]})
    result = replay_module.extend_candidate_features(
        frame,
        context,
        np.stack([first, second]),
        ["locf", "other"],
        np.zeros(3),
        np.ones(3),
        [2, 0],
        joint=False,
    )
    assert result.loc[0, "static.target_bin1.delta_max"] == 90
    assert result.loc[1, "static.target_bin1.delta_max"] == 0
    assert (result[list(COVARIATE_FEATURES)].to_numpy() == 0).all()
    changed_context, changed_bank = context.copy(), np.stack([first, second])
    changed_context[:, 1] *= 1000
    changed_bank[:, :, 1] *= 1000
    repeated = replay_module.extend_candidate_features(
        frame,
        changed_context,
        changed_bank,
        ["locf", "other"],
        np.zeros(3),
        np.ones(3),
        [2, 0],
        joint=False,
    )
    pd.testing.assert_frame_equal(result[list(TARGET_FEATURES)], repeated[list(TARGET_FEATURES)])


def test_ranked_contexts_keep_target_order_when_decision_rows_are_reversed(replay_module):
    class Selector:
        candidate_ids = ("a", "b", "c")
        baseline_id = "c"

        def pair_scores(self, frame):
            return ["later", "earlier"], [[-1, -1, -1], [1, 1, 1]], [(0, 1), (0, 2), (1, 2)]

    frame = pd.DataFrame({"episode_id": ["later", "earlier"], "target_slot": [1, 0]})
    assert replay_module.rank_context_actions(Selector(), frame, 3) == [
        ["a", "c"],
        ["b", "b"],
        ["c", "a"],
    ]
    assert replay_module.rank_context_actions(Selector(), frame, 1) == [["a", "c"]]


def test_forecast_features_keep_guarded_prediction_out_of_the_finite_pool(replay_module):
    frame = pd.DataFrame({"candidate_id": ["guarded_direct"] * 3, "target_slot": [-1, 0, 1]})
    points = np.repeat(np.array([[[0.0, 0.0]], [[2.0, 4.0]], [[10.0, 20.0]]]), 96, axis=1)
    features = replay_module.extend_forecast_features(frame, points, ["locf", "other"], np.zeros(2))
    np.testing.assert_allclose(features["response.pool_distance"], [13.5, 9.0, 18.0])
    assert (features["response.has_quantiles"] == 0).all()
    assert (features["response.interval_width"] == 0).all()


def test_candidate_queries_preserve_guard_fallback_order_and_shared_scales(replay_module):
    context, bank, names = context_bank()

    class Runner:
        def predict_missing(self, values, spec):
            self.values = values
            return SimpleNamespace(point=values.sum(axis=1)[:, None, [0, 2]])

    runner = Runner()
    points, unique = replay_module.query_candidate_points(
        runner,
        None,
        context,
        bank,
        names,
        [0, 2],
        np.array([1.0, 0.0, 4.0]),
        np.array([2.0, 1.0, 4.0]),
        joint=True,
    )
    assert unique == len(runner.values) == 2
    np.testing.assert_allclose(points, [[[6.0, 2.0]], [[11.0, 2.0]], [[6.0, 2.0]]])
