from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.routing.recent_feedback import (
    combine_imputations,
    observed_ensemble_weights,
    observed_forecast_risk,
    plan_recent_probes,
    select_feedback,
    validate_feedback_end,
)


def source_manifest():
    def episode(origin):
        return {
            "episode_id": f"d|i|{origin}|validation|random_point|0.1|7",
            "origin": origin,
            "dataset_id": "d",
            "family_id": "f",
            "item_id": "i",
            "split": "validation",
            "mechanism": "random_point",
            "missing_rate": 0.1,
            "mask_seed": 7,
            "period": 4,
        }

    return {
        "identity": {"config": {"context_length": 4, "horizon": 2}},
        "datasets": [{"dataset_id": "d", "items": [{"item_id": "i", "prefix_end": 8}]}],
        "episodes": [episode(14), episode(16)],
    }


def test_probe_end_never_exceeds_decision_and_training_prefix_is_excluded():
    source = source_manifest()
    plan = plan_recent_probes(source)
    probes = {probe["probe_id"]: probe for probe in plan["probes"]}
    assert len(plan["skipped"]) == 1
    for link in plan["links"]:
        probe = probes[link["probe_id"]]
        assert probe["origin"] - 4 >= 8
        validate_feedback_end(probe["origin"], probe["horizon"], link["origin"])
    assert any(
        probe["reusable_episode_id"] == source["episodes"][0]["episode_id"]
        for probe in plan["probes"]
    )
    with pytest.raises(ValueError, match="not arrived"):
        validate_feedback_end(14, 3, 16)


def test_screening_decisions_are_explicit_and_do_not_expand_the_evaluation():
    source = source_manifest()
    selected = [source["episodes"][1]["episode_id"]]
    plan = plan_recent_probes(source, decision_ids=selected)
    assert plan["decision_episode_ids"] == selected
    assert {link["episode_id"] for link in plan["links"]} == set(selected)
    with pytest.raises(ValueError, match="existing validation"):
        plan_recent_probes(source, decision_ids=["unknown"])


def test_only_observed_probe_values_enter_mae_mse_and_counts():
    truth = np.array([[0.0, np.nan], [2.0, 4.0], [np.nan, 6.0]])
    point = np.array([[[1.0, 999.0], [4.0, 5.0], [-999.0, 8.0]]])
    risks, counts, valid = observed_forecast_risk(point, truth, minimum=2)
    np.testing.assert_equal(counts, [2, 2])
    np.testing.assert_equal(valid, [True, True])
    np.testing.assert_allclose(risks["mae"], [[1.5, 1.5]])
    np.testing.assert_allclose(risks["mse"], [[2.5, 2.5]])
    point[0, 0, 1], point[0, 2, 0] = -1e20, 1e20
    second, _, _ = observed_forecast_risk(point, truth, minimum=2)
    np.testing.assert_equal(second["mse"], risks["mse"])


def test_incomplete_feedback_uses_reference_for_joint_and_only_missing_targets_for_independent():
    risks = {
        "mae": np.array([[2.0, np.nan], [1.0, np.nan]]),
        "mse": np.array([[4.0, np.nan], [1.0, np.nan]]),
    }
    normalizers = {"mae": 1.0, "mse": 1.0}
    assert select_feedback(risks, "joint", normalizers, 0, per_target=False) == 0
    assert select_feedback(risks, "joint", normalizers, 0, per_target=True) == (1, 0)
    with pytest.raises(ValueError, match="normalizers"):
        select_feedback(risks, "joint", {"mae": 0.0, "mse": 1.0}, 0, per_target=False)


def test_empty_feedback_cannot_be_mistaken_for_zero_risk():
    risks, counts, valid = observed_forecast_risk(np.empty((2, 0, 2)), np.empty((0, 2)), minimum=1)
    assert np.isnan(risks["mae"]).all()
    np.testing.assert_equal(counts, [0, 0])
    assert not valid.any()
    weights = observed_ensemble_weights(
        np.empty((2, 0, 2)), np.empty((0, 2)), minimum=1, reference=1
    )
    np.testing.assert_equal(weights, [0, 1])


def test_observed_ensemble_uses_balanced_target_errors_and_valid_convex_weights():
    point = np.array([[[2.0, 2.0], [2.0, 2.0]], [[-1.0, -1.0], [-1.0, -1.0]]])
    truth = np.zeros((2, 2))
    weights = observed_ensemble_weights(point, truth, minimum=1, reference=0)
    np.testing.assert_allclose(weights, [1 / 3, 2 / 3], atol=1e-5)
    shrunk = observed_ensemble_weights(point, truth, minimum=1, reference=0, shrinkage=0.5)
    np.testing.assert_allclose(shrunk, 0.5 * weights + 0.25, atol=1e-5)
    assert weights.min() >= 0 and abs(weights.sum() - 1) < 1e-10


def test_imputation_mixtures_preserve_observations_and_support_explicit_target_weights():
    context = np.array([[1.0, np.nan, 7.0], [np.nan, 2.0, 8.0]])
    candidates = np.array(
        [[[1.0, 10.0, 7.0], [3.0, 2.0, 8.0]], [[1.0, 20.0, 7.0], [9.0, 2.0, 8.0]]]
    )
    mixed = combine_imputations(context, candidates, np.array([0.25, 0.75]))
    np.testing.assert_allclose(mixed, [[1, 17.5, 7], [7.5, 2, 8]])
    target = combine_imputations(
        context, candidates, np.array([[1.0, 0.0], [0.0, 1.0]]), targets=[0, 1]
    )
    np.testing.assert_equal(target, [[1, 20, 7], [3, 2, 8]])
    with pytest.raises(ValueError, match="sum to one"):
        combine_imputations(context, candidates, np.array([1.0, 1.0]))
