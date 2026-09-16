import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_matched_replay import independent_choices  # noqa: E402
from evaluate_matched_replay import replacement_probabilities, score_points  # noqa: E402
from matched_replay_core import (  # noqa: E402
    history_decisions,
    mask_rules,
    observed_risks,
    run_signature,
)
from matched_replay_sources import anchor_support  # noqa: E402


def test_mask_interventions_preserve_counts_and_position_control_structure():
    mask = np.zeros((96, 3), bool)
    mask[7:13, :2] = True
    mask[50:55, 2] = True
    variants, definition = mask_rules(mask, "known-query")
    np.testing.assert_array_equal(variants[2], mask)
    assert not np.array_equal(variants[1], mask)
    assert run_signature(variants[1]) == run_signature(mask)
    for variant in variants:
        np.testing.assert_array_equal(variant.sum(0), mask.sum(0))
    np.testing.assert_array_equal(
        variants[1].astype(int).T @ variants[1].astype(int), mask.astype(int).T @ mask.astype(int)
    )
    same, duplicate = mask_rules(mask, "known-query")
    np.testing.assert_array_equal(same, variants)
    assert definition == duplicate


def test_anchor_support_is_causal_and_requires_original_observations():
    values = np.zeros((2200, 3))
    mask = np.zeros((96, 3), bool)
    mask[-12:] = True
    first = anchor_support(values, mask, 1500, 100, 1024)
    assert len(first["selected"]) == 8
    assert all(origin + 96 <= 1500 and origin - 96 >= 100 for origin in first["selected"])
    changed = values.copy()
    changed[1500:] = np.nan
    assert anchor_support(changed, mask, 1500, 100, 1024) == first
    changed[1308:1404, :2] = np.nan
    after = anchor_support(changed, mask, 1500, 100, 1024)
    assert 1404 not in after["complete_context_origins"]
    assert 1308 not in after["future_observed_origins"]


def test_paired_choices_respect_default_joint_and_independent_units():
    loss = np.ones((8, 9, 2))
    choices, _, _ = history_decisions(loss, True)
    np.testing.assert_array_equal(choices["forced"], [0, 0])
    np.testing.assert_array_equal(choices["erm"], [8, 8])
    loss[:, 0, 0] = 0.4
    loss[:, 1, 1] = 0.4
    for joint in (True, False):
        choice, mean, uncertainty = history_decisions(loss, joint)
        independent, other_mean, other_uncertainty = independent_choices(loss, joint)
        np.testing.assert_array_equal(np.stack(list(choice.values())), independent)
        np.testing.assert_allclose(mean, other_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(uncertainty, other_uncertainty, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(choice["erm"], [0, 0] if joint else [0, 1])


def test_unknown_past_targets_never_count_as_imputed_truth():
    prediction = np.zeros((8, 9, 96, 2))
    truth = np.ones((8, 96, 2))
    truth[:, 48:, 1] = np.nan
    mae, mse = observed_risks(prediction, truth, np.ones(2))
    np.testing.assert_array_equal(mae, 1.0)
    np.testing.assert_array_equal(mse, 1.0)
    prediction[:, :, 48:, 1] = 1e12
    changed = observed_risks(prediction, truth, np.ones(2))
    np.testing.assert_array_equal(changed[0], mae)
    np.testing.assert_array_equal(changed[1], mse)


def test_random_action_risk_is_not_the_loss_of_the_average_prediction():
    bank = np.zeros((9, 96, 2))
    bank[:4] = -1
    bank[4:8] = 1
    _, mse = score_points(bank, np.zeros((96, 2)))
    probability = replacement_probabilities([0, 8])
    expected = (probability * mse).sum(0)
    np.testing.assert_array_equal(expected, [1.0, 0.0])
    _, averaged = score_points(bank[:8].mean(0)[None], np.zeros((96, 2)))
    np.testing.assert_array_equal(averaged, 0.0)
