import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reliability_attention_core import (  # noqa: E402
    conditional_reliability,
    patch_weights,
    weighted_mask,
)


def test_conditional_information_and_original_observation_limits():
    covariance = np.asarray([[1.000001, 0.8], [0.8, 1.000001]])
    observed = np.asarray([[True, False], [False, False], [True, True]])
    r, v = conditional_reliability(observed, covariance)
    np.testing.assert_allclose(r[0], [1, 0.64 / 1.000001])
    np.testing.assert_array_equal(r[1], [0, 0])
    np.testing.assert_array_equal(r[2], [1, 1])
    np.testing.assert_allclose(v[0, 1], 1 - 0.64 / 1.000001)


def test_constant_missing_variable_does_not_gain_information_from_regularization():
    r, v = conditional_reliability(np.asarray([[False, True]]), np.diag([1e-6, 1.000001]))
    np.testing.assert_array_equal(r, [[0, 1]])
    np.testing.assert_array_equal(v, [[0, 0]])


def test_unit_reliability_preserves_every_original_mask_entry():
    original = torch.tensor([[[[0.0, -1e30, 0.0, 0.0]]]])
    changed, fallback = weighted_mask(original, torch.ones((1, 1, 1, 4)))
    assert torch.equal(original, changed) and fallback == 0


def test_group_mask_layout_and_neutral_signed_zeros_are_preserved():
    original = torch.zeros((3, 3, 5)).permute(2, 0, 1).unsqueeze(1) * -1.0
    for weight in (1.0, 0.5):
        changed, _ = weighted_mask(original, torch.full((5, 1, 1, 3), weight))
        assert changed.stride() == original.stride()
        if weight == 1:
            assert torch.equal(torch.signbit(original), torch.signbit(changed))


def test_key_weighting_matches_normalized_probability_product_and_keeps_forbidden_keys():
    mask = torch.tensor([[[[0.0, 0.0, -1e30, 0.0]]]], dtype=torch.float64)
    weights = torch.tensor([[[[0.5, 0.25, 1.0, 1.0]]]], dtype=torch.float64)
    changed, fallback = weighted_mask(mask, weights)
    scores = torch.tensor([[[[0.3, -0.4, 3.0, 0.1]]]], dtype=torch.float64)
    original = torch.softmax(scores + mask, dim=-1)
    expected = original * weights
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.softmax(scores + changed, dim=-1), expected, rtol=1e-12, atol=1e-12
    )
    assert changed[0, 0, 0, 2] == mask[0, 0, 0, 2] and changed[0, 0, 0, 3] == 0
    assert fallback == 0


def test_no_evidence_group_rows_fall_back_without_opening_cross_group_links():
    low = torch.finfo(torch.float32).min
    original = torch.tensor([[[[0, 0, low], [0, 0, low], [low, low, 0]]]], dtype=torch.float32)
    changed, fallback = weighted_mask(original, torch.tensor([[[[0, 0, 1.0]]]]))
    assert torch.equal(original, changed) and fallback == 2


def test_shift_control_preserves_each_channels_patch_weight_multiset():
    values = np.arange(96).reshape(3, 32) / 100
    data = {"observed": np.ones((3, 32), dtype=bool), "reliability": values}
    ordinary, shifted = patch_weights(data, "conditional"), patch_weights(data, "shifted")
    np.testing.assert_array_equal(np.sort(ordinary, axis=1), np.sort(shifted, axis=1))
    assert not np.array_equal(ordinary, shifted)
