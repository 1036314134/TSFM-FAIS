import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import pytest
import torch
from chronos.chronos_bolt import InstanceNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from normalization_interface import (  # noqa: E402
    choose_statistics,
    numpy_statistics,
    statistical_mask,
)


@pytest.mark.parametrize("arcsinh", [False, True])
def test_imputed_values_do_not_change_observed_statistics(arcsinh):
    norm = InstanceNorm(use_arcsinh=arcsinh)
    x = torch.tensor([[1.0, 3.0, 2.0, 8.0], [-2.0, -1.0, 4.0, 5.0]])
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    a, _ = choose_statistics(norm, x, mask, "observed")
    changed = torch.where(mask, x, x * 100 + 90)
    b, _ = choose_statistics(norm, changed, mask, "observed")
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with statistical_mask(norm, mask, "observed"):
        scaled, stats = norm(x)
    torch.testing.assert_close(norm.inverse(scaled, stats), x, rtol=1e-6, atol=1e-6)


def test_insufficient_and_constant_observed_rows_keep_the_ordinary_statistics():
    norm = InstanceNorm()
    x = torch.tensor([[1.0, 4.0, 9.0], [2.0, 2.0, 8.0], [4.0, 2.0, 6.0]])
    mask = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
    _, ordinary = norm(x)
    actual, details = choose_statistics(norm, x, mask, "observed")
    assert details["fallback"].all()
    for left, right in zip(actual, ordinary, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_fully_observed_path_and_hook_removal_are_exact():
    norm = InstanceNorm(use_arcsinh=True)
    x = torch.tensor([[1.0, float("nan"), 5.0, 7.0], [0.0, 0.0, 0.0, 0.0]])
    before, stats = norm(x)
    hooks = len(norm._forward_pre_hooks)
    with statistical_mask(norm, torch.isfinite(x), "observed") as captured:
        actual, after_stats = norm(x)
        assert captured["calls"] == 1
    torch.testing.assert_close(actual, before, rtol=0, atol=0, equal_nan=True)
    for a, b in zip(stats, after_stats, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(norm._forward_pre_hooks) == hooks
    with pytest.raises(RuntimeError), statistical_mask(norm, torch.isfinite(x), "observed"):
        raise RuntimeError("deliberate interruption")
    assert len(norm._forward_pre_hooks) == hooks


@pytest.mark.parametrize("mode", ["observed", "location", "scale", "shifted"])
def test_torch_statistics_match_independent_reference_and_shift_preserves_counts(mode):
    rng = np.random.default_rng(341)
    x = rng.normal(size=(3, 192)).astype(np.float32)
    mask = rng.random(x.shape) > 0.3
    norm = InstanceNorm()
    actual, details = choose_statistics(norm, torch.tensor(x), torch.tensor(mask), mode)
    loc, scale, expected_mask, fallback = numpy_statistics(x, mask, mode, norm.eps)
    np.testing.assert_allclose(actual[0].numpy(), loc, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1].numpy(), scale, rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(details["effective_mask"].numpy(), expected_mask)
    np.testing.assert_array_equal(details["fallback"].numpy(), fallback)
    np.testing.assert_array_equal(expected_mask.sum(1), mask.sum(1))
    if mode == "shifted":
        np.testing.assert_array_equal(expected_mask[2], mask[2])
