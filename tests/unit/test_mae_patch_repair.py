import sys
from pathlib import Path

import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from patch_repair_runtime import repair_loss as joint_loss  # noqa: E402
from train_mae_patch_repair import repair_loss  # noqa: E402


def test_mae_loss_has_bounded_outlier_gradient_and_no_remaining_mse_component():
    point = torch.tensor([[1.0, 100.0]], dtype=torch.float64, requires_grad=True)
    target, mean, scale = torch.zeros_like(point), torch.zeros(2), torch.ones(2)
    loss = repair_loss(point, target, mean, scale, point.new_zeros(()))
    gradient = torch.autograd.grad(loss, point)[0]
    assert bool((gradient.abs() <= 0.5).all())
    assert abs(float(gradient[0, 0] - gradient[0, 1])) < 1e-6
    other = point.detach().clone().requires_grad_()
    old_gradient = torch.autograd.grad(
        joint_loss(other, target, mean, scale, other.new_zeros(())), other
    )[0]
    assert float(old_gradient[0, 1]) > 50


def test_prefix_units_and_repair_energy_keep_the_registered_meaning():
    point = torch.tensor([[1.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[3.0, 1.0]], dtype=torch.float64)
    mean, scale = torch.tensor([2.0, 5.0]), torch.tensor([2.0, 3.0])
    penalty = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    original = repair_loss(point, target, mean, scale, penalty, weight=2.0)
    transformed = repair_loss(
        7 * point + 11, 7 * target + 11, 7 * mean + 11, 7 * scale, penalty, weight=2.0
    )
    torch.testing.assert_close(original, transformed, rtol=1e-12, atol=1e-12)
    derivative = torch.autograd.grad(original, penalty)[0]
    assert float(derivative) == 0.001
