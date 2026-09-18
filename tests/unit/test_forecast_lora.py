import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from forecast_lora_core import ForecastLoRA, smooth_mae  # noqa: E402


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.ModuleDict(
            {name: nn.Linear(3, 3, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.output_patch_embedding = nn.ModuleDict({"output_layer": nn.Linear(3, 2)})

    def forward(self, value):
        for layer in self.self_attention.values():
            value = layer(value)
        return self.output_patch_embedding["output_layer"](value)


def test_zero_adapter_identity_and_original_parameter_isolation():
    model = Toy().eval().requires_grad_(False)
    value = torch.randn(4, 3)
    original = model(value).clone()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    bank = ForecastLoRA(model)
    with bank.installed():
        torch.testing.assert_close(model(value), original, rtol=0, atol=0)
        model(value).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters())
        torch.optim.SGD(bank.parameters(), lr=0.1).step()
        assert not torch.equal(model(value), original)
    torch.testing.assert_close(model(value), original, rtol=0, atol=0)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)


def test_hook_cleanup_after_exception_and_no_duplicate_installation():
    model = Toy().eval().requires_grad_(False)
    bank = ForecastLoRA(model)
    with pytest.raises(RuntimeError, match="example"):
        with bank.installed():
            with pytest.raises(ValueError, match="already"):
                with bank.installed():
                    pass
            raise RuntimeError("example")
    assert not bank.handles
    assert all(not module._forward_hooks for _, module in bank.bindings)


def test_missing_future_labels_have_no_gradient_or_loss_contribution():
    point = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    truth = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    loss = smooth_mae(point, truth)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(point.grad).all()
    assert point.grad[0, 0] == 0 and point.grad[1, 1] == 0


def test_initialization_is_identical_between_capacity_controls():
    model = Toy().requires_grad_(False)
    one, two = ForecastLoRA(model), ForecastLoRA(model)
    assert one.projection_names == two.projection_names
    for first, second in zip(one.parameters(), two.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
