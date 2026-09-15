from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def test_mixture_gradients_change_only_missing_inputs_and_preserve_the_simplex():
    script = Path(__file__).parents[2] / "scripts/probe_differentiable_imputation.py"
    spec = importlib.util.spec_from_file_location("differentiable_imputation_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = torch.tensor([[1.0, float("nan")], [float("nan"), 2.0]])
    candidates = torch.tensor([[[1.0, 4.0], [3.0, 2.0]], [[1.0, 8.0], [9.0, 2.0]]])
    logits = torch.zeros((2, 2), requires_grad=True)
    mixed, weights = module.compose(candidates, context, logits)
    assert torch.equal(mixed, torch.tensor([[1.0, 6.0], [6.0, 2.0]]))
    known = torch.isfinite(context)
    mixed[known].sum().backward(retain_graph=True)
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    logits.grad = None
    mixed[~known].square().sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert torch.allclose(weights.sum(0), torch.ones(2))


def test_teacher_target_is_detached_and_does_not_read_source_future():
    script = Path(__file__).parents[2] / "scripts/probe_differentiable_imputation.py"
    spec = importlib.util.spec_from_file_location("teacher_imputation_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecasts = torch.tensor([[[0.0]], [[8.0]], [[2.0]]], requires_grad=True)
    first = module.optimization_target(torch.zeros(1, 1), forecasts, "forecast_median")
    changed = module.optimization_target(torch.full((1, 1), 1e9), forecasts, "forecast_median")
    assert torch.equal(first, changed)
    assert torch.equal(first, torch.tensor([[2.0]]))
    assert not first.requires_grad
