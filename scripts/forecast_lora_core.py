"""Standard external LoRA branches using the installed Chronos-2 target scope."""

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

SEED, RANK, ALPHA, STEPS_PER_SOURCE = 7402, 8, 16, 864
MODES = ("natural_lora", "corruption_lora")
SUFFIXES = (
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)


class LowRankBranch(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.a = nn.Parameter(torch.empty(RANK, inputs))
        self.b = nn.Parameter(torch.zeros(outputs, RANK))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, value):
        return F.linear(F.linear(value, self.a), self.b) * (ALPHA / RANK)


class ForecastLoRA(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.bindings = [
            (name, module)
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(SUFFIXES)
        ]
        if not self.bindings or not any(name == SUFFIXES[-1] for name, _ in self.bindings):
            raise ValueError("the registered Chronos LoRA projection scope is unavailable")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            self.branches = nn.ModuleDict(
                {
                    f"p{index:03d}": LowRankBranch(module.in_features, module.out_features)
                    for index, (_, module) in enumerate(self.bindings)
                }
            )
        self.handles = []

    @property
    def projection_names(self):
        return [name for name, _ in self.bindings]

    @contextmanager
    def installed(self):
        if self.handles:
            raise ValueError("LoRA hooks are already installed")
        for index, (_, module) in enumerate(self.bindings):
            branch = self.branches[f"p{index:03d}"]

            def hook(_module, arguments, result, *, adapter=branch):
                return result + adapter(arguments[0])

            self.handles.append(module.register_forward_hook(hook))
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


def smooth_mae(point, truth):
    valid = torch.isfinite(truth)
    count = valid.sum(0)
    if (count == 0).any() or not torch.isfinite(point).all():
        raise ValueError("finite predictions and observed targets are required")
    target = torch.where(valid, truth, torch.zeros_like(truth))
    values = F.smooth_l1_loss(point, target, beta=0.01, reduction="none")
    return ((values * valid).sum(0) / count).mean()


def predict_tensor(backbone, pipeline, context, horizon):
    return backbone(
        context=context,
        group_ids=torch.zeros(len(context), device=context.device, dtype=torch.long),
        num_output_patches=math.ceil(horizon / pipeline.model_output_patch_size),
    ).quantile_preds.float()


def update_once(bank, optimizer, backbone, pipeline, example, mode):
    context = torch.tensor(
        example["natural" if mode == "natural_lora" else "corrupted"], device=backbone.device
    )
    truth = torch.tensor(example["future"], device=backbone.device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    quantiles = predict_tensor(backbone, pipeline, context, 24)
    point = quantiles[: truth.shape[1], pipeline.quantiles.index(0.5), :24].T
    loss = smooth_mae(point, truth)
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters()):
        raise ValueError("an adapter gradient is missing or nonfinite")
    norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    optimizer.step()
    return float(loss.detach().cpu()), float(norm.detach().cpu())


def optimizer_for(bank):
    return torch.optim.AdamW(bank.parameters(), lr=1e-5, weight_decay=0.01)
