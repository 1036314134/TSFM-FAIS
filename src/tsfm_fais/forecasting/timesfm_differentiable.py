# Copyright 2025 Google LLC
# Modifications for differentiable finite-input evaluation.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""TimesFM 2.5 median inference with input gradients for horizons up to 128.

The running-statistics equations follow the installed Google TimesFM backend.
Only the zero-variance derivative is defined explicitly; forward values remain
unchanged. Full-model parity still requires a real-checkpoint verification.
"""

from __future__ import annotations

import torch


class _SafeSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        root = value.sqrt()
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, gradient):
        (root,) = ctx.saved_tensors
        return torch.where(root > 0, gradient / (2 * root), torch.zeros_like(gradient))


def running_stats(n, mean, scale, values, mask):
    valid = ~mask
    added_n = valid.to(values.dtype).sum(-1)
    safe_added = torch.where(added_n == 0, 1.0, added_n)
    added_mean = (values * valid).sum(-1) / safe_added
    added_mean = torch.where(added_n == 0, 0.0, added_mean)
    added_var = (((values - added_mean.unsqueeze(-1)) ** 2) * valid).sum(-1) / safe_added
    added_var = torch.where(added_n == 0, 0.0, added_var)
    added_scale = _SafeSqrt.apply(added_var)
    total = n + added_n
    safe_total = torch.where(total == 0, 1.0, total)
    updated_mean = (n * mean + added_mean * added_n) / safe_total
    updated_mean = torch.where(total == 0, 0.0, updated_mean)
    variance = (
        n * scale.pow(2)
        + added_n * added_scale.pow(2)
        + n * (mean - updated_mean).pow(2)
        + added_n * (added_mean - updated_mean).pow(2)
    ) / safe_total
    variance = torch.where(total == 0, 0.0, variance)
    return total, updated_mean, _SafeSqrt.apply(torch.clamp(variance, min=0.0))


def timesfm_median(core, context, horizon, targets):
    """Match the repository's normalize/flip/positive median settings.

    Context must be completed [L,D]. No future values enter this function.
    Continuous-quantile adjustments and crossing repair leave channel 5 intact.
    The caller freezes the model parameters and keeps it in evaluation mode.
    """
    from timesfm.torch.util import revin

    if context.ndim != 2 or not targets or len(set(targets)) != len(targets):
        raise ValueError("context and distinct forecast targets are required")
    if min(targets) < 0 or max(targets) >= context.shape[1]:
        raise ValueError("forecast target is outside the context")
    if not 1 <= horizon <= core.o or core.q != 10 or core.aridx != 5:
        raise ValueError("this median path supports one TimesFM 2.5 output patch only")
    values = context[:, list(targets)].T.float()
    if not bool(torch.isfinite(values).all()) or values.shape[1] < 2:
        raise ValueError("differentiable TimesFM requires finite completed target histories")
    padding = (-values.shape[1]) % core.p
    values = torch.nn.functional.pad(values, (padding, 0))
    masks = torch.zeros_like(values, dtype=torch.bool)
    masks[:, :padding] = True
    positive = (values >= 0).all(-1, keepdim=True)
    mean = values.mean(-1, keepdim=True)
    scale = values.std(-1, keepdim=True)
    normalized = revin(values, mean, scale)

    def prefill(inputs):
        batch = inputs.shape[0]
        patches = inputs.reshape(batch, -1, core.p)
        patch_masks = masks.reshape(batch, -1, core.p)
        n = inputs.new_zeros(batch)
        mu, sigma = n.clone(), n.clone()
        means, scales = [], []
        for index in range(patches.shape[1]):
            n, mu, sigma = running_stats(n, mu, sigma, patches[:, index], patch_masks[:, index])
            means.append(mu)
            scales.append(sigma)
        means, scales = torch.stack(means, 1), torch.stack(scales, 1)
        inputs = torch.where(patch_masks, 0.0, revin(patches, means, scales))
        (_, _, output, _), _ = core(inputs, patch_masks, decode_caches=None)
        restored = revin(output, means, scales, reverse=True).reshape(batch, -1, core.o, core.q)
        return restored[:, -1, :horizon, 5]

    point = (prefill(normalized) - prefill(-normalized)) / 2
    point = revin(point, mean, scale, reverse=True)
    point = torch.where(positive, point.clamp_min(0), point)
    return point.T
