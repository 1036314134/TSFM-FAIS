"""Canonical Chronos-2 input layout for differentiable median forecasting."""

import math

import torch


def chronos_median(pipeline, context, horizon, targets):
    if context.ndim != 2 or horizon < 1 or not targets or len(set(targets)) != len(targets):
        raise ValueError("a [L,D] context, positive horizon and distinct targets are required")
    if min(targets) < 0 or max(targets) >= context.shape[1]:
        raise ValueError("forecast target is outside the context")
    patches = math.ceil(horizon / pipeline.model_output_patch_size)
    if patches > pipeline.max_output_patches:
        raise ValueError("the differentiable path does not implement long-horizon unrolling")
    median = [index for index, value in enumerate(pipeline.quantiles) if abs(value - 0.5) < 1e-8]
    if len(median) != 1:
        raise ValueError("a unique trained median quantile is required")
    # The public dataset builder concatenates [D,L] inputs into contiguous storage.
    # A bare transpose produced different raw-model results in a verified case.
    canonical = context.to(device=pipeline.model.device, dtype=torch.float32).T.contiguous()
    prediction = pipeline.model(
        context=canonical,
        group_ids=torch.zeros(canonical.shape[0], device=canonical.device, dtype=torch.long),
        num_output_patches=patches,
    ).quantile_preds
    if prediction.shape[:2] != (context.shape[1], len(pipeline.quantiles)):
        raise ValueError("Chronos output must follow [D,Q,H]")
    return prediction[:, median[0], :horizon].T[:, targets].float()
