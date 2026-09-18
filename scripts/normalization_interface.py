"""Temporary statistical masks, independent of Chronos token availability."""

from contextlib import contextmanager

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch

MODES = ("observed", "location", "scale", "shifted")


def choose_statistics(module, context, original_mask, mode, target_rows=(0, 1)):
    if mode not in MODES or original_mask.shape != context.shape:
        raise ValueError("registered mode and aligned statistical mask required")
    mask = original_mask.to(device=context.device, dtype=torch.bool).clone()
    if mode == "shifted":
        for row in target_rows:
            mask[row] = torch.roll(mask[row], shifts=37, dims=-1)
    mask &= torch.isfinite(context)
    # Direct forward bypasses module hooks and preserves the installed numerical rule.
    _, ordinary = module.forward(context)
    _, observed = module.forward(torch.where(mask, context, torch.full_like(context, torch.nan)))
    count = mask.sum(-1, keepdim=True)
    fallback = (count < 2) | (observed[1] <= module.eps)
    observed_loc = torch.where(fallback, ordinary[0], observed[0])
    observed_scale = torch.where(fallback, ordinary[1], observed[1])
    loc = ordinary[0] if mode == "scale" else observed_loc
    scale = ordinary[1] if mode == "location" else observed_scale
    return (loc, scale), {
        "effective_mask": mask,
        "count": count,
        "fallback": fallback,
        "ordinary_loc": ordinary[0],
        "ordinary_scale": ordinary[1],
        "observed_loc": observed[0],
        "observed_scale": observed[1],
        "chosen_loc": loc,
        "chosen_scale": scale,
    }


@contextmanager
def statistical_mask(module, original_mask, mode, target_rows=(0, 1)):
    captured = {"calls": 0}

    def before(norm, args, kwargs):
        existing = args[1] if len(args) > 1 else kwargs.get("loc_scale")
        if existing is not None:
            return None
        if not args:
            raise ValueError("the registered Chronos context must be positional")
        chosen, details = choose_statistics(norm, args[0], original_mask, mode, target_rows)
        captured.update(details)
        captured["calls"] += 1
        return (args[0], chosen), {k: v for k, v in kwargs.items() if k != "loc_scale"}

    handle = module.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def constant_statistics(module, loc, scale):
    """Replay interface: no mask logic or statistical calculation is shared."""

    def before(_norm, args, kwargs):
        existing = args[1] if len(args) > 1 else kwargs.get("loc_scale")
        if existing is not None:
            return None
        return (args[0], (loc, scale)), {k: v for k, v in kwargs.items() if k != "loc_scale"}

    handle = module.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def numpy_statistics(context, original_mask, mode, eps, target_rows=(0, 1)):
    """Float64 reference independent of the installed float32 reductions."""
    x = np.asarray(context, dtype=float)
    mask = np.asarray(original_mask, dtype=bool).copy()
    if mode == "shifted":
        mask[list(target_rows)] = np.roll(mask[list(target_rows)], 37, axis=-1)
    mask &= np.isfinite(x)

    def moments(valid):
        count = valid.sum(-1, keepdims=True)
        loc = np.where(valid, x, 0).sum(-1, keepdims=True) / np.maximum(count, 1)
        var = np.where(valid, (x - loc) ** 2, 0).sum(-1, keepdims=True) / np.maximum(count, 1)
        unit = np.where(count == 0, 1, np.sqrt(var))
        return loc, np.where(unit == 0, eps, unit)

    ordinary = moments(np.isfinite(x))
    observed = moments(mask)
    fallback = (mask.sum(-1, keepdims=True) < 2) | (observed[1] <= eps)
    loc = np.where(fallback, ordinary[0], observed[0])
    scale = np.where(fallback, ordinary[1], observed[1])
    if mode == "location":
        scale = ordinary[1]
    if mode == "scale":
        loc = ordinary[0]
    return loc, scale, mask, fallback
