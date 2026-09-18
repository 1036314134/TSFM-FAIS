"""Fixed conditional-information key weighting for a frozen Chronos encoder."""

from contextlib import contextmanager

import numpy as np
import torch

MODES = {
    "gaussian_conditional_attention": ("gaussian", "conditional", "both", False),
    "gaussian_time_attention": ("gaussian", "conditional", "time", False),
    "gaussian_group_attention": ("gaussian", "conditional", "group", False),
    "gaussian_observation_attention": ("gaussian", "observed", "both", False),
    "gaussian_shifted_attention": ("gaussian", "shifted", "both", False),
    "native_observation_attention": ("native", "observed", "both", False),
    "gaussian_provenance": ("gaussian", "conditional", "none", True),
    "gaussian_provenance_attention": ("gaussian", "conditional", "both", True),
}


def conditional_reliability(observed, covariance):
    prior = covariance - 1e-6 * np.eye(len(covariance))
    diagonal = np.diag(prior)
    if (diagonal < -1e-10).any():
        raise ValueError("negative prior marginal variance")
    reliability = observed.astype(float)
    variances = np.zeros(observed.shape, dtype=float)
    for t, mask in enumerate(observed):
        seen, missing = np.flatnonzero(mask), np.flatnonzero(~mask)
        if not len(missing):
            continue
        conditional = prior[np.ix_(missing, missing)].copy()
        if len(seen):
            cross = prior[np.ix_(missing, seen)]
            conditional -= cross @ np.linalg.solve(covariance[np.ix_(seen, seen)], cross.T)
        values = np.diag(conditional)
        if (values < -1e-10).any():
            raise ValueError("negative conditional marginal variance")
        variances[t, missing] = np.maximum(values, 0)
        active = diagonal[missing] > 1e-12
        reliability[t, missing[active]] = np.clip(
            1 - values[active] / diagonal[missing[active]], 0, 1
        )
    return reliability, variances


def patch_weights(data, kind):
    cells = data["observed"].astype(float) if kind == "observed" else data["reliability"]
    weights = cells.reshape(len(cells), -1, 16).mean(-1)
    return np.roll(weights, 1, axis=1).copy() if kind == "shifted" else weights


def weighted_mask(mask, key_weights):
    if not torch.isfinite(key_weights).all() or (key_weights < 0).any() or (key_weights > 1).any():
        raise ValueError("reliability weights must lie in [0,1]")
    available = mask == 0
    weights = key_weights.expand_as(mask)
    valid = available & (weights > 0)
    positive = torch.where(weights > 0, weights, torch.ones_like(weights))
    bias = torch.where(
        weights > 0, torch.log(positive), torch.full_like(weights, torch.finfo(mask.dtype).min)
    )
    changed = torch.where(available & (weights != 1), bias, mask)
    fallback = ~valid.any(-1, keepdim=True)
    result = torch.empty_strided(mask.size(), mask.stride(), dtype=mask.dtype, device=mask.device)
    result.copy_(torch.where(fallback, mask, changed))
    return result, int((fallback & available.any(-1, keepdim=True)).sum().item())


@contextmanager
def attention_intervention(backbone, data, mode, records=None, *, neutral=False):
    _, kind, scope, provenance = MODES[mode]
    raw_weights = np.ones_like(patch_weights(data, kind)) if neutral else patch_weights(data, kind)
    weights = torch.tensor(raw_weights, device=backbone.device, dtype=backbone.dtype)
    observed = torch.tensor(data["observed"], device=backbone.device, dtype=backbone.dtype)
    n_patches, channels = weights.shape[1], len(weights)
    saved = records if records is not None else {}
    calls = {"blocks": 0, "embeddings": 0}

    def before_block(_block, args, kwargs):
        original_time, original_group = kwargs["attention_mask"], kwargs["group_time_mask"]
        key_count = original_time.shape[-1]
        if key_count < n_patches or original_time.shape[0] != channels:
            raise ValueError("unexpected context attention shape")
        extended = torch.cat(
            [
                weights,
                torch.ones(
                    (channels, key_count - n_patches), device=weights.device, dtype=weights.dtype
                ),
            ],
            dim=-1,
        )
        time_mask, time_fallback = original_time, 0
        group_mask, group_fallback = original_group, 0
        if scope in ("time", "both"):
            time_mask, time_fallback = weighted_mask(original_time, extended[:, None, None, :])
        if scope in ("group", "both"):
            group_mask, group_fallback = weighted_mask(original_group, extended.T[:, None, None, :])
        if calls["blocks"] == 0:
            for name, value in (
                ("original_time_mask", original_time),
                ("original_group_mask", original_group),
                ("time_mask", time_mask),
                ("group_mask", group_mask),
            ):
                saved[name] = value.detach().cpu().numpy().copy()
            saved["time_fallback_rows"] = np.asarray(time_fallback)
            saved["group_fallback_rows"] = np.asarray(group_fallback)
        calls["blocks"] += 1
        return args, {**kwargs, "attention_mask": time_mask, "group_time_mask": group_mask}

    def before_embedding(_block, args):
        fields = args[0]
        calls["embeddings"] += 1
        if calls["embeddings"] != 1:
            return args
        if fields.shape[:2] != (channels, n_patches) or fields.shape[-1] != 48:
            raise ValueError("the registered context input block changed")
        updated = fields.clone()
        updated[..., 32:] = observed.reshape(channels, n_patches, 16)
        saved["original_embedding_fields"] = fields.detach().cpu().numpy().copy()
        saved["embedding_fields"] = updated.detach().cpu().numpy().copy()
        return (updated, *args[1:])

    handles = [
        block.register_forward_pre_hook(before_block, with_kwargs=True)
        for block in backbone.encoder.block
    ]
    if provenance:
        handles.append(backbone.input_patch_embedding.register_forward_pre_hook(before_embedding))
    try:
        yield saved
    finally:
        for handle in handles:
            handle.remove()
    if calls["blocks"] != len(backbone.encoder.block) or (provenance and calls["embeddings"] != 2):
        raise ValueError("the registered encoder invocation count changed")
