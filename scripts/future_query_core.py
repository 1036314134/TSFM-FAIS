"""Fixed future-key role masks for the existing Chronos encoder geometry."""

import hashlib
from contextlib import contextmanager

import numpy as np
import torch

POLICIES = ("aux_both", "aux_time", "aux_group", "readonly")
DEFINITIONS = {
    "query_aux_both_native_long": ("native_long", "aux_both"),
    "query_aux_time_native_long": ("native_long", "aux_time"),
    "query_aux_group_native_long": ("native_long", "aux_group"),
    "query_readonly_native_long": ("native_long", "readonly"),
    "query_aux_both_native_short": ("native_short", "aux_both"),
    "query_aux_both_gaussian_short": ("gaussian_short", "aux_both"),
}


def exact_clone(value):
    copy = torch.empty_strided(value.size(), value.stride(), dtype=value.dtype, device=value.device)
    return copy.copy_(value)


def role_masks(time_mask, group_mask, targets, future_start, policy):
    rows, _, _, size = time_mask.shape
    if time_mask.shape != (rows, 1, 1, size) or group_mask.shape != (size, 1, rows, rows):
        raise ValueError("unregistered Chronos attention geometry")
    if not 1 <= targets <= rows or not 0 < future_start < size:
        raise ValueError("invalid target set or future boundary")
    if policy not in (*POLICIES, "none"):
        raise ValueError("unregistered future-key policy")
    if policy == "none" or (targets == rows and policy != "readonly"):
        return time_mask, group_mask
    time, group = time_mask, group_mask
    if policy in ("aux_time", "aux_both", "readonly"):
        time = exact_clone(time_mask)
        start = 0 if policy == "readonly" else targets
        time[start:, ..., future_start:] = torch.finfo(time.dtype).min
    if policy in ("aux_group", "aux_both", "readonly"):
        group = exact_clone(group_mask)
        if policy == "readonly":
            off_diagonal = ~torch.eye(rows, device=group.device, dtype=torch.bool)
            group[future_start:].masked_fill_(
                off_diagonal[None, None], torch.finfo(group.dtype).min
            )
        else:
            group[future_start:, ..., targets:] = torch.finfo(group.dtype).min
    return time, group


def array_digest(value):
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def field_digests(fields):
    return {name: array_digest(value) for name, value in fields.items()}


@contextmanager
def future_key_policy(backbone, context, horizon, targets, policy, records=None, perturb=False):
    patch = backbone.chronos_config.input_patch_size
    future_start = int(np.ceil(context.shape[1] / patch)) + int(
        backbone.chronos_config.use_reg_token
    )
    expected_future = int(np.ceil(horizon / backbone.chronos_config.output_patch_size))
    trace = records if records is not None else {}
    count = [0]

    def before_block(_block, args, kwargs):
        original_time, original_group = kwargs["attention_mask"], kwargs["group_time_mask"]
        if original_time.shape[-1] != future_start + expected_future:
            raise ValueError("the future-query boundary changed")
        time, group = role_masks(original_time, original_group, targets, future_start, policy)
        if count[0] == 0:
            for name, value in (
                ("original_time_mask", original_time),
                ("original_group_mask", original_group),
                ("time_mask", time),
                ("group_mask", group),
            ):
                trace[name] = value.detach().cpu().numpy().copy()
            for name, value in (
                ("original_time_stride", original_time),
                ("original_group_stride", original_group),
                ("time_stride", time),
                ("group_stride", group),
            ):
                trace[name] = np.asarray(value.stride(), dtype=np.int64)
            trace["future_start"] = np.asarray(future_start)
            trace["targets"] = np.asarray(targets)
        count[0] += 1
        return args, {**kwargs, "attention_mask": time, "group_time_mask": group}

    def perturb_future(_encoder, args, kwargs):
        embeddings = kwargs["inputs_embeds"]
        if targets >= len(embeddings):
            raise ValueError("the preflight perturbation requires an auxiliary row")
        altered = exact_clone(embeddings)
        index = torch.arange(embeddings.shape[-1], device=embeddings.device, dtype=torch.float32)
        change = (0.125 * torch.sin(0.37 * index)).to(embeddings.dtype)
        altered[targets:, future_start:] += change
        return args, {**kwargs, "inputs_embeds": altered}

    handles = [
        block.register_forward_pre_hook(before_block, with_kwargs=True)
        for block in backbone.encoder.block
    ]
    if perturb:
        handles.append(backbone.encoder.register_forward_pre_hook(perturb_future, with_kwargs=True))
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()
    if count[0] != len(backbone.encoder.block):
        raise ValueError("not every registered encoder block was visited exactly once")
