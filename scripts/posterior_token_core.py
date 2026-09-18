"""Static conditional uncertainty and moment propagation for the frozen ReLU input block."""

import hashlib
import math
from contextlib import contextmanager

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
import torch.nn.functional as functional
from dynamic_posterior_core import covariance_root
from normalization_interface import constant_statistics


def conditional_uncertainty(context, covariance, keep, jitter=1e-6):
    observed = np.isfinite(context)
    selected = np.flatnonzero(keep)
    covariances = np.empty((len(context), len(selected), len(selected)))
    roots = np.empty_like(covariances)
    cache = {}
    for t, mask in enumerate(observed):
        key = tuple(mask.tolist())
        if key not in cache:
            obs, missing = np.flatnonzero(mask), np.flatnonzero(~mask)
            posterior = covariance.copy()
            if len(obs):
                posterior -= covariance[:, obs] @ np.linalg.solve(
                    covariance[np.ix_(obs, obs)], covariance[obs]
                )
            posterior[obs, :] = 0
            posterior[:, obs] = 0
            posterior[missing, missing] -= jitter
            posterior = (posterior + posterior.T) / 2
            marginal = posterior[np.ix_(selected, selected)]
            root = covariance_root(marginal)
            root[mask[selected]] = 0
            marginal = root @ root.T
            cache[key] = marginal, root
        covariances[t], roots[t] = cache[key]
    unconditional = np.maximum(np.diag(covariance)[selected] - jitter, 0)[:, None] * (
        ~observed[:, selected].T
    )
    return covariances, roots, unconditional


def static_samples(mean, roots, observed, case_id):
    seed = hashlib.sha256(f"r37|6103|{case_id}".encode()).hexdigest()[:16]
    random = np.random.default_rng(int(seed, 16))
    length, dimension = roots.shape[:2]
    noise = random.standard_normal((8, length, dimension))
    delta = np.zeros_like(noise)
    for t in range(length):
        delta[:, t] = noise[:, t] @ roots[t].T
    delta[:, observed.T] = 0
    deviations = np.stack([delta, -delta], axis=1).reshape(16, length, dimension).transpose(0, 2, 1)
    samples = mean.astype(float)[None] + deviations
    samples[:, observed] = mean[observed]
    np.testing.assert_allclose(
        (samples[0::2] + samples[1::2]) / 2,
        np.broadcast_to(mean, samples[0::2].shape),
        rtol=0,
        atol=1e-12,
    )
    return np.ascontiguousarray(samples, dtype=np.float32), seed


def posterior_normalization(module, mean, variance):
    _, (loc, ordinary_scale) = module.forward(mean)
    value = mean.float()
    var_mean = (value - loc).square().nanmean(-1, keepdim=True)
    addition = variance.float().mean(-1, keepdim=True) * ((mean.shape[-1] - 1) / mean.shape[-1])
    scale = torch.where(addition > 0, torch.sqrt(var_mean + addition), ordinary_scale)
    return loc, scale, ordinary_scale, addition


def transformed_moments(module, mean, variance, loc, scale):
    point, _ = module.forward(mean, (loc, scale))
    center = (mean.double() - loc.double()) / scale.double()
    spread = variance.double() / scale.double().square()
    if module.use_arcsinh:
        nodes, weights = np.polynomial.hermite.hermgauss(9)
        nodes = torch.tensor(nodes, dtype=torch.float64, device=mean.device)
        weights = torch.tensor(weights / np.sqrt(np.pi), dtype=torch.float64, device=mean.device)
        transformed = torch.asinh(center[..., None] + torch.sqrt(2 * spread)[..., None] * nodes)
        average = (transformed * weights).sum(-1)
        var = ((transformed - average[..., None]).square() * weights).sum(-1)
    else:
        average, var = center, spread
    average = torch.where(variance > 0, average, point.double())
    var = torch.where(variance > 0, var, torch.zeros_like(var))
    return average, var


def gaussian_relu_mean(mean, variance):
    active = variance > 0
    scale = torch.sqrt(torch.where(active, variance, torch.ones_like(variance)))
    z = mean / scale
    cdf = 0.5 * torch.erfc(-z / math.sqrt(2))
    density = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    expected = torch.clamp_min(scale * density + mean * cdf, 0)
    return torch.where(active, expected, torch.relu(mean))


def moment_embedding(block, patches, value_mean, value_variance, mode):
    if block.use_layer_norm or not isinstance(block.act, torch.nn.ReLU) or block.training:
        raise ValueError(
            "R37 requires the registered eval-mode ReLU residual input block without LayerNorm"
        )
    size = patches.shape[-1] // 3
    expected = patches.clone()
    if mode in ("mean_only", "moment"):
        expected[..., size : 2 * size] = value_mean.reshape(*patches.shape[:-1], size).to(
            patches.dtype
        )
    if mode == "mean_only":
        output = block.forward(expected)
    elif mode in ("variance_only", "moment"):
        hidden_mean = block.hidden_layer(expected)
        value_weights = block.hidden_layer.weight[:, size : 2 * size].double()
        hidden_variance = functional.linear(
            value_variance.reshape(*patches.shape[:-1], size), value_weights.square()
        )
        activation_mean = gaussian_relu_mean(hidden_mean.double(), hidden_variance).to(
            patches.dtype
        )
        output = block.output_layer(activation_mean) + block.residual_layer(expected)
    else:
        raise ValueError("unregistered posterior embedding mode")
    original = block.forward(patches)
    active = value_variance.reshape(*patches.shape[:-1], size).sum(-1, keepdim=True) > 0
    return torch.where(active, output, original)


def mc_embedding(block, patches, normalized_samples, value_variance):
    size = patches.shape[-1] // 3
    samples = patches[None].expand(len(normalized_samples), *patches.shape).clone()
    samples[..., size : 2 * size] = normalized_samples.reshape(
        len(samples), *patches.shape[:-1], size
    ).to(patches.dtype)
    encoded = block.forward(samples)
    expected = encoded.double().mean(0).to(patches.dtype)
    active = value_variance.reshape(*patches.shape[:-1], size).sum(-1, keepdim=True) > 0
    return torch.where(active, expected, block.forward(patches))


@contextmanager
def fixed_embedding(block, replacement):
    record = {"calls": 0}

    def after(_module, args, output):
        record["calls"] += 1
        if record["calls"] == 1:
            if output.shape != replacement.shape:
                raise ValueError("posterior encoding changed the input block shape")
            return replacement
        return output

    handle = block.register_forward_hook(after)
    try:
        yield record
    finally:
        handle.remove()


def predicted_quantiles(
    backbone, context, horizon, patch_size, loc=None, scale=None, replacement=None
):
    from contextlib import nullcontext

    normalization = (
        constant_statistics(backbone.instance_norm, loc, scale)
        if loc is not None
        else nullcontext()
    )
    encoding = (
        fixed_embedding(backbone.input_patch_embedding, replacement)
        if replacement is not None
        else nullcontext()
    )
    with torch.inference_mode(), normalization, encoding as record:
        result = (
            backbone(
                context=context,
                group_ids=torch.zeros(len(context), dtype=torch.long, device=context.device),
                num_output_patches=int(np.ceil(horizon / patch_size)),
            )
            .quantile_preds.float()
            .cpu()
            .numpy()
        )
    if record is not None and record["calls"] != 2:
        raise ValueError("the registered two-call context/future input interface changed")
    if not np.isfinite(result).all():
        raise ValueError("posterior token prediction is nonfinite")
    return result
