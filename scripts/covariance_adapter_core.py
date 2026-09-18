"""Bounded low-rank covariance correction on an immutable conditional-mean anchor."""

import math

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from dynamic_posterior_core import condition_state, regular_covariance


def fit_static(prefix):
    mean, scale = np.nanmean(prefix, 0), np.nanstd(prefix, 0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    z = (prefix - mean) / scale
    valid = np.isfinite(z).all(1)
    if valid.sum() < 128:
        raise ValueError("the static source covariance lacks complete original observations")
    return {
        "mean": mean,
        "scale": scale,
        "center": z[valid].mean(0),
        "covariance": regular_covariance(z[valid]),
        "support": np.asarray(valid.sum()),
    }


def static_values(context, model):
    z = (context - model["mean"]) / model["scale"]
    repaired = np.stack(
        [condition_state(model["center"], model["covariance"], row)[0] for row in z]
    )
    values = repaired * model["scale"] + model["mean"]
    observed = np.isfinite(context)
    values[observed] = context[observed]
    return values


class CovarianceAdapter(torch.nn.Module):
    def __init__(self, dimension=17, rank=2, seed=5101):
        super().__init__()
        projection = np.linalg.qr(
            np.random.default_rng(seed).standard_normal((dimension, rank)), mode="reduced"
        )[0]
        self.correction = torch.nn.Parameter(torch.zeros((dimension, rank), dtype=torch.float64))
        self.register_buffer("projection", torch.tensor(projection, dtype=torch.float64))
        self.register_buffer("identity", torch.eye(dimension, dtype=torch.float64))
        self.amplitude = 0.5 / math.sqrt(dimension * rank)

    def transform(self):
        return self.identity + self.amplitude * torch.tanh(self.correction) @ self.projection.T

    def forward(self, covariance):
        transform = self.transform()
        return (transform @ covariance @ transform.T).contiguous()

    def penalty(self):
        return (self.transform() - self.identity).square().sum()


def geometry(data, model, device="cuda"):
    z = (data["context"] - data["mean"]) / data["scale"]
    mask = np.isfinite(z)
    original = torch.tensor(z, dtype=torch.float64, device=device).contiguous()
    covariance = torch.tensor(model["covariance"], dtype=torch.float64, device=device).contiguous()
    center = torch.tensor(model["center"], dtype=torch.float64, device=device)
    groups = []
    patterns, assignment = np.unique(mask, axis=0, return_inverse=True)
    for index, pattern in enumerate(patterns):
        obs, missing = np.flatnonzero(pattern), np.flatnonzero(~pattern)
        if len(obs) == 0 or len(missing) == 0:
            continue
        times = torch.tensor(np.flatnonzero(assignment == index), device=device, dtype=torch.long)
        obs = torch.tensor(obs, device=device, dtype=torch.long)
        missing = torch.tensor(missing, device=device, dtype=torch.long)
        rhs = (original[times][:, obs] - center[obs]).T.contiguous()
        base = covariance[missing][:, obs] @ torch.linalg.solve(
            covariance[obs][:, obs].contiguous(), rhs
        )
        groups.append(
            {"times": times, "observed": obs, "missing": missing, "rhs": rhs, "base": base}
        )
    base_z = ((data["base_values"] - data["mean"]) / data["scale"]).astype(np.float32)
    return {
        "original": original.float(),
        "mask": torch.tensor(mask, device=device),
        "anchor": torch.tensor(base_z, device=device).contiguous(),
        "covariance": covariance,
        "groups": groups,
        "keep": torch.tensor(data["keep"], device=device),
    }


def repaired_context(adapter, prepared):
    updated = adapter(prepared["covariance"])
    delta = torch.zeros_like(prepared["original"], dtype=torch.float64)
    for group in prepared["groups"]:
        observed, missing, times = group["observed"], group["missing"], group["times"]
        change = (
            updated[missing][:, observed]
            @ torch.linalg.solve(updated[observed][:, observed].contiguous(), group["rhs"])
            - group["base"]
        )
        delta = delta.index_put((times[:, None], missing[None, :]), change.T)
    result = torch.where(prepared["mask"], prepared["original"], prepared["anchor"] + delta.float())
    return result


def masked_smooth_mae(prediction, truth):
    valid = torch.isfinite(truth)
    count = valid.sum(0)
    active = count > 0
    if not bool(active.any()):
        raise ValueError("a source objective has no original observed labels")
    safe = torch.where(valid, truth, torch.zeros_like(truth))
    errors = torch.sqrt((prediction - safe).square() + 1e-6)
    per_channel = (errors * valid).sum(0) / count.clamp_min(1)
    return per_channel[active].mean()


def forecast_mix(bank, weights):
    """Canonical sequential reduction; weights follow [method,target]."""
    if bank.ndim != 3 or weights.shape != (len(bank), bank.shape[-1]):
        raise ValueError("forecast portfolio and target weights do not align")
    result = np.zeros(bank.shape[1:], dtype=float)
    for values, weight in zip(bank, weights, strict=True):
        result += values * weight[None, :]
    return result
