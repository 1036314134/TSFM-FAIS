"""A small median-initialized predictor bounded by the candidate forecast envelope."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def position_inputs(base_features, points):
    """Inputs are [N,7,33] summaries and [N,7,H] candidate point forecasts."""
    base, points = np.asarray(base_features, float), np.asarray(points, float)
    if base.shape != (len(points), 7, 33) or points.ndim != 3 or points.shape[1] != 7:
        raise ValueError("aligned candidate summaries and forecast trajectories are required")
    if not np.isfinite(base).all() or not np.isfinite(points).all():
        raise ValueError("all observed inputs and bounded candidate forecasts must be finite")
    lower, upper, median = points.min(1), points.max(1), np.median(points, axis=1)
    span = upper - lower
    relative = ((points - median[:, None]) / np.maximum(span[:, None], 1e-6)).transpose(0, 2, 1)
    changes = np.diff(relative, axis=1, prepend=relative[:, :1])
    horizon = points.shape[2]
    position = np.broadcast_to(np.linspace(0, 1, horizon)[None, :, None], (len(points), horizon, 1))
    local = np.concatenate([relative, changes, np.log1p(span)[:, :, None], position], axis=2)
    context = np.concatenate([base.mean(1), local.mean(1)], axis=1)
    return {
        "context": np.ascontiguousarray(context, dtype=np.float32),
        "local": np.ascontiguousarray(local, dtype=np.float32),
        "lower": np.ascontiguousarray(lower),
        "upper": np.ascontiguousarray(upper),
        "median": np.ascontiguousarray(median),
    }


class PositionalPortfolio(nn.Module):
    def __init__(self, mode="local"):
        super().__init__()
        if mode not in ("local", "pooled"):
            raise ValueError("unknown matched position-input mode")
        self.mode = mode
        self.register_buffer("context_mean", torch.zeros(1, 49))
        self.register_buffer("context_scale", torch.ones(1, 49))
        self.register_buffer("local_mean", torch.zeros(1, 1, 16))
        self.register_buffer("local_scale", torch.ones(1, 1, 16))
        self.context_encoder = nn.Linear(49, 8)
        self.local_encoder = nn.Linear(24, 8)
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def fit_normalization(self, context, local, weights):
        context, local, weights = (
            np.asarray(context, float),
            np.asarray(local, float),
            np.asarray(weights, float),
        )
        if weights.shape != (len(context),) or not np.all(weights > 0):
            raise ValueError("one positive source weight per target decision is required")
        context_mean = (context * weights[:, None]).sum(0) / weights.sum()
        context_var = ((context - context_mean) ** 2 * weights[:, None]).sum(0) / weights.sum()
        denominator = weights.sum() * local.shape[1]
        local_mean = (local * weights[:, None, None]).sum((0, 1)) / denominator
        local_var = ((local - local_mean) ** 2 * weights[:, None, None]).sum((0, 1)) / denominator
        with torch.no_grad():
            self.context_mean.copy_(torch.as_tensor(context_mean, dtype=torch.float32)[None])
            self.context_scale.copy_(
                torch.as_tensor(np.maximum(np.sqrt(context_var), 1e-6), dtype=torch.float32)[None]
            )
            self.local_mean.copy_(torch.as_tensor(local_mean, dtype=torch.float32)[None, None])
            self.local_scale.copy_(
                torch.as_tensor(np.maximum(np.sqrt(local_var), 1e-6), dtype=torch.float32)[
                    None, None
                ]
            )

    def offsets(self, context, local):
        context = ((context - self.context_mean) / self.context_scale).clamp(-10, 10)
        local = ((local - self.local_mean) / self.local_scale).clamp(-10, 10)
        if self.mode == "pooled":
            local = local.mean(1, keepdim=True).expand(-1, local.shape[1], -1)
        encoded = F.relu(self.context_encoder(context))[:, None].expand(-1, local.shape[1], -1)
        hidden = F.relu(self.local_encoder(torch.cat([encoded, local], dim=-1)))
        return self.output(hidden).squeeze(-1)

    def forward(self, context, local, lower, upper, median):
        offset = self.offsets(context, local).to(torch.float64)
        return torch.clamp(median + offset, min=lower, max=upper)


def replay_offsets(state, context, local, mode):
    with torch.no_grad():
        context = torch.as_tensor(context, dtype=torch.float32)
        local = torch.as_tensor(local, dtype=torch.float32)
        context = ((context - state["context_mean"]) / state["context_scale"]).clamp(-10, 10)
        local = ((local - state["local_mean"]) / state["local_scale"]).clamp(-10, 10)
        if mode == "pooled":
            local = local.mean(1, keepdim=True).expand(-1, local.shape[1], -1)
        global_hidden = F.relu(
            F.linear(context, state["context_encoder.weight"], state["context_encoder.bias"])
        )
        inputs = torch.cat([global_hidden[:, None].expand(-1, local.shape[1], -1), local], dim=-1)
        hidden = F.relu(
            F.linear(inputs, state["local_encoder.weight"], state["local_encoder.bias"])
        )
        return (
            F.linear(hidden, state["output.weight"], state["output.bias"])
            .squeeze(-1)
            .numpy()
            .astype(float)
        )


def predict_position(model, inputs, indices=None):
    indices = np.arange(len(inputs["context"])) if indices is None else np.asarray(indices)
    outputs = []
    with torch.no_grad():
        for selection in np.array_split(indices, max(1, (len(indices) + 255) // 256)):
            context, local = inputs["context"][selection], inputs["local"][selection]
            offset = (
                model.offsets(torch.as_tensor(context), torch.as_tensor(local))
                .numpy()
                .astype(float)
            )
            np.testing.assert_array_equal(
                offset, replay_offsets(model.state_dict(), context, local, model.mode)
            )
            lower, upper, median = (
                inputs[name][selection] for name in ("lower", "upper", "median")
            )
            point = np.clip(median + offset, lower, upper)
            outputs.append(point)
    return np.concatenate(outputs)


def fit_position(inputs, teacher, weights, *, mode, seed, settings):
    torch.manual_seed(seed)
    model = PositionalPortfolio(mode)
    initial_sha = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    if sum(value.numel() for value in model.parameters()) != 609:
        raise ValueError("matched position-model capacity changed")
    model.fit_normalization(inputs["context"], inputs["local"], weights)
    tensors = {
        name: torch.as_tensor(
            value, dtype=torch.float32 if name in ("context", "local") else torch.float64
        )
        for name, value in inputs.items()
    }
    target, sample_weights = (
        torch.as_tensor(teacher, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
    )
    if target.shape != tensors["median"].shape or not bool(torch.isfinite(target).all()):
        raise ValueError("source teacher coordinates must match each target trajectory")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(settings["epochs"]):
        total, count = 0.0, 0
        for indices in torch.randperm(len(teacher), generator=generator).split(
            settings["batch_size"]
        ):
            point = model(**{name: value[indices] for name, value in tensors.items()})
            loss = (((point - target[indices]) ** 2).mean(1) * sample_weights[indices]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite positional teacher objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite positional teacher gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        history.append({"epoch": epoch + 1, "mean_training_teacher_mse": total / count})
    model.eval().requires_grad_(False)
    return model, history, initial_sha
