"""Scale a bounded positional correction by visible forecast disagreement."""

import hashlib

import numpy as np
import torch
from position_objective import position_loss
from positional_forecast_portfolio import PositionalPortfolio, replay_offsets

SCALES = ("unit", "range", "mad")


def forecast_scale(points, kind):
    points = np.asarray(points, float)
    if points.ndim != 3 or points.shape[1] != 7 or not np.isfinite(points).all():
        raise ValueError("finite seven-candidate target trajectories are required")
    if kind == "unit":
        return np.ones((len(points), points.shape[2]))
    if kind == "range":
        return points.max(1) - points.min(1)
    if kind == "mad":
        return np.median(abs(points - np.median(points, axis=1)[:, None]), axis=1)
    raise ValueError("unregistered disagreement scale")


class ScaledPortfolio(PositionalPortfolio):
    def forward(self, context, local, lower, upper, median, scale):
        offset = self.offsets(context, local).to(torch.float64)
        return torch.clamp(median + scale * torch.tanh(offset), min=lower, max=upper)


def decode(inputs, scale, offset):
    return np.clip(inputs["median"] + scale * np.tanh(offset), inputs["lower"], inputs["upper"])


def predict_scaled(model, inputs, scale, indices):
    if scale.shape != inputs["median"].shape or not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("a visible forecast scale is invalid")
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
            outputs.append(
                decode(
                    {name: value[selection] for name, value in inputs.items()},
                    scale[selection],
                    offset,
                )
            )
    return np.concatenate(outputs)


def fit_scaled(inputs, scale, target, weights, *, mode, seed, settings):
    if scale.shape != inputs["median"].shape or (scale < 0).any() or not np.isfinite(scale).all():
        raise ValueError("the source forecast scale is invalid")
    torch.manual_seed(seed)
    model = ScaledPortfolio(mode)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    if sum(value.numel() for value in model.parameters()) != 609:
        raise ValueError("scaling changed model capacity")
    model.fit_normalization(inputs["context"], inputs["local"], weights)
    tensors = {
        name: torch.as_tensor(
            value, dtype=torch.float32 if name in ("context", "local") else torch.float64
        )
        for name, value in {**inputs, "scale": scale}.items()
    }
    target, weights = (
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
    )
    if target.shape != tensors["median"].shape or not bool(torch.isfinite(target).all()):
        raise ValueError("unavailable source labels entered a scaled fit")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(settings["epochs"]):
        total, count = 0.0, 0
        for indices in torch.randperm(len(target), generator=generator).split(
            settings["batch_size"]
        ):
            prediction = model(**{name: value[indices] for name, value in tensors.items()})
            loss = (position_loss(prediction, target[indices], "joint") * weights[indices]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite scaled objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite scaled gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        history.append({"epoch": epoch + 1, "mean_training_loss": total / count})
    return model.eval().requires_grad_(False), history, initial
