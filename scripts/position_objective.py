"""Keep the registered positional model while changing its supervised loss."""

import hashlib

import torch
from positional_forecast_portfolio import PositionalPortfolio


def position_loss(point, target, objective):
    error = point - target
    if objective == "mse":
        return ((point - target) ** 2).mean(1)
    if objective == "joint":
        return (torch.sqrt(error.square() + 0.001**2) + error.square()).mean(1) / 2
    raise ValueError("unregistered positional loss")


def fit_position_loss(inputs, teacher, weights, *, mode, seed, settings, objective):
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
            loss = (
                position_loss(point, target[indices], objective) * sample_weights[indices]
            ).mean()
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
        history.append({"epoch": epoch + 1, "mean_training_loss": total / count})
    model.eval().requires_grad_(False)
    return model, history, initial_sha
