"""Matched forecast gates with candidate identity and observed forecast geometry."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_shared_forecast_gate import SETTINGS  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate, gate_objective  # noqa: E402
from tsfm_fais.routing.forecast_projection import forecast_geometry  # noqa: E402


def geometry_features(base, points, *, mode):
    base, points = np.asarray(base), np.asarray(points)
    if base.shape != (len(points), 7, 33) or points.ndim != 3 or points.shape[1] != 7:
        raise ValueError("aligned seven-candidate feature and forecast arrays are required")
    if mode not in ("full", "diagonal") or not np.isfinite(base).all():
        raise ValueError("a declared geometry mode and finite base features are required")
    _, _, _, gram = forecast_geometry(points)
    if mode == "diagonal":
        gram = gram * np.eye(7)[None]
    relation = np.sign(gram) * np.log1p(abs(gram))
    identity = np.broadcast_to(np.eye(7), (len(points), 7, 7))
    result = np.ascontiguousarray(
        np.concatenate([base, identity, relation], axis=2), dtype=np.float32
    )
    if result.shape != (len(points), 7, 47) or not np.isfinite(result).all():
        raise ValueError("geometry feature layout changed")
    return result


def fit_geometry_gate(features, gram, alignment, family_weights, *, seed):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=47, hidden=SETTINGS["hidden"])
    initial_sha = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    if sum(value.numel() for value in model.parameters()) != 1320:
        raise ValueError("matched gate capacity changed")
    model.fit_normalization(features, family_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    tensors = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features, gram, alignment, family_weights)
    ]
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(SETTINGS["epochs"]):
        order = torch.randperm(len(features), generator=generator)
        total, count = 0.0, 0
        for indices in order.split(SETTINGS["batch_size"]):
            x, g, b, weights = [value[indices] for value in tensors]
            loss = (gate_objective(model(x), g, b, "ensemble") * weights).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite geometry-gate objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite geometry-gate gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        history.append(
            {"epoch": epoch + 1, "mean_training_batch_relative_objective": total / count}
        )
    model.eval().requires_grad_(False)
    return model, history, initial_sha
