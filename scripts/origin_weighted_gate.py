"""Keep source normalization fixed while isolating training-loss row weights."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_shared_forecast_gate import SETTINGS  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate, gate_objective  # noqa: E402


def fit_weighted_gate(features, gram, alignment, normalization_weights, loss_weights, *, seed):
    if np.asarray(normalization_weights).shape != (len(features),) or np.asarray(
        loss_weights
    ).shape != (len(features),):
        raise ValueError("one normalization and loss weight per training decision is required")
    if not np.all(np.asarray(loss_weights) > 0):
        raise ValueError("training loss weights must remain positive")
    torch.manual_seed(seed)
    model = SharedForecastGate(hidden=SETTINGS["hidden"])
    model.fit_normalization(features, normalization_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    tensors = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features, gram, alignment, loss_weights)
    ]
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(SETTINGS["epochs"]):
        order = torch.randperm(len(features), generator=generator)
        total, count, clipped, batches = 0.0, 0, 0, 0
        for indices in order.split(SETTINGS["batch_size"]):
            x, g, b, sample = [value[indices] for value in tensors]
            probability = model(x)
            loss = (gate_objective(probability, g, b, "ensemble") * sample).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite weighted-gate objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite weighted-gate gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
            clipped += int(float(norm) > SETTINGS["gradient_norm"])
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_batch_relative_objective": total / count,
                "clipped_batch_fraction": clipped / batches,
            }
        )
    model.eval()
    return model, history
