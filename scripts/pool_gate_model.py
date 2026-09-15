"""Matched source fitting and inference for an expanded forecast candidate pool."""

import hashlib

import numpy as np
import torch
from audit_shared_forecast_gate import replay_network
from metric_source_gate import metric_objective
from train_shared_forecast_gate import predict_weights

from tsfm_fais.routing.forecast_gate import SharedForecastGate
from tsfm_fais.routing.utility import _family_weights


def fit_pool_gate(frame, features, points, target, gram, alignment, indices, seed, kind):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97, candidates=points.shape[1])
    weights = _family_weights(frame.iloc[indices])
    model.fit_normalization(features[indices], weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    x, g, b, w = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features[indices], gram[indices], alignment[indices], weights)
    ]
    p, y = (
        torch.tensor(points[indices], dtype=torch.float64),
        torch.tensor(target[indices], dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(25):
        total = 0.0
        for batch in torch.randperm(len(indices), generator=generator).split(128):
            loss = (
                metric_objective(model(x[batch]), p[batch], y[batch], g[batch], b[batch], kind)
                * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite matched metric objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite matched metric gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch + 1, "relative_training_loss": total / len(indices)})
    return {
        "state_dict": model.state_dict(),
        "initial_parameter_sha256": initial,
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest(),
        "training_origins": sorted(frame.iloc[indices].origin_id.unique()),
        "training_families": sorted(frame.iloc[indices].family_id.unique()),
        "history": history,
    }


def pool_probability(state, features):
    model = (
        SharedForecastGate(features=features.shape[2], candidates=features.shape[1])
        .eval()
        .requires_grad_(False)
    )
    model.load_state_dict(state)
    probability = predict_weights(model, features)
    np.testing.assert_array_equal(probability, replay_network(state, features))
    return probability
