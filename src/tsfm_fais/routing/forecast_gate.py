"""Small shared gates trained on the risk of the returned forecast mixture."""

import numpy as np
import torch
from torch import nn

from .forecast_projection import forecast_geometry


def teacher_quadratics(points, relative_single_risks):
    """Recover exact mixture-risk coefficients from audited single-candidate risks."""
    _, _, energy, gram = forecast_geometry(points)
    risks = np.asarray(relative_single_risks, float)
    if risks.shape != energy.shape or not np.isfinite(risks).all():
        raise ValueError("one finite relative teacher risk per candidate is required")
    alignment = (energy - risks) / 2
    return gram, alignment


def gate_objective(weights, gram, alignment, kind):
    """Both losses omit the same parameter-independent reference-teacher MSE."""
    if kind == "ensemble":
        return torch.einsum("na,nab,nb->n", weights, gram, weights) - 2 * (weights * alignment).sum(
            1
        )
    if kind == "member":
        return (weights * (gram.diagonal(dim1=1, dim2=2) - 2 * alignment)).sum(1)
    raise ValueError("unknown matched forecast-gate objective")


def compose_forecasts(points, weights):
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if points.ndim != 3 or weights.shape != points.shape[:2]:
        raise ValueError("forecast vectors and weights must align")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.any(weights.sum(1) <= 0)
    ):
        raise ValueError("finite forecasts and nonnegative nonzero mixture weights are required")
    weights = weights / weights.sum(1, keepdims=True)
    return points[:, 0] + np.einsum("na,naq->nq", weights, points - points[:, :1])


class SharedForecastGate(nn.Module):
    """Use 33 visible features per candidate and a pooled candidate representation."""

    def __init__(self, features=33, candidates=7, hidden=16):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(1, 1, features))
        self.register_buffer("feature_scale", torch.ones(1, 1, features))
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.action_bias = nn.Parameter(torch.zeros(candidates))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def fit_normalization(self, train_features, family_weights):
        values, weights = np.asarray(train_features, float), np.asarray(family_weights, float)
        if (
            values.ndim != 3
            or weights.shape != (len(values),)
            or not np.isfinite(values).all()
            or not np.all(weights > 0)
        ):
            raise ValueError("normalization requires finite training features and positive weights")
        denominator = weights.sum() * values.shape[1]
        mean = (values * weights[:, None, None]).sum(axis=(0, 1)) / denominator
        variance = ((values - mean) ** 2 * weights[:, None, None]).sum(axis=(0, 1)) / denominator
        with torch.no_grad():
            self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32)[None, None])
            self.feature_scale.copy_(
                torch.as_tensor(np.maximum(np.sqrt(variance), 1e-6), dtype=torch.float32)[
                    None, None
                ]
            )

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (
            len(self.action_bias),
            self.feature_mean.shape[-1],
        ):
            raise ValueError("the fixed candidate-feature layout changed")
        normalized = ((features - self.feature_mean) / self.feature_scale).clamp(-10, 10)
        encoded = self.encoder(normalized)
        pooled = encoded.mean(1, keepdim=True).expand_as(encoded)
        logits = self.score(torch.cat([encoded, pooled], dim=2))[..., 0] + self.action_bias
        return logits.softmax(1)
