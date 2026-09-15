"""Observed-future joint risk with the original full-observation calculation preserved."""

import hashlib

import numpy as np
import torch
from metric_source_gate import EPSILON, metric_objective, polish_simplex
from native_source_transfer_io import masked_geometry, observed_weights
from scipy.optimize import minimize

from tsfm_fais.routing.forecast_gate import SharedForecastGate, gate_objective
from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights


def observed_geometry(points, truth, observed, joint):
    full = np.asarray(observed, bool).all(1)
    gram = np.empty((len(points), 8, 8))
    alignment = np.empty((len(points), 8))
    if full.any():
        gram[full] = forecast_geometry(points[full])[3]
        alignment[full] = projection_targets(points[full], truth[full])["raw_projection"]
    if (~full).any():
        gram[~full], alignment[~full] = masked_geometry(
            points[~full], truth[~full], observed[~full], joint=joint
        )
    return gram, alignment, observed_weights(observed, joint=joint, minimum=48)


def observed_objective(probability, points, target, gram, alignment, observed, coordinates):
    if bool(observed.all()):
        return metric_objective(probability, points, target, gram, alignment, "joint")
    target = torch.where(observed, target, torch.zeros_like(target))
    prediction = points[:, 0] + torch.einsum(
        "na,naq->nq", probability.to(points.dtype), points - points[:, :1]
    )
    absolute = (torch.sqrt((prediction - target).square() + EPSILON**2) * coordinates).sum(1)
    return (absolute + gate_objective(probability, gram, alignment, "ensemble")) / 2


def fit_observed_gate(frame, data, indices, seed, updates):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97, candidates=8)
    weights = _family_weights(frame.iloc[indices])
    model.fit_normalization(data["features"][indices], weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    x, g, b, w = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (
            data["features"][indices],
            data["gram"][indices],
            data["alignment"][indices],
            weights,
        )
    ]
    p = torch.tensor(data["points"][indices], dtype=torch.float64)
    y = torch.tensor(
        np.where(data["observed"][indices], data["truth"][indices], 0.0), dtype=torch.float64
    )
    observed = torch.as_tensor(data["observed"][indices], dtype=torch.bool)
    coordinates = torch.tensor(data["coordinates"][indices], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 100000)
    steps, epoch, history = 0, 0, []
    while steps < updates:
        epoch += 1
        total, count = 0.0, 0
        for batch in torch.randperm(len(indices), generator=generator).split(128):
            loss = (
                observed_objective(
                    model(x[batch]),
                    p[batch],
                    y[batch],
                    g[batch],
                    b[batch],
                    observed[batch],
                    coordinates[batch],
                )
                * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite observed-future objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite observed-future gradient")
            optimizer.step()
            steps += 1
            total += float(loss.detach()) * len(batch)
            count += len(batch)
            if steps == updates:
                break
        history.append({"epoch": epoch, "relative_training_loss": total / count})
    return {
        "state_dict": model.state_dict(),
        "initial_parameter_sha256": initial,
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest(),
        "training_origins": sorted(frame.iloc[indices].origin_id.unique()),
        "training_families": sorted(frame.iloc[indices].family_id.unique()),
        "history": history,
        "updates": steps,
    }


def observed_fixed_objective(points, truth, coordinates, row_weights):
    weights = np.asarray(row_weights, float)
    weights = weights / weights.sum()
    median = np.median(points, axis=1)
    eligible = coordinates.ravel() > 0
    matrix = np.ascontiguousarray(
        (points - median[:, None]).transpose(0, 2, 1).reshape(-1, 8)[eligible]
    )
    residual = (median - np.where(coordinates > 0, truth, 0.0)).ravel()[eligible]
    weight = (coordinates * weights[:, None]).ravel()[eligible]

    def objective(probability):
        error = residual + matrix @ probability
        magnitude = np.sqrt(error**2 + EPSILON**2)
        return float(weight @ ((magnitude + error**2) / 2)), matrix.T @ (
            weight * (0.5 * error / magnitude + error)
        )

    return objective


def fit_observed_fixed(frame, data, indices):
    weights = _family_weights(frame.iloc[indices])
    probability = weights / weights.sum()
    g = np.einsum("n,nab->ab", probability, data["gram"][indices])
    b = np.einsum("n,na->a", probability, data["alignment"][indices])
    initial = simplex_quadratic_weights(g[None], b[None])[0][0]
    objective = observed_fixed_objective(
        data["points"][indices], data["truth"][indices], data["coordinates"][indices], weights
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=[(0, 1)] * 8,
        constraints=[
            {
                "type": "eq",
                "fun": lambda value: value.sum() - 1,
                "jac": lambda value: np.ones_like(value),
            }
        ],
        options={"maxiter": 2000, "ftol": 1e-13},
    )
    values = np.maximum(result.x, 0)
    values /= values.sum()
    if np.max(abs(values - result.x)) > 1e-8:
        raise ValueError("observed fixed optimizer returned infeasible weights")
    values, polish = polish_simplex(objective, values)
    value, gradient = objective(values)
    gap = float((gradient @ values - gradient.min()) / max(abs(gradient).max(), 1.0))
    if gap > 1e-7 or not np.isfinite(value):
        raise ValueError("observed fixed optimality failed")
    singles = [objective(np.eye(8)[index])[0] for index in range(8)]
    return {
        "weights": values.tolist(),
        "single_index": int(np.argmin(singles)),
        "single_objectives": singles,
        "objective": value,
        "optimality_gap": max(gap, 0.0),
        "polishing_steps": polish,
        "solver_success": bool(result.success),
    }
