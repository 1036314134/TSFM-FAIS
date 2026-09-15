"""Matched conditional Gaussian supervision and convex portfolio controls."""

import hashlib

import numpy as np
import torch
from metric_source_gate import EPSILON, metric_objective, polish_simplex
from scipy.optimize import linprog, minimize
from scipy.special import ndtr

from tsfm_fais.routing.forecast_gate import SharedForecastGate, gate_objective
from tsfm_fais.routing.forecast_projection import (
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights

CONDITIONS = ("conditional_expected", "conditional_factual", "source_steps_matched")


def conditional_objective(
    probability, points, target, variance, gram, alignment, simulated, expected
):
    if not bool(simulated.any()):
        return metric_objective(probability, points, target, gram, alignment, "joint")
    result = torch.empty(len(points), dtype=points.dtype, device=points.device)
    original = ~simulated
    if bool(original.any()):
        result[original] = metric_objective(
            probability[original],
            points[original],
            target[original],
            gram[original],
            alignment[original],
            "joint",
        )
    p, w = points[simulated], probability[simulated]
    prediction = p[:, 0] + torch.einsum("na,naq->nq", w.to(p.dtype), p - p[:, :1])
    error = prediction - target[simulated]
    if expected:
        sigma = torch.sqrt(variance[simulated])
        if not bool((sigma > 0).all()):
            raise ValueError("registered conditional futures have positive marginal variances")
        z = error / sigma
        absolute = sigma * np.sqrt(2 / np.pi) * torch.exp(-z.square() / 2) + error * torch.erf(
            z / np.sqrt(2)
        )
    else:
        absolute = error.abs()
    result[simulated] = (
        absolute.mean(1) + gate_objective(w, gram[simulated], alignment[simulated], "ensemble")
    ) / 2
    return result


def condition_data(data, expected):
    target = data["truth"].copy()
    if expected:
        target[data["simulated"]] = data["mean"][data["simulated"]]
    alignment = np.full(data["points"].shape[:2], np.nan)
    eligible = np.isfinite(target).all(1)
    alignment[eligible] = projection_targets(data["points"][eligible], target[eligible])[
        "raw_projection"
    ]
    return target, alignment


def fit_conditional_gate(frame, data, indices, seed, updates, expected):
    target, alignment = condition_data(data, expected)
    if not np.isfinite(target[indices]).all():
        raise ValueError("unavailable labels entered a training index")
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
        for value in (data["features"][indices], data["gram"][indices], alignment[indices], weights)
    ]
    p, y, v = [
        torch.tensor(value, dtype=torch.float64)
        for value in (data["points"][indices], target[indices], data["variance"][indices])
    ]
    simulated = torch.as_tensor(data["simulated"][indices], dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed + 100000)
    steps, epoch, history = 0, 0, []
    while steps < updates:
        epoch += 1
        total, count = 0.0, 0
        for batch in torch.randperm(len(indices), generator=generator).split(128):
            loss = (
                conditional_objective(
                    model(x[batch]),
                    p[batch],
                    y[batch],
                    v[batch],
                    g[batch],
                    b[batch],
                    simulated[batch],
                    expected,
                )
                * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite conditional-supervision objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite conditional-supervision gradient")
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


def fixed_terms(points, target, variance, simulated, row_weights, expected):
    weights = np.array(row_weights, dtype=float, copy=True)
    weights /= weights.sum()
    anchor = np.median(points, axis=1)
    matrix = np.ascontiguousarray((points - anchor[:, None]).transpose(0, 2, 1).reshape(-1, 8))
    weight = np.repeat(weights / points.shape[2], points.shape[2])
    kind = np.repeat(np.where(simulated, 2 if expected else 1, 0), points.shape[2])
    var = np.where(kind == 2, variance.ravel(), 0.0)
    return matrix, (anchor - target).ravel(), weight, kind, var


def fixed_value_gradient(terms, probability, certificate=False):
    matrix, residual, weight, kind, variance = terms
    error = residual + matrix @ probability
    magnitude = np.sqrt(error**2 + EPSILON**2)
    slope = error / magnitude
    factual = kind == 1
    magnitude[factual], slope[factual] = abs(error[factual]), np.sign(error[factual])
    expected = kind == 2
    if expected.any():
        sigma = np.sqrt(variance[expected])
        z = error[expected] / sigma
        slope[expected] = 2 * ndtr(z) - 1
        magnitude[expected] = (
            sigma * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + error[expected] * slope[expected]
        )
    value = float(weight @ ((magnitude + error**2 + variance) / 2))
    ties = np.flatnonzero(factual & (abs(error) <= 1e-10)) if certificate else np.empty(0, int)
    if len(ties):
        slope[ties] = 0.0
    gradient = matrix.T @ (weight * (0.5 * slope + error))
    details = {"tie_indices": [], "tie_subgradient": [], "maximum_tie_residual": 0.0}
    if len(ties):
        effect = matrix[ties].T * (0.5 * weight[ties])
        constraint = probability @ effect - effect
        solution = linprog(
            np.r_[np.zeros(len(ties)), 1.0],
            A_ub=np.c_[constraint, -np.ones(8)],
            b_ub=-(gradient @ probability - gradient),
            bounds=[(-1, 1)] * len(ties) + [(0, None)],
            method="highs",
        )
        if not solution.success:
            raise ValueError("the absolute-loss subgradient certificate failed")
        gradient += effect @ solution.x[:-1]
        details = {
            "tie_indices": ties.tolist(),
            "tie_subgradient": solution.x[:-1].tolist(),
            "maximum_tie_residual": float(abs(error[ties]).max()),
        }
    return value, gradient, details


def fit_conditional_fixed(frame, data, indices, expected):
    target, alignment = condition_data(data, expected)
    weights = _family_weights(frame.iloc[indices])
    distribution = weights / weights.sum()
    g = np.einsum("n,nab->ab", distribution, data["gram"][indices])
    b = np.einsum("n,na->a", distribution, alignment[indices])
    initial = simplex_quadratic_weights(g[None], b[None])[0][0]
    terms = fixed_terms(
        data["points"][indices],
        target[indices],
        data["variance"][indices],
        data["simulated"][indices],
        weights,
        expected,
    )

    def objective(probability):
        return fixed_value_gradient(terms, probability)[:2]

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
    probability = np.maximum(result.x, 0)
    probability /= probability.sum()
    if abs(probability - result.x).max() > 1e-8:
        raise ValueError("the conditional fixed solver returned infeasible weights")
    value, gradient, certificate = fixed_value_gradient(terms, probability, certificate=True)
    gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    polish = 0
    if gap > 1e-7:
        probability, polish = polish_simplex(objective, probability)
        value, gradient, certificate = fixed_value_gradient(terms, probability, certificate=True)
        gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    if (
        not np.isfinite(value)
        or gap > 1e-7
        or probability.min() < 0
        or abs(probability.sum() - 1) > 1e-10
    ):
        raise ValueError(f"conditional fixed optimality failed: {gap}")
    singles = [objective(np.eye(8)[i])[0] for i in range(8)]
    return {
        "weights": probability.tolist(),
        "objective": value,
        "optimality_gap": max(gap, 0.0),
        "certificate": certificate,
        "polishing_steps": polish,
        "solver_success": bool(result.success),
        "single_index": int(np.argmin(singles)),
        "single_objectives": singles,
    }
