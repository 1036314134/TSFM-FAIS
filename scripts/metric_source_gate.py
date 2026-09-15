"""Matched differentiable MAE and joint forecast objectives with convex controls."""

import numpy as np
import torch
from scipy.optimize import brentq, minimize

from tsfm_fais.routing.forecast_gate import gate_objective
from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)

EPSILON = 0.001
CONDITIONS = ("mae_teacher", "mae_future", "joint_teacher", "joint_future")


def metric_objective(probability, points, target, gram, alignment, kind):
    if kind not in ("mae", "joint"):
        raise ValueError("unknown matched metric objective")
    point = points[:, 0] + torch.einsum(
        "na,naq->nq", probability.to(points.dtype), points - points[:, :1]
    )
    absolute = torch.sqrt((point - target).square() + EPSILON**2).mean(1)
    return (
        absolute
        if kind == "mae"
        else (absolute + gate_objective(probability, gram, alignment, "ensemble")) / 2
    )


def fixed_objective(points, target, weights, kind):
    points, target = np.asarray(points, float), np.asarray(target, float)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    median = np.median(points, axis=1)
    matrix = np.ascontiguousarray(
        (points - median[:, None]).transpose(0, 2, 1).reshape(-1, points.shape[1])
    )
    residual = (median - target).ravel()
    coordinate_weights = np.repeat(weights / points.shape[2], points.shape[2])
    alpha, beta = (1.0, 0.0) if kind == "mae" else (0.5, 0.5)

    def objective(probability):
        error = residual + matrix @ probability
        magnitude = np.sqrt(error**2 + EPSILON**2)
        loss = float(coordinate_weights @ (alpha * magnitude + beta * error**2))
        gradient = matrix.T @ (coordinate_weights * (alpha * error / magnitude + 2 * beta * error))
        return loss, gradient

    return objective


def polish_simplex(objective, probability):
    """Resolve residual stationarity by feasible two-coordinate line searches."""
    probability = np.array(probability, dtype=float, copy=True)
    starting_value, gradient = objective(probability)
    gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    if gap <= 1e-7:
        return probability, 0
    steps = 0
    for _ in range(200):
        active = np.flatnonzero(probability > 0)
        donor = int(active[np.argmax(gradient[active])])
        receiver = int(np.argmin(gradient))
        if donor == receiver:
            raise ValueError("positive stationarity gap without a feasible descent direction")
        upper = float(probability[donor])

        def derivative(amount, donor=donor, receiver=receiver):
            candidate = probability.copy()
            candidate[donor] -= amount
            candidate[receiver] += amount
            _, current = objective(candidate)
            return float(current[receiver] - current[donor])

        amount = (
            upper
            if derivative(upper) <= 0
            else brentq(derivative, 0.0, upper, xtol=1e-15, rtol=1e-12)
        )
        probability[donor] -= amount
        probability[receiver] += amount
        steps += 1
        value, gradient = objective(probability)
        if value > starting_value + 1e-12 * max(abs(starting_value), 1.0):
            raise ValueError("fixed-control precision refinement increased its objective")
        gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
        if gap <= 1e-8:
            break
    return probability, steps


def fit_fixed_metric(points, target, weights, kind):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, target)["raw_projection"]
    g = np.einsum("n,nab->ab", weights, gram)
    b = np.einsum("n,na->a", weights, alignment)
    initial, _, _ = simplex_quadratic_weights(g[None], b[None])
    objective = fixed_objective(points, target, weights, kind)
    count = points.shape[1]
    result = minimize(
        objective,
        initial[0],
        method="SLSQP",
        jac=True,
        bounds=[(0, 1)] * count,
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
    if np.max(abs(probability - result.x)) > 1e-8:
        raise ValueError("the fixed optimizer returned an infeasible solution")
    initial_value, initial_gradient = objective(probability)
    initial_gap = float(
        (initial_gradient @ probability - initial_gradient.min())
        / max(abs(initial_gradient).max(), 1.0)
    )
    probability, polishing_steps = polish_simplex(objective, probability)
    value, gradient = objective(probability)
    gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
    if (
        not np.isfinite(value)
        or not np.isfinite(gradient).all()
        or gap > 1e-7
        or (probability < 0).any()
        or abs(probability.sum() - 1) > 1e-10
    ):
        raise ValueError(
            f"matched fixed {kind} optimality failed: gap={gap}, solver={result.message}"
        )
    single_values = [objective(np.eye(count)[index])[0] for index in range(count)]
    return {
        "weights": probability.tolist(),
        "single_index": int(np.argmin(single_values)),
        "single_objectives": single_values,
        "objective": value,
        "optimality_gap": max(gap, 0),
        "initial_optimality_gap": max(initial_gap, 0),
        "initial_objective": initial_value,
        "polishing_steps": polishing_steps,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
    }
