"""Learn teacher residual projections and retain the observed forecast geometry."""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .forecast_response import FORECAST_FEATURES, forecast_response_inputs
from .utility import _family_weights

NORM_EPS = 1e-12


def forecast_offsets(points, *, anchor=None):
    """Center forecast vectors at an explicit reference or their coordinate median."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or min(points.shape) < 1 or not np.isfinite(points).all():
        raise ValueError("finite candidate forecast vectors are required")
    anchor = np.median(points, axis=1) if anchor is None else np.asarray(anchor, float)
    if anchor.shape != (points.shape[0], points.shape[2]) or not np.isfinite(anchor).all():
        raise ValueError("the forecast reference must have aligned finite coordinates")
    changes = points - anchor[:, None]
    energy = np.mean(changes**2, axis=2)
    return anchor, changes, energy


def forecast_geometry(points):
    """Candidate vectors [N,A,Q], centered at their coordinate median."""
    anchor, changes, energy = forecast_offsets(points)
    gram = np.einsum("naq,nbq->nab", changes, changes) / changes.shape[2]
    return anchor, changes, energy, gram


def projection_targets(points, teacher, *, anchor=None):
    anchor, changes, energy = forecast_offsets(points, anchor=anchor)
    teacher = np.asarray(teacher, dtype=float)
    if teacher.shape != anchor.shape or not np.isfinite(teacher).all():
        raise ValueError("the teacher must match the candidate forecast coordinates")
    alignment = np.mean(changes * (teacher - anchor)[:, None], axis=2)
    norm = np.sqrt(energy)
    directional = np.divide(alignment, norm, out=np.zeros_like(norm), where=norm > NORM_EPS)
    return {
        "direct_risk": energy - 2 * alignment,
        "raw_projection": alignment,
        "unit_projection": directional,
    }


def simplex_quadratic_weights(gram, alignment):
    """Minimize w' G w - 2 b' w by enumerating simplex faces (at most 8 vertices)."""
    gram, alignment = np.asarray(gram, float), np.asarray(alignment, float)
    if (
        gram.ndim != 3
        or gram.shape[1] != gram.shape[2]
        or alignment.shape != gram.shape[:2]
        or not 1 <= gram.shape[1] <= 8
        or not np.isfinite(gram).all()
        or not np.isfinite(alignment).all()
    ):
        raise ValueError(
            "finite aligned quadratic objectives with at most eight vertices are required"
        )
    scale = np.maximum(
        np.maximum(np.abs(gram).max(axis=(1, 2)), np.abs(alignment).max(axis=1)), 1e-12
    )
    matrix, linear = gram / scale[:, None, None], alignment / scale[:, None]
    if not np.allclose(matrix, matrix.transpose(0, 2, 1), rtol=0, atol=1e-10):
        raise ValueError("the forecast Gram matrix must be symmetric")
    matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
    if np.linalg.eigvalsh(matrix).min() < -1e-9:
        raise ValueError("the forecast Gram matrix must be positive semidefinite")
    count, vertices = linear.shape
    weights = np.zeros_like(linear)
    weights[:, 0] = 1.0
    best = matrix[:, 0, 0] - 2 * linear[:, 0]
    for size in range(1, vertices + 1):
        for face in combinations(range(vertices), size):
            indices = np.asarray(face)
            local = matrix[:, indices][:, :, indices]
            payoff = linear[:, indices]
            if size == 1:
                candidate = np.ones((count, 1))
                feasible = np.ones(count, dtype=bool)
            else:
                system = np.zeros((count, size + 1, size + 1))
                system[:, :size, :size] = 2 * local
                system[:, :size, -1] = system[:, -1, :size] = 1.0
                rhs = np.concatenate([2 * payoff, np.ones((count, 1))], axis=1)
                solution = (np.linalg.pinv(system, rcond=1e-12, hermitian=True) @ rhs[..., None])[
                    ..., 0
                ]
                candidate = solution[:, :size]
                residual = np.max(np.abs((system @ solution[..., None])[..., 0] - rhs), axis=1)
                feasible = (candidate.min(axis=1) >= -1e-9) & (residual <= 1e-8)
                candidate = np.maximum(candidate, 0.0)
                candidate /= np.maximum(candidate.sum(axis=1, keepdims=True), 1e-30)
            objective = np.einsum("ni,nij,nj->n", candidate, local, candidate) - 2 * np.sum(
                payoff * candidate, axis=1
            )
            improved = feasible & (objective < best)
            selected = np.flatnonzero(improved)
            weights[selected] = 0.0
            weights[np.ix_(selected, indices)] = candidate[selected]
            best[improved] = objective[improved]
    gradient = 2 * (np.einsum("nij,nj->ni", matrix, weights) - linear)
    gap = np.sum(gradient * weights, axis=1) - gradient.min(axis=1)
    if (
        gap.max() > 1e-7
        or weights.min() < 0
        or not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("the simplex solution did not pass its numerical optimality check")
    return weights, np.maximum(gap, 0.0), scale


def compose_from_estimates(points, estimates, *, target_kind):
    anchor, changes, energy, gram = forecast_geometry(points)
    estimates = np.asarray(estimates, float)
    if estimates.shape != energy.shape or not np.isfinite(estimates).all():
        raise ValueError("one finite estimate per candidate is required")
    if target_kind == "unit_projection":
        alignment = np.sqrt(energy) * estimates
    elif target_kind == "raw_projection":
        alignment = estimates.copy()
    elif target_kind == "direct_risk":
        alignment = (energy - estimates) / 2
    else:
        raise ValueError("unknown forecast-regression target")
    alignment[energy <= NORM_EPS**2] = 0.0
    full_gram = np.pad(gram, ((0, 0), (1, 0), (1, 0)))
    full_alignment = np.pad(alignment, ((0, 0), (1, 0)))
    weights, gap, scale = simplex_quadratic_weights(full_gram, full_alignment)
    point = anchor + np.einsum("na,naq->nq", weights[:, 1:], changes)
    return point, weights, gap, scale


@dataclass
class ForecastProjectionRegressor:
    seed: int = 5101
    n_estimators: int = 160
    candidate_ids: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = FORECAST_FEATURES
    model: Any = None

    def _matrix(self, frame):
        frame = forecast_response_inputs(frame)
        if set(frame.candidate_id) != set(self.candidate_ids):
            raise ValueError("the candidate pool changed")
        values = np.column_stack(
            [
                frame[list(self.feature_names)].to_numpy(dtype=float),
                *[
                    (frame.candidate_id == name).to_numpy(dtype=float)
                    for name in self.candidate_ids
                ],
            ]
        )
        if not np.isfinite(values).all():
            raise ValueError("regressor inputs must be finite")
        return pd.DataFrame(
            values,
            columns=[*self.feature_names, *["action." + name for name in self.candidate_ids]],
            index=frame.index,
        )

    def fit(self, frame, labels):
        from lightgbm import LGBMRegressor

        labels = np.asarray(labels, float)
        if labels.shape != (len(frame),) or not np.isfinite(labels).all():
            raise ValueError("finite aligned training labels are required")
        self.candidate_ids = tuple(sorted(frame.candidate_id.unique()))
        self.model = LGBMRegressor(
            objective="regression",
            n_estimators=self.n_estimators,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=25,
            reg_lambda=5.0,
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        ).fit(self._matrix(frame), labels, sample_weight=_family_weights(frame))
        return self

    def predict(self, frame):
        if self.model is None:
            raise ValueError("fit the forecast projection regressor first")
        return self.model.predict(self._matrix(frame))
