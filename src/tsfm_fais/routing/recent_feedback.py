"""Causal historical probes and forecast-risk decisions for incomplete series."""

from __future__ import annotations

import hashlib

import numpy as np


def plan_recent_probes(source: dict, offsets=(1, 2), horizons=None, decision_ids=None) -> dict:
    config = source["identity"]["config"]
    length, goal_horizon = config["context_length"], config["horizon"]
    offsets = tuple(offsets)
    horizons = (goal_horizon,) if horizons is None else tuple(horizons)
    if not offsets or len(set(offsets)) != len(offsets) or any(offset < 1 for offset in offsets):
        raise ValueError("probe offsets must be distinct positive integers")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(horizon < 1 for horizon in horizons)
    ):
        raise ValueError("probe horizons must be distinct positive integers")
    prefixes = {
        (dataset["dataset_id"], item["item_id"]): item["prefix_end"]
        for dataset in source["datasets"]
        for item in dataset["items"]
        if "prefix_end" in item
    }
    existing = {record["episode_id"] for record in source["episodes"]}
    probes, links, skipped = {}, [], []
    requested = None if decision_ids is None else set(decision_ids)
    decisions = []
    for current in source["episodes"]:
        if current["split"] != "validation":
            continue
        if requested is not None and current["episode_id"] not in requested:
            continue
        decisions.append(current["episode_id"])
        prefix = prefixes[(current["dataset_id"], current["item_id"])]
        for horizon in horizons:
            for offset in offsets:
                origin = current["origin"] - offset * horizon
                if origin - length < prefix:
                    skipped.append(
                        {
                            "episode_id": current["episode_id"],
                            "horizon": horizon,
                            "offset": offset,
                            "reason": "insufficient_post_fit_history",
                        }
                    )
                    continue
                origin_id = f"{current['dataset_id']}|{current['item_id']}|{origin}"
                source_episode = f"{origin_id}|{current['split']}|{current['mechanism']}|{current['missing_rate']}|{current['mask_seed']}"
                probe_id = f"{source_episode}|H={horizon}"
                record = {
                    key: current[key]
                    for key in (
                        "dataset_id",
                        "family_id",
                        "item_id",
                        "split",
                        "mechanism",
                        "missing_rate",
                        "mask_seed",
                        "period",
                    )
                }
                record.update(
                    probe_id=probe_id,
                    origin_id=origin_id,
                    origin=origin,
                    horizon=horizon,
                    prefix_end=prefix,
                    path="episodes/" + hashlib.sha256(probe_id.encode()).hexdigest()[:24] + ".npz",
                    reusable_episode_id=source_episode if source_episode in existing else None,
                )
                probes[probe_id] = record
                links.append(
                    {
                        "episode_id": current["episode_id"],
                        "origin": current["origin"],
                        "probe_id": probe_id,
                        "probe_end": origin + horizon,
                        "probe_horizon": horizon,
                        "offset": offset,
                    }
                )
    if requested is not None and requested != set(decisions):
        raise ValueError("requested decisions must be existing validation episodes")
    return {
        "evidence_role": "development",
        "context_length": length,
        "goal_horizon": goal_horizon,
        "offsets": list(offsets),
        "horizons": list(horizons),
        "decision_episode_ids": decisions,
        "probes": list(probes.values()),
        "links": links,
        "skipped": skipped,
    }


def validate_feedback_end(probe_origin: int, horizon: int, decision_origin: int) -> None:
    if horizon < 1 or probe_origin + horizon > decision_origin:
        raise ValueError("historical feedback has not arrived before the decision")


def observed_forecast_risk(point, observed_truth, minimum: int = 12):
    """Use common observed cells; return [A,K] MAE/MSE and counts per target."""
    prediction, truth = np.asarray(point, float), np.asarray(observed_truth, float)
    if (
        prediction.ndim != 3
        or truth.shape != prediction.shape[1:]
        or minimum < 1
        or prediction.shape[0] < 1
        or prediction.shape[2] < 1
    ):
        raise ValueError("feedback expects [A,T,K] forecasts and [T,K] observed truth")
    if not np.isfinite(prediction).all() or np.isinf(truth).any():
        raise ValueError("forecasts must be finite; unavailable observations must be NaN")
    observed = np.isfinite(truth)
    counts = observed.sum(axis=0)
    valid = counts >= minimum
    residual = np.where(observed[None], prediction - truth[None], 0.0)
    risks = {}
    for name, errors in (("mae", np.abs(residual)), ("mse", residual**2)):
        result = np.full((prediction.shape[0], prediction.shape[2]), np.nan)
        np.divide(errors.sum(axis=1), counts[None], out=result, where=valid[None])
        risks[name] = result
    return risks, counts, valid


def select_feedback(risks, objective: str, normalizers: dict, reference: int, *, per_target: bool):
    if objective == "joint":
        if any(
            not np.isfinite(normalizers[key]) or normalizers[key] <= 0 for key in ("mae", "mse")
        ):
            raise ValueError("joint-risk normalizers must be positive and finite")
        scores = 0.5 * (risks["mae"] / normalizers["mae"] + risks["mse"] / normalizers["mse"])
    elif objective in ("mae", "mse"):
        scores = risks[objective]
    else:
        raise ValueError("unknown forecast objective")
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or not 0 <= reference < len(scores):
        raise ValueError("feedback scores must cover actions and targets")
    valid = np.isfinite(scores).all(axis=0)
    if np.any(np.isfinite(scores).any(axis=0) != valid):
        raise ValueError("all candidates must use the same observed cells")

    def pick(values):
        best = np.min(values)
        return reference if values[reference] == best else int(np.argmin(values))

    if per_target:
        return tuple(
            pick(scores[:, slot]) if available else reference
            for slot, available in enumerate(valid)
        )
    return pick(scores.mean(axis=1)) if valid.all() else reference


def simplex_mse_weights(errors):
    from scipy.optimize import minimize

    matrix = np.asarray(errors, float)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("ensemble fitting requires finite observed historical residuals")
    gram = matrix.T @ matrix / len(matrix)
    normalized = gram / max(float(np.max(np.diag(gram))), 1e-12)
    n = gram.shape[0]
    result = minimize(
        lambda w: float(w @ normalized @ w),
        np.full(n, 1 / n),
        jac=lambda w: 2 * normalized @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={
            "type": "eq",
            "fun": lambda w: float(w.sum() - 1),
            "jac": lambda w: np.ones(n),
        },
        options={"ftol": 1e-10, "maxiter": 300},
    )
    alternatives = [np.full(n, 1 / n), np.eye(n)[int(np.argmin(np.diag(gram)))]]
    if np.isfinite(result.x).all() and result.x.min() >= -1e-7 and abs(result.x.sum() - 1) < 1e-6:
        weights = np.maximum(result.x, 0)
        alternatives.append(weights / weights.sum())
    weights = min(alternatives, key=lambda w: float(w @ gram @ w))
    return weights, bool(result.success)


def observed_ensemble_weights(
    point, observed_truth, minimum=12, reference=0, *, per_target=False, shrinkage=0.0
):
    prediction, truth = np.asarray(point, float), np.asarray(observed_truth, float)
    _, counts, valid = observed_forecast_risk(prediction, truth, minimum)
    n_actions, _, n_targets = prediction.shape
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0,1]")
    fallback = np.eye(n_actions)[reference]

    def fit(slots):
        parts = []
        for slot in slots:
            present = np.isfinite(truth[:, slot])
            errors = (prediction[:, present, slot] - truth[present, slot]).T
            parts.append(errors / np.sqrt(counts[slot]))
        weights, _ = simplex_mse_weights(np.concatenate(parts))
        return (1 - shrinkage) * weights + shrinkage / n_actions

    if per_target:
        return np.stack([fit([slot]) if valid[slot] else fallback for slot in range(n_targets)])
    return fit(range(n_targets)) if valid.all() else fallback


def combine_imputations(context, completions, weights, targets=None):
    values, candidates, mixture = (
        np.asarray(context, float),
        np.asarray(completions, float),
        np.asarray(weights, float),
    )
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != values.shape
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("finite [A,L,D] completions must match the observed context")
    if (
        not np.isfinite(mixture).all()
        or np.any(mixture < -1e-10)
        or not np.allclose(mixture.sum(axis=-1), 1)
    ):
        raise ValueError("imputation weights must be nonnegative and sum to one")
    if mixture.ndim == 1 and len(mixture) == len(candidates):
        completed = np.einsum("a,ald->ld", mixture, candidates)
    elif (
        mixture.ndim == 2
        and targets is not None
        and mixture.shape == (len(targets), len(candidates))
    ):
        if len(set(targets)) != len(targets) or min(targets) < 0 or max(targets) >= values.shape[1]:
            raise ValueError("target weights require distinct valid target indices")
        completed = candidates.mean(axis=0)
        for slot, target in enumerate(targets):
            completed[:, target] = mixture[slot] @ candidates[:, :, target]
    else:
        raise ValueError("weights must define a sequence mixture or an explicit target mixture")
    observed = np.isfinite(values)
    completed[observed] = values[observed]
    return completed
