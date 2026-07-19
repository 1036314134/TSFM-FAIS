"""Tune routing blends on one held-out family without touching test forecasts."""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tsfm_fais.routing.models import RouterBundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--router-folds", type=Path, required=True)
    parser.add_argument("--tuning-family", default="ett")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--shortlist-size", type=int, default=6)
    return parser.parse_args()


def _rows(path: Path, family_id: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("family_id") != family_id:
                continue
            context = {
                f"dataset_id::{row['dataset_id']}": 1.0,
                f"family_id::{row['family_id']}": 1.0,
                "forecast_origin_log1p": float(np.log1p(row["forecast_origin"])),
            }
            for field in ("prior_features", "unary_features"):
                row[field].update(context)
            selected.append(row)
    return selected


def _normalize(values: np.ndarray, *, higher_is_better: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, 0.5, dtype=float)
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    normalized = np.full(values.shape, 0.5, dtype=float)
    if high - low > 1e-12:
        normalized[finite] = (values[finite] - low) / (high - low)
    else:
        normalized[finite] = 0.0
    if higher_is_better:
        normalized[finite] = 1.0 - normalized[finite]
    return normalized


def _weight_grid(step: float, dimensions: int) -> list[tuple[float, ...]]:
    units = round(1.0 / step)
    if units < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one exactly")
    weights: list[tuple[float, ...]] = []
    for values in itertools.product(range(units + 1), repeat=dimensions):
        if sum(values) == units:
            weights.append(tuple(value / units for value in values))
    return weights


def _matrix(rows: list[dict[str, Any]], field: str, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row[field].get(name, 0.0)) for name in names] for row in rows],
        dtype=float,
    )


def _group_evidence(
    rows: list[dict[str, Any]], bundle: RouterBundle
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    evidence: list[dict[str, Any]] = []
    model_priors = bundle.metadata.get("candidate_global_priors", {})
    for group_rows in grouped.values():
        candidates = [str(row["candidate_id"]) for row in group_rows]
        forecaster_id = str(group_rows[0]["forecaster_id"])
        priors = model_priors.get(forecaster_id, {})
        prior_predictions = bundle.prior.predict(
            _matrix(group_rows, "prior_features", bundle.feature_names)
        )
        unary_predictions = bundle.unary.predict(
            _matrix(group_rows, "unary_features", bundle.feature_names)
        )
        targets = np.asarray(
            [float(row.get("routing_target", row["degradation"])) for row in group_rows],
            dtype=float,
        )
        evidence.append(
            {
                "forecaster_id": forecaster_id,
                "candidates": candidates,
                "target": targets,
                "target_normalized": _normalize(targets),
                "r0": _normalize(prior_predictions, higher_is_better=True),
                "r1": _normalize(unary_predictions, higher_is_better=True),
                "proxy": _normalize(
                    np.asarray(
                        [row["unary_features"]["proxy_global_mae"] for row in group_rows],
                        dtype=float,
                    )
                ),
                "global_prior": _normalize(
                    np.asarray([priors.get(candidate, np.nan) for candidate in candidates])
                ),
            }
        )
    return evidence


def _episode_evidence(
    rows: list[dict[str, Any]], bundle: RouterBundle
) -> list[dict[str, Any]]:
    """Aggregate block predictions against full-candidate forecast loss."""

    prior_predictions = bundle.prior.predict(
        _matrix(rows, "prior_features", bundle.feature_names)
    )
    unary_predictions = bundle.unary.predict(
        _matrix(rows, "unary_features", bundle.feature_names)
    )
    aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row, r0, r1 in zip(
        rows, prior_predictions, unary_predictions, strict=True
    ):
        try:
            full_candidate_loss = float(row["full_candidate_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(full_candidate_loss):
            continue
        key = (
            str(row["forecaster_id"]),
            str(row["episode_id"]),
            str(row["candidate_id"]),
        )
        entry = aggregated.setdefault(
            key,
            {"r0": [], "r1": [], "proxy": [], "loss": full_candidate_loss},
        )
        entry["r0"].append(float(r0))
        entry["r1"].append(float(r1))
        entry["proxy"].append(float(row["unary_features"]["proxy_global_mae"]))

    episodes: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (forecaster_id, episode_id, candidate_id), entry in aggregated.items():
        episodes[(forecaster_id, episode_id)].append((candidate_id, entry))
    model_priors = bundle.metadata.get("candidate_global_priors", {})
    evidence: list[dict[str, Any]] = []
    for (forecaster_id, _), candidates_and_entries in episodes.items():
        candidates = [candidate for candidate, _ in candidates_and_entries]
        entries = [entry for _, entry in candidates_and_entries]
        targets = np.asarray([entry["loss"] for entry in entries], dtype=float)
        priors = model_priors.get(forecaster_id, {})
        evidence.append(
            {
                "forecaster_id": forecaster_id,
                "candidates": candidates,
                "target": targets,
                "target_normalized": _normalize(targets),
                "r0": _normalize(
                    np.asarray([np.mean(entry["r0"]) for entry in entries]),
                    higher_is_better=True,
                ),
                "r1": _normalize(
                    np.asarray([np.mean(entry["r1"]) for entry in entries]),
                    higher_is_better=True,
                ),
                "proxy": _normalize(
                    np.asarray([np.mean(entry["proxy"]) for entry in entries])
                ),
                "global_prior": _normalize(
                    np.asarray([priors.get(candidate, np.nan) for candidate in candidates])
                ),
            }
        )
    return evidence


def _shortlist(
    group: dict[str, Any],
    *,
    prior_weight: float,
    forced_prior_count: int,
    size: int,
) -> np.ndarray:
    candidates = group["candidates"]
    forced = [
        candidates.index(candidate)
        for candidate in ("locf", "linear_interp")
        if candidate in candidates
    ]
    if forced_prior_count:
        for index in np.argsort(group["global_prior"], kind="stable"):
            if int(index) not in forced:
                forced.append(int(index))
            if len(forced) >= 2 + forced_prior_count:
                break
    score = (1.0 - prior_weight) * group["r0"] + prior_weight * group["global_prior"]
    selected = list(forced[:size])
    for index in np.argsort(score, kind="stable"):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == size:
            break
    return np.asarray(selected, dtype=int)


def _mean_regret(groups: list[dict[str, Any]], selections: list[int]) -> float:
    return float(
        np.mean(
            [
                group["target_normalized"][selection]
                - np.min(group["target_normalized"])
                for group, selection in zip(groups, selections, strict=True)
            ]
        )
    )


def _tune_model(
    groups: list[dict[str, Any]], step: float, shortlist_size: int
) -> dict[str, Any]:
    shortlist_trials: list[tuple[float, float, int, list[np.ndarray]]] = []
    for prior_weight in np.arange(0.0, 1.0 + step / 2.0, step):
        for forced_prior_count in range(3):
            shortlists = [
                _shortlist(
                    group,
                    prior_weight=float(prior_weight),
                    forced_prior_count=forced_prior_count,
                    size=shortlist_size,
                )
                for group in groups
            ]
            regret = float(
                np.mean(
                    [
                        np.min(group["target_normalized"][indices])
                        - np.min(group["target_normalized"])
                        for group, indices in zip(groups, shortlists, strict=True)
                    ]
                )
            )
            shortlist_trials.append(
                (regret, float(prior_weight), forced_prior_count, shortlists)
            )
    shortlist_regret, prior_weight, forced_prior_count, shortlists = min(
        shortlist_trials, key=lambda value: (value[0], value[2], value[1])
    )

    best: tuple[float, tuple[float, ...], list[int]] | None = None
    for weights in _weight_grid(step, 4):
        selections: list[int] = []
        for group, indices in zip(groups, shortlists, strict=True):
            score = (
                weights[0] * group["r0"][indices]
                + weights[1] * group["r1"][indices]
                + weights[2] * group["proxy"][indices]
                + weights[3] * group["global_prior"][indices]
            )
            selections.append(int(indices[int(np.argmin(score))]))
        regret = _mean_regret(groups, selections)
        trial = (regret, weights, selections)
        if best is None or trial[:2] < best[:2]:
            best = trial
    assert best is not None
    regret, weights, selections = best
    accuracy = float(
        np.mean(
            [
                np.isclose(
                    group["target_normalized"][selection],
                    np.min(group["target_normalized"]),
                )
                for group, selection in zip(groups, selections, strict=True)
            ]
        )
    )
    return {
        "groups": len(groups),
        "shortlist": {
            "r0_global_prior_weight": prior_weight,
            "forced_global_prior_count": forced_prior_count,
            "mean_normalized_oracle_regret": shortlist_regret,
        },
        "final_blend": {
            "r0": weights[0],
            "r1": weights[1],
            "proxy": weights[2],
            "global_prior": weights[3],
            "mean_normalized_regret": regret,
            "top1_accuracy": accuracy,
        },
    }


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings(
        "ignore", message="X does not have valid feature names, but LGBM"
    )
    rows = _rows(args.labels, args.tuning_family)
    if not rows:
        raise ValueError(f"no labels found for tuning family {args.tuning_family!r}")
    bundle = RouterBundle.load(args.router_folds / args.tuning_family)
    evidence = _group_evidence(rows, bundle)
    episode_evidence = _episode_evidence(rows, bundle)
    result = {
        "tuning_family": args.tuning_family,
        "selection_protocol": "held_out_family_only",
        "models": {},
        "episode_models": {},
    }
    for forecaster_id in sorted({group["forecaster_id"] for group in evidence}):
        model_groups = [
            group for group in evidence if group["forecaster_id"] == forecaster_id
        ]
        result["models"][forecaster_id] = _tune_model(
            model_groups, args.step, args.shortlist_size
        )
    for forecaster_id in sorted(
        {group["forecaster_id"] for group in episode_evidence}
    ):
        model_groups = [
            group
            for group in episode_evidence
            if group["forecaster_id"] == forecaster_id
        ]
        result["episode_models"][forecaster_id] = _tune_model(
            model_groups, args.step, args.shortlist_size
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
