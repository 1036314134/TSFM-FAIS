"""Screen label-free forecast-consensus value ensembles on a development split.

Candidate weights are derived only from routing-time forecast-consensus scores.
The clean future is used after the imputed contexts are fixed, solely to score
development policies.  This script must not be used to tune on confirmatory
evaluation datasets.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_fais.config import load_config
from tsfm_fais.data import stable_seed
from tsfm_fais.evaluation import (
    _build_forecast_runner,
    _forecast_spec,
    _load_saved_candidates,
    _set_forecast_seed,
)
from tsfm_fais.pipeline import _forecast_consensus_scores


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--temperatures", type=float, nargs="*", default=[0.1, 0.25, 0.5])
    parser.add_argument("--disagreement-thresholds", type=float, nargs="*", default=[])
    parser.add_argument(
        "--third-relative-gap-thresholds",
        type=float,
        nargs="*",
        default=[],
    )
    parser.add_argument("--recompute-shortlist-scores", action="store_true")
    return parser.parse_args()


def _records(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _strict_candidate_best(metrics: pd.DataFrame) -> dict[str, float]:
    eligible = metrics["metric_eligible"].astype(str).str.lower().eq("true")
    native = metrics["native_valid"].astype(str).str.lower().eq("true")
    episode_counts = (
        metrics.loc[eligible & metrics["method"].eq("b_fais")]
        .groupby("dataset_id")["episode_id"]
        .nunique()
    )
    candidates = (
        metrics.loc[
            eligible
            & native
            & metrics["method_role"].isin(("baseline", "missing_anchor"))
        ]
        .groupby(["dataset_id", "method"])
        .agg(mase=("mase", "mean"), count=("episode_id", "nunique"))
        .reset_index()
    )
    best: dict[str, float] = {}
    for dataset_id, count in episode_counts.items():
        complete = candidates.loc[
            candidates["dataset_id"].eq(dataset_id) & candidates["count"].eq(count)
        ]
        if complete.empty:
            raise ValueError(f"no full-coverage candidate for {dataset_id}")
        best[str(dataset_id)] = float(complete["mase"].min())
    return best


def _consensus_diagnostics(record: Mapping[str, object]) -> Mapping[str, object] | None:
    routing = record.get("routing_metadata")
    if not isinstance(routing, Mapping):
        raise ValueError("routing record lacks routing_metadata")
    diagnostics = routing.get("forecast_consensus")
    if diagnostics is None:
        return None
    if not isinstance(diagnostics, Mapping):
        raise ValueError("forecast consensus diagnostics must be a mapping")
    return diagnostics if bool(diagnostics.get("active", False)) else None


def _numeric_scores(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for candidate_id, raw_score in value.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            scores[str(candidate_id)] = score
    return scores


def _candidate_weights(
    scores: Mapping[str, float],
    available: set[str],
    *,
    top_k: int,
    temperature: float | None,
) -> dict[str, float]:
    ordered = sorted(
        (
            (float(score), candidate_id)
            for candidate_id, score in scores.items()
            if candidate_id in available and np.isfinite(float(score))
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ordered:
        return {}
    selected = ordered[: min(top_k, len(ordered))]
    if temperature is None or len(selected) == 1:
        weight = 1.0 / len(selected)
        return {candidate_id: weight for _, candidate_id in selected}
    values = np.asarray([score for score, _ in selected], dtype=float)
    span = float(np.max(values) - np.min(values))
    normalized = np.zeros_like(values) if span <= 1e-12 else (values - np.min(values)) / span
    logits = -normalized / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return {
        candidate_id: float(weight)
        for (_, candidate_id), weight in zip(selected, weights, strict=True)
    }


def _ensemble_context(
    base: np.ndarray,
    observed_mask: np.ndarray,
    saved: Mapping[str, Mapping[str, object]],
    diagnostics: Mapping[str, object] | None,
    *,
    forecast_mode: str,
    target_indices: tuple[int, ...],
    top_k: int,
    temperature: float | None,
    disagreement_threshold: float | None = None,
    third_relative_gap_threshold: float | None = None,
    channel_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    completed = np.asarray(base, dtype=float).copy()
    if diagnostics is None:
        return completed, {}
    available = {
        candidate_id
        for candidate_id, metadata in saved.items()
        if bool(metadata.get("native_valid", False))
    }
    global_scores = _numeric_scores(
        diagnostics.get("selection_scores", diagnostics.get("scores", {}))
    )
    raw_target_scores = diagnostics.get("target_selection_scores", {})
    target_scores = raw_target_scores if isinstance(raw_target_scores, Mapping) else {}
    channels = target_indices if forecast_mode == "independent_univariate" else tuple(
        range(observed_mask.shape[1])
    )
    weights_by_channel: dict[str, dict[str, float]] = {}
    for channel in channels:
        scores = _numeric_scores(target_scores.get(str(channel), {})) or global_scores
        selected_top_k = top_k
        if third_relative_gap_threshold is not None:
            ordered_scores = sorted(
                float(score)
                for candidate_id, score in scores.items()
                if candidate_id in available and np.isfinite(float(score))
            )
            if len(ordered_scores) >= 3:
                relative_gap = (
                    (ordered_scores[2] - ordered_scores[1])
                    / max(abs(ordered_scores[1]), 1e-8)
                )
                if relative_gap <= third_relative_gap_threshold:
                    selected_top_k = 3
        weights = _candidate_weights(
            scores,
            available,
            top_k=selected_top_k,
            temperature=temperature,
        )
        if not weights:
            continue
        missing = ~observed_mask[:, channel]
        if not missing.any():
            continue
        candidate_series: dict[str, np.ndarray] = {}
        blended = np.zeros(observed_mask.shape[0], dtype=float)
        for candidate_id, weight in weights.items():
            values = np.asarray(saved[candidate_id]["values"], dtype=float)
            candidate_series[candidate_id] = values[:, channel]
            blended += weight * values[:, channel]
        if disagreement_threshold is None or len(candidate_series) < 2:
            completed[missing, channel] = blended[missing]
        else:
            if channel_scale is None:
                raise ValueError("gated ensemble requires per-channel scales")
            scale = max(abs(float(channel_scale[channel])), 1e-8)
            first = next(iter(candidate_series.values()))
            completed[missing, channel] = first[missing]
            missing_indices = np.flatnonzero(missing)
            split_points = np.flatnonzero(np.diff(missing_indices) > 1) + 1
            for run in np.split(missing_indices, split_points):
                stacked = np.stack(
                    [series[run] for series in candidate_series.values()],
                    axis=0,
                )
                disagreement = float(np.mean(np.ptp(stacked, axis=0)) / scale)
                if disagreement <= disagreement_threshold:
                    completed[run, channel] = blended[run]
        weights_by_channel[str(channel)] = weights
    return completed, weights_by_channel


def _shortlist_diagnostics(
    saved: Mapping[str, Mapping[str, object]],
    shortlist: Iterable[str],
    runner: object,
    spec: object,
    mase_scale: np.ndarray,
) -> dict[str, object] | None:
    candidate_ids = tuple(
        candidate_id
        for candidate_id in dict.fromkeys(map(str, shortlist))
        if candidate_id in saved and bool(saved[candidate_id].get("native_valid", False))
    )
    if not candidate_ids:
        return None
    candidate_values = {
        candidate_id: np.asarray(saved[candidate_id]["values"], dtype=float)[None, ...]
        for candidate_id in candidate_ids
    }
    scores, target_scores = _forecast_consensus_scores(
        candidate_values,
        spec,
        runner.predict,
        mase_scale,
    )
    target_selection_scores: dict[str, dict[str, float]] = {}
    if spec.mode == "independent_univariate":
        for offset, target_index in enumerate(spec.target_indices or ()):
            target_selection_scores[str(target_index)] = {
                candidate_id: float(target_scores[candidate_id][offset])
                for candidate_id in candidate_ids
            }
    return {
        "active": True,
        "selection_scores": scores,
        "target_selection_scores": target_selection_scores,
        "eligible_candidates": list(candidate_ids),
        "selection_source": "recomputed_shortlist",
    }


def main() -> int:
    args = _arguments()
    if any(value < 2 for value in args.top_k):
        raise ValueError("top-k values must be at least two")
    if any(not np.isfinite(value) or value <= 0 for value in args.temperatures):
        raise ValueError("temperatures must be finite and positive")
    if any(
        not np.isfinite(value) or value <= 0 for value in args.disagreement_thresholds
    ):
        raise ValueError("disagreement thresholds must be finite and positive")
    if any(
        not np.isfinite(value) or value < 0
        for value in args.third_relative_gap_thresholds
    ):
        raise ValueError("third-candidate relative gaps must be finite and non-negative")
    config = load_config(args.config)
    impute_root = Path(args.impute_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.evaluation)
    model_ids = set(metrics["forecaster_id"].astype(str))
    if model_ids != {args.forecaster_id}:
        raise ValueError(
            f"evaluation forecasters {sorted(model_ids)} do not match {args.forecaster_id!r}"
        )
    strict_best = _strict_candidate_best(metrics)
    runner = _build_forecast_runner(
        args.forecaster_id,
        Path(args.forecaster_artifact),
        device=config.runtime.device,
        batch_size=config.experiment.forecast_batch_size,
    )
    policies: list[
        tuple[str, int, float | None, float | None, float | None]
    ] = []
    for top_k in args.top_k:
        policies.append((f"mean_k{top_k}", top_k, None, None, None))
        policies.extend(
            (f"soft_k{top_k}_t{temperature:g}", top_k, temperature, None, None)
            for temperature in args.temperatures
        )
        policies.extend(
            (
                f"gated_mean_k{top_k}_tau{threshold:g}",
                top_k,
                None,
                threshold,
                None,
            )
            for threshold in args.disagreement_thresholds
        )
    policies.extend(
        (
            f"gap_mean_k2_k3_tau{threshold:g}",
            2,
            None,
            None,
            threshold,
        )
        for threshold in args.third_relative_gap_thresholds
    )

    rows: list[dict[str, object]] = []
    assignments = impute_root / "routing_assignments.jsonl"
    for record in _records(assignments):
        episode_id = str(record["episode_id"])
        dataset_id = str(record["dataset_id"])
        relative = Path(str(record["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe imputation path: {relative}")
        with np.load(impute_root / "imputations" / relative, allow_pickle=False) as archive:
            observed_mask = np.asarray(archive["observed_mask"], dtype=bool)
            base = np.asarray(archive["values"], dtype=float)
            saved = _load_saved_candidates(archive, observed_mask)
            spec = _forecast_spec(config, args.forecaster_id, base.shape[1])
            full_scale = np.asarray(archive["mase_scale"], dtype=float)
            scale = full_scale[
                list(spec.target_indices or ())
            ]
            if args.recompute_shortlist_scores:
                _set_forecast_seed(
                    stable_seed(
                        config.seed,
                        args.forecaster_id,
                        episode_id,
                        "shortlist-consensus",
                    )
                )
                diagnostics = _shortlist_diagnostics(
                    saved,
                    record.get("shortlist", ()),
                    runner,
                    spec,
                    scale,
                )
            else:
                diagnostics = _consensus_diagnostics(record)
            contexts: list[np.ndarray] = []
            policy_weights: list[dict[str, dict[str, float]]] = []
            for (
                _,
                top_k,
                temperature,
                disagreement_threshold,
                third_relative_gap_threshold,
            ) in policies:
                context, weights = _ensemble_context(
                    base,
                    observed_mask,
                    saved,
                    diagnostics,
                    forecast_mode=spec.mode,
                    target_indices=tuple(spec.target_indices or ()),
                    top_k=top_k,
                    temperature=temperature,
                    disagreement_threshold=disagreement_threshold,
                    third_relative_gap_threshold=third_relative_gap_threshold,
                    channel_scale=full_scale,
                )
                context[observed_mask] = np.asarray(archive["clean_context"])[observed_mask]
                contexts.append(context)
                policy_weights.append(weights)
            future = np.asarray(archive["clean_future"], dtype=float)[
                :, list(spec.target_indices or ())
            ]
        _set_forecast_seed(
            stable_seed(config.seed, args.forecaster_id, episode_id, "evaluation")
        )
        point = np.asarray(runner.predict(np.stack(contexts), spec).point, dtype=float)
        losses = np.mean(
            np.mean(np.abs(point - future[None, ...]), axis=1) / scale[None, :],
            axis=1,
        )
        for (
            policy,
            top_k,
            temperature,
            disagreement_threshold,
            third_relative_gap_threshold,
        ), loss, weights in zip(
            policies, losses, policy_weights, strict=True
        ):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "episode_id": episode_id,
                    "policy": policy,
                    "top_k": top_k,
                    "temperature": temperature,
                    "disagreement_threshold": disagreement_threshold,
                    "third_relative_gap_threshold": third_relative_gap_threshold,
                    "mase": float(loss),
                    "weights_by_channel": json.dumps(weights, sort_keys=True),
                }
            )

    episodes = pd.DataFrame.from_records(rows)
    episodes.to_csv(output_dir / "episode_ensemble_metrics.csv", index=False)
    summary = (
        episodes.groupby(["policy", "dataset_id"], as_index=False)
        .agg(mase=("mase", "mean"), episode_count=("episode_id", "nunique"))
    )
    summary["strict_best_candidate_mase"] = summary["dataset_id"].map(strict_best)
    summary["delta"] = summary["mase"] - summary["strict_best_candidate_mase"]
    summary["strict_win"] = summary["delta"] < 0
    summary.to_csv(output_dir / "dataset_ensemble_summary.csv", index=False)
    aggregate = (
        summary.groupby("policy", as_index=False)
        .agg(
            strict_wins=("strict_win", "sum"),
            dataset_count=("dataset_id", "nunique"),
            macro_mase=("mase", "mean"),
            macro_delta=("delta", "mean"),
            worst_delta=("delta", "max"),
        )
        .sort_values(
            ["strict_wins", "macro_delta", "worst_delta", "policy"],
            ascending=[False, True, True, True],
        )
    )
    aggregate.to_csv(output_dir / "ensemble_summary.csv", index=False)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPer-dataset best policy")
    print(
        summary.sort_values(["dataset_id", "delta", "policy"])
        .groupby("dataset_id", as_index=False)
        .first()[["dataset_id", "policy", "mase", "strict_best_candidate_mase", "delta"]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
