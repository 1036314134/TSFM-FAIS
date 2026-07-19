"""Tune forecast-consensus prior weight on a designated development split."""

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
from tsfm_fais.pipeline import _regularized_forecast_consensus_candidate


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--weights",
        default=",".join(f"{weight / 20:.2f}" for weight in range(21)),
    )
    return parser.parse_args()


def _records(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _strict_candidate_best(metrics: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    eligible = metrics[metrics["metric_eligible"].astype(str).str.lower().eq("true")]
    method_rows = eligible[eligible["method"].eq("b_fais")]
    episode_counts = method_rows.groupby("dataset_id")["episode_id"].nunique()
    actual = method_rows.groupby("dataset_id")["mase"].mean().astype(float).to_dict()
    candidates = (
        eligible[eligible["method_role"].isin(("baseline", "missing_anchor"))]
        .groupby(["dataset_id", "method"])
        .agg(mase=("mase", "mean"), count=("episode_id", "nunique"))
        .reset_index()
    )
    best: dict[str, float] = {}
    for dataset_id, episode_count in episode_counts.items():
        full = candidates[
            candidates["dataset_id"].eq(dataset_id) & candidates["count"].eq(episode_count)
        ].sort_values(["mase", "method"])
        if full.empty:
            raise ValueError(f"no full-coverage candidate for {dataset_id}")
        best[str(dataset_id)] = float(full.iloc[0]["mase"])
    return actual, best


def _diagnostics(record: Mapping[str, object]) -> Mapping[str, object] | None:
    routing = record.get("routing_metadata")
    if not isinstance(routing, Mapping):
        raise ValueError("routing record lacks routing_metadata")
    diagnostics = routing.get("forecast_consensus")
    if diagnostics is None:
        return None
    if not isinstance(diagnostics, Mapping):
        raise ValueError("forecast consensus diagnostics must be a mapping")
    return diagnostics if diagnostics.get("active") else None


def main() -> int:
    args = _arguments()
    config = load_config(args.config)
    impute_root = Path(args.impute_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = tuple(float(value) for value in args.weights.split(","))
    if not weights or any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("weights must be a non-empty comma-separated subset of [0, 1]")

    metrics = pd.read_csv(args.evaluation)
    actual, strict_best = _strict_candidate_best(metrics)
    eligible_metrics = metrics[metrics["metric_eligible"].astype(str).str.lower().eq("true")]
    actual_episode = (
        eligible_metrics[eligible_metrics["method"].eq("b_fais")]
        .set_index("episode_id")["mase"]
        .astype(float)
        .to_dict()
    )
    runner = _build_forecast_runner(
        args.forecaster_id,
        Path(args.forecaster_artifact),
        device=config.runtime.device,
        batch_size=config.experiment.forecast_batch_size,
    )
    candidate_rows: list[dict[str, object]] = []
    diagnostics_by_episode: dict[str, Mapping[str, object]] = {}
    fixed_loss_by_episode: dict[str, float] = {}
    dataset_lookup: dict[str, str] = {}
    for record in _records(impute_root / "routing_assignments.jsonl"):
        episode_id = str(record["episode_id"])
        dataset_id = str(record["dataset_id"])
        dataset_lookup[episode_id] = dataset_id
        diagnostics = _diagnostics(record)
        if diagnostics is None:
            fixed_loss_by_episode[episode_id] = float(actual_episode[episode_id])
            continue
        raw_scores = diagnostics.get("scores")
        raw_priors = diagnostics.get("candidate_priors", {})
        dynamic_ids = set(map(str, diagnostics.get("dataset_prior_candidates", ())))
        if not isinstance(raw_scores, Mapping) or not isinstance(raw_priors, Mapping):
            raise ValueError("forecast consensus scores and priors must be mappings")
        candidate_ids = tuple(sorted(map(str, raw_scores)))
        with np.load(
            impute_root / "imputations" / str(record["file"]),
            allow_pickle=False,
        ) as archive:
            observed = np.asarray(archive["observed_mask"], dtype=bool)
            saved = _load_saved_candidates(archive, observed)
            missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in saved]
            if missing:
                raise ValueError(f"saved candidate tensors missing for {episode_id}: {missing}")
            contexts = np.stack(
                [
                    np.asarray(saved[candidate_id]["values"], dtype=float)
                    for candidate_id in candidate_ids
                ]
            )
            spec = _forecast_spec(config, args.forecaster_id, contexts.shape[2])
            forecast_seed = stable_seed(
                config.seed,
                args.forecaster_id,
                episode_id,
                "consensus_weight_sweep",
            )
            _set_forecast_seed(forecast_seed)
            point = np.asarray(runner.predict(contexts, spec).point, dtype=float)
            future = np.asarray(archive["clean_future"], dtype=float)[:, list(spec.target_indices)]
            scale = np.asarray(archive["mase_scale"], dtype=float)[list(spec.target_indices)]
        losses = np.mean(
            np.mean(np.abs(point - future[None, ...]), axis=1) / scale[None, :],
            axis=1,
        )
        diagnostics_by_episode[episode_id] = diagnostics
        for candidate_id, loss in zip(candidate_ids, losses, strict=True):
            candidate_rows.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": dataset_id,
                    "candidate_id": candidate_id,
                    "mase": float(loss),
                    "medoid_score": float(raw_scores[candidate_id]),
                    "prior": (
                        float(raw_priors[candidate_id]) if candidate_id in raw_priors else np.nan
                    ),
                    "dynamic_candidate": candidate_id in dynamic_ids,
                }
            )

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_dir / "candidate_forecast_losses.csv", index=False)
    loss_lookup = candidate_frame.set_index(["episode_id", "candidate_id"])["mase"].to_dict()
    sweep_rows: list[dict[str, object]] = []
    for weight in weights:
        selections: list[dict[str, object]] = []
        selections.extend(
            {
                "dataset_id": dataset_lookup[episode_id],
                "mase": loss,
            }
            for episode_id, loss in fixed_loss_by_episode.items()
        )
        for episode_id, diagnostics in diagnostics_by_episode.items():
            raw_scores = {
                str(candidate_id): float(score)
                for candidate_id, score in diagnostics["scores"].items()
            }
            raw_priors = {
                str(candidate_id): float(score)
                for candidate_id, score in diagnostics.get("candidate_priors", {}).items()
            }
            selected, _ = _regularized_forecast_consensus_candidate(
                raw_scores,
                raw_priors,
                weight,
            )
            selections.append(
                {
                    "dataset_id": dataset_lookup[episode_id],
                    "mase": float(loss_lookup[(episode_id, selected)]),
                }
            )
        by_dataset = pd.DataFrame(selections).groupby("dataset_id")["mase"].mean()
        for dataset_id, mase in by_dataset.items():
            sweep_rows.append(
                {
                    "weight": weight,
                    "dataset_id": dataset_id,
                    "mase": float(mase),
                    "strict_best_candidate_mase": strict_best[dataset_id],
                    "strict_win": bool(mase < strict_best[dataset_id]),
                    "actual_b_fais_mase": actual[dataset_id],
                }
            )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(output_dir / "weight_sweep.csv", index=False)
    summary = (
        sweep.groupby("weight")
        .agg(
            strict_wins=("strict_win", "sum"),
            dataset_count=("dataset_id", "nunique"),
            macro_mase=("mase", "mean"),
            macro_actual_b_fais=("actual_b_fais_mase", "mean"),
        )
        .reset_index()
        .sort_values(["strict_wins", "macro_mase", "weight"], ascending=[False, True, True])
    )
    summary.to_csv(output_dir / "weight_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
