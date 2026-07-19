"""Evaluate validity-gated primary and fallback candidate shrinkage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
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
from tsfm_fais.routing.blocks import detect_missing_blocks


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument("--primary-weight", required=True, type=float)
    parser.add_argument("--fallback-candidate", required=True)
    parser.add_argument(
        "--fallback-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0],
    )
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
        metrics.loc[eligible & native & metrics["method_role"].isin(("baseline", "missing_anchor"))]
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


def _safe_imputation_path(root: Path, record: dict[str, object]) -> Path:
    relative = Path(str(record["file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe imputation path: {relative}")
    return root / "imputations" / relative


def _validate_weight(name: str, value: float) -> float:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return float(value)


def main() -> int:
    args = _arguments()
    primary_weight = _validate_weight("primary weight", args.primary_weight)
    fallback_weights = tuple(
        dict.fromkeys(_validate_weight("fallback weight", value) for value in args.fallback_weights)
    )
    if args.primary_candidate == args.fallback_candidate:
        raise ValueError("primary and fallback candidates must differ")

    config = load_config(args.config)
    root = Path(args.impute_artifact).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.evaluation)
    if set(metrics["forecaster_id"].astype(str)) != {args.forecaster_id}:
        raise ValueError("evaluation forecaster does not match the requested model")
    strict_best = _strict_candidate_best(metrics)
    runner = _build_forecast_runner(
        args.forecaster_id,
        Path(args.forecaster_artifact),
        device=config.runtime.device,
        batch_size=config.experiment.forecast_batch_size,
    )

    rows: list[dict[str, object]] = []
    for record in _records(root / "routing_assignments.jsonl"):
        episode_id = str(record["episode_id"])
        dataset_id = str(record["dataset_id"])
        with np.load(_safe_imputation_path(root, record), allow_pickle=False) as archive:
            observed = np.asarray(archive["observed_mask"], dtype=bool)
            base = np.asarray(archive["values"], dtype=float)
            clean_context = np.asarray(archive["clean_context"], dtype=float)
            saved = _load_saved_candidates(archive, observed)
            required = (args.primary_candidate, args.fallback_candidate)
            missing = [candidate_id for candidate_id in required if candidate_id not in saved]
            if missing:
                raise ValueError(f"saved candidates lack {missing} for {episode_id}")
            saved_ids = tuple(str(value) for value in np.asarray(archive["candidate_ids"]).tolist())
            saved_index = {candidate_id: index for index, candidate_id in enumerate(saved_ids)}
            saved_native = np.asarray(archive["candidate_native_valid"], dtype=bool)
            primary = np.asarray(saved[args.primary_candidate]["values"], dtype=float)
            fallback = np.asarray(saved[args.fallback_candidate]["values"], dtype=float)
            primary_native = saved_native[saved_index[args.primary_candidate]]
            fallback_native = saved_native[saved_index[args.fallback_candidate]]
            spec = _forecast_spec(config, args.forecaster_id, base.shape[1])
            targets = list(spec.target_indices or ())
            visible_channels = (
                set(range(base.shape[1])) if spec.mode == "joint_multivariate" else set(targets)
            )
            blocks = detect_missing_blocks(observed)
            contexts: list[np.ndarray] = []
            for fallback_weight in fallback_weights:
                context = base.copy()
                for block in blocks:
                    if block.channel not in visible_channels:
                        continue
                    selector = (slice(block.start, block.end), block.channel)
                    if primary_native[selector].all():
                        context[selector] = (1.0 - primary_weight) * base[
                            selector
                        ] + primary_weight * primary[selector]
                    elif fallback_native[selector].all() and fallback_weight > 0.0:
                        context[selector] = (1.0 - fallback_weight) * base[
                            selector
                        ] + fallback_weight * fallback[selector]
                context[observed] = clean_context[observed]
                contexts.append(context)
            future = np.asarray(archive["clean_future"], dtype=float)[:, targets]
            scale = np.asarray(archive["mase_scale"], dtype=float)[targets]

        _set_forecast_seed(stable_seed(config.seed, args.forecaster_id, episode_id, "evaluation"))
        point = np.asarray(runner.predict(np.stack(contexts), spec).point, dtype=float)
        losses = np.mean(
            np.mean(np.abs(point - future[None, ...]), axis=1) / scale[None, :],
            axis=1,
        )
        rows.extend(
            {
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "primary_candidate": args.primary_candidate,
                "primary_weight": primary_weight,
                "fallback_candidate": args.fallback_candidate,
                "fallback_weight": fallback_weight,
                "mase": float(loss),
            }
            for fallback_weight, loss in zip(fallback_weights, losses, strict=True)
        )

    episodes = pd.DataFrame.from_records(rows)
    episodes.to_csv(output / "episode_shrinkage_metrics.csv", index=False)
    datasets = episodes.groupby(["fallback_weight", "dataset_id"], as_index=False).agg(
        mase=("mase", "mean"), episode_count=("episode_id", "nunique")
    )
    datasets["strict_best_candidate_mase"] = datasets["dataset_id"].map(strict_best)
    datasets["delta"] = datasets["mase"] - datasets["strict_best_candidate_mase"]
    datasets["strict_win"] = datasets["delta"] < 0.0
    datasets.to_csv(output / "dataset_shrinkage_summary.csv", index=False)
    aggregate = (
        datasets.groupby("fallback_weight", as_index=False)
        .agg(
            strict_wins=("strict_win", "sum"),
            dataset_count=("dataset_id", "nunique"),
            macro_mase=("mase", "mean"),
            macro_delta=("delta", "mean"),
            worst_delta=("delta", "max"),
        )
        .sort_values(
            ["strict_wins", "worst_delta", "macro_delta", "fallback_weight"],
            ascending=[False, True, True, True],
        )
    )
    aggregate.to_csv(output / "shrinkage_summary.csv", index=False)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
