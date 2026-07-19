"""Evaluate fixed blends of two inference-only imputation selectors on development data."""

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
    _set_forecast_seed,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--primary-impute-artifact", required=True)
    parser.add_argument("--secondary-impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--secondary-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5],
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


def _safe_imputation_path(root: Path, record: dict[str, object]) -> Path:
    relative = Path(str(record["file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe imputation path: {relative}")
    return root / "imputations" / relative


def main() -> int:
    args = _arguments()
    weights = tuple(dict.fromkeys(map(float, args.secondary_weights)))
    if any(not np.isfinite(weight) or not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("secondary weights must be finite and lie in [0, 1]")
    config = load_config(args.config)
    primary_root = Path(args.primary_impute_artifact).resolve()
    secondary_root = Path(args.secondary_impute_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    secondary_records = {
        str(record["episode_id"]): record
        for record in _records(secondary_root / "routing_assignments.jsonl")
    }

    rows: list[dict[str, object]] = []
    for primary_record in _records(primary_root / "routing_assignments.jsonl"):
        episode_id = str(primary_record["episode_id"])
        dataset_id = str(primary_record["dataset_id"])
        if episode_id not in secondary_records:
            raise ValueError(f"secondary artifact lacks episode {episode_id}")
        secondary_record = secondary_records[episode_id]
        if str(primary_record["forecaster_id"]) != args.forecaster_id or str(
            secondary_record["forecaster_id"]
        ) != args.forecaster_id:
            raise ValueError(f"routing forecaster mismatch for {episode_id}")
        with np.load(
            _safe_imputation_path(primary_root, primary_record),
            allow_pickle=False,
        ) as primary_archive, np.load(
            _safe_imputation_path(secondary_root, secondary_record),
            allow_pickle=False,
        ) as secondary_archive:
            primary = np.asarray(primary_archive["values"], dtype=float)
            secondary = np.asarray(secondary_archive["values"], dtype=float)
            observed = np.asarray(primary_archive["observed_mask"], dtype=bool)
            if not np.array_equal(observed, secondary_archive["observed_mask"]):
                raise ValueError(f"observed-mask mismatch for {episode_id}")
            clean_context = np.asarray(primary_archive["clean_context"], dtype=float)
            if not np.array_equal(clean_context, secondary_archive["clean_context"]):
                raise ValueError(f"clean-context mismatch for {episode_id}")
            clean_future = np.asarray(primary_archive["clean_future"], dtype=float)
            if not np.array_equal(clean_future, secondary_archive["clean_future"]):
                raise ValueError(f"clean-future mismatch for {episode_id}")
            full_scale = np.asarray(primary_archive["mase_scale"], dtype=float)
            if not np.array_equal(full_scale, secondary_archive["mase_scale"]):
                raise ValueError(f"MASE-scale mismatch for {episode_id}")
            spec = _forecast_spec(config, args.forecaster_id, primary.shape[1])
            contexts: list[np.ndarray] = []
            for weight in weights:
                context = (1.0 - weight) * primary + weight * secondary
                context[observed] = clean_context[observed]
                contexts.append(context)
            targets = list(spec.target_indices or ())
            future = clean_future[:, targets]
            scale = full_scale[targets]
        _set_forecast_seed(
            stable_seed(config.seed, args.forecaster_id, episode_id, "selector-blend")
        )
        point = np.asarray(runner.predict(np.stack(contexts), spec).point, dtype=float)
        losses = np.mean(
            np.mean(np.abs(point - future[None, ...]), axis=1) / scale[None, :],
            axis=1,
        )
        rows.extend(
            {
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "policy": f"secondary_weight_{weight:g}",
                "secondary_weight": weight,
                "mase": float(loss),
            }
            for weight, loss in zip(weights, losses, strict=True)
        )

    episodes = pd.DataFrame.from_records(rows)
    episodes.to_csv(output_dir / "episode_blend_metrics.csv", index=False)
    summary = (
        episodes.groupby(["policy", "secondary_weight", "dataset_id"], as_index=False)
        .agg(mase=("mase", "mean"), episode_count=("episode_id", "nunique"))
    )
    summary["strict_best_candidate_mase"] = summary["dataset_id"].map(strict_best)
    summary["delta"] = summary["mase"] - summary["strict_best_candidate_mase"]
    summary["strict_win"] = summary["delta"] < 0
    summary.to_csv(output_dir / "dataset_blend_summary.csv", index=False)
    aggregate = (
        summary.groupby(["policy", "secondary_weight"], as_index=False)
        .agg(
            strict_wins=("strict_win", "sum"),
            dataset_count=("dataset_id", "nunique"),
            macro_mase=("mase", "mean"),
            macro_delta=("delta", "mean"),
            worst_delta=("delta", "max"),
        )
        .sort_values(
            ["strict_wins", "worst_delta", "macro_delta", "policy"],
            ascending=[False, True, True, True],
        )
    )
    aggregate.to_csv(output_dir / "blend_summary.csv", index=False)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
