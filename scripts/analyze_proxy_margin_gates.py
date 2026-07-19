"""Screen label-free proxy-risk margin gates on a development split.

The saved B-FAIS context already contains the configured proxy blend.  For a
block whose proxy advantage is below a policy threshold, this script restores
the primary candidate value.  Future observations are used only to score the
fixed development policies after every context has been assembled.
"""

from __future__ import annotations

import argparse
import json
import re
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

_BLOCK_ID = re.compile(r"^n(?P<batch>\d+):d(?P<channel>\d+):(?P<start>\d+)-(?P<end>\d+)$")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--impute-artifact", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--forecaster-id", required=True)
    parser.add_argument("--forecaster-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--absolute-thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--relative-thresholds",
        type=float,
        nargs="*",
        default=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
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
    result: dict[str, float] = {}
    for dataset_id, count in episode_counts.items():
        complete = candidates.loc[
            candidates["dataset_id"].eq(dataset_id) & candidates["count"].eq(count)
        ]
        if complete.empty:
            raise ValueError(f"no full-coverage candidate for {dataset_id}")
        result[str(dataset_id)] = float(complete["mase"].min())
    return result


def _policies(
    absolute: Iterable[float],
    relative: Iterable[float],
) -> tuple[tuple[str, str, float], ...]:
    policies = tuple(
        (f"absolute_margin_{threshold:g}", "absolute", float(threshold))
        for threshold in dict.fromkeys(absolute)
    ) + tuple(
        (f"relative_margin_{threshold:g}", "relative", float(threshold))
        for threshold in dict.fromkeys(relative)
    )
    if not policies or any(
        not np.isfinite(threshold) or threshold < 0.0
        for _, _, threshold in policies
    ):
        raise ValueError("proxy margin thresholds must be finite and non-negative")
    return policies


def _proxy_records(record: Mapping[str, object]) -> Mapping[str, object]:
    routing = record.get("routing_metadata", {})
    if not isinstance(routing, Mapping):
        raise ValueError("routing metadata must be a mapping")
    records = routing.get("proxy_blend_block_assignments", {})
    if not isinstance(records, Mapping):
        raise ValueError("proxy blend records must be a mapping")
    return records


def _gated_context(
    base: np.ndarray,
    observed: np.ndarray,
    candidates: Mapping[str, Mapping[str, object]],
    block_records: Mapping[str, object],
    *,
    mode: str,
    threshold: float,
) -> tuple[np.ndarray, int]:
    context = np.asarray(base, dtype=float).copy()
    disabled = 0
    for block_id, raw_record in block_records.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"invalid proxy blend record for {block_id}")
        match = _BLOCK_ID.fullmatch(str(block_id))
        if match is None or int(match.group("batch")) != 0:
            raise ValueError(f"unsupported block ID {block_id!r}")
        primary_id = str(raw_record["primary_assignment"])
        if primary_id not in candidates:
            raise ValueError(f"saved candidates lack primary {primary_id!r}")
        margin = float(raw_record["proxy_risk_margin"])
        primary_risk = float(raw_record["primary_proxy_risk"])
        evidence = (
            margin
            if mode == "absolute"
            else margin / max(abs(primary_risk), 1e-8)
        )
        if evidence + 1e-12 >= threshold:
            continue
        channel = int(match.group("channel"))
        start = int(match.group("start"))
        end = int(match.group("end"))
        primary = np.asarray(candidates[primary_id]["values"], dtype=float)
        context[start:end, channel] = primary[start:end, channel]
        disabled += 1
    context[observed] = base[observed]
    return context, disabled


def main() -> int:
    args = _arguments()
    policies = _policies(args.absolute_thresholds, args.relative_thresholds)
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
        relative_path = Path(str(record["file"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe imputation path: {relative_path}")
        with np.load(root / "imputations" / relative_path, allow_pickle=False) as archive:
            observed = np.asarray(archive["observed_mask"], dtype=bool)
            base = np.asarray(archive["values"], dtype=float)
            clean_context = np.asarray(archive["clean_context"], dtype=float)
            candidates = _load_saved_candidates(archive, observed)
            spec = _forecast_spec(config, args.forecaster_id, base.shape[1])
            targets = list(spec.target_indices or ())
            future = np.asarray(archive["clean_future"], dtype=float)[:, targets]
            scale = np.asarray(archive["mase_scale"], dtype=float)[targets]
            contexts: list[np.ndarray] = []
            disabled_counts: list[int] = []
            for _, mode, threshold in policies:
                context, disabled = _gated_context(
                    base,
                    observed,
                    candidates,
                    _proxy_records(record),
                    mode=mode,
                    threshold=threshold,
                )
                context[observed] = clean_context[observed]
                contexts.append(context)
                disabled_counts.append(disabled)
        _set_forecast_seed(
            stable_seed(config.seed, args.forecaster_id, episode_id, "evaluation")
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
                "policy": policy,
                "margin_mode": mode,
                "threshold": threshold,
                "disabled_block_count": disabled,
                "mase": float(loss),
            }
            for (policy, mode, threshold), disabled, loss in zip(
                policies,
                disabled_counts,
                losses,
                strict=True,
            )
        )

    episodes = pd.DataFrame.from_records(rows)
    episodes.to_csv(output / "episode_gate_metrics.csv", index=False)
    datasets = (
        episodes.groupby(
            ["policy", "margin_mode", "threshold", "dataset_id"],
            as_index=False,
        )
        .agg(
            mase=("mase", "mean"),
            episode_count=("episode_id", "nunique"),
            disabled_blocks=("disabled_block_count", "sum"),
        )
    )
    datasets["strict_best_candidate_mase"] = datasets["dataset_id"].map(strict_best)
    datasets["delta"] = datasets["mase"] - datasets["strict_best_candidate_mase"]
    datasets["strict_win"] = datasets["delta"] < 0.0
    datasets.to_csv(output / "dataset_gate_summary.csv", index=False)
    aggregate = (
        datasets.groupby(["policy", "margin_mode", "threshold"], as_index=False)
        .agg(
            strict_wins=("strict_win", "sum"),
            dataset_count=("dataset_id", "nunique"),
            macro_mase=("mase", "mean"),
            macro_delta=("delta", "mean"),
            worst_delta=("delta", "max"),
            disabled_blocks=("disabled_blocks", "sum"),
        )
        .sort_values(
            ["strict_wins", "macro_delta", "worst_delta", "policy"],
            ascending=[False, True, True, True],
        )
    )
    aggregate.to_csv(output / "gate_summary.csv", index=False)
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
