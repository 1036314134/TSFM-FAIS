"""Select imputations using only completed, originally observed forecast probes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.routing.recent_feedback import (  # noqa: E402
    observed_ensemble_weights,
    observed_forecast_risk,
    select_feedback,
    validate_feedback_end,
)
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def evaluate_prediction(point, truth):
    residual = np.asarray(point, float) - np.asarray(truth, float)
    if not np.isfinite(residual).all():
        raise ValueError("final evaluation requires finite predictions and complete truth")
    return {"mae": float(np.mean(np.abs(residual))), "mse": float(np.mean(residual**2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    parser.add_argument("--counts", default="1,2")
    parser.add_argument("--output-name", default="analysis-v001")
    args = parser.parse_args()
    source_root, root, accuracy_root = (
        args.source_root.resolve(),
        args.probe_root.resolve(),
        args.accuracy_root.resolve(),
    )
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    prepared = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    source_sha = file_sha256(source_root / "episodes_manifest.json")
    if (
        plan["source_manifest_sha256"] != source_sha
        or accuracy["source_episode_manifest_sha256"] != source_sha
    ):
        raise ValueError("probes and current forecasts belong to different source episodes")
    if not args.output_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in args.output_name
    ):
        parser.error("output-name must be a simple directory name")
    output = root / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("probe analysis is already complete; preserve its evidence")
    probes = {probe["probe_id"]: probe for probe in plan["probes"]}
    prepared_records = {probe["probe_id"]: probe for probe in prepared["probes"]}
    links = defaultdict(list)
    for link in plan["links"]:
        links[link["episode_id"]].append(link)
    indices = {record["episode_id"]: i for i, record in enumerate(source["episodes"])}
    decision_ids = set(plan["decision_episode_ids"])
    current_records = [
        record for record in source["episodes"] if record["episode_id"] in decision_ids
    ]
    counts = tuple(map(int, args.counts.split(",")))
    if not counts or min(counts) < 1 or max(counts) > max(plan["offsets"]):
        parser.error("counts must be positive and covered by the probe plan")
    models = args.models.split(",")
    references = pd.read_parquet(
        accuracy_root / "candidate_accuracy.parquet",
        columns=[
            "model_id",
            "split",
            "target_slot",
            "candidate_id",
            "family_id",
            "dataset_id",
            "mae",
            "mse",
        ],
    )
    references = references[
        (references.split == "train")
        & (references.target_slot == -1)
        & (references.candidate_id == "locf")
    ]
    results, mixtures, historical_risks = [], [], []
    eval_truth = np.load(accuracy_root / "truth_z.npy", mmap_mode="r")
    for model in models:
        manifest = json.loads((root / model / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"]["accuracy_manifest_sha256"] != file_sha256(
            accuracy_root / "manifest.json"
        ):
            raise ValueError("historical and current predictions use different accuracy exports")
        forecasts = {record["probe_id"]: record for record in manifest["probes"]}
        actions = manifest["action_ids"]
        ref = actions.index("locf")
        finite = [i for i, action in enumerate(actions) if action != "guarded_direct"]
        current_points = np.load(accuracy_root / f"{model}_point_z.npy", mmap_mode="r")
        current_positions = [accuracy["action_orders"][model].index(action) for action in actions]
        for current in current_records:
            index = indices[current["episode_id"]]
            candidates = np.asarray(current_points[index][current_positions])
            meta = {
                key: current[key]
                for key in (
                    "episode_id",
                    "origin_id",
                    "family_id",
                    "dataset_id",
                    "item_id",
                    "origin",
                    "mechanism",
                    "missing_rate",
                    "mask_seed",
                )
            }
            meta["model_id"] = model
            training = references[
                (references.model_id == model) & (references.family_id != current["family_id"])
            ]
            if training.empty:
                raise ValueError("joint-risk normalization needs other-family training episodes")
            normalizers = {
                metric: max(
                    float(
                        training.groupby(["family_id", "dataset_id"])[metric]
                        .mean()
                        .groupby(level="family_id")
                        .mean()
                        .mean()
                    ),
                    1e-12,
                )
                for metric in ("mae", "mse")
            }
            collected = {}
            for link in links[current["episode_id"]]:
                probe = probes[link["probe_id"]]
                validate_feedback_end(probe["origin"], probe["horizon"], current["origin"])
                for field in (
                    "dataset_id",
                    "item_id",
                    "split",
                    "mechanism",
                    "missing_rate",
                    "mask_seed",
                ):
                    if probe[field] != current[field]:
                        raise ValueError(
                            "probe does not share the decision's missingness trajectory"
                        )
                prepared_record = prepared_records[link["probe_id"]]
                forecast_record = forecasts[link["probe_id"]]
                prepared_path = root / prepared_record["path"]
                forecast_path = root / forecast_record["path"]
                if (
                    file_sha256(prepared_path) != prepared_record["sha256"]
                    or file_sha256(forecast_path) != forecast_record["sha256"]
                ):
                    raise ValueError("probe artifact changed")
                with (
                    np.load(prepared_path, allow_pickle=False) as observed,
                    np.load(forecast_path, allow_pickle=False) as predicted,
                ):
                    if (
                        str(predicted["probe_sha256"]) != prepared_record["sha256"]
                        or predicted["action_ids"].tolist() != actions
                    ):
                        raise ValueError("probe prediction and observed history do not match")
                    collected[(link["probe_horizon"], link["offset"])] = (
                        predicted["point_z"],
                        observed["observed_future_z"],
                    )
            for horizon in plan["horizons"]:
                for number in counts:
                    selected = [
                        value
                        for (probe_horizon, offset), value in collected.items()
                        if probe_horizon == horizon and offset <= number
                    ]
                    point = (
                        np.concatenate([value[0] for value in selected], axis=1)
                        if selected
                        else np.empty((len(actions), 0, candidates.shape[2]))
                    )
                    observed = (
                        np.concatenate([value[1] for value in selected], axis=0)
                        if selected
                        else np.empty((0, candidates.shape[2]))
                    )
                    minimum = max(1, horizon // 8)
                    risk, n_observed, valid = observed_forecast_risk(point, observed, minimum)
                    base = meta | {
                        "probe_horizon": horizon,
                        "probe_count": number,
                        "available_probes": len(selected),
                        "feedback_target_fraction": float(valid.mean()),
                        "observed_feedback_cells": int(n_observed.sum()),
                    }
                    for action_index, action in enumerate(actions):
                        for target in range(candidates.shape[2]):
                            historical_risks.append(
                                base
                                | {
                                    "candidate_id": action,
                                    "target_slot": target,
                                    "observed_count": int(n_observed[target]),
                                    "feedback_valid": bool(valid[target]),
                                    "historical_mae": float(risk["mae"][action_index, target]),
                                    "historical_mse": float(risk["mse"][action_index, target]),
                                    "source_mae_normalizer": normalizers["mae"],
                                    "source_mse_normalizer": normalizers["mse"],
                                }
                            )

                    # Decisions and weights above/below only read the observed
                    # completed probes. The current future is used solely here.
                    def record(
                        method,
                        objective,
                        prediction,
                        shrinkage=0.0,
                        *,
                        base=base,
                        index=index,
                        selected_actions=(),
                        forecast_weights=None,
                    ):
                        results.append(
                            base
                            | {
                                "method": method,
                                "objective": objective,
                                "shrinkage": shrinkage,
                                "selected_action_ids": json.dumps(selected_actions),
                                "forecast_weights": json.dumps(forecast_weights),
                            }
                            | evaluate_prediction(prediction, eval_truth[index])
                        )

                    for objective in ("mae", "mse", "joint"):
                        choice = select_feedback(
                            risk, objective, normalizers, ref, per_target=False
                        )
                        record(
                            "recent_sequence",
                            objective,
                            candidates[choice],
                            selected_actions=[actions[choice]],
                        )
                        if model == "timesfm2p5":
                            choices = select_feedback(
                                risk, objective, normalizers, ref, per_target=True
                            )
                            prediction = np.column_stack(
                                [candidates[action, :, slot] for slot, action in enumerate(choices)]
                            )
                            record(
                                "recent_target",
                                objective,
                                prediction,
                                selected_actions=[actions[action] for action in choices],
                            )
                    for pool_name, positions in (
                        ("finite", finite),
                        ("guarded", list(range(len(actions)))),
                    ):
                        pool_actions = [actions[position] for position in positions]
                        pool_ref = pool_actions.index("locf")
                        for per_target in (False, True) if model == "timesfm2p5" else (False,):
                            granularity = "target" if per_target else "sequence"
                            for shrinkage in (0.0, 0.5):
                                weights = observed_ensemble_weights(
                                    point[positions],
                                    observed,
                                    minimum,
                                    pool_ref,
                                    per_target=per_target,
                                    shrinkage=shrinkage,
                                )
                                prediction = (
                                    np.einsum("ka,ahk->hk", weights, candidates[positions])
                                    if per_target
                                    else np.einsum("a,ahk->hk", weights, candidates[positions])
                                )
                                record(
                                    f"recent_forecast_{pool_name}_{granularity}",
                                    "mse",
                                    prediction,
                                    shrinkage,
                                    selected_actions=pool_actions,
                                    forecast_weights=weights.tolist(),
                                )
                                if pool_name == "finite":
                                    mixtures.append(
                                        base
                                        | {
                                            "method": f"recent_imputation_{granularity}",
                                            "objective": "mse",
                                            "shrinkage": shrinkage,
                                            "per_target": per_target,
                                            "action_ids": json.dumps(pool_actions),
                                            "weights": json.dumps(weights.tolist()),
                                        }
                                    )
        print(json.dumps({"model": model, "status": "analyzed"}), flush=True)
    frame = pd.DataFrame(results)
    keys = ["model_id", "probe_horizon", "probe_count", "method", "objective", "shrinkage"]
    if frame.duplicated(keys + ["episode_id"]).any():
        raise ValueError("duplicate probe evaluation episodes")
    family = (
        frame.groupby(keys + ["family_id", "dataset_id"])[
            ["mae", "mse", "feedback_target_fraction"]
        ]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse", "feedback_target_fraction"]].mean().reset_index()
    frame.to_parquet(output / "episode_results.parquet", index=False)
    pd.DataFrame(historical_risks).to_parquet(output / "historical_risks.parquet", index=False)
    pd.DataFrame(mixtures).to_parquet(output / "input_mixture_weights.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "script_sha256": file_sha256(Path(__file__)),
            "feedback_module_sha256": file_sha256(
                ROOT / "src/tsfm_fais/routing/recent_feedback.py"
            ),
            "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
            "plan_sha256": file_sha256(root / "plan.json"),
            "prepared_manifest_sha256": file_sha256(root / "prepared_manifest.json"),
            "forecast_manifests": {
                model: file_sha256(root / model / "manifest.json") for model in models
            },
            "candidate_imputation_combinations": "weights are saved here; their downstream results require actual model evaluation of the combined imputed contexts",
            "information": "only originally observed values in completed historical probe horizons influence decisions; all current future values are evaluation-only",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
