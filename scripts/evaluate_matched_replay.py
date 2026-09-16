"""Freeze past-feedback actions before reading current forecast outcomes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from aligned_portfolio_io import decision_truth
from latent_source_inputs import ROOT, read_json
from matched_replay_core import RULES, choose_current, history_decisions, observed_risks
from metric_source_gate import fit_fixed_metric
from pool_gate_inputs import load_pool_inputs

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def source_defaults(output):
    root = ROOT / "artifacts/iclr27-r12/motm-pool-source-v001"
    pool = ROOT / "artifacts/iclr27-r12/motm-pool-inputs-v001"
    original = read_json(root / "manifest.json")
    audit = read_json(ROOT / "artifacts/iclr27-r12/motm-pool-audit-v001/manifest.json")
    accuracy = ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    truth_record = read_json(accuracy / "manifest.json")
    identity = {
        "source_study_sha256": file_sha256(root / "manifest.json"),
        "pool_sha256": file_sha256(pool / "manifest.json"),
        "truth_sha256": file_sha256(accuracy / "truth_z.npy"),
        "loss_module_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
    }
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != identity["source_study_sha256"]
        or truth_record["prediction_arrays"]["truth_z.npy"] != identity["truth_sha256"]
    ):
        raise ValueError("a source-default provenance check failed")
    path = output / "source_defaults.json"
    if path.exists():
        result = read_json(path)
        if result["identity"] != identity:
            raise ValueError("source-default definitions changed")
        return result
    result = {"identity": identity, "models": {}}
    for model_id in ("chronos2", "timesfm2p5"):
        manifest, frame, arrays = load_pool_inputs(pool, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        points = arrays["vectors"][training]
        target = decision_truth(
            frame.iloc[training], np.load(accuracy / "truth_z.npy", mmap_mode="r")
        )
        weights = _family_weights(frame.iloc[training])
        mae = np.einsum("n,na->a", weights / weights.sum(), abs(points - target[:, None]).mean(2))
        fixed = fit_fixed_metric(points, target, weights, "mae")
        fold = next(
            row
            for row in original["folds"]
            if row["model_id"] == model_id and row["held_family"] is None
        )
        if file_sha256(root / fold["control_path"]) != fold["control_sha256"]:
            raise ValueError("the source joint fixed control changed")
        joint = read_json(root / fold["control_path"])
        with np.load(
            pool / model_id / manifest["episodes"][0]["path"], allow_pickle=False
        ) as saved:
            actions = saved["actions"].tolist()
        result["models"][model_id] = {
            "actions": actions,
            "single_index": int(mae.argmin()),
            "single_mae": mae.tolist(),
            "fixed_mae": fixed,
            "fixed_joint_weights": joint["weights"],
            "source_origins": sorted(frame.iloc[training].origin_id.unique()),
            "source_families": sorted(frame.iloc[training].family_id.unique()),
        }
    _write_json(path, result)
    return result


def replacement_probabilities(choices):
    result = np.zeros((9, 2))
    for slot, choice in enumerate(choices):
        if choice == 8:
            result[8, slot] = 1
        else:
            result[:8, slot] = 1 / 8
    return result


def score_points(points, truth_z):
    observed = np.isfinite(truth_z)
    counts = observed.sum(0)
    if (counts < 48).any():
        raise ValueError("the current common future mask lost support")
    error = np.where(observed[None], points - truth_z[None], 0.0)
    return abs(error).sum(1) / counts, (error**2).sum(1) / counts


def aggregate_groups(frame):
    metrics = ["mae", "mse"]
    series = (
        frame.groupby(["model_id", "method", "group_id", "family_id", "dataset_id", "item_id"])[
            metrics
        ]
        .mean()
        .reset_index()
    )
    datasets = (
        series.groupby(["model_id", "method", "group_id", "family_id", "dataset_id"])[metrics]
        .mean()
        .reset_index()
    )
    groups = datasets.groupby(["model_id", "method", "group_id"])[metrics].mean().reset_index()
    summary = groups.groupby(["model_id", "method"])[metrics].mean().reset_index()
    return series, datasets, groups, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed matched-replay results")
    plan_root = ROOT / "artifacts/iclr27-r19/pilot-plan-v001"
    prepared_root = ROOT / "artifacts/iclr27-r19/replay-inputs-v001"
    forecast_root = ROOT / "artifacts/iclr27-r19/replay-forecasts-v001"
    plan, prepared, forecasts = [
        read_json(root / "manifest.json") for root in (plan_root, prepared_root, forecast_root)
    ]
    if (
        any(record["status"] != "completed" for record in (plan, prepared, forecasts))
        or len(forecasts["cases"]) != 48
    ):
        raise ValueError("complete all frozen current and historical predictions first")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "core_module_sha256": file_sha256(ROOT / "scripts/matched_replay_core.py"),
        "plan_sha256": file_sha256(plan_root / "manifest.json"),
        "prepared_sha256": file_sha256(prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
        "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R19_MATCHED_REPLAY_PROTOCOL.md"),
        "primary": "matched_erm",
        "primary_metric": "mae",
        "deployment": "past target feedback available",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial replay evaluation definitions changed")
    _write_json(output / "identity.json", identity)
    torch.set_num_threads(1)
    defaults = source_defaults(output)
    metadata = {row["case_id"]: row for row in plan["cases"]}
    input_map = {row["case_id"]: row for row in prepared["cases"]}
    decisions = []
    for entry in forecasts["cases"]:
        model_id, case_id = entry["model_id"], entry["case_id"]
        input_record = input_map[case_id]
        if (
            file_sha256(prepared_root / input_record["path"]) != input_record["sha256"]
            or file_sha256(forecast_root / entry["path"]) != entry["sha256"]
        ):
            raise ValueError("a frozen replay input or forecast changed")
        with (
            np.load(prepared_root / input_record["path"], allow_pickle=False) as inputs,
            np.load(forecast_root / entry["path"], allow_pickle=False) as prediction,
        ):
            mean, scale = inputs["mean"][:2], inputs["scale"][:2]
            past_truth = (inputs["historical_future"] - mean) / scale
            current = prediction["current"]
            bank = np.concatenate([current, np.median(current, axis=0)[None]])
            actions = prediction["actions"].tolist()
            control = defaults["models"][model_id]
            if actions != control["actions"]:
                raise ValueError("the source and replay candidate identities differ")
            point_methods = {action: current[i] for i, action in enumerate(actions)}
            point_methods.update(
                median8=bank[8],
                mean8=current.mean(0),
                source_single_mae=current[control["single_index"]],
                source_fixed_mae=(
                    current * np.asarray(control["fixed_mae"]["weights"])[:, None, None]
                ).sum(0),
                source_fixed_joint=(
                    current * np.asarray(control["fixed_joint_weights"])[:, None, None]
                ).sum(0),
            )
            point_methods.update(
                zip(prediction["extra_names"].tolist(), prediction["extras"], strict=True)
            )
            random_methods = {"random_forced": replacement_probabilities([0, 0])}
            all_mae, all_mse, choices_all, means, uncertainties = [], [], [], [], []
            for rule, historical in zip(RULES, prediction["historical"], strict=True):
                histories = np.concatenate(
                    [historical, np.median(historical, axis=1)[:, None]], axis=1
                )
                mae, mse = observed_risks(histories, past_truth, np.ones(2))
                choices, paired_mean, uncertainty = history_decisions(mae, model_id == "chronos2")
                all_mae.append(mae)
                all_mse.append(mse)
                means.append(paired_mean)
                uncertainties.append(uncertainty)
                choices_all.append(np.stack(list(choices.values())))
                for mode, selected in choices.items():
                    point_methods[f"{rule}_{mode}"] = choose_current(bank, selected)
                    if mode != "forced":
                        random_methods[f"{rule}_{mode}_random_replacement"] = (
                            replacement_probabilities(selected)
                        )
            if len(point_methods) != 25 or len(random_methods) != 7:
                raise ValueError("the registered decision and control population changed")
            path = output / "decisions" / model_id / f"{case_id}.npz"
            _save_npz(
                path,
                current_bank=bank,
                methods=np.asarray(list(point_methods)),
                points=np.stack(list(point_methods.values())),
                random_methods=np.asarray(list(random_methods)),
                random_probabilities=np.stack(list(random_methods.values())),
                historical_mae=np.stack(all_mae),
                historical_mse=np.stack(all_mse),
                choices=np.stack(choices_all),
                paired_means=np.stack(means),
                uncertainties=np.stack(uncertainties),
            )
        decisions.append(
            {
                "model_id": model_id,
                "case_id": case_id,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    _write_json(
        output / "decisions_frozen.json",
        {
            "identity": identity,
            "decisions": decisions,
            "source_defaults_sha256": file_sha256(output / "source_defaults.json"),
            "current_future_values_read": False,
        },
    )
    scores, switches = [], []
    for entry in decisions:
        row = metadata[entry["case_id"]]
        input_record = input_map[entry["case_id"]]
        if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
            raise ValueError("an original current outcome file changed")
        with (
            np.load(row["current_artifact"], allow_pickle=False) as original,
            np.load(prepared_root / input_record["path"], allow_pickle=False) as inputs,
        ):
            truth = original["future"][:96]
            observed = original["future_observed"][:96]
            np.testing.assert_array_equal(observed, np.isfinite(truth))
            truth_z = (truth - inputs["mean"][:2]) / inputs["scale"][:2]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            maes, mses = score_points(saved["points"], truth_z)
            losses = {
                name: (float(mae.mean()), float(mse.mean()))
                for name, mae, mse in zip(saved["methods"].tolist(), maes, mses, strict=True)
            }
            bank_mae, bank_mse = score_points(saved["current_bank"], truth_z)
            for name, probability in zip(
                saved["random_methods"].tolist(), saved["random_probabilities"], strict=True
            ):
                losses[name] = (
                    float((probability * bank_mae).sum(0).mean()),
                    float((probability * bank_mse).sum(0).mean()),
                )
            for r, rule in enumerate(RULES):
                for mode_index, mode in enumerate(("forced", "erm", "conservative")):
                    switches.append(
                        {
                            "case_id": entry["case_id"],
                            "model_id": entry["model_id"],
                            "rule": rule,
                            "mode": mode,
                            "replacement_fraction": float(
                                (saved["choices"][r, mode_index] != 8).mean()
                            ),
                            "group_id": row["group_id"],
                            "dataset_id": row["dataset_id"],
                            "item_id": row["item_id"],
                        }
                    )
        for method, (mae, mse) in losses.items():
            scores.append(
                {
                    **{
                        name: row[name]
                        for name in (
                            "group_id",
                            "family_id",
                            "dataset_id",
                            "item_id",
                            "episode_id",
                            "origin_id",
                        )
                    },
                    "model_id": entry["model_id"],
                    "case_id": entry["case_id"],
                    "method": method,
                    "mae": mae,
                    "mse": mse,
                }
            )
    frame = pd.DataFrame(scores)
    if len(frame) != 1536:
        raise ValueError("the pilot scoring population changed")
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(switches).to_csv(output / "replacement_rates.csv", index=False)
    for name, table in zip(
        ("series", "datasets", "groups", "summary"), aggregate_groups(frame), strict=True
    ):
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "decisions": decisions,
            "score_rows": len(frame),
            "source_groups": 4,
            "cases": 24,
            "primary": "matched_erm",
            "primary_metric": "mae",
            "random_controls_are_expected_action_losses": True,
            "limits": "previously used, support-qualified development pilot; independent audit required",
        },
    )


if __name__ == "__main__":
    main()
