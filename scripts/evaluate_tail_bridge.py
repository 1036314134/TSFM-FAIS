"""Freeze fixed tail-action controls, then score original observed future targets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

BASE = ROOT / "artifacts/iclr27-r20"
DEFAULTS = ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json"


def fixed_points(prediction, control):
    actions = prediction["actions"].tolist()
    if actions != control["actions"]:
        raise ValueError("the frozen source and current action orders differ")
    result = {}
    for prefix, bank in (("", prediction["normal"]), ("budget_", prediction["budget"])):
        result.update({prefix + action: bank[i] for i, action in enumerate(actions)})
        result[prefix + "mean8"] = bank.mean(0)
        result[prefix + "median8"] = np.median(bank, axis=0)
        result[prefix + "source_single_mae"] = bank[control["single_index"]]
        for name, weights in (
            ("source_fixed_mae", control["fixed_mae"]["weights"]),
            ("source_fixed_joint", control["fixed_joint_weights"]),
        ):
            result[prefix + name] = (bank * np.asarray(weights)[:, None, None]).sum(0)
    result.update(
        bridge_native=prediction["bridge"],
        native_long1024=prediction["long"][0],
        native_long4096=prediction["long"][1],
    )
    if len(result) != 29:
        raise ValueError("the registered fixed method population changed")
    return result


def expanded_panels(frame):
    result = []
    for panel in ("native_common", "native_target_only"):
        selected = frame[frame.panel == panel].copy()
        selected["evaluation_panel"] = panel
        result.append(selected)
    selected = frame[(frame.panel != "synthetic") & (frame.model_id == "timesfm2p5")].copy()
    selected["evaluation_panel"] = "native_target_all"
    result.append(selected)
    synthetic = frame[frame.panel == "synthetic"]
    for gap in (8, 24, 48, None):
        selected = synthetic.copy() if gap is None else synthetic[synthetic.gap == gap].copy()
        selected["evaluation_panel"] = "synthetic_all" if gap is None else f"synthetic_g{gap}"
        result.append(selected)
    return pd.concat(result, ignore_index=True)


def aggregate(frame):
    common = ["evaluation_panel", "model_id", "method"]
    metrics = ["mae", "mse"]
    table = expanded_panels(frame)
    outputs = {}
    for name, remaining in (
        ("histories", ["group_id", "family_id", "dataset_id", "item_id", "base_id"]),
        ("series", ["group_id", "family_id", "dataset_id", "item_id"]),
        ("datasets", ["group_id", "family_id", "dataset_id"]),
        ("groups", ["group_id"]),
        ("summary", []),
    ):
        table = table.groupby(common + remaining)[metrics].mean().reset_index()
        outputs[name] = table
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed tail-action results")
    roots = [BASE / name for name in ("tail-plan-v001", "tail-inputs-v001", "tail-forecasts-v001")]
    plan, prepared, forecasts = [read_json(root / "manifest.json") for root in roots]
    if any(row["status"] != "completed" for row in (plan, prepared, forecasts)):
        raise ValueError("finish every registered input and forecast first")
    if len(forecasts["cases"]) != 100:
        raise ValueError("the frozen forecast population changed")
    identity = {
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [
                Path(__file__),
                *[root / "manifest.json" for root in roots],
                DEFAULTS,
                ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_PROTOCOL.md",
                ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_EXECUTION_ADDENDUM.md",
            ]
        },
        "primary": "bridge_native",
        "primary_metric": "mae",
        "main_panel": "native_common",
        "independent_confirmation": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial tail evaluation definitions changed")
    _write_json(output / "identity.json", identity)
    defaults = read_json(DEFAULTS)
    decisions = []
    for entry in forecasts["cases"]:
        path = roots[2] / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen tail forecast changed")
        with np.load(path, allow_pickle=False) as prediction:
            points = fixed_points(prediction, defaults["models"][entry["model_id"]])
        target = output / "predictions" / entry["model_id"] / f"{entry['case_id']}.npz"
        _save_npz(target, methods=np.asarray(list(points)), points=np.stack(list(points.values())))
        decisions.append(
            {
                "model_id": entry["model_id"],
                "case_id": entry["case_id"],
                "path": str(target.relative_to(output)),
                "sha256": file_sha256(target),
            }
        )
    _write_json(
        output / "predictions_frozen.json",
        {
            "identity": identity,
            "predictions": decisions,
            "current_future_values_read": False,
            "source_defaults_sha256": file_sha256(DEFAULTS),
        },
    )
    from matched_replay_sources import native_sources

    sources = native_sources()
    source_map = {(row["cohort"], row["dataset_id"], row["item_id"]): row for row in sources}
    metadata = {row["case_id"]: row for row in plan["cases"]}
    inputs = {row["case_id"]: row for row in prepared["cases"]}
    scores, targets = [], []
    for entry in decisions:
        row, inp = metadata[entry["case_id"]], inputs[entry["case_id"]]
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        truth = source["values"][row["origin"] : row["origin"] + 96, :2]
        path = roots[1] / inp["path"]
        if file_sha256(path) != inp["sha256"]:
            raise ValueError("a frozen tail input changed")
        with np.load(path, allow_pickle=False) as data:
            truth_z = (truth - data["mean"][:2]) / data["scale"][:2]
        observed = np.isfinite(truth_z)
        counts = observed.sum(0)
        if (counts < 48).any():
            raise ValueError("insufficient original observed future targets")
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            error = np.where(observed[None], saved["points"] - truth_z[None], 0.0)
            mae, mse = abs(error).sum(1) / counts, (error**2).sum(1) / counts
            for index, method in enumerate(saved["methods"].tolist()):
                record = {
                    **{
                        name: row[name]
                        for name in (
                            "case_id",
                            "base_id",
                            "panel",
                            "gap",
                            "origin",
                            "group_id",
                            "family_id",
                            "dataset_id",
                            "item_id",
                            "prior_use",
                        )
                    },
                    "model_id": entry["model_id"],
                    "method": method,
                    "mae": float(mae[index].mean()),
                    "mse": float(mse[index].mean()),
                }
                scores.append(record)
                for slot in (0, 1):
                    targets.append(
                        {
                            **record,
                            "target_slot": slot,
                            "target_gap": row["gaps"][entry["model_id"]][slot],
                            "observed_count": int(counts[slot]),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if len(scores) != 2900 or len(targets) != 5800:
        raise ValueError("the registered scoring population changed")
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    for name, table in aggregate(frame).items():
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "predictions": decisions,
            "score_rows": len(scores),
            "target_score_rows": len(targets),
            "primary": "bridge_native",
            "primary_metric": "mae",
            "main_panel": "native_common",
            "limits": "previously used development events; independent audit required",
            "source_files": [row["source_files"] for row in sources],
        },
    )


if __name__ == "__main__":
    main()
