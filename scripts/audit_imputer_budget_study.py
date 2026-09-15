"""Audit budget-study inputs and forecasts, then compare matched downstream errors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def verify_evaluation_coverage(frame, prepared):
    expected_ids = {record["episode_id"] for record in prepared["cases"]}
    expected_count = prepared.get("evaluation_episode_count", 18)
    if len(expected_ids) != expected_count or expected_count < 1:
        raise ValueError("the prepared panel disagrees with its declared episode count")
    for _, group in frame.groupby(["model_id", "budget", "method"]):
        if len(group) != expected_count or set(group.episode_id) != expected_ids:
            raise ValueError("a budget variant has different evaluation coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "evaluation-root", "source-root", "accuracy-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    args = parser.parse_args()
    models = args.models.split(",")
    if len(set(models)) != len(models) or set(models) - {"chronos2", "timesfm2p5", "tirex"}:
        raise ValueError("distinct supported forecasting models are required")
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed budget audits")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    if (
        prepared["status"] != "completed"
        or len(prepared["fits"]) != 18
        or file_sha256(args.source_root / "episodes_manifest.json")
        != prepared["identity"]["source_manifest_sha256"]
    ):
        raise ValueError("complete the registered preparation without changing its source")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    batch_map = {}
    for record in prepared["training_batches"]:
        path = args.prepared_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a historical training batch changed")
        batch_map[(record["dataset_id"], record["windows"])] = path
    for dataset in prepared["identity"]["datasets"]:
        with (
            np.load(batch_map[(dataset, 64)], allow_pickle=False) as small,
            np.load(batch_map[(dataset, 512)], allow_pickle=False) as large,
        ):
            np.testing.assert_array_equal(small["item_ids"], large["item_ids"][:64])
            np.testing.assert_array_equal(small["values"], large["values"][:64])
            np.testing.assert_array_equal(small["observed"], large["observed"][:64])
            if len(set(large["item_ids"].tolist())) != 512:
                raise ValueError("the larger training batch duplicated descriptors")
    fit_records = []
    structures = {}
    for item in prepared["fits"]:
        path = args.prepared_root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a fitted budget model record changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        structure = {
            name: value for name, value in record["constructor_params"].items() if name != "epochs"
        }
        key = (record["dataset_id"], record["candidate_id"])
        if key in structures and structure != structures[key]:
            raise ValueError("a model changed structure together with its budget")
        structures[key] = structure
        for entry in record["files"]:
            if file_sha256(path.parent / record["candidate_id"] / entry["path"]) != entry["sha256"]:
                raise ValueError("a fitted imputer checkpoint changed")
        fit_records.append(
            {
                name: record[name]
                for name in (
                    "dataset_id",
                    "budget",
                    "candidate_id",
                    "seconds",
                    "training_best_loss",
                )
            }
        )
    originals = {record["episode_id"]: record for record in source["episodes"]}
    for case in prepared["cases"]:
        original = originals[case["episode_id"]]
        if (
            file_sha256(args.prepared_root / case["path"]) != case["sha256"]
            or file_sha256(args.source_root / original["path"]) != original["sha256"]
        ):
            raise ValueError("a budget input or its original changed")
        with (
            np.load(args.prepared_root / case["path"], allow_pickle=False) as changed,
            np.load(args.source_root / original["path"], allow_pickle=False) as baseline,
        ):
            actions, context = baseline["candidate_ids"].tolist(), baseline["context"]
            np.testing.assert_array_equal(changed["candidate_ids"], baseline["candidate_ids"])
            for index, name in enumerate(actions):
                values = changed["candidate_values"][index]
                np.testing.assert_array_equal(
                    values[np.isfinite(context)], context[np.isfinite(context)]
                )
                if name not in {"saits", "timemixerpp"}:
                    np.testing.assert_array_equal(values, baseline["candidate_values"][index])
    rows, sources, max_difference = [], {}, 0.0
    for model in models:
        directory = args.evaluation_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["status"] != "completed"
            or not manifest["parameters_unchanged"]
            or manifest["identity"]["prepared_manifest_sha256"]
            != file_sha256(args.prepared_root / "manifest.json")
        ):
            raise ValueError("complete both frozen-forecaster budget evaluations")
        sources[model] = file_sha256(directory / "manifest.json")
        reported = pd.read_parquet(directory / "episode_results.parquet")
        for record in manifest["predictions"]:
            path = directory / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a budget forecast changed")
            original = originals[record["episode_id"]]
            scaler = scalers[(original["dataset_id"], original["item_id"])]
            with np.load(args.source_root / original["path"], allow_pickle=False) as saved:
                truth = (saved["future"][:, :2] - np.asarray(scaler["mean"])[:2]) / np.asarray(
                    scaler["scale"]
                )[:2]
            with np.load(path, allow_pickle=False) as saved:
                methods, points = saved["methods"].tolist(), saved["point_z"]
            for method, point in zip(methods, points, strict=True):
                error = point - truth
                values = np.array([np.abs(error).mean(), np.square(error).mean()])
                selected = reported[
                    (reported.episode_id == record["episode_id"])
                    & (reported.budget == record["budget"])
                    & (reported.method == method)
                ]
                if len(selected) != 1:
                    raise ValueError("a budget score is missing or duplicated")
                np.testing.assert_allclose(
                    values, selected[["mae", "mse"]].to_numpy()[0], rtol=0, atol=1e-10
                )
                max_difference = max(
                    max_difference,
                    float(np.abs(values - selected[["mae", "mse"]].to_numpy()[0]).max()),
                )
        rows.append(reported)
    frame = pd.concat(rows, ignore_index=True)
    verify_evaluation_coverage(frame, prepared)
    summary = (
        frame.groupby(["model_id", "dataset_id", "budget", "method"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    comparisons = []
    for (model, dataset, method), group in summary.groupby(["model_id", "dataset_id", "method"]):
        base = group[group.budget == "epochs10_windows64"].iloc[0]
        for row in group.itertuples(index=False):
            comparisons.append(
                {
                    "model_id": model,
                    "dataset_id": dataset,
                    "method": method,
                    "budget": row.budget,
                    "delta_mae_vs_refit10_64": row.mae - base.mae,
                    "delta_mse_vs_refit10_64": row.mse - base.mse,
                }
            )
    pd.DataFrame(comparisons).to_csv(output / "paired_budget_changes.csv", index=False)
    pd.DataFrame(fit_records).to_csv(output / "fit_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": sources,
            "script_sha256": file_sha256(Path(__file__)),
            "maximum_metric_difference": max_difference,
            "nested_training_windows_verified": True,
            "unchanged_model_structures_verified": True,
            "unchanged_classical_inputs_verified": True,
            "limits": prepared.get(
                "panel_description",
                "three development datasets with one fixed evaluation origin each; no independent-confirmation or algorithm-capacity claim",
            ),
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        summary[summary.method.isin(["saits", "timemixerpp", "forecast_median_guarded"])].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
