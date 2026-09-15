"""Verify shared draws and recompute the Gaussian reference forecast metrics."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "input-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed audits")
    output.mkdir(parents=True, exist_ok=True)
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = {row["episode_id"]: row for row in source["episodes"]}
    targets = source["identity"]["config"]["target_indices"]
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    identities, shared, metrics, comparisons = {}, {}, [], []
    max_metric_difference = 0.0
    for model in ("chronos2", "timesfm2p5"):
        directory = args.input_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if (
            manifest["status"] != "completed"
            or not manifest["parameters_unchanged"]
            or identity["source_manifest_sha256"] != file_sha256(source_path)
            or identity["script_sha256"] != file_sha256(directory / "script_snapshot.py")
            or identity["module_sha256"] != file_sha256(directory / "module_snapshot.py")
        ):
            raise ValueError("the completed experiment or saved source identity is invalid")
        identities[model] = file_sha256(directory / "manifest.json")
        family_values = defaultdict(list)
        for item in manifest["predictions"]:
            episode = item["episode_id"]
            record = records[episode]
            path = directory / item["path"]
            source_file = args.source_root / record["path"]
            if file_sha256(path) != item["sha256"] or file_sha256(source_file) != record["sha256"]:
                raise ValueError("a cached prediction or source task changed")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            with np.load(source_file, allow_pickle=False) as saved:
                truth = (saved["future"][:, targets] - mean[targets]) / scale[targets]
                context = (saved["context"] - mean) / scale
            with np.load(path, allow_pickle=False) as saved:
                current = (str(saved["input_bank_sha256"]), str(saved["seed"]))
                if model == "chronos2":
                    shared[episode] = current
                elif shared[episode] != current:
                    raise ValueError("the two models did not receive identical Gaussian samples")
                if str(saved["parameter_sha256"]) != manifest["parameter_sha256"]:
                    raise ValueError("a cached forecast has different model parameters")
                observed = np.isfinite(context)
                np.testing.assert_array_equal(
                    saved["conditional_mean_z"][observed], context[observed]
                )
                point = saved["point_z"]
            if point.shape != (33, 96, len(targets)) or not np.isfinite(point).all():
                raise ValueError("a Gaussian forecast has invalid values or dimensions")
            methods = {"gaussian_conditional_mean_input": point[0]}
            for count in (8, 16, 32):
                methods[f"gaussian_forecast_mean_k{count}"] = (
                    np.add.reduce(point[1 : count + 1], axis=0) / count
                )
                ordered = np.sort(point[1 : count + 1], axis=0)
                methods[f"gaussian_forecast_median_k{count}"] = (
                    ordered[count // 2 - 1] + ordered[count // 2]
                ) / 2
            for name, prediction in methods.items():
                error = prediction - truth
                # Calculate target-wise errors first, as specified in the protocol.
                mae = np.abs(error).sum(axis=0) / len(truth)
                mse = np.square(error).sum(axis=0) / len(truth)
                family_values[(name, record["family_id"])].append((mae.mean(), mse.mean()))
        if len(manifest["predictions"]) != 90:
            raise ValueError("the completed panel must contain 90 cases")
        per_method = defaultdict(list)
        for (name, _family), values in family_values.items():
            per_method[name].append(np.mean(values, axis=0))
        with (directory / "macro_metrics.csv").open(encoding="utf-8", newline="") as handle:
            reported = {row["method"]: row for row in csv.DictReader(handle)}
        for name, values in per_method.items():
            result = np.mean(values, axis=0)
            reference = np.array([float(reported[name][key]) for key in ("mae", "mse")])
            delta = float(np.abs(result - reference).max())
            max_metric_difference = max(max_metric_difference, delta)
            np.testing.assert_allclose(result, reference, rtol=0, atol=1e-10)
            metrics.append({"model": model, "method": name, "mae": result[0], "mse": result[1]})
        family = pd.read_csv(directory / "family_metrics.csv")
        for method in (
            "gaussian_forecast_mean_k8",
            "gaussian_forecast_mean_k32",
            "gaussian_forecast_median_k32",
        ):
            for baseline in (
                "gaussian_conditional_mean_input",
                "guarded_direct",
                "candidate_forecast_median",
            ):
                pair = family[family.method == method].merge(
                    family[family.method == baseline],
                    on=["model", "family_id"],
                    suffixes=("", "_baseline"),
                    validate="one_to_one",
                )
                mae, mse = pair.mae - pair.mae_baseline, pair.mse - pair.mse_baseline
                comparisons.append(
                    {
                        "model": model,
                        "method": method,
                        "baseline": baseline,
                        "families": len(pair),
                        "delta_mae": float(mae.mean()),
                        "delta_mse": float(mse.mean()),
                        "both_metrics_win_count": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                    }
                )
    pd.DataFrame(comparisons).to_csv(output / "family_comparisons.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "recomputed_metrics.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "input_manifest_sha256": identities,
            "script_sha256": file_sha256(Path(__file__)),
            "matching_sample_banks": len(shared),
            "observations_preserved": True,
            "forecasters_unchanged": True,
            "maximum_macro_metric_difference": max_metric_difference,
            "metrics": metrics,
            "family_comparisons": comparisons,
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "status": "completed",
                "matching_sample_banks": len(shared),
                "max_metric_difference": max_metric_difference,
                "family_comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
