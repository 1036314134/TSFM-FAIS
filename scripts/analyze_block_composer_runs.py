"""Compare completed composers and exactly matched input-standardized controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the existing composer readout")
    accuracy_path = args.accuracy_root / "manifest.json"
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    accuracy_sha = file_sha256(accuracy_path)
    reference = json.loads((args.reference_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        reference.get("status") != "completed"
        or reference["source_manifests"]["accuracy_manifest"] != accuracy_sha
    ):
        raise ValueError("reference readout does not match the completed accuracy export")
    source_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("source episode manifest changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_records = {
        row["episode_id"]: (index, row) for index, row in enumerate(source["episodes"])
    }
    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    frames, origins, provenance, normalizers = [], None, {}, {}
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or not manifest.get("parameter_digest_unchanged"):
            raise ValueError("every composer must have completed its frozen-parameter check")
        if manifest["identity"]["accuracy_manifest_sha256"] != accuracy_sha:
            raise ValueError("composer and baselines have different source forecasts")
        selected = set(manifest["identity"]["validation_ids"])
        if origins is not None and selected != origins:
            raise ValueError("composer runs use different development tasks")
        origins = selected
        data = pd.read_parquet(run / "selected_validation.parquet")
        data["method"] = run.name + ":" + data.method
        prior = json.loads((run / "training_prior.json").read_text(encoding="utf-8"))
        for method in data.method.unique():
            normalizers[method] = prior["normalizers"]
        data["information"] = "complete_source_training_labels"
        frames.append(data)
        provenance[str(run)] = file_sha256(run / "manifest.json")
    baseline = pd.read_parquet(args.reference_root / "episode_results.parquet")
    baseline = baseline[
        baseline.episode_id.isin(origins)
        & baseline.method.isin(
            [
                "forecast_median_guarded",
                "fixed_seasonal_lag",
                "prefix_input_z_direct",
                "prefix_input_z_guarded_direct",
                "prefix_input_z_locf",
                "history_288_prefix_z_direct",
                "history_1024_prefix_z_direct",
            ]
        )
    ].copy()
    baseline["information"] = np.where(
        baseline.method.str.startswith("history_"), "extended_history", "fixed_context_control"
    )
    frames.append(baseline)
    normalized = []
    actions = source["identity"]["config"]["candidate_ids"]
    for model in sorted(baseline.model_id.unique()):
        directory = args.controls_root / model
        control = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            control.get("status") != "completed"
            or control["identity"]["accuracy_manifest_sha256"] != accuracy_sha
        ):
            raise ValueError("input controls do not match the accuracy export")
        provenance[str(directory)] = file_sha256(directory / "manifest.json")
        for record in control["episodes"]:
            if record["episode_id"] not in origins:
                continue
            path = directory / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("input-standardized prediction cache changed")
            index, meta = source_records[record["episode_id"]]
            with np.load(path, allow_pickle=False) as saved:
                methods = saved["methods"].tolist()
                point = saved["point_z"]
                finite = [methods.index("prefix_input_z_" + action) for action in actions]
                guarded = finite + [methods.index("prefix_input_z_guarded_direct")]
                native = finite + [methods.index("prefix_input_z_direct")]
                combinations = {
                    "forecast_median_input_z_finite": np.median(point[finite], axis=0),
                    "forecast_median_input_z_guarded": np.median(point[guarded], axis=0),
                    "forecast_median_input_z_native": np.median(point[native], axis=0),
                    "forecast_mean_input_z_guarded": np.mean(point[guarded], axis=0),
                }
                for name, prediction in combinations.items():
                    error = prediction - truth[index]
                    if not np.isfinite(error).all():
                        raise ValueError("all matched-control errors must be finite")
                    normalized.append(
                        {
                            key: meta[key]
                            for key in ("episode_id", "origin_id", "family_id", "dataset_id")
                        }
                        | {
                            "model_id": model,
                            "method": name,
                            "information": "fixed_context_control",
                            "mae": float(np.mean(np.abs(error))),
                            "mse": float(np.mean(error**2)),
                        }
                    )
    frames.append(pd.DataFrame(normalized))
    data = pd.concat(frames, ignore_index=True)
    keys = ["model_id", "method", "information"]
    if (
        data.duplicated(keys + ["episode_id"]).any()
        or not np.isfinite(data[["mae", "mse"]].to_numpy()).all()
    ):
        raise ValueError("comparison contains duplicated or invalid observations")
    for _, group in data.groupby(keys):
        if set(group.episode_id) != origins:
            raise ValueError("all methods must cover exactly the same tasks")
    family = (
        data.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    comparisons = []
    generator = np.random.default_rng(6101)
    for (model, method, information), group in family.groupby(keys):
        if ":" not in method:
            continue
        ref = family[
            (family.model_id == model) & (family.method == "forecast_median_input_z_guarded")
        ].set_index("family_id")
        paired = group.set_index("family_id").join(
            ref[["mae", "mse"]], rsuffix="_reference", validate="one_to_one"
        )
        draws = generator.integers(0, len(paired), size=(2000, len(paired)))
        for metric in ("mae", "mse"):
            delta = (paired[metric] - paired[metric + "_reference"]).to_numpy()
            low, high = np.quantile(delta[draws].mean(1), [0.025, 0.975])
            comparisons.append(
                {
                    "model_id": model,
                    "method": method,
                    "information": information,
                    "metric": metric,
                    "mean_difference": float(delta.mean()),
                    "lower": float(low),
                    "upper": float(high),
                    "family_wins": int((delta < -1e-8).sum()),
                    "family_losses": int((delta > 1e-8).sum()),
                    "family_count": len(paired),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    oracles = []
    for (model, method), group in data[data.method.str.contains(":", regex=False)].groupby(
        ["model_id", "method"]
    ):
        native = data[(data.model_id == model) & (data.method == "prefix_input_z_direct")]
        paired = group.merge(
            native[["episode_id", "mae", "mse"]],
            on="episode_id",
            suffixes=("", "_native"),
            validate="one_to_one",
        )
        scale = normalizers[method]
        imputed_risk = paired.mae / scale["mae"] + paired.mse / scale["mse"]
        native_risk = paired.mae_native / scale["mae"] + paired.mse_native / scale["mse"]
        choose_native = native_risk <= imputed_risk
        chosen = paired.assign(
            mae=np.where(choose_native, paired.mae_native, paired.mae),
            mse=np.where(choose_native, paired.mse_native, paired.mse),
            native_selected=choose_native.astype(float),
        )
        means = (
            chosen.groupby(["family_id", "dataset_id"])[["mae", "mse", "native_selected"]]
            .mean()
            .groupby(level="family_id")
            .mean()
            .mean()
        )
        oracles.append(
            {
                "model_id": model,
                "composer": method,
                "oracle_mae": float(means.mae),
                "oracle_mse": float(means.mse),
                "native_fraction": float(means.native_selected),
                "interpretation": "future-label oracle over native and composed forecasts; diagnostic only, not a deployable policy or an attainable-gain claim",
            }
        )
    pd.DataFrame(oracles).to_csv(output / "native_choice_oracle.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "decision_count": len(origins),
            "script_sha256": file_sha256(Path(__file__)),
            "source_manifests": provenance,
            "accuracy_manifest_sha256": accuracy_sha,
            "reference_manifest_sha256": file_sha256(args.reference_root / "manifest.json"),
            "interpretation": "input-standardized controls share the exact tasks and scoring units; bootstrap intervals are descriptive because development outcomes selected checkpoints; extended-history controls use additional history",
        },
    )
    print(summary.sort_values(["model_id", "mse"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
