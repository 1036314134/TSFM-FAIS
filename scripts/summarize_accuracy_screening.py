"""Compare completed screening methods on exactly the same decision episodes."""

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


def completed_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"incomplete screening stage: {directory}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("screening readout already exists; use a new output directory")
    plan = json.loads((args.probe_root / "plan.json").read_text(encoding="utf-8"))
    decisions = set(plan["decision_episode_ids"])
    if not decisions or len(decisions) != len(plan["decision_episode_ids"]):
        raise ValueError("screening decisions must be nonempty and unique")
    accuracy_manifest = json.loads(
        (args.accuracy_root / "manifest.json").read_text(encoding="utf-8")
    )
    if accuracy_manifest["source_episode_manifest_sha256"] != plan["source_manifest_sha256"]:
        raise ValueError("accuracy results and screening plan have different source episodes")
    accuracy_sha = file_sha256(args.accuracy_root / "manifest.json")
    plan_sha = file_sha256(args.probe_root / "plan.json")
    common = ["model_id", "episode_id", "origin_id", "family_id", "dataset_id", "mae", "mse"]
    parts = []
    sources = {}
    candidate_path = args.accuracy_root / "candidate_accuracy.parquet"
    candidate = pd.read_parquet(
        candidate_path,
        columns=common + ["candidate_id"],
        filters=[("split", "==", "validation"), ("target_slot", "==", -1)],
    )
    candidate = candidate[candidate.episode_id.isin(decisions)].copy()
    candidate["method"] = "fixed_" + candidate.candidate_id
    candidate["setting"] = "original_context"
    parts.append(candidate[common + ["method", "setting"]])
    baseline_path = args.accuracy_root / "control_accuracy.parquet"
    baseline = pd.read_parquet(
        baseline_path,
        columns=common + ["method"],
        filters=[("split", "==", "validation"), ("target_slot", "==", -1)],
    )
    baseline = baseline[baseline.episode_id.isin(decisions)].copy()
    baseline["setting"] = "original_context"
    baseline.loc[baseline.method == "clean", "setting"] = "unavailable_clean_context_reference"
    baseline.loc[baseline.method.str.endswith("_legacy"), "setting"] = (
        "superseded_layout_diagnostic"
    )
    parts.append(baseline[common + ["method", "setting"]])
    sources["accuracy_manifest"] = accuracy_sha
    for model in ("chronos2", "timesfm2p5"):
        directory = args.controls_root / model
        if (directory / "manifest.json").exists():
            manifest = completed_manifest(directory)
            if (
                manifest["identity"]["accuracy_manifest_sha256"] != accuracy_sha
                or manifest["identity"]["plan_sha256"] != plan_sha
            ):
                raise ValueError("history controls belong to another screening run")
            data = pd.read_parquet(directory / "episode_results.parquet")
            data["setting"] = "history_or_input_scaling_control"
            parts.append(data[common + ["method", "setting"]])
            sources[str(directory)] = file_sha256(directory / "manifest.json")
    analysis = args.probe_root / "analysis-v001"
    if (analysis / "manifest.json").exists():
        manifest = completed_manifest(analysis)
        if (
            manifest["accuracy_manifest_sha256"] != accuracy_sha
            or manifest["plan_sha256"] != plan_sha
        ):
            raise ValueError("recent feedback belongs to another screening run")
        data = pd.read_parquet(analysis / "episode_results.parquet")
        data["method"] = (
            data.method
            + "|objective="
            + data.objective
            + "|H="
            + data.probe_horizon.astype(str)
            + "|P="
            + data.probe_count.astype(str)
            + "|shrink="
            + data.shrinkage.astype(str)
        )
        data["setting"] = "observed_recent_feedback"
        parts.append(data[common + ["method", "setting"]])
        sources[str(analysis)] = file_sha256(analysis / "manifest.json")
    for model in ("chronos2", "timesfm2p5"):
        directory = args.probe_root / (model + "-input-mixtures")
        if (directory / "manifest.json").exists():
            manifest = completed_manifest(directory)
            if manifest["identity"]["analysis_manifest_sha256"] != sources.get(str(analysis)):
                raise ValueError("input mixtures do not match the completed feedback analysis")
            data = pd.read_parquet(directory / "episode_results.parquet")
            data["method"] = (
                data.method
                + "|objective="
                + data.objective
                + "|H="
                + data.probe_horizon.astype(str)
                + "|P="
                + data.probe_count.astype(str)
                + "|shrink="
                + data.shrinkage.astype(str)
            )
            data["setting"] = "actual_imputation_mixture"
            parts.append(data[common + ["method", "setting"]])
            sources[str(directory)] = file_sha256(directory / "manifest.json")
    frame = pd.concat(parts, ignore_index=True)
    if not np.isfinite(frame[["mae", "mse"]].to_numpy()).all():
        raise ValueError("screening scores must be finite; do not average away missing results")
    if (frame[["mae", "mse"]] < 0).any().any():
        raise ValueError("forecast error cannot be negative")
    keys = ["model_id", "setting", "method"]
    if frame.duplicated(keys + ["episode_id"]).any():
        raise ValueError("duplicate screening results")
    for _, group in frame.groupby(keys):
        if set(group.episode_id) != decisions:
            raise ValueError("a screening method omits or adds decision episodes")
    family = (
        frame.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    comparisons = []
    for (model, setting, method), group in family.groupby(keys):
        for reference in (
            "forecast_median_guarded",
            "fixed_seasonal_lag",
            "fixed_guarded_direct",
            "prefix_input_z_direct",
            "history_288_prefix_z_direct",
            "history_1024_prefix_z_direct",
        ):
            baseline = family[(family.model_id == model) & (family.method == reference)].set_index(
                "family_id"
            )
            if baseline.empty:
                continue
            for metric in ("mae", "mse"):
                paired = group.set_index("family_id")[[metric]].join(
                    baseline[[metric]], rsuffix="_baseline", validate="one_to_one"
                )
                relative = paired[metric] / paired[metric + "_baseline"] - 1
                comparisons.append(
                    {
                        "model_id": model,
                        "setting": setting,
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "mean_error": float(paired[metric].mean()),
                        "relative_change_of_macro": float(
                            paired[metric].mean() / paired[metric + "_baseline"].mean() - 1
                        ),
                        "median_family_relative_change": float(relative.median()),
                        "family_wins": int((relative < -1e-10).sum()),
                        "family_losses": int((relative > 1e-10).sum()),
                        "worst_family_relative_change": float(relative.max()),
                        "family_count": len(paired),
                    }
                )
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "decision_count": len(decisions),
            "source_manifests": sources,
            "plan_sha256": file_sha256(args.probe_root / "plan.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "interpretation": "all compared methods cover the same predeclared screening decisions; results are exploratory; historical-information and input-scaling settings are labeled separately",
        },
    )
    print(summary.sort_values(["model_id", "mse"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
