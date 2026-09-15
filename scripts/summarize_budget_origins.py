"""Combine audited old and additional budget panels using historical-origin weights."""

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
    for name in (
        "initial-root",
        "additional-root",
        "initial-audit-root",
        "initial-tirex-audit-root",
        "additional-audit-root",
        "source-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed origin readouts")
    audits = {}
    for name in ("initial_audit_root", "initial_tirex_audit_root", "additional_audit_root"):
        path = getattr(args, name) / "manifest.json"
        audits[name] = json.loads(path.read_text(encoding="utf-8"))
        if audits[name]["status"] != "completed" or audits[name]["maximum_metric_difference"] != 0:
            raise ValueError("complete the score audits before combining the panels")
    original_sources = {
        **audits["initial_audit_root"]["source_manifests"],
        **audits["initial_tirex_audit_root"]["source_manifests"],
    }
    sources, frames, accuracy_hashes = {}, [], set()
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    metadata = pd.DataFrame(source["episodes"])[["episode_id", "dataset_id", "origin", "split"]]
    for panel, directory, expected in (
        ("initial", args.initial_root, original_sources),
        ("additional", args.additional_root, audits["additional_audit_root"]["source_manifests"]),
    ):
        for model in ("chronos2", "timesfm2p5", "tirex"):
            path = directory / model / "manifest.json"
            if file_sha256(path) != expected[model]:
                raise ValueError("an audited prediction manifest changed")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            accuracy_hashes.add(manifest["identity"]["accuracy_manifest_sha256"])
            sources[f"{panel}/{model}"] = file_sha256(path)
            frame = pd.read_parquet(directory / model / "episode_results.parquet")
            if len(frame) != (18 if panel == "initial" else 54) * 4 * 9:
                raise ValueError("the score table changed method or budget coverage")
            if (
                set(frame.model_id) != {model}
                or not np.isfinite(frame[["mae", "mse"]].to_numpy()).all()
            ):
                raise ValueError("the score table has incomplete or mislabelled model results")
            frame = frame.merge(
                metadata, on=["episode_id", "dataset_id"], how="left", validate="many_to_one"
            )
            if set(frame.split) != {"validation"}:
                raise ValueError("training histories entered the evaluation readout")
            frame["panel"] = panel
            frames.append(frame)
    if len(accuracy_hashes) != 1:
        raise ValueError("the panels use different source standardization records")
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["model_id", "episode_id", "budget", "method"]).any():
        raise ValueError("the initial and additional panels overlap")
    counts = frame.groupby(["panel", "model_id"]).episode_id.nunique()
    if not all(value == (18 if panel == "initial" else 54) for (panel, _), value in counts.items()):
        raise ValueError("a panel changed episode coverage")
    keys = ["panel", "model_id", "dataset_id", "origin", "budget", "method"]
    if not (frame.groupby(keys).episode_id.nunique() == 6).all():
        raise ValueError("each historical origin needs all six mask conditions")
    per_origin = frame.groupby(keys)[["mae", "mse"]].mean().reset_index()
    summary_keys = ["panel", "model_id", "dataset_id", "budget", "method"]
    summaries = per_origin.groupby(summary_keys)[["mae", "mse"]].mean().reset_index()
    combined = (
        per_origin.assign(panel="combined")
        .groupby(summary_keys)[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summaries = pd.concat([summaries, combined], ignore_index=True)
    pair_keys = ["panel", "model_id", "dataset_id", "origin", "method"]
    baseline = per_origin[per_origin.budget == "epochs10_windows64"][pair_keys + ["mae", "mse"]]
    changed = per_origin[per_origin.budget.isin(["epochs50_windows64", "epochs50_windows512"])]
    differences = changed.merge(
        baseline, on=pair_keys, suffixes=("", "_baseline"), validate="many_to_one"
    )
    for metric in ("mae", "mse"):
        differences["delta_" + metric] = differences[metric] - differences[metric + "_baseline"]
    differences["both_metrics_improved"] = (differences.delta_mae < -1e-12) & (
        differences.delta_mse < -1e-12
    )
    differences["both_metrics_worsened"] = (differences.delta_mae > 1e-12) & (
        differences.delta_mse > 1e-12
    )
    paired = (
        differences.groupby(["model_id", "dataset_id", "budget", "method"])
        .agg(
            origins=("origin", "nunique"),
            delta_mae=("delta_mae", "mean"),
            delta_mse=("delta_mse", "mean"),
            improved_origins=("both_metrics_improved", "sum"),
            worsened_origins=("both_metrics_worsened", "sum"),
        )
        .reset_index()
    )
    if not (paired.origins == 4).all():
        raise ValueError("the combined study must contain four origins per dataset")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("per_origin.csv", per_origin),
        ("summary.csv", summaries),
        ("origin_budget_differences.csv", differences),
        ("paired_budget_differences.csv", paired),
    ):
        table.to_csv(output / name, index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": sources,
            "source_metadata_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "independent_families": 3,
            "historical_origins": 12,
            "masked_episodes": 72,
            "aggregation": "six masks per history, then equal historical-origin weights within dataset",
            "limits": "development follow-up; three datasets with one evaluated item each; no new independent confirmation or budget selection",
        },
    )
    print(
        paired[paired.method.isin(["saits", "timemixerpp", "forecast_median_guarded"])].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
