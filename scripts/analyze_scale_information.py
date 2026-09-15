"""Separate missing-target scale effects from candidate-selection effects."""

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
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--readout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("scale analysis already completed; preserve the existing evidence")
    readout = json.loads((args.readout_root / "manifest.json").read_text(encoding="utf-8"))
    if readout.get("status") != "completed" or readout["source_manifests"][
        "accuracy_manifest"
    ] != file_sha256(args.accuracy_root / "manifest.json"):
        raise ValueError("readout does not match the completed accuracy export")
    keys = ["model_id", "episode_id"]
    features = ["static.empty_target_fraction", "static.empty_channel_fraction"]
    metadata = pd.read_parquet(
        args.accuracy_root / "candidate_accuracy.parquet",
        columns=keys + features,
        filters=[
            ("split", "==", "validation"),
            ("target_slot", "==", -1),
            ("candidate_id", "==", "locf"),
        ],
    )
    scores = pd.read_parquet(args.readout_root / "episode_results.parquet")
    scores = scores.merge(metadata, on=keys, how="left", validate="many_to_one")
    if not np.isfinite(scores[features].to_numpy()).all():
        raise ValueError("every screening episode must have explicit context observability")
    if ((scores[features] < 0) | (scores[features] > 1)).any().any():
        raise ValueError("empty-channel fractions must be between zero and one")
    records, family_rows, detail = [], [], []
    for model, model_scores in scores.groupby("model_id"):
        direct = "fixed_native_missing" if model == "chronos2" else "fixed_vendor_missing"
        for reference, alternative in (
            (direct, "prefix_input_z_direct"),
            (direct, "fixed_guarded_direct"),
            ("prefix_input_z_direct", "prefix_input_z_guarded_direct"),
            ("prefix_input_z_direct", "history_288_prefix_z_direct"),
            ("prefix_input_z_direct", "history_1024_prefix_z_direct"),
            ("fixed_knn_multivariate", "prefix_input_z_standardized_knn"),
        ):
            before = model_scores[model_scores.method == reference]
            after = model_scores[model_scores.method == alternative]
            if before.empty or after.empty:
                continue
            paired = before.merge(
                after[keys + ["mae", "mse"]],
                on=keys,
                validate="one_to_one",
                suffixes=("_reference", "_alternative"),
            )
            if len(paired) != len(before) or len(paired) != len(after):
                raise ValueError("scale contrasts must use exactly the same episodes")
            empty_target = paired[features[0]] > 0
            empty_channel = paired[features[1]] > 0
            masks = {
                "all": np.ones(len(paired), bool),
                "any_empty_target": empty_target,
                "only_auxiliary_empty": empty_channel & ~empty_target,
                "no_empty_channel": ~empty_channel,
            }
            for stratum, mask in masks.items():
                subset = paired[mask]
                if subset.empty:
                    continue
                identity = {
                    "model_id": model,
                    "reference": reference,
                    "alternative": alternative,
                    "stratum": stratum,
                }
                columns = ["mae_reference", "mse_reference", "mae_alternative", "mse_alternative"]
                family = (
                    subset.groupby(["family_id", "dataset_id"])[columns]
                    .mean()
                    .groupby(level="family_id")
                    .mean()
                )
                for metric in ("mae", "mse"):
                    before_mean = float(family[metric + "_reference"].mean())
                    after_mean = float(family[metric + "_alternative"].mean())
                    difference = family[metric + "_alternative"] - family[metric + "_reference"]
                    records.append(
                        identity
                        | {
                            "metric": metric,
                            "reference_error": before_mean,
                            "alternative_error": after_mean,
                            "relative_change": after_mean / before_mean - 1
                            if before_mean
                            else None,
                            "family_wins": int((difference < -1e-8).sum()),
                            "family_losses": int((difference > 1e-8).sum()),
                            "family_count": len(family),
                            "episode_count": len(subset),
                            "origin_count": subset.origin_id.nunique(),
                        }
                    )
                family_rows.extend(
                    identity | row for row in family.reset_index().to_dict("records")
                )
                detail.extend(
                    identity | row
                    for row in subset[
                        keys + ["family_id", "origin_id"] + features + columns
                    ].to_dict("records")
                )
    if not records:
        raise ValueError("no completed scale-information contrasts are available")
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(family_rows).to_csv(output / "family_results.csv", index=False)
    pd.DataFrame(detail).to_parquet(output / "episode_results.parquet", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
            "readout_manifest_sha256": file_sha256(args.readout_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "interpretation": "strata use current observation masks only; each contrast is paired within stratum; family populations differ between strata, so their errors must not be compared as causal effects",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
