"""Report prespecified source-paired R6 effects without pooling horizons or mask panels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def source_resampling_interval(primary, baseline, seed=9101):
    """Describe reweightings of the observed source families, not an unseen population."""
    if len(primary) < 3:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(primary), size=(20000, len(primary)))
    values = (primary - baseline)[indices].mean(axis=1)
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "audit-root", "prepared-root", "cohort-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed R6 readouts")
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        audit["status"] != "completed"
        or audit["verified_method_vectors"] != cohort["task_count"] * 2 * 2 * 23
    ):
        raise ValueError("complete the full registered policy audit before interpretation")
    families, summaries, comparisons, origin_records, condition_records = [], [], [], [], []
    for entry in audit["models"]:
        model_id = entry["model_id"]
        directory = args.policy_root / model_id
        if file_sha256(directory / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("policy results changed after the audit")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for horizon_entry in manifest["horizons"]:
            horizon = horizon_entry["horizon"]
            scores = pd.read_parquet(
                directory / horizon_entry["directory"] / "episode_results.parquet"
            )
            for panel_name, panel in result_panels(scores):
                keys, metrics = (
                    ["method", "family_id", "dataset_id", "item_id"],
                    ["mae", "mse", "raw_mae", "raw_mse"],
                )
                origin = panel.groupby([*keys, "origin_id"])[metrics].mean().reset_index()
                origin_records.append(
                    origin.assign(model_id=model_id, horizon=horizon, panel=panel_name)
                )
                items = origin.groupby(keys)[metrics].mean()
                datasets = items.groupby(keys[:-1])[metrics].mean()
                family = datasets.groupby(keys[:-2])[metrics].mean()
                # Equal mask multiplicity makes origin-first and registered window-first means agree.
                original = (
                    panel.groupby(keys)[metrics]
                    .mean()
                    .groupby(keys[:-1])
                    .mean()
                    .groupby(keys[:-2])
                    .mean()
                )
                np.testing.assert_allclose(family, original, rtol=1e-12, atol=1e-12)
                families.append(
                    family.reset_index().assign(
                        model_id=model_id, horizon=horizon, panel=panel_name
                    )
                )
                summary = family.groupby("method")[metrics].mean()
                summaries.append(
                    summary.reset_index().assign(
                        model_id=model_id,
                        horizon=horizon,
                        panel=panel_name,
                        families=panel.family_id.nunique(),
                        origins=panel.origin_id.nunique(),
                    )
                )
                primary = family.loc["ensemble_gate", ["mae", "mse"]]
                for method in sorted(set(panel.method) - {"ensemble_gate"}):
                    baseline = family.loc[method, ["mae", "mse"]].loc[primary.index]
                    delta = primary - baseline
                    p, b = primary.mean(), baseline.mean()
                    comparisons.append(
                        {
                            "model_id": model_id,
                            "horizon": horizon,
                            "panel": panel_name,
                            "comparator": method,
                            "families": len(primary),
                            "origins": panel.origin_id.nunique(),
                            "primary_mae": float(p.mae),
                            "primary_mse": float(p.mse),
                            "baseline_mae": float(b.mae),
                            "baseline_mse": float(b.mse),
                            "mae_delta": float(p.mae - b.mae),
                            "mse_delta": float(p.mse - b.mse),
                            "mae_relative_percent": float(100 * (p.mae / b.mae - 1))
                            if b.mae > 0
                            else None,
                            "mse_relative_percent": float(100 * (p.mse / b.mse - 1))
                            if b.mse > 0
                            else None,
                            "strict_joint_family_wins": int(
                                ((delta.mae < 0) & (delta.mse < 0)).sum()
                            ),
                            "strict_joint_family_losses": int(
                                ((delta.mae > 0) & (delta.mse > 0)).sum()
                            ),
                            "mae_source_resampling_range": source_resampling_interval(
                                primary.mae.to_numpy(), baseline.mae.to_numpy()
                            ),
                            "mse_source_resampling_range": source_resampling_interval(
                                primary.mse.to_numpy(), baseline.mse.to_numpy()
                            ),
                        }
                    )
            synthetic = scores[scores.panel == "new_synthetic"]
            keys = [
                "model_id",
                "horizon",
                "method",
                "family_id",
                "dataset_id",
                "item_id",
                "mechanism",
                "missing_rate",
                "mask_seed",
            ]
            condition_records.append(synthetic.groupby(keys)[["mae", "mse"]].mean().reset_index())
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    statuses = pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "reason": candidate["failure_reason"] or "",
            }
            for row in prep["episodes"]
            for candidate in row["candidate_statuses"]
        ]
    )
    tables = {
        "summary.csv": pd.concat(summaries, ignore_index=True),
        "family_metrics.csv": pd.concat(families, ignore_index=True),
        "origin_metrics.csv": pd.concat(origin_records, ignore_index=True),
        "primary_comparisons.csv": pd.DataFrame(comparisons),
        "synthetic_conditions.csv": pd.concat(condition_records, ignore_index=True),
        "candidate_status_counts.csv": statuses.groupby(
            ["dataset_id", "candidate_id", "status", "reason"], dropna=False
        )
        .size()
        .rename("windows")
        .reset_index(),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    fallacies = {
        "1_aggregation_reversal": "Source means and joint win/loss counts are retained; mixed source directions are not described as universal gains.",
        "2_ecological_inference": "Family averages do not establish improvements for every individual forecast window.",
        "3_filtered_population": "Eligibility conditions on future observability and complete synthetic histories; conclusions concern these conditional cohorts.",
        "4_conditioning_bias": "No method-specific case filtering is used. Observability-based selection still limits claims about unobserved future outcomes.",
        "5_base_rates": "Stratified missing/complete counts are not population missingness estimates; no diagnostic predictive-value claim is made.",
        "6_regression_to_mean": "New sources were chosen and frozen before their forecast accuracy was inspected; all prior failures remain documented.",
        "7_survivorship": "Every registered method/window is required by the audit; pre-fallback imputer statuses are retained.",
        "8_multiple_comparisons": "All 22 comparators, two horizons and registered panels are retained; no unadjusted significance claims are made.",
        "9_researcher_choices": "Method and cohort freezes preserve local preregistration; preprocessing, secondary horizon and label-supervised control are explicit.",
        "10_causal_scope": "Controlled algorithm comparisons describe this computational evaluation; cross-model differences do not isolate architecture or input-scope causes.",
        "11_temporal_direction": "Prefix fitting and future scoring boundaries were verified; environment-level causal relationships are not inferred.",
    }
    main_rows = [
        row
        for row in comparisons
        if row["panel"] == "new_synthetic_all"
        and row["horizon"] == 96
        and row["comparator"]
        in {
            "forecast_median_guarded",
            "forecast_median_with_motm",
            "gate_source_fixed_convex",
            "member_gate",
            "source_future_gate",
        }
    ]
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "audit_sha256": file_sha256(args.audit_root / "manifest.json"),
            "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
            "tables_sha256": {name: file_sha256(output / name) for name in tables},
            "primary_horizon": 96,
            "secondary_horizon": 192,
            "main_comparisons": main_rows,
            "fallacy_scan_coverage": "11/11 checked",
            "fallacy_scope_notes": fallacies,
            "uncertainty_scope": "Central 95% ranges from 20000 paired reweightings of the observed source families, seed 9101; descriptive only. Fewer than three families produce no range. These are not population confidence or significance claims.",
            "limits": "Four purposively selected sources; Beijing stations are one family. Horizons and masks share histories. Original-NA and nominal time-grid gaps remain separate. The data are used after this readout.",
        },
    )
    print(json.dumps(main_rows), flush=True)


if __name__ == "__main__":
    main()
