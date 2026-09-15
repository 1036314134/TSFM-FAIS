"""Summarize all registered follow-up panels after the independent policy audit."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402

COMPARATORS = (
    "forecast_median_guarded",
    "member_risk",
    "old_teacher_rank3",
    "source_fixed_median_risk",
    "old_source_fixed3",
    "forecast_mean_guarded",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "audit-root", "prepared-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed follow-up readouts")
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed" or audit["maximum_prediction_difference"] != 0:
        raise ValueError("complete the exact policy prediction audit before interpretation")
    summaries, family_deltas, overview, breakdowns = [], [], [], []
    for entry in audit["models"]:
        model, directory = entry["model_id"], args.policy_root / entry["model_id"]
        if file_sha256(directory / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("a policy evaluation changed after the audit")
        summaries.append(pd.read_csv(directory / "summary.csv"))
        scores = pd.read_parquet(directory / "episode_results.parquet")
        for panel_name, panel in result_panels(scores):
            keys = ["method", "family_id", "dataset_id", "item_id"]
            families = (
                panel.groupby(keys)[["mae", "mse"]]
                .mean()
                .groupby(keys[:-1])
                .mean()
                .groupby(keys[:-2])
                .mean()
            )
            primary = families.loc["median_risk"]
            for comparator in COMPARATORS:
                baseline = families.loc[comparator].loc[primary.index]
                delta = primary - baseline
                for family in delta.index:
                    family_deltas.append(
                        {
                            "model_id": model,
                            "panel": panel_name,
                            "family_id": family,
                            "comparator": comparator,
                            "mae_delta": float(delta.loc[family, "mae"]),
                            "mse_delta": float(delta.loc[family, "mse"]),
                            "mae_relative_percent": float(
                                100 * delta.loc[family, "mae"] / baseline.loc[family, "mae"]
                            )
                            if baseline.loc[family, "mae"] > 0
                            else None,
                            "mse_relative_percent": float(
                                100 * delta.loc[family, "mse"] / baseline.loc[family, "mse"]
                            )
                            if baseline.loc[family, "mse"] > 0
                            else None,
                        }
                    )
                mean_primary, mean_baseline = primary.mean(), baseline.mean()
                overview.append(
                    {
                        "model_id": model,
                        "panel": panel_name,
                        "comparator": comparator,
                        "families": len(primary),
                        "origins": panel.origin_id.nunique(),
                        "items": len(panel[["dataset_id", "item_id"]].drop_duplicates()),
                        "primary_mae": float(mean_primary.mae),
                        "primary_mse": float(mean_primary.mse),
                        "comparator_mae": float(mean_baseline.mae),
                        "comparator_mse": float(mean_baseline.mse),
                        "mae_relative_percent": float(
                            100 * (mean_primary.mae / mean_baseline.mae - 1)
                        )
                        if mean_baseline.mae > 0
                        else None,
                        "mse_relative_percent": float(
                            100 * (mean_primary.mse / mean_baseline.mse - 1)
                        )
                        if mean_baseline.mse > 0
                        else None,
                        "strict_both_metric_family_wins": int(
                            ((delta.mae < 0) & (delta.mse < 0)).sum()
                        ),
                        "strict_both_metric_family_losses": int(
                            ((delta.mae > 0) & (delta.mse > 0)).sum()
                        ),
                        "both_metric_exact_ties": int(((delta.mae == 0) & (delta.mse == 0)).sum()),
                        "uncertainty_scope": "descriptive observed-family effects; too few new families for a broad generalization claim",
                    }
                )
        synthetic = scores[scores.panel == "new_synthetic"]
        keys = [
            "model_id",
            "method",
            "family_id",
            "dataset_id",
            "item_id",
            "mechanism",
            "missing_rate",
            "mask_seed",
        ]
        breakdowns.append(synthetic.groupby(keys)[["mae", "mse"]].mean().reset_index())
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    statuses = []
    for row in prep["episodes"]:
        for candidate in row["candidate_statuses"]:
            statuses.append(
                {
                    "dataset_id": row["dataset_id"],
                    "panel": row["panel"],
                    "candidate_id": candidate["candidate_id"],
                    "status": candidate["status"],
                    "failure_reason": candidate["failure_reason"] or "",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary.csv": pd.concat(summaries, ignore_index=True),
        "primary_comparisons.csv": pd.DataFrame(overview),
        "family_differences.csv": pd.DataFrame(family_deltas),
        "synthetic_conditions.csv": pd.concat(breakdowns, ignore_index=True),
        "candidate_status_counts.csv": pd.DataFrame(statuses)
        .groupby(["dataset_id", "panel", "candidate_id", "status", "failure_reason"], dropna=False)
        .size()
        .rename("windows")
        .reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    main_rows = [
        row
        for row in overview
        if row["comparator"] == "forecast_median_guarded"
        and row["panel"] in {"held_items_missing", "new_native_missing", "new_synthetic_all"}
    ]
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "audit_sha256": file_sha256(args.audit_root / "manifest.json"),
            "tables_sha256": {name: file_sha256(output / name) for name in tables},
            "main_comparisons": main_rows,
            "future_outcomes_now_used": True,
            "limits": [
                "held-out items in known families, new-source native data and new-source synthetic cases are distinct panels",
                "new native and synthetic panels each contain only two source families; no population-level significance claim is made",
                "multiple masks of one history do not create independent histories",
                "both MAE and MSE, individual-family effects and all registered controls must be considered",
                "missing future measurements are unscored; this estimates error on the originally observed future entries",
            ],
        },
    )
    print(json.dumps(main_rows), flush=True)


if __name__ == "__main__":
    main()
