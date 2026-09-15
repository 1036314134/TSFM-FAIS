"""Compare audited shared-gate source results with every declared strong control."""

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
    for name in ("study-root", "audit-root", "matched-root", "original-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed shared-gate readouts")
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        audit["status"] != "completed"
        or audit["verified_seed_checkpoints"] != 180
        or not audit["matched_inputs_and_capacity"]
    ):
        raise ValueError("complete all matched source-gate checks first")
    tables = []
    reference = None
    for kind in ("ensemble", "member"):
        directory = args.study_root / kind
        if file_sha256(directory / "manifest.json") != audit["study_manifest_sha256"][kind]:
            raise ValueError("gate results changed after their audit")
        frame = pd.read_csv(directory / "family_metrics.csv")
        base = (
            frame[frame.method == "forecast_median_guarded"]
            .set_index(["model_id", "family_id"])[["mae", "mse"]]
            .sort_index()
        )
        if reference is None:
            reference = base
        else:
            pd.testing.assert_frame_equal(base, reference)
        tables.append(
            frame if kind == "ensemble" else frame[frame.method.str.startswith("member_")]
        )
    for directory, names in (
        (args.matched_root, {"median_risk": "pairwise_portfolio"}),
        (
            args.original_root,
            {
                "clean_forecast_mse_rank3": "original_teacher_top3",
                "source_fixed3_clean_forecast_mse": "original_source_fixed3",
            },
        ),
    ):
        frame = pd.read_csv(directory / "family_metrics.csv")
        frame = frame[frame.panel == "full_development"]
        base = (
            frame[frame.method == "forecast_median_guarded"]
            .set_index(["model_id", "family_id"])[["mae", "mse"]]
            .sort_index()
        )
        np.testing.assert_allclose(base.loc[reference.index], reference, rtol=1e-12, atol=1e-12)
        frame = frame[frame.method.isin(names)].copy()
        frame["method"] = frame.method.map(names)
        tables.append(frame[["model_id", "method", "family_id", "mae", "mse"]])
    family = pd.concat(tables, ignore_index=True)
    if family.duplicated(["model_id", "method", "family_id"]).any():
        raise ValueError("a comparison family row was duplicated")
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    comparisons = []
    for model in ("chronos2", "timesfm2p5"):
        local = family[family.model_id == model].set_index(["method", "family_id"])[["mae", "mse"]]
        primary = local.loc["ensemble_gate"]
        for method in (
            "member_gate",
            "forecast_mean_guarded",
            "forecast_median_guarded",
            "source_fixed_convex",
            "source_fixed_single",
            "pairwise_portfolio",
            "original_teacher_top3",
            "original_source_fixed3",
        ):
            comparator = local.loc[method].loc[primary.index]
            delta = primary - comparator
            p, b = primary.mean(), comparator.mean()
            comparisons.append(
                {
                    "model_id": model,
                    "comparator": method,
                    "families": len(primary),
                    "primary_mae": p.mae,
                    "primary_mse": p.mse,
                    "comparator_mae": b.mae,
                    "comparator_mse": b.mse,
                    "mae_relative_percent": 100 * (p.mae / b.mae - 1),
                    "mse_relative_percent": 100 * (p.mse / b.mse - 1),
                    "strict_joint_family_wins": int(((delta.mae < 0) & (delta.mse < 0)).sum()),
                    "strict_joint_family_losses": int(((delta.mae > 0) & (delta.mse > 0)).sum()),
                }
            )
    seed_rows = []
    for model in ("chronos2", "timesfm2p5"):
        for kind in ("ensemble", "member"):
            values = summary[
                (summary.model_id == model) & summary.method.str.startswith(kind + "_seed")
            ]
            if len(values) != 3:
                raise ValueError("all three prespecified seed results must be retained")
            seed_rows.append(
                {
                    "model_id": model,
                    "objective": kind,
                    "seed_count": 3,
                    "seed_mae_mean": values.mae.mean(),
                    "seed_mae_std": values.mae.std(ddof=1),
                    "seed_mse_mean": values.mse.mean(),
                    "seed_mse_std": values.mse.std(ddof=1),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "primary_comparisons.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_variation.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "audit_sha256": file_sha256(args.audit_root / "manifest.json"),
            "comparison_sources_sha256": {
                str(path): file_sha256(path)
                for path in (
                    args.matched_root / "family_metrics.csv",
                    args.original_root / "family_metrics.csv",
                )
            },
            "main_comparisons": comparisons,
            "parameters_per_seed": 1096,
            "parameters_in_prespecified_three_seed_ensemble": 3288,
            "limits": "exploratory source-development comparisons after prior method search; seed variation is not uncertainty over new data families; no independent follow-up evaluation of this gate",
        },
    )
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
