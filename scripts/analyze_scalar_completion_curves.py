"""Read the completed scalar-completion diagnostic without refitting forecasts."""

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
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed diagnostic readout")
    output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, manifests = [], [], {}
    for model in ("chronos2", "timesfm2p5"):
        directory = args.input_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed" or not manifest["parameters_unchanged"]:
            raise ValueError("both model diagnostics must be complete and fixed")
        for record in manifest["curves"]:
            if file_sha256(directory / record["path"]) != record["sha256"]:
                raise ValueError("a response curve changed")
        manifests[model] = file_sha256(directory / "manifest.json")
        diagnostics = pd.read_parquet(directory / "curve_diagnostics.parquet")
        scores = pd.read_parquet(directory / "source_forecast_errors.parquet")
        expected = set(manifest["identity"]["source_episodes"])
        for (teacher, horizon), group in diagnostics.groupby(["teacher", "horizon"]):
            if set(group.episode_id) != expected or len(group) != len(expected):
                raise ValueError("the response diagnostic has incomplete coverage")
            informative = group[group.curve_variance > 1e-16]
            ratios = informative.residual_variance_ratio.to_numpy()
            summaries.append(
                {
                    "model": model,
                    "teacher": teacher,
                    "horizon": int(horizon),
                    "count": len(group),
                    "nonconstant_curves": len(informative),
                    "mean_curve_variance": float(group.curve_variance.mean()),
                    "mean_verified_projection_mse": float(group.verified_projection_mse.mean()),
                    "median_residual_variance_ratio": float(np.median(ratios))
                    if len(ratios)
                    else None,
                    "ratio_over_one_percent_count": int((ratios > 0.01).sum()),
                    "curve_variance_over_1e_6_count": int((group.curve_variance > 1e-6).sum()),
                    "ratio_over_one_percent_with_variance_over_1e_6_count": int(
                        (
                            (group.curve_variance > 1e-6) & (group.residual_variance_ratio > 0.01)
                        ).sum()
                    ),
                    "median_coarse_minus_fine_grid_mse": float(
                        (group.coarse_grid_mse - group.grid_mse).median()
                    ),
                    "positive_condition_change_count": int(
                        group.positive_inference_condition_changes.sum()
                    ),
                    "degenerate_observed_scale_count": int(group.degenerate_observed_scale.sum()),
                }
            )
        for method, group in scores.groupby("method"):
            if set(group.episode_id) != expected or len(group) != len(expected):
                raise ValueError("the source forecast comparison has incomplete coverage")
            for baseline in (
                "locf",
                "native",
                "input_projection_mean",
                "input_projection_coordinate_median",
            ):
                reference = scores[scores.method == baseline]
                paired = group.merge(
                    reference,
                    on=["model", "family_id", "episode_id"],
                    suffixes=("", "_base"),
                    validate="one_to_one",
                )
                mae, mse = paired.mae - paired.mae_base, paired.mse - paired.mse_base
                comparisons.append(
                    {
                        "model": model,
                        "method": method,
                        "baseline": baseline,
                        "families": len(paired),
                        "mae": float(group.mae.mean()),
                        "mse": float(group.mse.mean()),
                        "delta_mae": float(mae.mean()),
                        "delta_mse": float(mse.mean()),
                        "both_metrics_win_count": int(((mae < 0) & (mse < 0)).sum()),
                    }
                )
    pd.DataFrame(summaries).to_csv(output / "geometry_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "source_accuracy_comparisons.csv", index=False)
    payload = {
        "status": "completed",
        "script_sha256": file_sha256(Path(__file__)),
        "input_manifest_sha256": manifests,
        "geometry": summaries,
        "accuracy_comparisons": comparisons,
        "interpretation_limits": [
            "one source-only controlled gap per family; no independent accuracy confirmation",
            "uniform input-grid scenario has no verified posterior interpretation",
            "positive numerical residual does not certify a global unattainability bound",
            "the one-percent ratio is a descriptive summary, not a significance test",
            "a source accuracy benefit alone cannot establish a new deployable method",
        ],
    }
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    _write_json(output / "manifest.json", payload)
    print(
        json.dumps(
            {
                "geometry": summaries,
                "accuracy_comparisons": [
                    row
                    for row in comparisons
                    if row["method"].startswith("output_") and row["baseline"] in ("locf", "native")
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
