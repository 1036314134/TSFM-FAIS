"""Report paired family differences for the prespecified native confirmation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402

BASELINES = (
    "forecast_median_guarded",
    "forecast_mean_guarded",
    "source_fixed3_clean_forecast_mse",
    "source_fixed3_future_mse",
    "future_supervised_rank3",
    "input_median_finite",
    "guarded_direct",
    "motm_reference",
    "forecast_median_with_motm",
    "native_guarded_prefix_z96",
    "native_guarded_prefix_z1024",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "audit-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed confirmation summaries")
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed" or audit["verified_model_episodes"] != 746:
        raise ValueError("complete the independent prediction and score audit")
    comparisons, differences, summaries = [], [], []
    for model in ("chronos2", "timesfm2p5"):
        directory = args.input_root / model
        if file_sha256(directory / "manifest.json") != audit["sources"][model]:
            raise ValueError("an audited confirmation result changed")
        summary = pd.read_csv(directory / "summary.csv", float_precision="round_trip")
        summaries.append(summary)
        for panel, count in (("all_registered", 9), ("naturally_missing", 7)):
            family = pd.read_csv(directory / f"{panel}_family_metrics.csv", float_precision="round_trip")
            primary = family[family.method == "teacher_rank3"]
            if len(primary) != count or primary.family_id.nunique() != count:
                raise ValueError("the primary confirmation population changed")
            rng = np.random.default_rng(4101)
            resamples = rng.integers(0, count, size=(20000, count))
            for baseline in BASELINES:
                reference = family[family.method == baseline]
                paired = primary.merge(
                    reference, on="family_id", suffixes=("", "_reference"), validate="one_to_one"
                ).sort_values("family_id")
                if len(paired) != count:
                    raise ValueError("a comparison has different family coverage")
                if not np.isfinite(
                    paired[["mae", "mse", "mae_reference", "mse_reference"]].to_numpy()
                ).all():
                    comparisons.append(
                        {
                            "model_id": model,
                            "panel": panel,
                            "baseline": baseline,
                            "status": "incomplete_method",
                        }
                    )
                    continue
                mae = paired.mae.to_numpy() - paired.mae_reference.to_numpy()
                mse = paired.mse.to_numpy() - paired.mse_reference.to_numpy()
                record = {
                    "model_id": model,
                    "panel": panel,
                    "baseline": baseline,
                    "status": "completed",
                    "families": count,
                    "both_metrics_win_families": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                    "both_metrics_lose_families": int(((mae > 1e-12) & (mse > 1e-12)).sum()),
                }
                for metric, delta in (("mae", mae), ("mse", mse)):
                    denominator = paired[metric + "_reference"].mean()
                    low, high = np.quantile(delta[resamples].mean(axis=1), [0.025, 0.975])
                    record.update(
                        {
                            "delta_" + metric: float(delta.mean()),
                            "relative_change_" + metric + "_percent": float(
                                100 * delta.mean() / denominator
                            )
                            if denominator > 0
                            else None,
                            "bootstrap_" + metric + "_low": float(low),
                            "bootstrap_" + metric + "_high": float(high),
                        }
                    )
                comparisons.append(record)
                for index, row in enumerate(paired.itertuples(index=False)):
                    differences.append(
                        {
                            "model_id": model,
                            "panel": panel,
                            "baseline": baseline,
                            "family_id": row.family_id,
                            "delta_mae": float(mae[index]),
                            "delta_mse": float(mse[index]),
                        }
                    )
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "family_differences.csv", index=False)
    primary = [row for row in comparisons if row["baseline"] == "forecast_median_guarded"]
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "audit_manifest_sha256": file_sha256(args.audit_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "primary_method": "teacher_rank3",
            "primary_comparison": primary,
            "bootstrap": {
                "unit": "family",
                "resamples": 20000,
                "seed": 4101,
                "interval": "2.5th and 97.5th percentiles of paired macro differences",
            },
            "limits": [
                "nine prespecified families; seven in the natural-missing subset",
                "intervals describe sensitivity to family resampling and do not establish universal generalization",
                "multiple exploratory baseline comparisons are not multiplicity-corrected",
                "no configuration is selected using these confirmation scores",
            ],
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == "__main__":
    main()
