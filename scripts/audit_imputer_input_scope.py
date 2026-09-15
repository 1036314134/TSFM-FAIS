"""Recompute input-scope contrast scores from the stored forecast vectors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.forecasting.input_scope_controls import VARIANTS  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "source-root", "accuracy-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed input-scope audits")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    originals = {record["episode_id"]: record for record in source["episodes"]}
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    sources, rows, interactions, maximum_difference = {}, [], [], 0.0
    for model in ("chronos2", "timesfm2p5"):
        directory = args.input_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["status"] != "completed"
            or not manifest["parameters_unchanged"]
            or len(manifest["predictions"]) != 72
        ):
            raise ValueError("complete every registered intervention")
        sources[model] = file_sha256(directory / "manifest.json")
        reported = pd.read_parquet(directory / "episode_results.parquet")
        for record in manifest["predictions"]:
            path = directory / record["path"]
            original = originals[record["episode_id"]]
            source_path = args.source_root / original["path"]
            if (
                file_sha256(path) != record["sha256"]
                or file_sha256(source_path) != original["sha256"]
            ):
                raise ValueError("a contrast prediction or its source changed")
            scaler = scalers[(original["dataset_id"], original["item_id"])]
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["identity_sha256"]) != manifest["identity_sha256"]:
                    raise ValueError("an intervention identity changed")
                points = saved["point_z"]
            with np.load(source_path, allow_pickle=False) as saved:
                truth = (saved["future"][:, :2] - np.asarray(scaler["mean"])[:2]) / np.asarray(
                    scaler["scale"]
                )[:2]
            if model == "timesfm2p5":
                np.testing.assert_array_equal(points[0], points[2])
                np.testing.assert_array_equal(points[1], points[3])
            mae, mse = (
                np.abs(points - truth).mean(axis=(1, 2)),
                np.square(points - truth).mean(axis=(1, 2)),
            )
            for index, variant in enumerate(VARIANTS):
                row = reported[
                    (reported.episode_id == record["episode_id"])
                    & (reported.budget == record["budget"])
                    & (reported.imputer == record["imputer"])
                    & (reported.variant == variant)
                ]
                if len(row) != 1:
                    raise ValueError("a contrast score is missing or duplicated")
                expected = np.array([mae[index], mse[index]])
                np.testing.assert_allclose(
                    row[["mae", "mse"]].to_numpy()[0], expected, rtol=0, atol=1e-10
                )
                maximum_difference = max(
                    maximum_difference,
                    float(np.abs(row[["mae", "mse"]].to_numpy()[0] - expected).max()),
                )
                rows.append(
                    {
                        "model_id": model,
                        "dataset_id": original["dataset_id"],
                        "episode_id": record["episode_id"],
                        "budget": record["budget"],
                        "imputer": record["imputer"],
                        "variant": variant,
                        "mae": mae[index],
                        "mse": mse[index],
                    }
                )
            interactions.append(
                {
                    "model_id": model,
                    "dataset_id": original["dataset_id"],
                    "episode_id": record["episode_id"],
                    "budget": record["budget"],
                    "imputer": record["imputer"],
                    "forecast_interaction_rms_z": float(
                        np.sqrt(np.square(points[3] - points[1] - points[2] + points[0]).mean())
                    ),
                    "delta_mse_targets": mse[1] - mse[0],
                    "delta_mse_covariates": mse[2] - mse[0],
                    "delta_mse_both": mse[3] - mse[0],
                    "mse_interaction": mse[3] - mse[1] - mse[2] + mse[0],
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 576:
        raise ValueError("the full intervention panel changed")
    frame.groupby(["model_id", "dataset_id", "budget", "imputer", "variant"])[
        ["mae", "mse"]
    ].mean().reset_index().to_csv(output / "summary.csv", index=False)
    pd.DataFrame(interactions).groupby(["model_id", "dataset_id", "budget", "imputer"])[
        [
            "forecast_interaction_rms_z",
            "delta_mse_targets",
            "delta_mse_covariates",
            "delta_mse_both",
            "mse_interaction",
        ]
    ].mean().reset_index().to_csv(output / "interactions.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": sources,
            "script_sha256": file_sha256(Path(__file__)),
            "verified_forecast_vectors": len(frame),
            "maximum_metric_difference": maximum_difference,
            "independent_target_invariance": True,
            "limits": "exploratory crossed-input contrasts on three fixed development histories; forecast and loss interactions are distinct quantities",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
