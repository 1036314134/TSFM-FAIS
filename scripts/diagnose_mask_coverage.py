"""Compare observed mask coverage and prediction-boundary behavior without reading futures."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from latent_source_inputs import ROOT, read_json
from position_objective_inputs import arguments, load_inputs

from tsfm_fais.utility_experiment import _write_json, file_sha256


def summary_stats(point, predictions):
    lower, upper, median = point.min(0), point.max(0), np.median(point, axis=0)
    span = upper - lower
    mad = np.median(abs(point - median), axis=0)
    result = {
        "mean_span": float(span.mean()),
        "median_span": float(np.median(span)),
        "mean_mad": float(mad.mean()),
        "active_coordinate_fraction": float((span > 1e-6).mean()),
        "identical_forecasts": bool(np.all(span == 0)),
        "zero_mad_fraction": float((mad == 0).mean()),
    }
    if predictions is not None:
        active = span > 1e-6
        for name, prediction in predictions.items():
            boundary = (prediction == lower) | (prediction == upper)
            result[name + "_boundary_fraction"] = (
                float(boundary[active].mean()) if active.any() else np.nan
            )
            result[name + "_beyond_mad_fraction"] = (
                float((abs(prediction - median)[active] > mad[active]).mean())
                if active.any()
                else np.nan
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed coverage diagnostics")
    source_root = ROOT / "artifacts/iclr27-r3/development-expanded-v001"
    source = read_json(source_root / "episodes_manifest.json")
    accuracy_root = ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    accuracy = read_json(accuracy_root / "manifest.json")
    if accuracy["source_episode_manifest_sha256"] != file_sha256(
        source_root / "episodes_manifest.json"
    ):
        raise ValueError("source input ordering changed")
    target_roots = {
        "legacy_native": ROOT / "artifacts/iclr27-r5/native-confirmation-v001/prepared",
        "r6_native": ROOT / "artifacts/iclr27-r6/confirmation-v001/prepared",
    }
    transfer_root = ROOT / "artifacts/iclr27-r17/position-transfer-v001"
    transfer = read_json(transfer_root / "manifest.json")
    source_study = read_json(ROOT / "artifacts/iclr27-r17/position-objectives-v001/manifest.json")
    source_args = arguments("Source inputs").parse_args(["--output-root", str(output)])
    names = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    cases = {}
    for index, row in enumerate(source["episodes"]):
        cases[("source_" + row["split"], row["episode_id"])] = (source_root, row, index)
    for cohort, root in target_roots.items():
        manifest = read_json(root / "manifest.json")
        for index, row in enumerate(manifest["episodes"]):
            if not row["window"]["context_has_missing"] or (
                cohort == "r6_native" and row["mechanism"] != "native"
            ):
                continue
            cases[(cohort, row["episode_id"])] = (root, row, index)
    mask_stats = {}
    for key, (root, row, _) in cases.items():
        path = root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("an observed diagnostic input changed")
        with np.load(path, allow_pickle=False) as saved:
            missing = ~np.isfinite(saved["context"])
        mask_stats[key] = {
            "context_missing_fraction": float(missing.mean()),
            "target_missing_fraction": float(missing[:, :2].mean()),
            "both_targets_complete": bool(not missing[:, :2].any()),
            "maximum_target_missing_fraction": float(missing[:, :2].mean(0).max()),
        }
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        source_point_path = accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(source_point_path) != accuracy["prediction_arrays"][source_point_path.name]:
            raise ValueError("a source candidate bank changed")
        source_points = np.load(source_point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in names]
        ]
        frame, _, _, _, _ = load_inputs(source_args, model_id)
        fold = next(row for row in source_study["folds"] if row["model_id"] == model_id)
        path = ROOT / "artifacts/iclr27-r17/position-objectives-v001" / fold["prediction_path"]
        if file_sha256(path) != fold["prediction_sha256"]:
            raise ValueError("a positional source prediction changed")
        selected_methods = [
            "position17_local_joint_future",
            *[f"position17_local_joint_future_seed{seed}" for seed in (5101, 5102, 5103)],
        ]
        source_predictions = {}
        with np.load(path, allow_pickle=False) as saved:
            decisions = frame.iloc[saved["validation_indices"]]
            for method in selected_methods:
                values = saved["point"][saved["methods"].tolist().index(method)]
                for position, row in enumerate(decisions.itertuples()):
                    source_predictions.setdefault(row.source_episode_id, {}).setdefault(
                        method, np.empty((96, 2))
                    )[:, row.target_slot] = values[position]
        native_points, native_predictions = {}, {}
        for cohort in target_roots:
            if cohort == "r6_native":
                root = ROOT / f"artifacts/iclr27-r6/policy-results-v002/{model_id}/h96"
                marker = read_json(root / "predictions_frozen.json")
                path = root / "policy_predictions.npz"
                expected_sha = marker["prediction_sha256"]
            else:
                root = ROOT / "artifacts/iclr27-r6/legacy-native-gates-v001"
                freeze = read_json(root / "prediction_freeze.json")
                entry = next(row for row in freeze["banks"] if row["model_id"] == model_id)
                path, expected_sha = root / entry["path"], entry["sha256"]
            if file_sha256(path) != expected_sha:
                raise ValueError("an audited native candidate bank changed")
            with np.load(path, allow_pickle=False) as saved:
                native_points[cohort] = saved["point_z"][
                    :, [saved["methods"].tolist().index(name) for name in names]
                ]
            entry = next(
                row
                for row in transfer["prediction_banks"]
                if row["model_id"] == model_id
                and row["horizon"] == 96
                and row["cohort"] == ("r6" if cohort == "r6_native" else cohort)
            )
            path = transfer_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a target prediction bank changed")
            with np.load(path, allow_pickle=False) as saved:
                native_predictions[cohort] = {
                    name: saved["point_z"][:, saved["methods"].tolist().index(name)]
                    for name in selected_methods
                }
        for key, (_, row, index) in cases.items():
            cohort, episode = key
            point = (
                source_points[index]
                if cohort.startswith("source_")
                else native_points[cohort][index]
            )
            predictions = (
                source_predictions.get(episode)
                if cohort.startswith("source_")
                else {name: values[index] for name, values in native_predictions[cohort].items()}
            )
            rows.append(
                {
                    "model_id": model_id,
                    "cohort": cohort,
                    "episode_id": episode,
                    "origin_id": row["origin_id"],
                    "family_id": row["family_id"],
                    **mask_stats[key],
                    **summary_stats(point, predictions),
                }
            )
    result = pd.DataFrame(rows)
    metrics = [name for name in result if result[name].dtype.kind in "fbiu"]
    summary = result.groupby(["model_id", "cohort"])[metrics].mean().reset_index()
    quantiles = (
        result.groupby(["model_id", "cohort"])[
            ["target_missing_fraction", "context_missing_fraction", "mean_span", "mean_mad"]
        ]
        .quantile([0.1, 0.5, 0.9])
        .reset_index()
    )
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / "observed_diagnostics.parquet", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    quantiles.to_csv(output / "quantiles.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "source_sha256": file_sha256(source_root / "episodes_manifest.json"),
            "source_study_sha256": file_sha256(
                ROOT / "artifacts/iclr27-r17/position-objectives-v001/manifest.json"
            ),
            "transfer_sha256": file_sha256(transfer_root / "manifest.json"),
            "rows": len(result),
            "future_outcomes_used": False,
            "new_fits": 0,
            "new_forecaster_calls": 0,
            "limits": "descriptive observed-input and fitted-prediction diagnostic on previously used populations",
        },
    )


if __name__ == "__main__":
    main()
