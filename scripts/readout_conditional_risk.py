"""Separate conditional selection gains from paired hindsight optimism in simulation."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from conditional_future import (
    ForecastHull,
    expected_risks,
    future_moments,
    sample_futures,
    scenarios,
)
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def standard_error(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def evaluate_decision(points, mean, variance, future, conditional, hull):
    expected_mae, expected_mse = expected_risks(points, mean, variance)
    error = points[None] - future[:, None]
    mae = np.abs(error).mean(2)
    mse = (error**2).mean(2)
    winner_mae = mae.argmin(1)
    winner_mse = mse.argmin(1)
    row = np.arange(len(future))
    best_mae = int(expected_mae.argmin())
    best_mse = int(expected_mse.argmin())
    gain_mae = mae[:, best_mae] - mae[row, winner_mae]
    gain_mse = mse[:, best_mse] - mse[row, winner_mse]
    weights, hindsight, gap, fallback = hull.solve(future)
    _, conditional_prediction, _, _ = conditional
    conditional_prediction = conditional_prediction[0]
    conditional_risk = float(expected_risks(conditional_prediction, mean, variance)[1])
    hindsight_loss = ((hindsight - future) ** 2).mean(1)
    same_future_loss = ((conditional_prediction - future) ** 2).mean(1)
    convex_gain = same_future_loss - hindsight_loss
    if convex_gain.min() < -1e-8:
        raise ValueError("hindsight convex optimization lost to a feasible conditional decision")
    innovation = future - mean
    direction = points - points[best_mse]
    projected = 2 * (innovation @ direction.T) / points.shape[1]
    margins = expected_mse - expected_mse[best_mse]
    reconstructed = np.max(projected - margins, axis=1)
    np.testing.assert_allclose(reconstructed, gain_mse, rtol=1e-10, atol=1e-10)
    zscores = []
    for samples, expected in ((mae, expected_mae), (mse, expected_mse)):
        error = np.abs(samples.mean(0) - expected)
        se = np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
        zscores.append(float(np.max(error / np.maximum(se, 1e-10))))
    if max(zscores) > 10:
        raise ValueError("conditional Monte Carlo risks fail the prespecified moment check")
    uniform_mae, uniform_mse = expected_risks(points.mean(0), mean, variance)
    median_mae, median_mse = expected_risks(np.median(points, axis=0), mean, variance)
    metrics = {
        "conditional_single_mae": float(expected_mae[best_mae]),
        "conditional_single_mse": float(expected_mse[best_mse]),
        "hindsight_single_mae": float(mae[row, winner_mae].mean()),
        "hindsight_single_mse": float(mse[row, winner_mse].mean()),
        "optimism_single_mae": float(gain_mae.mean()),
        "optimism_single_mse": float(gain_mse.mean()),
        "optimism_single_mae_se": standard_error(gain_mae),
        "optimism_single_mse_se": standard_error(gain_mse),
        "hindsight_single_independent_mae": float(expected_mae[winner_mae].mean()),
        "hindsight_single_independent_mse": float(expected_mse[winner_mse].mean()),
        "conditional_convex_mse": conditional_risk,
        "hindsight_convex_mse": float(hindsight_loss.mean()),
        "optimism_convex_mse": float(convex_gain.mean()),
        "optimism_convex_mse_se": standard_error(convex_gain),
        "hindsight_convex_independent_mse": float(
            expected_risks(hindsight, mean, variance)[1].mean()
        ),
        "uniform_mae": float(uniform_mae),
        "uniform_mse": float(uniform_mse),
        "median_mae": float(median_mae),
        "median_mse": float(median_mse),
        "convex_attainable_vs_uniform_mse": float(uniform_mse) - conditional_risk,
        "noise_projection_width": float(np.max(projected, axis=1).mean()),
        "projection_identity_maximum_difference": float(abs(reconstructed - gain_mse).max()),
        "candidate_mc_maximum_zscore_mae": zscores[0],
        "candidate_mc_maximum_zscore_mse": zscores[1],
        "maximum_hindsight_optimality_gap": float(gap.max()),
        "fallback_solves": fallback,
    }
    witnesses = {
        "candidate_expected_mae": expected_mae,
        "candidate_expected_mse": expected_mse,
        "winner_mae": winner_mae,
        "winner_mse": winner_mse,
        "conditional_weights": conditional[0][0],
        "hindsight_weights": weights,
        "paired_gain_mae": gain_mae,
        "paired_gain_mse": gain_mse,
        "paired_gain_convex_mse": convex_gain,
    }
    return metrics, witnesses


def summaries(frame, risks):
    keys = ["model_id", "process", "mechanism", "missing_rate", "noise", "pool_size", "target_slot"]
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        risk_mae = np.stack([risks[index][0] for index in group.index])
        risk_mse = np.stack([risks[index][1] for index in group.index])
        metrics = group.select_dtypes(include=[np.number]).mean().to_dict()
        fixed_mae = float(risk_mae.mean(0).min())
        fixed_mse = float(risk_mse.mean(0).min())
        attainable_mae = fixed_mae - metrics["conditional_single_mae"]
        attainable_mse = fixed_mse - metrics["conditional_single_mse"]
        if min(attainable_mae, attainable_mse) < -1e-10:
            raise ValueError("conditional choices lost to a feasible panel-fixed candidate")
        for name, attainable, fixed in (
            ("mae", attainable_mae, fixed_mae),
            ("mse", attainable_mse, fixed_mse),
        ):
            apparent = attainable + metrics[f"optimism_single_{name}"]
            metrics.update(
                {
                    f"panel_fixed_{name}": fixed,
                    f"attainable_single_{name}": attainable,
                    f"apparent_single_gain_{name}": apparent,
                    f"optimism_fraction_{name}": metrics[f"optimism_single_{name}"] / apparent
                    if apparent > 1e-12
                    else np.nan,
                }
            )
        rows.append(
            {**dict(zip(keys, key, strict=True)), **metrics, "histories": group.origin_id.nunique()}
        )
    panels = pd.DataFrame(rows)
    target_keys = ["model_id", "process", "mechanism", "missing_rate", "noise", "pool_size"]
    per_process = panels.groupby(target_keys).mean(numeric_only=True).reset_index()
    summary = (
        per_process.groupby(["model_id", "mechanism", "missing_rate", "noise", "pool_size"])
        .mean(numeric_only=True)
        .reset_index()
    )
    return panels, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r14/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r14/conditional-forecasts-v001",
        "protocol": "docs/iclr2027/R14_CONDITIONAL_RISK_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditional-risk results")
    prep = read_json(args.prepared_root / "manifest.json")
    forecasts = read_json(args.forecast_root / "manifest.json")
    if forecasts["status"] != "completed" or forecasts["identity"][
        "prepared_sha256"
    ] != file_sha256(args.prepared_root / "manifest.json"):
        raise ValueError("freeze all candidate forecasts before future sampling")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "process_module_sha256": file_sha256(ROOT / "scripts/conditional_future.py"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(args.forecast_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "future_samples": 512,
        "noise_scales": [0.0, 0.5, 1.0, 2.0],
        "candidate_pool_sizes": [7, 8],
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial conditional-risk definitions changed")
    _write_json(output / "identity.json", identity)
    banks = {}
    for entry in forecasts["models"]:
        path = args.forecast_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen forecast manifest changed")
        banks[entry["model_id"]] = {row["episode_id"]: row for row in read_json(path)["episodes"]}
    models = scenarios()
    records, risks, witness_files = [], {}, []
    for number, row in enumerate(prep["episodes"]):
        model = models[row["generator"]]
        path = args.prepared_root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("an observed-history input changed")
        with np.load(path, allow_pickle=False) as saved:
            posterior_mean, posterior_cov = saved["posterior_mean"], saved["posterior_covariance"]
        scaler = prep["prefixes"][row["generator"]]
        center, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
        predictors = {}
        hulls = {}
        conditional = {}
        first_mean = future_moments(
            model, posterior_mean, posterior_cov, row["phase"] + 96, 96, 0.0
        )[0][:, :2]
        first_mean = (first_mean - center) / scale
        for model_id in ("chronos2", "timesfm2p5"):
            entry = banks[model_id][row["episode_id"]]
            path = args.forecast_root / model_id / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a candidate forecast changed after freeze")
            with np.load(path, allow_pickle=False) as saved:
                point = saved["point_z"]
                names = saved["actions"].tolist()
            for slot in [-1] if model_id == "chronos2" else [0, 1]:
                for size in (7, 8):
                    selected = (
                        np.arange(8)
                        if size == 8
                        else np.array(
                            [index for index, name in enumerate(names) if name != "motm_reference"]
                        )
                    )
                    points = (
                        point[selected].reshape(size, -1)
                        if slot == -1
                        else point[selected, :, slot]
                    )
                    key = (model_id, slot, size)
                    predictors[key] = points
                    hulls[key] = ForecastHull(points)
                    conditional[key] = hulls[key].solve(
                        first_mean.reshape(-1) if slot == -1 else first_mean[:, slot]
                    )
        for noise in identity["noise_scales"]:
            mean, missing, innovation = future_moments(
                model, posterior_mean, posterior_cov, row["phase"] + 96, 96, noise
            )
            mean = (mean[:, :2] - center) / scale
            missing = missing[:, :2] / scale**2
            innovation = innovation[:, :2] / scale**2
            variance = missing + innovation
            np.testing.assert_array_equal(mean, first_mean)
            samples = sample_futures(
                model,
                posterior_mean,
                posterior_cov,
                row["phase"] + 96,
                96,
                noise,
                512,
                row["future_seed"],
            )
            samples = (samples[:, :, :2] - center) / scale
            if noise == 0 and row["mechanism"] == "complete":
                np.testing.assert_array_equal(samples, np.repeat(samples[:1], 512, axis=0))
            for key, points in predictors.items():
                model_id, slot, size = key
                mu = mean.reshape(-1) if slot == -1 else mean[:, slot]
                var = variance.reshape(-1) if slot == -1 else variance[:, slot]
                future = samples.reshape(512, -1) if slot == -1 else samples[:, :, slot]
                metrics, witness = evaluate_decision(
                    points, mu, var, future, conditional[key], hulls[key]
                )
                future_variance = innovation.reshape(-1) if slot == -1 else innovation[:, slot]
                metrics.update(
                    noise_floor_mse=float(var.mean()),
                    noise_floor_mae=float((np.sqrt(var) * np.sqrt(2 / np.pi)).mean()),
                    future_only_floor_mse=float(future_variance.mean()),
                    future_only_floor_mae=float(
                        (np.sqrt(future_variance) * np.sqrt(2 / np.pi)).mean()
                    ),
                )
                if (
                    row["mechanism"] == "complete"
                    and max(
                        abs(metrics[name])
                        for name in (
                            "optimism_single_mae",
                            "optimism_single_mse",
                            "optimism_convex_mse",
                        )
                    )
                    > 1e-10
                ):
                    raise ValueError(
                        "complete-context duplicate predictions created a false oracle gain"
                    )
                index = len(records)
                risks[index] = (
                    witness["candidate_expected_mae"],
                    witness["candidate_expected_mse"],
                )
                label = f"{row['episode_id']}|{model_id}|{slot}|{size}|{noise}"
                path = (
                    output
                    / "witnesses"
                    / (hashlib.sha256(label.encode()).hexdigest()[:24] + ".npz")
                )
                _save_npz(path, **witness)
                witness_files.append(
                    {
                        "row": index,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                records.append(
                    {
                        "model_id": model_id,
                        "process": model["name"],
                        "origin_id": row["origin_id"],
                        "episode_id": row["episode_id"],
                        "mechanism": row["mechanism"],
                        "missing_rate": row["missing_rate"],
                        "noise": noise,
                        "pool_size": size,
                        "target_slot": slot,
                        **metrics,
                    }
                )
        if (number + 1) % 12 == 0:
            print(f"conditional-risk readout: {number + 1}/180 observed inputs", flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != 4320:
        raise ValueError("conditional-risk comparison coverage changed")
    panels, summary = summaries(frame, risks)
    frame.to_parquet(output / "decision_results.parquet", index=False)
    panels.to_csv(output / "process_panels.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "witnesses": witness_files,
            "decision_rows": len(frame),
            "independent_histories": 36,
            "observed_inputs": 180,
            "new_forecaster_calls": 0,
            "limits": "known-process controlled diagnostic; oracle optimism is established prior work; no real-data identifiability claim",
        },
    )


if __name__ == "__main__":
    main()
