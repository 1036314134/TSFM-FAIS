"""Change future dependence while keeping every fixed forecast's expected losses unchanged."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from conditional_future import ForecastHull, future_moments, scenarios
from future_dependence import (
    covariance_intervention,
    flip_probabilities,
    matched_future_components,
    projected_variances,
)
from latent_source_inputs import ROOT, read_json
from readout_conditional_risk import evaluate_decision, summaries

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def dependence_summaries(frame, risks):
    panels, summaries_all = [], []
    for _, group in frame.groupby("correlation"):
        panel, summary = summaries(group, risks)
        for name in ("mae", "mse"):
            old_name = f"optimism_fraction_{name}"
            summary = summary.rename(columns={old_name: f"mean_process_{old_name}"})
            denominator = summary[f"attainable_single_{name}"] + summary[f"optimism_single_{name}"]
            summary[f"ratio_of_mean_gains_{name}"] = np.where(
                denominator > 1e-12, summary[f"optimism_single_{name}"] / denominator, np.nan
            )
        panels.append(panel)
        summaries_all.append(summary)
    return pd.concat(panels, ignore_index=True), pd.concat(summaries_all, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r14/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r14/conditional-forecasts-v001",
        "protocol": "docs/iclr2027/R15_DEPENDENCE_PROTOCOL.md",
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
    original_root = ROOT / "artifacts/iclr27-r14/conditional-risk-v001"
    original_audit = read_json(
        ROOT / "artifacts/iclr27-r14/conditional-risk-audit-v001/manifest.json"
    )
    if original_audit["status"] != "completed" or original_audit["study_sha256"] != file_sha256(
        original_root / "manifest.json"
    ):
        raise ValueError("finish the original controlled risk audit first")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "process_module_sha256": file_sha256(ROOT / "scripts/conditional_future.py"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(args.forecast_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "future_samples": 512,
        "noise_scale": 1.0,
        "correlations": [0.0, 0.5, 1.0],
        "dependence_module_sha256": file_sha256(ROOT / "scripts/future_dependence.py"),
        "risk_module_sha256": file_sha256(ROOT / "scripts/readout_conditional_risk.py"),
        "audit_script_sha256": file_sha256(ROOT / "scripts/audit_future_dependence.py"),
        "reference_sha256": file_sha256(original_root / "manifest.json"),
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
        mean, variance, correlated, independent = matched_future_components(
            model,
            posterior_mean,
            posterior_cov,
            row["phase"] + 96,
            center,
            scale,
            row["future_seed"],
        )
        _, missing, innovation = future_moments(
            model, posterior_mean, posterior_cov, row["phase"] + 96, 96, 1.0
        )
        missing = missing[:, :2] / scale**2
        innovation = innovation[:, :2] / scale**2
        np.testing.assert_array_equal(mean, first_mean)
        for correlation in identity["correlations"]:
            noise = 1.0
            samples = covariance_intervention(mean, correlated, independent, correlation)
            for key, points in predictors.items():
                model_id, slot, size = key
                mu = mean.reshape(-1) if slot == -1 else mean[:, slot]
                var = variance.reshape(-1) if slot == -1 else variance[:, slot]
                future = samples.reshape(512, -1) if slot == -1 else samples[:, :, slot]
                metrics, witness = evaluate_decision(
                    points, mu, var, future, conditional[key], hulls[key]
                )
                best = int(witness["candidate_expected_mse"].argmin())
                directions = points - points[best]
                correlated_var, diagonal_var = projected_variances(
                    model, posterior_cov, directions, var, scale, slot
                )
                projected_var = correlation * correlated_var + (1 - correlation) * diagonal_var
                margins = (
                    witness["candidate_expected_mse"] - witness["candidate_expected_mse"][best]
                )
                projection = 2 * ((future - mu) @ directions.T) / points.shape[1]
                empirical_variance = projection.var(axis=0, ddof=1)
                variance_z = float(
                    np.max(
                        abs(empirical_variance - projected_var)
                        / np.maximum(projected_var * np.sqrt(2 / 511), 1e-12)
                    )
                )
                probabilities = flip_probabilities(margins, projected_var)
                empirical = np.mean(projection > margins, axis=0)
                probability_z = float(
                    np.max(
                        abs(empirical - probabilities)
                        / np.maximum(np.sqrt(probabilities * (1 - probabilities) / 512), 1 / 512)
                    )
                )
                if max(variance_z, probability_z) > 10:
                    raise ValueError("the projected dependence or ranking-flip check failed")
                metrics.update(
                    projected_variance_max_z=variance_z,
                    pair_flip_max_z=probability_z,
                    mean_pair_flip_probability=float(probabilities.mean()),
                    mean_projected_variance=float(projected_var.mean()),
                )
                witness.update(
                    projected_variance=projected_var,
                    pair_flip_probability=probabilities,
                    empirical_pair_flip=empirical,
                    empirical_projected_variance=empirical_variance,
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
                label = f"{row['episode_id']}|{model_id}|{slot}|{size}|{noise}|{correlation}"
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
                        "correlation": correlation,
                        "pool_size": size,
                        "target_slot": slot,
                        **metrics,
                    }
                )
        if (number + 1) % 12 == 0:
            print(f"conditional-risk readout: {number + 1}/180 observed inputs", flush=True)
    frame = pd.DataFrame(records)
    if len(frame) != 3240:
        raise ValueError("conditional-risk comparison coverage changed")
    panels, summary = dependence_summaries(frame, risks)
    original = pd.read_parquet(original_root / "decision_results.parquet")
    original = original[original.noise == 1.0]
    keys = ["model_id", "episode_id", "noise", "pool_size", "target_slot"]
    current = frame[frame.correlation == 1.0].set_index(keys).loc[original.set_index(keys).index]
    reference = original.set_index(keys)
    columns = reference.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(current[columns], reference[columns], rtol=1e-10, atol=1e-10)
    reference_delta = float(
        np.max(abs(current[columns].to_numpy() - reference[columns].to_numpy()))
    )
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
            "maximum_reference_replay_difference": reference_delta,
            "independent_histories": 36,
            "observed_inputs": 180,
            "new_forecaster_calls": 0,
            "files": [
                {"path": name, "sha256": file_sha256(output / name)}
                for name in ("decision_results.parquet", "process_panels.csv", "summary.csv")
            ],
            "limits": "known-process controlled diagnostic; oracle optimism is established prior work; no real-data identifiability claim",
        },
    )


if __name__ == "__main__":
    main()
