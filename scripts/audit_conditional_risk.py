"""Check the generative information boundary and conditional oracle witnesses."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from conditional_future import (
    condition_history,
    draw_history,
    future_moments,
    sample_futures,
    scenarios,
)
from latent_source_inputs import ROOT, read_json
from readout_conditional_risk import summaries
from replay_preforecast_student import assemble_selected_context
from scipy.special import erf

from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_risks(points, mean, variance):
    difference = points - mean
    sigma = np.sqrt(np.maximum(variance, 0.0))
    positive = sigma > 0
    absolute = abs(difference).copy()
    z = np.divide(difference, sigma, out=np.zeros_like(difference), where=positive)
    absolute[..., positive] = (
        sigma * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + difference * erf(z / np.sqrt(2))
    )[..., positive]
    return absolute.mean(-1), (difference**2 + variance).mean(-1)


def audit_weight(points, target, weights):
    if (weights < 0).any() or not np.allclose(weights.sum(-1), 1, rtol=0, atol=1e-10):
        raise ValueError("oracle weights are infeasible")
    anchor = np.median(points, axis=0)
    delta = points - anchor
    prediction = points[0] + weights @ (points - points[:1])
    gram = delta @ delta.T / points.shape[1]
    alignment = (target - anchor) @ delta.T / points.shape[1]
    gradient = 2 * (prediction - target) @ delta.T / points.shape[1]
    scale = np.maximum(np.maximum(abs(alignment).max(-1), abs(gram).max()), 1e-12)
    gap = (np.sum(gradient * weights, axis=-1) - gradient.min(-1)) / scale
    if np.max(gap) > 1e-7:
        raise ValueError("an oracle fails direct gradient optimality")
    return prediction, float(max(np.max(gap), 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r14/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r14/conditional-forecasts-v001",
        "study-root": "artifacts/iclr27-r14/conditional-risk-v001",
        "protocol": "docs/iclr2027/R14_CONDITIONAL_RISK_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed controlled audits")
    prep = read_json(args.prepared_root / "manifest.json")
    forecast = read_json(args.forecast_root / "manifest.json")
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["prepared_sha256"] != file_sha256(args.prepared_root / "manifest.json")
        or study["identity"]["forecast_sha256"] != file_sha256(args.forecast_root / "manifest.json")
        or study["identity"]["process_module_sha256"]
        != file_sha256(ROOT / "scripts/conditional_future.py")
        or study["identity"]["script_sha256"]
        != file_sha256(ROOT / "scripts/readout_conditional_risk.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("controlled-risk definitions changed")
    models = scenarios()
    inputs = {}
    banks = {}
    queries = set()
    input_records = {row["episode_id"]: row for row in prep["episodes"]}
    for generator, model in enumerate(models):
        record = prep["prefixes"][generator]
        path = args.prepared_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a calibration prefix changed")
        prefix = np.load(path)
        np.testing.assert_array_equal(prefix, draw_history(model, 6144, 0, 14400 + generator))
        np.testing.assert_allclose(prefix.mean(0), record["mean"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(prefix.std(0), record["scale"], rtol=1e-12, atol=1e-12)
    for entry in prep["imputer_fits"]:
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a calibration imputer record changed")
        fitted = read_json(path)
        if fitted["status"] != "fitted" or fitted["training_windows"] != 64:
            raise ValueError("a prescribed calibration fit failed")
        for record in fitted["files"]:
            if (
                file_sha256(path.parent / entry["candidate_id"] / record["path"])
                != record["sha256"]
            ):
                raise ValueError("a calibration imputer artifact changed")
    for record in prep["episodes"]:
        model = models[record["generator"]]
        path = args.prepared_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a controlled observation changed")
        clean = draw_history(
            model, 96, record["phase"], 14500 + 100 * record["generator"] + record["history"]
        )
        context = clean.copy()
        rng = np.random.default_rng(
            14600 + 1000 * record["generator"] + 10 * record["history"] + record["condition"]
        )
        if record["mechanism"] == "random_point":
            context[rng.random(context.shape) < record["missing_rate"]] = np.nan
        elif record["mechanism"] == "tail_block":
            context[-int(round(96 * record["missing_rate"])) :] = np.nan
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["context"], context)
            np.testing.assert_array_equal(saved["clean_context"], clean)
            mean, covariance = condition_history(model, context, record["phase"], scalar=True)
            np.testing.assert_allclose(saved["posterior_mean"], mean, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                saved["posterior_covariance"], covariance, rtol=1e-10, atol=1e-10
            )
            candidates = np.concatenate([saved["candidate_values"], saved["motm_values"][None]])
            for candidate in candidates:
                np.testing.assert_array_equal(
                    candidate[np.isfinite(context)], context[np.isfinite(context)]
                )
            inputs[record["episode_id"]] = (
                context,
                candidates,
                [*saved["candidate_ids"].tolist(), "motm_reference"],
                saved["posterior_mean"],
                saved["posterior_covariance"],
            )
    for model_entry in forecast["models"]:
        model_id = model_entry["model_id"]
        root = args.forecast_root / model_id
        if file_sha256(args.forecast_root / model_entry["path"]) != model_entry["sha256"]:
            raise ValueError("a forecasting manifest changed")
        model_manifest = read_json(args.forecast_root / model_entry["path"])
        for entry in model_manifest["episodes"]:
            path = root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a frozen forecast bank changed")
            context, candidates, actions, _, _ = inputs[entry["episode_id"]]
            record = input_records[entry["episode_id"]]
            scaler = prep["prefixes"][record["generator"]]
            center, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            with np.load(path, allow_pickle=False) as saved:
                names = saved["actions"].tolist()
                points = saved["point_z"]
                for action, key, point in zip(
                    names, saved["query_keys"].tolist(), points, strict=True
                ):
                    effective = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if model_id == "chronos2" else [action, action],
                        [0, 1],
                        joint=model_id == "chronos2",
                    )
                    effective = np.asarray(
                        effective if model_id == "chronos2" else effective[:, :2], np.float32
                    ).copy(order="C")
                    effective[np.isnan(effective)] = np.nan
                    if (
                        hashlib.sha256(
                            str(effective.shape).encode() + effective.tobytes()
                        ).hexdigest()
                        != key
                    ):
                        raise ValueError("a controlled forecast used a different input")
                    with np.load(root / "queries" / f"{key}.npz", allow_pickle=False) as query:
                        np.testing.assert_array_equal(query["effective_input"], effective)
                        np.testing.assert_array_equal((query["point"] - center) / scale, point)
                        if str(query["parameter_sha256"]) != model_manifest["parameter_sha256"]:
                            raise ValueError("forecasting parameters changed")
                    queries.add((model_id, key))
                if record["mechanism"] == "complete":
                    np.testing.assert_array_equal(points, np.repeat(points[:1], 8, axis=0))
                banks[(model_id, entry["episode_id"])] = (points, names)
    frame = pd.read_parquet(args.study_root / "decision_results.parquet")
    risks = {}
    conditional_cache = {}
    sample_cache = {}
    maximum_gap = 0.0
    maximum_difference = 0.0
    for entry in study["witnesses"]:
        index = entry["row"]
        row = frame.iloc[index]
        record = input_records[row.episode_id]
        model = models[record["generator"]]
        key = (row.episode_id, float(row.noise))
        if key not in sample_cache:
            _, _, _, posterior_mean, posterior_covariance = inputs[row.episode_id]
            mean, missing, innovation = future_moments(
                model,
                posterior_mean,
                posterior_covariance,
                record["phase"] + 96,
                96,
                float(row.noise),
            )
            scaler = prep["prefixes"][record["generator"]]
            center, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            sampled = sample_futures(
                model,
                posterior_mean,
                posterior_covariance,
                record["phase"] + 96,
                96,
                float(row.noise),
                512,
                record["future_seed"],
            )
            sample_cache = {
                key: (
                    (mean[:, :2] - center) / scale,
                    (missing + innovation)[:, :2] / scale**2,
                    (sampled[:, :, :2] - center) / scale,
                )
            }
        mu, var, sampled = sample_cache[key]
        bank, names = banks[(row.model_id, row.episode_id)]
        chosen = (
            np.arange(8)
            if row.pool_size == 8
            else np.array([i for i, name in enumerate(names) if name != "motm_reference"])
        )
        slot, size = int(row.target_slot), int(row.pool_size)
        points = bank[chosen].reshape(size, -1) if slot == -1 else bank[chosen, :, slot]
        mean = mu.reshape(-1) if slot == -1 else mu[:, slot]
        variance = var.reshape(-1) if slot == -1 else var[:, slot]
        future = sampled.reshape(512, -1) if slot == -1 else sampled[:, :, slot]
        path = args.study_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("an oracle witness changed")
        with np.load(path, allow_pickle=False) as witness:
            expected_mae, expected_mse = direct_risks(points, mean, variance)
            risks[index] = (witness["candidate_expected_mae"], witness["candidate_expected_mse"])
            np.testing.assert_allclose(expected_mae, risks[index][0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(expected_mse, risks[index][1], rtol=1e-12, atol=1e-12)
            actual_mae = abs(points[None] - future[:, None]).mean(2)
            actual_mse = ((points[None] - future[:, None]) ** 2).mean(2)
            win_mae = actual_mae.argmin(1)
            win_mse = actual_mse.argmin(1)
            rows = np.arange(512)
            np.testing.assert_array_equal(win_mae, witness["winner_mae"])
            np.testing.assert_array_equal(win_mse, witness["winner_mse"])
            gain_mae = actual_mae[:, expected_mae.argmin()] - actual_mae[rows, win_mae]
            gain_mse = actual_mse[:, expected_mse.argmin()] - actual_mse[rows, win_mse]
            np.testing.assert_allclose(gain_mae, witness["paired_gain_mae"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(gain_mse, witness["paired_gain_mse"], rtol=1e-12, atol=1e-12)
            conditional, gap = audit_weight(points, mean, witness["conditional_weights"])
            maximum_gap = max(maximum_gap, gap)
            hindsight, gap = audit_weight(points, future, witness["hindsight_weights"])
            maximum_gap = max(maximum_gap, gap)
            hindsight_loss = ((hindsight - future) ** 2).mean(1)
            gain = ((conditional - future) ** 2).mean(1) - hindsight_loss
            np.testing.assert_allclose(
                gain, witness["paired_gain_convex_mse"], rtol=1e-11, atol=1e-11
            )
            condition_key = (row.model_id, row.episode_id, slot, size)
            if condition_key in conditional_cache:
                np.testing.assert_array_equal(
                    conditional_cache[condition_key], witness["conditional_weights"]
                )
            else:
                conditional_cache[condition_key] = witness["conditional_weights"].copy()
            checks = {
                "conditional_single_mae": float(expected_mae.min()),
                "conditional_single_mse": float(expected_mse.min()),
                "optimism_single_mae": float(gain_mae.mean()),
                "optimism_single_mse": float(gain_mse.mean()),
                "optimism_convex_mse": float(gain.mean()),
                "hindsight_convex_mse": float(hindsight_loss.mean()),
                "conditional_convex_mse": float(direct_risks(conditional, mean, variance)[1]),
                "hindsight_convex_independent_mse": float(
                    direct_risks(hindsight, mean, variance)[1].mean()
                ),
                "hindsight_single_independent_mae": float(expected_mae[win_mae].mean()),
                "hindsight_single_independent_mse": float(expected_mse[win_mse].mean()),
                "noise_floor_mse": float(variance.mean()),
                "noise_floor_mae": float((np.sqrt(variance) * np.sqrt(2 / np.pi)).mean()),
            }
            for name, value in checks.items():
                maximum_difference = max(maximum_difference, abs(float(row[name]) - value))
                np.testing.assert_allclose(row[name], value, rtol=1e-10, atol=1e-10)
    if (
        len(frame) != 4320
        or len(study["witnesses"]) != 4320
        or len({row["origin_id"] for row in prep["episodes"]}) != 36
    ):
        raise ValueError("controlled diagnostic population changed")
    panels, summary = summaries(frame, risks)
    pd.testing.assert_frame_equal(
        panels,
        pd.read_csv(args.study_root / "process_panels.csv", float_precision="round_trip"),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "decision_results.parquet", index=False)
    panels.to_csv(output / "process_panels.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_decisions": 4320,
            "independent_histories": 36,
            "verified_effective_forecast_inputs": len(queries),
            "maximum_oracle_optimality_gap": maximum_gap,
            "maximum_metric_reconstruction_difference": maximum_difference,
            "limits": "known-process simulation only; established oracle-bias principle; no real-data noise attribution or novelty guarantee",
        },
    )


if __name__ == "__main__":
    main()
