"""Independently reconstruct covariance interventions and conditional-risk witnesses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from audit_conditional_risk import audit_weight, direct_risks
from conditional_future import future_moments, sample_futures, scenarios
from future_dependence import dense_future_covariance
from latent_source_inputs import ROOT, read_json
from readout_future_dependence import dependence_summaries
from scipy.special import erf

from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r14/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r14/conditional-forecasts-v001",
        "study-root": "artifacts/iclr27-r15/dependence-v001",
        "protocol": "docs/iclr2027/R15_DEPENDENCE_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed dependence audits")
    prep = read_json(args.prepared_root / "manifest.json")
    forecast = read_json(args.forecast_root / "manifest.json")
    study = read_json(args.study_root / "manifest.json")
    reference_root = ROOT / "artifacts/iclr27-r14/conditional-risk-v001"
    reference = read_json(reference_root / "manifest.json")
    original_audit = read_json(
        ROOT / "artifacts/iclr27-r14/conditional-risk-audit-v001/manifest.json"
    )
    identity = study["identity"]
    bindings = {
        "prepared_sha256": args.prepared_root / "manifest.json",
        "forecast_sha256": args.forecast_root / "manifest.json",
        "process_module_sha256": ROOT / "scripts/conditional_future.py",
        "dependence_module_sha256": ROOT / "scripts/future_dependence.py",
        "risk_module_sha256": ROOT / "scripts/readout_conditional_risk.py",
        "script_sha256": ROOT / "scripts/readout_future_dependence.py",
        "audit_script_sha256": ROOT / "scripts/audit_future_dependence.py",
        "reference_sha256": reference_root / "manifest.json",
        "protocol_sha256": args.protocol,
    }
    if study["status"] != "completed" or any(
        identity[key] != file_sha256(path) for key, path in bindings.items()
    ):
        raise ValueError("dependence study definitions changed")
    if (
        original_audit["status"] != "completed"
        or original_audit["study_sha256"] != identity["reference_sha256"]
        or original_audit["script_sha256"]
        != file_sha256(ROOT / "scripts/audit_conditional_risk.py")
        or any(
            reference["identity"][key] != identity[key]
            for key in ("prepared_sha256", "forecast_sha256", "process_module_sha256")
        )
        or reference["identity"]["script_sha256"] != identity["risk_module_sha256"]
        or identity["correlations"] != [0.0, 0.5, 1.0]
        or identity["future_samples"] != 512
        or identity["noise_scale"] != 1.0
        or identity["candidate_pool_sizes"] != [7, 8]
    ):
        raise ValueError("the audited reference or registered intervention changed")
    for entry in study["files"]:
        if file_sha256(args.study_root / entry["path"]) != entry["sha256"]:
            raise ValueError("a dependence result file changed")
    inputs = {}
    records = {row["episode_id"]: row for row in prep["episodes"]}
    for record in prep["episodes"]:
        path = args.prepared_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a previously audited observation changed")
        with np.load(path, allow_pickle=False) as saved:
            inputs[record["episode_id"]] = saved["posterior_mean"], saved["posterior_covariance"]
    banks = {}
    for entry in forecast["models"]:
        path = args.forecast_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen forecast manifest changed")
        for record in read_json(path)["episodes"]:
            path = args.forecast_root / entry["model_id"] / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("an audited forecast changed")
            with np.load(path, allow_pickle=False) as saved:
                banks[(entry["model_id"], record["episode_id"])] = (
                    saved["point_z"],
                    saved["actions"].tolist(),
                )
    frame = pd.read_parquet(args.study_root / "decision_results.parquet")
    if (
        len(frame) != 3240
        or len(study["witnesses"]) != 3240
        or [entry["row"] for entry in study["witnesses"]] != list(range(3240))
        or len({row["origin_id"] for row in prep["episodes"]}) != 36
        or len(records) != 180
    ):
        raise ValueError("dependence population coverage changed")
    models = scenarios()
    risks, invariant = {}, {}
    cached_episode = None
    maximum_gap = maximum_metric_difference = maximum_covariance_difference = 0.0
    for entry in study["witnesses"]:
        index = entry["row"]
        row = frame.iloc[index]
        record = records[row.episode_id]
        if row.episode_id != cached_episode:
            model = models[record["generator"]]
            posterior_mean, posterior_covariance = inputs[row.episode_id]
            prefix = prep["prefixes"][record["generator"]]
            center, scale = np.asarray(prefix["mean"])[:2], np.asarray(prefix["scale"])[:2]
            mu_raw, missing, innovation = future_moments(
                model, posterior_mean, posterior_covariance, record["phase"] + 96, 96, 1.0
            )
            mu = (mu_raw[:, :2] - center) / scale
            var = missing[:, :2] / scale**2 + innovation[:, :2] / scale**2
            original = sample_futures(
                model,
                posterior_mean,
                posterior_covariance,
                record["phase"] + 96,
                96,
                1.0,
                512,
                record["future_seed"],
            )
            original = (original[:, :, :2] - center) / scale
            independent = (
                np.random.default_rng(record["future_seed"] + 500000).standard_normal(
                    original.shape
                )
                * np.sqrt(var)[None]
            )
            covariance = dense_future_covariance(model, posterior_covariance, 96, scale)
            np.testing.assert_allclose(np.diag(covariance), var.ravel(), rtol=1e-12, atol=1e-12)
            cached_episode = row.episode_id
        rho = float(row.correlation)
        sampled = (
            original
            if rho == 1
            else (mu + np.sqrt(rho) * (original - mu) + np.sqrt(1 - rho) * independent)
        )
        bank, names = banks[(row.model_id, row.episode_id)]
        size, slot = int(row.pool_size), int(row.target_slot)
        chosen = (
            np.arange(8)
            if size == 8
            else np.array([i for i, name in enumerate(names) if name != "motm_reference"])
        )
        indices = np.arange(192) if slot == -1 else np.arange(slot, 192, 2)
        points = bank[chosen].reshape(size, -1) if slot == -1 else bank[chosen, :, slot]
        mean, variance = mu.ravel()[indices], var.ravel()[indices]
        future = sampled.reshape(512, -1) if slot == -1 else sampled[:, :, slot]
        selected_covariance = covariance[np.ix_(indices, indices)]
        selected_covariance = rho * selected_covariance + (1 - rho) * np.diag(variance)
        path = args.study_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a dependence witness changed")
        with np.load(path, allow_pickle=False) as witness:
            expected_mae, expected_mse = direct_risks(points, mean, variance)
            risks[index] = witness["candidate_expected_mae"], witness["candidate_expected_mse"]
            np.testing.assert_allclose(expected_mae, risks[index][0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(expected_mse, risks[index][1], rtol=1e-12, atol=1e-12)
            key = (row.model_id, row.episode_id, slot, size)
            current = (*risks[index], witness["conditional_weights"])
            if key in invariant:
                for first, second in zip(invariant[key], current, strict=True):
                    np.testing.assert_array_equal(first, second)
            else:
                invariant[key] = current
            actual_mae = abs(points[None] - future[:, None]).mean(2)
            actual_mse = ((points[None] - future[:, None]) ** 2).mean(2)
            win_mae, win_mse = actual_mae.argmin(1), actual_mse.argmin(1)
            np.testing.assert_array_equal(win_mae, witness["winner_mae"])
            np.testing.assert_array_equal(win_mse, witness["winner_mse"])
            replicas = np.arange(512)
            gain_mae = actual_mae[:, expected_mae.argmin()] - actual_mae[replicas, win_mae]
            gain_mse = actual_mse[:, expected_mse.argmin()] - actual_mse[replicas, win_mse]
            conditional, gap = audit_weight(points, mean, witness["conditional_weights"])
            maximum_gap = max(maximum_gap, gap)
            hindsight, gap = audit_weight(points, future, witness["hindsight_weights"])
            maximum_gap = max(maximum_gap, gap)
            hindsight_loss = ((hindsight - future) ** 2).mean(1)
            gain_convex = ((conditional - future) ** 2).mean(1) - hindsight_loss
            for name, value in (("mae", gain_mae), ("mse", gain_mse), ("convex_mse", gain_convex)):
                np.testing.assert_allclose(
                    witness[f"paired_gain_{name}"], value, rtol=1e-10, atol=1e-10
                )
            best = int(expected_mse.argmin())
            directions = points - points[best]
            projected = (
                4
                * np.einsum("ai,ij,aj->a", directions, selected_covariance, directions)
                / len(indices) ** 2
            )
            maximum_covariance_difference = max(
                maximum_covariance_difference,
                float(abs(projected - witness["projected_variance"]).max()),
            )
            np.testing.assert_allclose(
                projected, witness["projected_variance"], rtol=1e-10, atol=1e-10
            )
            margins = expected_mse - expected_mse[best]
            probability = np.zeros(size)
            positive = projected > 0
            probability[positive] = 0.5 * (
                1 - erf(margins[positive] / np.sqrt(2 * projected[positive]))
            )
            np.testing.assert_allclose(
                probability, witness["pair_flip_probability"], rtol=1e-10, atol=1e-10
            )
            projection = 2 * ((future - mean) @ directions.T) / len(indices)
            np.testing.assert_allclose(
                projection.var(0, ddof=1),
                witness["empirical_projected_variance"],
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_array_equal(
                (projection > margins).mean(0), witness["empirical_pair_flip"]
            )
            checks = {
                "conditional_single_mae": float(expected_mae.min()),
                "conditional_single_mse": float(expected_mse.min()),
                "optimism_single_mae": float(gain_mae.mean()),
                "optimism_single_mse": float(gain_mse.mean()),
                "optimism_convex_mse": float(gain_convex.mean()),
                "hindsight_single_mae": float(actual_mae[replicas, win_mae].mean()),
                "hindsight_single_mse": float(actual_mse[replicas, win_mse].mean()),
                "hindsight_convex_mse": float(hindsight_loss.mean()),
                "conditional_convex_mse": float(direct_risks(conditional, mean, variance)[1]),
                "hindsight_convex_independent_mse": float(
                    direct_risks(hindsight, mean, variance)[1].mean()
                ),
                "hindsight_single_independent_mae": float(expected_mae[win_mae].mean()),
                "hindsight_single_independent_mse": float(expected_mse[win_mse].mean()),
                "noise_floor_mse": float(variance.mean()),
                "noise_floor_mae": float((np.sqrt(variance) * np.sqrt(2 / np.pi)).mean()),
                "mean_pair_flip_probability": float(probability.mean()),
                "mean_projected_variance": float(projected.mean()),
            }
            for name, value in checks.items():
                maximum_metric_difference = max(
                    maximum_metric_difference, abs(float(row[name]) - value)
                )
                np.testing.assert_allclose(row[name], value, rtol=1e-10, atol=1e-10)
    old = pd.read_parquet(reference_root / "decision_results.parquet")
    keys = ["model_id", "episode_id", "noise", "pool_size", "target_slot"]
    old = old[old.noise == 1.0].set_index(keys)
    current = frame[frame.correlation == 1.0].set_index(keys).loc[old.index]
    columns = old.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(current[columns], old[columns], rtol=1e-10, atol=1e-10)
    reference_difference = float(np.max(abs(current[columns].to_numpy() - old[columns].to_numpy())))
    panels, summary = dependence_summaries(frame, risks)
    for name, table in (("process_panels.csv", panels), ("summary.csv", summary)):
        pd.testing.assert_frame_equal(
            table,
            pd.read_csv(args.study_root / name, float_precision="round_trip"),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    if len(invariant) != 1080:
        raise ValueError("the three covariance conditions do not cover every original decision")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "registered_audit_sha256": identity["audit_script_sha256"],
            "repair_note_sha256": file_sha256(ROOT / "docs/iclr2027/R15_AUDIT_REPLAY_NOTE.md"),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_decisions": len(frame),
            "invariant_expected_risk_and_weight_sets": len(invariant),
            "maximum_oracle_optimality_gap": maximum_gap,
            "maximum_metric_reconstruction_difference": maximum_metric_difference,
            "maximum_dense_covariance_difference": maximum_covariance_difference,
            "maximum_reference_replay_difference": reference_difference,
            "new_forecaster_calls": 0,
            "limits": "known conditional future-law intervention on reused simulation histories; no real-data attribution or novelty guarantee",
        },
    )


if __name__ == "__main__":
    main()
