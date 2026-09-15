"""Compare conditional-mean filling with Gaussian predictive integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.imputers.gaussian_bridge import conditional_ar1  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "accuracy-root",
        "plan",
        "controls-root",
        "motm-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed Gaussian integration results")
    output.mkdir(parents=True, exist_ok=True)
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    controls_root = args.controls_root / args.model
    controls = json.loads((controls_root / "manifest.json").read_text(encoding="utf-8"))
    motm = json.loads((args.motm_root / "manifest.json").read_text(encoding="utf-8"))
    extra = json.loads((args.motm_root / args.model / "manifest.json").read_text(encoding="utf-8"))
    source_sha = file_sha256(source_path)
    accuracy_sha = file_sha256(args.accuracy_root / "manifest.json")
    if (
        accuracy["source_episode_manifest_sha256"] != source_sha
        or plan["source_manifest_sha256"] != source_sha
        or controls["identity"]["accuracy_manifest_sha256"] != accuracy_sha
        or motm["identity"]["accuracy_manifest_sha256"] != accuracy_sha
        or any(value["status"] != "completed" for value in (controls, motm, extra))
    ):
        raise ValueError("the source and completed controls do not share a protocol")
    config = source["identity"]["config"]
    records = {row["episode_id"]: row for row in source["episodes"]}
    selected = [records[episode] for episode in plan["decision_episode_ids"]]
    if len(selected) != 90 or any(row["split"] != "validation" for row in selected):
        raise ValueError("use the registered 90-task development panel")
    reference_index = {row["episode_id"]: row for row in controls["episodes"]}
    extra_index = {row["episode_id"]: row for row in extra["predictions"]}
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    rho = math.exp(-1 / 12)
    identity = {
        "source_manifest_sha256": source_sha,
        "accuracy_manifest_sha256": accuracy_sha,
        "plan_sha256": file_sha256(args.plan),
        "model": args.model,
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/gaussian_bridge.py"),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "controls_manifest_sha256": file_sha256(controls_root / "manifest.json"),
        "motm_manifest_sha256": file_sha256(args.motm_root / "manifest.json"),
        "motm_model_manifest_sha256": file_sha256(args.motm_root / args.model / "manifest.json"),
        "prior": "independent stationary Gaussian AR(1) in common prefix units; exact observations",
        "rho": rho,
        "stationary_mean": 0,
        "stationary_variance": 1,
        "correlation_length_steps": 12,
        "antithetic_draws": 32,
        "reported_draw_counts": [8, 16, 32],
        "seed_recipe": "first 16 hex digits of SHA256(6101|episode_id), local PCG64 generator",
        "interpretation": "established probabilistic reference; conditional distribution is a working assumption",
        "inference_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/evaluate_timesfm_vendor_missing.py",
                "src/tsfm_fais/forecasting/runner.py",
                "src/tsfm_fais/forecasting/adapters/chronos.py",
                "src/tsfm_fais/forecasting/adapters/timesfm.py",
            )
        },
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Gaussian integration identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "module_snapshot.py").write_bytes(
        (ROOT / "src/tsfm_fais/imputers/gaussian_bridge.py").read_bytes()
    )
    torch.set_num_threads(1)
    registry = default_forecast_registry()
    adapter = (
        TimesFMVendorMissingAdapter(
            model_name=config["forecaster_artifacts"][args.model], device="cuda", batch_size=4
        )
        if args.model == "timesfm2p5"
        else registry.build(
            args.model,
            model_name=config["forecaster_artifacts"][args.model],
            device="cuda",
            batch_size=4,
        )
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    before = parameter_digest(backbone)
    targets = config["target_indices"]
    spec = ForecastSpec(
        args.model,
        registry.get(args.model).mode,
        config["horizon"],
        context_length=config["context_length"],
        target_indices=targets,
    )
    rows, auxiliary, files = [], [], []
    for number, record in enumerate(selected):
        episode = record["episode_id"]
        path = args.source_root / record["path"]
        reference = reference_index[episode]
        addition = extra_index[episode]
        paths = [
            (path, record["sha256"]),
            (controls_root / reference["path"], reference["sha256"]),
            (args.motm_root / addition["path"], addition["sha256"]),
        ]
        if any(file_sha256(p) != digest for p, digest in paths):
            raise ValueError("a source context or matched control changed")
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        with np.load(path, allow_pickle=False) as saved:
            context = (saved["context"] - mean) / scale
        seed = int(hashlib.sha256(("6101|" + episode).encode()).hexdigest()[:16], 16)
        case = hashlib.sha256(episode.encode()).hexdigest()[:24]
        cache = output / "predictions" / f"{case}.npz"
        if not cache.exists():
            positive = np.random.default_rng(seed).normal(size=(16, *context.shape))
            noise = np.stack([positive, -positive], axis=1).reshape(32, *context.shape)
            conditional_mean, samples = conditional_ar1(context, noise, rho=rho)
            observed = np.isfinite(context)
            np.testing.assert_array_equal(conditional_mean[observed], context[observed])
            np.testing.assert_array_equal(
                samples[:, observed], np.repeat(context[observed][None], 32, axis=0)
            )
            for count in identity["reported_draw_counts"]:
                np.testing.assert_allclose(
                    samples[:count].mean(axis=0), conditional_mean, rtol=1e-12, atol=1e-12
                )
            bank = np.concatenate([conditional_mean[None], samples])
            point = runner.predict(bank, spec).point
            if point.shape != (33, config["horizon"], len(targets)) or not np.isfinite(point).all():
                raise ValueError("Gaussian predictive integration produced invalid forecasts")
            _save_npz(
                cache,
                point_z=point,
                conditional_mean_z=conditional_mean,
                input_bank_sha256=np.asarray(hashlib.sha256(bank.tobytes()).hexdigest()),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(before),
                source_sha256=np.asarray(record["sha256"]),
                seed=np.asarray(str(seed)),
            )
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != before
                or str(saved["source_sha256"]) != record["sha256"]
            ):
                raise ValueError("cached Gaussian forecasts have different provenance")
            point, conditional_mean = saved["point_z"], saved["conditional_mean_z"]
        forecasts = {"gaussian_conditional_mean_input": point[0]}
        for count in identity["reported_draw_counts"]:
            forecasts[f"gaussian_forecast_mean_k{count}"] = point[1 : count + 1].mean(axis=0)
            forecasts[f"gaussian_forecast_median_k{count}"] = np.median(
                point[1 : count + 1], axis=0
            )
        with (
            np.load(paths[1][0], allow_pickle=False) as saved,
            np.load(paths[2][0], allow_pickle=False) as added,
        ):
            names = saved["methods"].tolist()
            actions = config["candidate_ids"] + ["guarded_direct"]
            pool = saved["point_z"][[names.index("prefix_input_z_" + action) for action in actions]]
            added_point = added["point_z"][motm["identity"]["views"].index("motm_prefix_z")]
            for action, prediction in zip(actions, pool, strict=True):
                forecasts[action] = prediction
            forecasts["candidate_forecast_median"] = np.median(pool, axis=0)
            forecasts["candidate_forecast_mean"] = pool.mean(axis=0)
            forecasts["candidate_forecast_median_with_motm"] = np.median(
                np.concatenate([pool, added_point[None]]), axis=0
            )
        # The actual outcome and hidden reconstruction labels enter only for scoring.
        with np.load(path, allow_pickle=False) as saved:
            truth = (saved["future"][:, targets] - mean[targets]) / scale[targets]
            clean = (saved["clean_context"] - mean) / scale
        metadata = {
            name: record[name]
            for name in (
                "family_id",
                "dataset_id",
                "item_id",
                "mechanism",
                "missing_rate",
                "episode_id",
            )
        }
        for method, prediction in forecasts.items():
            error = prediction - truth
            rows.append(
                {
                    **metadata,
                    "model": args.model,
                    "method": method,
                    "mae": float(np.abs(error).mean()),
                    "mse": float((error**2).mean()),
                }
            )
        missing = np.isnan(context)
        reconstruction = conditional_mean[missing] - clean[missing]
        auxiliary.append(
            {
                **metadata,
                "missing_cells": int(missing.sum()),
                "imputation_mae": float(np.abs(reconstruction).mean())
                if len(reconstruction)
                else None,
                "imputation_mse": float((reconstruction**2).mean())
                if len(reconstruction)
                else None,
            }
        )
        files.append(
            {
                "episode_id": episode,
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        print(
            json.dumps(
                {
                    "status": "gaussian_case_completed",
                    "model": args.model,
                    "number": number + 1,
                    "total": len(selected),
                }
            ),
            flush=True,
        )
    if before != parameter_digest(backbone):
        raise ValueError("the fixed forecaster changed")
    scores = pd.DataFrame(rows)
    scores.to_parquet(output / "episode_results.parquet", index=False)
    pd.DataFrame(auxiliary).to_parquet(output / "imputation_auxiliary.parquet", index=False)
    family = scores.groupby(["model", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(["model", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "macro_metrics.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "identity_sha256": identity_sha,
            "parameters_unchanged": True,
            "parameter_sha256": before,
            "predictions": files,
            "macro_metrics": summary.to_dict("records"),
            "new_logical_forecast_contexts": 33 * len(selected),
            "draw_inputs_determined_without_future": True,
            "runtime_current_process_only": runner.resource_metrics(),
            "interpretation": "test of a fixed working Gaussian prior; no posterior-calibration, method novelty or confirmation claim",
        },
    )


if __name__ == "__main__":
    main()
