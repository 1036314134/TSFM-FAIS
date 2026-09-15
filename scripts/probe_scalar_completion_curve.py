"""Measure fixed-model response curves for one missing historical value."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from tsfm_fais.forecasting.completion_curve import nearest_curve_points  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed response-curve evidence")
    output.mkdir(parents=True, exist_ok=True)
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if accuracy["source_episode_manifest_sha256"] != file_sha256(source_path):
        raise ValueError("source histories and standardizers do not match")
    config = source["identity"]["config"]
    selected = {}
    for record in sorted(
        (row for row in source["episodes"] if row["split"] == "train"),
        key=lambda row: (row["origin"], row["dataset_id"], row["item_id"], row["episode_id"]),
    ):
        selected.setdefault(record["family_id"], record)
    selected = [selected[key] for key in sorted(selected)]
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    target = config["target_indices"][0]
    identity = {
        "source_manifest_sha256": file_sha256(source_path),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "curve_module_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/completion_curve.py"),
        "inference_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/evaluate_timesfm_vendor_missing.py",
                "src/tsfm_fais/forecasting/adapters/chronos.py",
                "src/tsfm_fais/forecasting/adapters/timesfm.py",
                "src/tsfm_fais/forecasting/runner.py",
            )
        },
        "model": args.model,
        "source_episodes": [record["episode_id"] for record in selected],
        "context_length": config["context_length"],
        "forecast_horizon": config["horizon"],
        "target": target,
        "missing_cell": "last context position of first forecast target; all other history observed",
        "interval": "LOCF plus/minus 2 population standard deviations of the other 95 target values",
        "grid_points": 401,
        "partial_horizons": [1, 12, 24, 96],
        "teachers": ["mean", "coordinate_median"],
        "distribution_interpretation": "specified equal-grid scenario, not an estimated posterior",
        "role": "source-only mechanism check; no confirmation or deployment claim",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("response-curve identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
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
    spec = ForecastSpec(
        args.model,
        registry.get(args.model).mode,
        config["horizon"],
        context_length=config["context_length"],
        target_indices=[target],
    )
    rows, scores, files = [], [], []
    for number, record in enumerate(selected):
        path = args.source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("source history changed")
        case = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24]
        cache = output / "curves" / f"{case}.npz"
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        with np.load(path, allow_pickle=False) as saved:
            complete = (saved["clean_context"] - mean) / scale
        if not np.isfinite(complete).all():
            raise ValueError("this controlled probe requires an originally complete source history")
        observed = complete.copy()
        observed[-1, target] = np.nan
        center = float(observed[-2, target])
        radius = float(2 * observed[:-1, target].std(ddof=0))
        if not cache.exists():
            if radius <= 1e-8:
                # Keep degenerate cases in the panel, with their zero response variability.
                grid = center + max(1, abs(center)) * np.linspace(-1e-8, 1e-8, 401)
            else:
                grid = center + radius * np.linspace(-1, 1, 401)
            bank = np.repeat(observed[None], len(grid), axis=0)
            bank[:, -1, target] = grid
            response = runner.predict(bank, spec).point[:, :, 0]
            anchors = [response.mean(axis=0), np.median(response, axis=0)]
            proposals, proposal_values = [], []
            for teacher, anchor in zip(identity["teachers"], anchors, strict=True):
                for horizon in identity["partial_horizons"]:
                    result = nearest_curve_points(grid, response[:, :horizon], anchor[:horizon])
                    coarse = nearest_curve_points(
                        grid[::2], response[::2, :horizon], anchor[:horizon]
                    )
                    proposals.append(
                        {
                            "teacher": teacher,
                            "horizon": horizon,
                            **result,
                            "coarse_grid_mse": coarse["grid_mse"],
                        }
                    )
                    proposal_values.append(result["interpolated_input"])
            proposal_bank = np.repeat(observed[None], len(proposals), axis=0)
            proposal_bank[:, -1, target] = proposal_values
            refined = runner.predict(proposal_bank, spec).point[:, :, 0]
            native = runner.predict_missing(observed[None], spec).point[0, :, 0]
            clean_point = runner.predict(complete[None], spec).point[0, :, 0]
            # Verify singleton and batched public inference agree at the center input.
            singleton = runner.predict(bank[200:201], spec).point[0, :, 0]
            np.testing.assert_allclose(singleton, response[200], rtol=2e-4, atol=2e-4)
            _save_npz(
                cache,
                grid=grid,
                response=response,
                anchors=np.stack(anchors),
                refined=refined,
                native=native,
                clean_point=clean_point,
                proposals=np.asarray(json.dumps(proposals)),
                identity_sha256=np.asarray(identity_sha),
                model_sha256=np.asarray(before),
                source_sha256=np.asarray(record["sha256"]),
            )
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["model_sha256"]) != before
                or str(saved["source_sha256"]) != record["sha256"]
            ):
                raise ValueError("cached response-curve provenance changed")
            grid, response, anchors = saved["grid"], saved["response"], saved["anchors"]
            refined, native, clean_point = saved["refined"], saved["native"], saved["clean_point"]
            proposals = json.loads(str(saved["proposals"]))
        if response.shape != (401, config["horizon"]) or not np.isfinite(response).all():
            raise ValueError("incomplete or invalid response curve")
        # Current future labels are opened only after every input and projection is fixed.
        with np.load(path, allow_pickle=False) as saved:
            truth = (saved["future"][:, target] - mean[target]) / scale[target]
        evaluated = {"locf": response[200], "native": native, "complete_history": clean_point}
        for teacher, anchor in zip(identity["teachers"], anchors, strict=True):
            evaluated["output_" + teacher] = anchor
        for index, proposal in enumerate(proposals):
            horizon = proposal["horizon"]
            teacher_index = identity["teachers"].index(proposal["teacher"])
            anchor = anchors[teacher_index]
            residual = float(((refined[index, :horizon] - anchor[:horizon]) ** 2).mean())
            choose_refined = residual < proposal["grid_mse"]
            chosen = refined[index] if choose_refined else response[proposal["grid_index"]]
            residual = min(residual, proposal["grid_mse"])
            variance = float(response[:, :horizon].var(axis=0, ddof=0).mean())
            rows.append(
                {
                    "model": args.model,
                    "family_id": record["family_id"],
                    "episode_id": record["episode_id"],
                    **proposal,
                    "verified_projection_mse": residual,
                    "curve_variance": variance,
                    "residual_variance_ratio": residual / variance if variance > 1e-16 else None,
                    "radius_z": radius,
                    "degenerate_observed_scale": radius <= 1e-8,
                    "positive_inference_condition_changes": bool(
                        np.all(observed[:-1, target] >= 0) and grid[0] < 0 < grid[-1]
                    ),
                    "projection_input_z": proposal["interpolated_input"]
                    if choose_refined
                    else proposal["grid_input"],
                }
            )
            if horizon == config["horizon"]:
                evaluated["input_projection_" + proposal["teacher"]] = chosen
        for method, prediction in evaluated.items():
            error = prediction - truth
            scores.append(
                {
                    "model": args.model,
                    "family_id": record["family_id"],
                    "episode_id": record["episode_id"],
                    "method": method,
                    "mae": float(np.abs(error).mean()),
                    "mse": float((error**2).mean()),
                }
            )
        files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        print(
            json.dumps(
                {
                    "status": "curve_completed",
                    "model": args.model,
                    "number": number + 1,
                    "total": len(selected),
                    "family": record["family_id"],
                }
            ),
            flush=True,
        )
    if before != parameter_digest(backbone):
        raise ValueError("the fixed forecaster changed")
    frame, score_frame = pd.DataFrame(rows), pd.DataFrame(scores)
    frame.to_parquet(output / "curve_diagnostics.parquet", index=False)
    score_frame.to_parquet(output / "source_forecast_errors.parquet", index=False)
    summary = score_frame.groupby("method")[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "source_macro_metrics.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity_sha256": identity_sha,
            "identity": identity,
            "parameters_unchanged": True,
            "parameter_sha256": before,
            "curves": files,
            "source_macro_metrics": summary.to_dict("records"),
            "runtime_counters_current_process_only": runner.resource_metrics(),
            "logical_context_queries_per_completed_curve": 412,
            "limitations": [
                "sampled and interpolated residuals are not certified lower bounds",
                "specified input ambiguity is not an estimated conditional distribution",
                "one source window per family and one missing value only",
                "this is not the registered development or confirmation evaluation",
            ],
        },
    )


if __name__ == "__main__":
    main()
