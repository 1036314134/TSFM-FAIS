"""Check frozen Chronos gradients on training-only imputation mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.forecasting.chronos_differentiable import chronos_median  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def compose(candidates, context, logits):
    import torch

    weights = torch.softmax(logits, dim=0)
    mixed = (candidates * weights[:, None, :]).sum(dim=0)
    observed = torch.isfinite(context)
    return torch.where(observed, torch.nan_to_num(context), mixed), weights


def optimization_target(source_future, candidate_forecasts, mode):
    if mode == "source_truth":
        return source_future.detach()
    if mode == "forecast_median":
        if candidate_forecasts.ndim != 3 or candidate_forecasts.shape[0] < 2:
            raise ValueError("the prediction teacher must contain [A,H,K] candidate forecasts")
        return candidate_forecasts.detach().quantile(0.5, dim=0)
    raise ValueError("unknown optimization target")


def parameter_digest(model):
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--accuracy-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--datasets", default="ETTh1,exchange_rate,Coastal_T_S_H")
    parser.add_argument(
        "--optimization-target", choices=("source_truth", "forecast_median"), default="source_truth"
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 40:
        parser.error("the feasibility probe is limited to 1-40 optimization steps")
    import torch
    from chronos import BaseChronosPipeline

    from tsfm_fais.routing.differentiable import missing_block_ids, mix_blocks

    torch.set_num_threads(1)
    torch.manual_seed(6101)
    source_root, accuracy_root, output = (
        args.source_root.resolve(),
        args.accuracy_root.resolve(),
        args.output_root.resolve(),
    )
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if accuracy["source_episode_manifest_sha256"] != file_sha256(
        source_root / "episodes_manifest.json"
    ):
        raise ValueError("training episodes and prefix standardizers must share the same source")
    config = source["identity"]["config"]
    datasets = tuple(args.datasets.split(","))
    if not 1 <= len(datasets) <= 5 or len(set(datasets)) != len(datasets):
        parser.error("select one to five distinct source datasets for feasibility checks")
    seed = min(config["mask_seeds"])
    selected = []
    for dataset in datasets:
        available = [
            r
            for r in source["episodes"]
            if r["dataset_id"] == dataset
            and r["split"] == "train"
            and r["mechanism"] == "independent_block"
            and np.isclose(r["missing_rate"], 0.3)
            and r["mask_seed"] == seed
        ]
        if not available:
            raise ValueError(f"predeclared training probe is unavailable: {dataset}")
        selected.append(min(available, key=lambda r: r["origin"]))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed gradient feasibility evidence")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "chronos_gradient_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/chronos_differentiable.py"
        ),
        "composer_module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/differentiable.py"),
        "source_manifest_sha256": file_sha256(source_root / "episodes_manifest.json"),
        "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
        "episodes": [r["episode_id"] for r in selected],
        "steps": args.steps,
        "optimization_target": args.optimization_target,
        "checkpoint": config["forecaster_artifacts"]["chronos2"],
        "wide_case_steps": 1,
        "full_step_variate_limit": 64,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("gradient probe identity changed; choose a new output root")
    _write_json(identity_path, identity)
    (output / "script_snapshot.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "composer_snapshot.py").write_text(
        (ROOT / "src/tsfm_fais/routing/differentiable.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
    }
    pipeline = BaseChronosPipeline.from_pretrained(identity["checkpoint"], device_map="cuda")
    model = pipeline.model.eval().requires_grad_(False)
    before_digest = parameter_digest(model)
    horizon, targets = config["horizon"], list(config["target_indices"])
    median_indices = np.flatnonzero(np.isclose(pipeline.quantiles, 0.5))
    if len(median_indices) != 1:
        raise ValueError("the checkpoint must expose exactly one median quantile")
    started, records = monotonic(), []

    def predict(context):
        return chronos_median(pipeline, context, horizon, targets)

    for source_record in selected:
        cache = output / (
            hashlib.sha256(source_record["episode_id"].encode()).hexdigest()[:20] + ".json"
        )
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if file_sha256(cache.with_suffix(".npz")) != cached["arrays_sha256"]:
                raise ValueError("completed projection arrays changed")
            records.append(cached)
            continue
        path = source_root / source_record["path"]
        if file_sha256(path) != source_record["sha256"]:
            raise ValueError("training candidate cache changed")
        scaler = scalers[(source_record["dataset_id"], source_record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        with np.load(path, allow_pickle=False) as episode:
            context = torch.tensor(
                (episode["context"] - mean) / scale, device="cuda", dtype=torch.float32
            )
            candidates = torch.tensor(
                (episode["candidate_values"] - mean) / scale, device="cuda", dtype=torch.float32
            )
            truth = torch.tensor(
                ((episode["future"] - mean) / scale)[:, targets], device="cuda", dtype=torch.float32
            )
            actions = episode["candidate_ids"].tolist()
        observed = torch.isfinite(context)
        if not torch.isfinite(candidates).all() or not torch.isfinite(truth).all():
            raise ValueError(
                "feasibility probe requires finite candidates and complete source training labels"
            )
        if not torch.equal(candidates[:, observed], context[observed].expand(len(candidates), -1)):
            raise ValueError("candidate completion changed a known observation")
        if bool(observed.all()):
            raise ValueError("predeclared case contains no missing values")
        initial, _ = compose(candidates, context, torch.zeros((len(candidates), 1), device="cuda"))
        native, _ = pipeline.predict_quantiles(
            inputs=[{"target": initial.detach().cpu().numpy().T}],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            predict_batches_jointly=False,
        )
        official = native[0].detach().cpu().numpy()[:, :, 1].T[:, targets]
        with torch.no_grad():
            direct = predict(initial).cpu().numpy()
        contract_error = float(np.max(np.abs(direct - official)))
        np.testing.assert_allclose(direct, official, rtol=2e-4, atol=2e-4)
        baseline_quantiles, _ = pipeline.predict_quantiles(
            inputs=[{"target": item.detach().cpu().numpy().T} for item in [*candidates, context]],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            predict_batches_jointly=False,
        )
        source_truth = truth.detach().cpu().numpy()
        baseline_mse = {
            action: float(
                np.mean(
                    (forecast.detach().cpu().numpy()[:, :, 1].T[:, targets] - source_truth) ** 2
                )
            )
            for action, forecast in zip(
                [*actions, "prefix_input_z_native"], baseline_quantiles, strict=True
            )
        }
        teacher_points = torch.stack(
            [forecast[:, :, 1].T[:, targets].float().to("cuda") for forecast in baseline_quantiles]
        )
        # Match the guarded native action used by the strong ensemble control.
        if bool((~observed.any(0)).any()):
            teacher_points[-1] = teacher_points[actions.index("locf")]
        teacher = optimization_target(truth, teacher_points, "forecast_median")
        fit_target = optimization_target(truth, teacher_points, args.optimization_target)
        teacher_error = teacher - truth
        saved_arrays = {
            "teacher": teacher.cpu().numpy(),
            "source_future": source_truth,
            "teacher_candidate_forecasts": teacher_points.cpu().numpy(),
            "initial_forecast": direct,
        }
        identifiers, variables = missing_block_ids(context)
        case_steps = args.steps if context.shape[1] <= 64 else 1
        variants = []
        granularities = [("sequence", 1), ("variate", context.shape[1])]
        if args.optimization_target == "forecast_median":
            granularities.append(("block", len(variables)))
        for granularity, width in granularities:
            logits = torch.zeros((len(candidates), width), device="cuda", requires_grad=True)
            optimizer = torch.optim.Adam([logits], lr=0.05)
            best_loss, best_weights, best_forecast_errors, trace = float("inf"), None, None, []
            for step in range(case_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                if granularity == "block":
                    weights = logits.softmax(0)
                    mixed = mix_blocks(candidates, context, identifiers, weights)
                else:
                    mixed, weights = compose(candidates, context, logits)
                if not torch.equal(mixed[observed], context[observed]):
                    raise ValueError("optimization changed a known observation")
                prediction = predict(mixed)
                loss = (prediction - fit_target).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite differentiable forecasting loss")
                value = float(loss.detach())
                if value < best_loss:
                    best_loss, best_weights = value, weights.detach().cpu().tolist()
                    error = prediction.detach() - truth
                    best_forecast_errors = {
                        "mae": float(error.abs().mean()),
                        "mse": float(error.square().mean()),
                    }
                    saved_arrays[granularity + "_best_forecast"] = prediction.detach().cpu().numpy()
                if step == case_steps:
                    trace.append({"step": step, "mse": value})
                    break
                loss.backward()
                if logits.grad is None or not torch.isfinite(logits.grad).all():
                    raise ValueError("forecast gradients do not reach the mixture weights")
                norm = float(logits.grad.norm())
                trace.append({"step": step, "mse": value, "gradient_norm": norm})
                optimizer.step()
            variants.append(
                {
                    "granularity": granularity,
                    "optimization_target": args.optimization_target,
                    "initial_objective_mse": trace[0]["mse"],
                    "best_objective_mse": best_loss,
                    "source_forecast_errors_at_best_objective": best_forecast_errors,
                    "best_weights": best_weights,
                    "trace": trace,
                }
            )
        np.savez_compressed(cache.with_suffix(".npz"), **saved_arrays)
        result = {
            "episode_id": source_record["episode_id"],
            "source_sha256": source_record["sha256"],
            "arrays_sha256": file_sha256(cache.with_suffix(".npz")),
            "actions": actions,
            "dimensions": context.shape[1],
            "contract_max_absolute_error": contract_error,
            "effective_steps": case_steps,
            "source_baseline_mse": baseline_mse,
            "teacher_forecast_errors": {
                "mae": float(teacher_error.abs().mean()),
                "mse": float(teacher_error.square().mean()),
            },
            "optimization_target": args.optimization_target,
            "role": "source-only fitting diagnostic; per-window weights are not a deployable learned selector; a finite optimization run does not prove a global attainable bound",
            "variants": variants,
        }
        _write_json(cache, result)
        records.append(result)
        print(
            json.dumps(
                {
                    "episode_id": result["episode_id"],
                    "contract_error": contract_error,
                    "objective_losses": [
                        (r["granularity"], r["initial_objective_mse"], r["best_objective_mse"])
                        for r in variants
                    ],
                }
            ),
            flush=True,
        )
    after_digest = parameter_digest(model)
    if before_digest != after_digest or any(
        p.requires_grad or p.grad is not None for p in model.parameters()
    ):
        raise ValueError("the frozen forecasting parameters changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "frozen_parameter_sha256": before_digest,
            "parameter_digest_unchanged": True,
            "records": records,
            "elapsed_seconds": monotonic() - started,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "interpretation": "checks output equivalence, finite gradients, source-fit optimization and frozen parameters; does not establish generalization or method superiority",
        },
    )


if __name__ == "__main__":
    main()
