"""Train small block composers through a frozen Chronos predictor."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.forecasting.adapters.timesfm import TimesFM2p5Adapter  # noqa: E402
from tsfm_fais.forecasting.chronos_differentiable import chronos_median  # noqa: E402
from tsfm_fais.forecasting.timesfm_differentiable import timesfm_median  # noqa: E402
from tsfm_fais.routing.differentiable import ContextBlockComposer, FixedBlockComposer  # noqa: E402
from tsfm_fais.routing.teacher_cache import collect_teacher_records  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def atomic_torch_save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    torch.load(temporary, map_location="cpu", weights_only=True)
    temporary.replace(path)


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def family_errors(frame):
    return (
        frame.groupby(["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level="family_id")
        .mean()
    )


def training_objective(prediction, truth, normalizers, *, teacher=None, teacher_weight=0.0):
    if not 0 <= teacher_weight <= 1:
        raise ValueError("teacher weight must lie in [0,1]")

    def joint(target):
        residual = prediction - target.detach()
        return 0.5 * (
            residual.abs().mean() / normalizers["mae"]
            + residual.square().mean() / normalizers["mse"]
        )

    if teacher_weight == 0:
        return joint(truth)
    if teacher is None:
        raise ValueError("a positive teacher weight requires a detached prediction teacher")
    if teacher_weight == 1:
        return joint(teacher)
    return (1 - teacher_weight) * joint(truth) + teacher_weight * joint(teacher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--accuracy-root", required=True, type=Path)
    parser.add_argument("--screening-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-family", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--forecaster-features", action="store_true")
    parser.add_argument("--target-conditioned", action="store_true")
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), default="chronos2")
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10 or not 1 <= args.per_family <= 96:
        parser.error("initial development runs allow 1-10 epochs and 1-96 tasks per family")
    if args.learning_rate <= 0 or args.regularization < 0:
        parser.error("learning rate must be positive and regularization nonnegative")
    if args.model == "timesfm2p5" and args.forecaster_features:
        parser.error("predictor-patch features are currently implemented only for Chronos")
    if not 0 <= args.teacher_weight <= 1:
        parser.error("teacher weight must lie in [0,1]")
    if args.teacher_weight and args.model != "chronos2":
        parser.error("native-input prediction teachers are currently enabled only for Chronos")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_root, accuracy_root, output = (
        args.source_root.resolve(),
        args.accuracy_root.resolve(),
        args.output_root.resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("training run already completed; preserve its evidence")
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    screening = json.loads(args.screening_plan.read_text(encoding="utf-8"))
    source_sha = file_sha256(source_root / "episodes_manifest.json")
    if (
        accuracy["source_episode_manifest_sha256"] != source_sha
        or screening["source_manifest_sha256"] != source_sha
    ):
        raise ValueError("training and evaluation must share the fixed source protocol")
    config = source["identity"]["config"]
    actions = config["candidate_ids"]
    records = {record["episode_id"]: record for record in source["episodes"]}
    by_family = defaultdict(list)
    for record in source["episodes"]:
        if record["split"] == "train":
            by_family[record["family_id"]].append(record["episode_id"])
    generator = np.random.default_rng(args.seed)
    training_ids = []
    for family in sorted(by_family):
        pool = sorted(by_family[family])
        positions = generator.choice(len(pool), min(args.per_family, len(pool)), replace=False)
        training_ids.extend(pool[int(index)] for index in sorted(positions))
    validation_ids = list(screening["decision_episode_ids"])
    if any(records[item]["split"] != "validation" for item in validation_ids):
        raise ValueError("evaluation must contain validation episodes only")
    train_origins = {records[item]["origin_id"] for item in training_ids}
    validation_origins = {records[item]["origin_id"] for item in validation_ids}
    if train_origins & validation_origins:
        raise ValueError("training and validation origins overlap")
    for family in by_family:
        earliest_evaluation = min(
            records[item]["origin"]
            for item in validation_ids
            if records[item]["family_id"] == family
        )
        if (
            max(
                records[item]["origin"] + config["horizon"]
                for item in training_ids
                if records[item]["family_id"] == family
            )
            > earliest_evaluation - config["context_length"]
        ):
            raise ValueError("source supervision overlaps an evaluation context")
    module_path = ROOT / "src/tsfm_fais/routing/differentiable.py"
    identity = {
        "source_manifest_sha256": source_sha,
        "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
        "screening_plan_sha256": file_sha256(args.screening_plan),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(module_path),
        "teacher_cache_module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/teacher_cache.py"),
        "checkpoint": config["forecaster_artifacts"][args.model],
        "model_id": args.model,
        "chronos_gradient_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/chronos_differentiable.py"
        ),
        "timesfm_gradient_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/timesfm_differentiable.py"
        )
        if args.model == "timesfm2p5"
        else None,
        "seed": args.seed,
        "epochs": args.epochs,
        "per_family": args.per_family,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "forecaster_features": args.forecaster_features,
        "target_conditioned": args.target_conditioned,
        "teacher_weight": args.teacher_weight,
        "teacher": "median of six finite candidate forecasts plus guarded native input; source training episodes only"
        if args.teacher_weight
        else None,
        "training_ids": training_ids,
        "validation_ids": validation_ids,
        "supervision": "complete earlier source outcomes; temporal development, not independent confirmation",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("training identity changed; use a new output directory")
    _write_json(identity_path, identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "composer_snapshot.py").write_bytes(module_path.read_bytes())
    labels = pd.read_parquet(
        accuracy_root / "candidate_accuracy.parquet",
        columns=["episode_id", "family_id", "dataset_id", "candidate_id", "mae", "mse"],
        filters=[
            ("model_id", "==", args.model),
            ("split", "==", "train"),
            ("target_slot", "==", -1),
            ("episode_id", "in", training_ids),
        ],
    )
    labels = labels[labels.candidate_id.isin(actions)]
    if (
        len(labels) != len(training_ids) * len(actions)
        or labels.duplicated(["episode_id", "candidate_id"]).any()
    ):
        raise ValueError(
            "source prior requires the full candidate pool for every sampled training episode"
        )
    normalizers = family_errors(labels[labels.candidate_id == "locf"]).mean().to_dict()
    if not all(np.isfinite(value) and value > 0 for value in normalizers.values()):
        raise ValueError("training loss normalizers must be finite and positive")
    training_scores = {
        action: family_errors(labels[labels.candidate_id == action]).mean().to_dict()
        for action in actions
    }
    prior_action = min(
        actions,
        key=lambda name: sum(
            training_scores[name][metric] / normalizers[metric] for metric in ("mae", "mse")
        ),
    )
    prior = np.full(len(actions), 0.1 / (len(actions) - 1))
    prior[actions.index(prior_action)] = 0.9
    _write_json(
        output / "training_prior.json",
        {
            "action": prior_action,
            "probabilities": prior.tolist(),
            "normalizers": normalizers,
            "candidate_training_errors": training_scores,
        },
    )
    scalers = {
        (item["dataset_id"], item["item_id"]): item
        for item in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
    }
    cache = {}

    def load_episode(episode_id):
        if episode_id not in cache:
            record = records[episode_id]
            path = source_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("source candidate episode changed")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            with np.load(path, allow_pickle=False) as episode:
                if episode["candidate_ids"].tolist() != actions:
                    raise ValueError("candidate ordering changed")
                context = torch.tensor((episode["context"] - mean) / scale, dtype=torch.float32)
                candidates = torch.tensor(
                    (episode["candidate_values"] - mean) / scale, dtype=torch.float32
                )
                truth = torch.tensor(
                    ((episode["future"] - mean) / scale)[:, list(config["target_indices"])],
                    dtype=torch.float64,
                )
            if not bool(torch.isfinite(candidates).all()) or not bool(torch.isfinite(truth).all()):
                raise ValueError(
                    "source training requires finite completions and complete outcomes"
                )
            known = torch.isfinite(context)
            if not torch.equal(candidates[:, known], context[known].expand(len(actions), -1)):
                raise ValueError("source completions changed known observations")
            cache[episode_id] = (candidates, context, truth)
        return tuple(value.to("cuda") for value in cache[episode_id])

    if args.model == "chronos2":
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(identity["checkpoint"], device_map="cuda")
        median_indices = np.flatnonzero(np.isclose(pipeline.quantiles, 0.5))
        if len(median_indices) != 1:
            raise ValueError("a unique median forecast is required")
    else:
        adapter = TimesFM2p5Adapter(model_name=identity["checkpoint"], device="cuda", batch_size=2)
        pipeline = adapter._ensure_backend()
    forecaster = pipeline.model.eval().requires_grad_(False)
    forecaster_digest = parameter_digest(forecaster)
    horizon, targets = config["horizon"], list(config["target_indices"])
    if args.model == "timesfm2p5" and horizon > forecaster.o:
        raise ValueError("the verified TimesFM gradient path covers one output patch")
    models = {
        "fixed": FixedBlockComposer(prior).cuda(),
        "adaptive": ContextBlockComposer(
            prior,
            forecaster_dim=forecaster.model_dim if args.forecaster_features else None,
            target_conditioned=args.target_conditioned,
        ).cuda(),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
        for name, model in models.items()
    }
    checkpoint_path = output / "checkpoint.pt"
    state = {
        "epoch": 1,
        "next_index": 0,
        "initial_evaluation_done": False,
        "best": {},
        "epoch_summaries": [],
        "training_compute_seconds": 0.0,
        "forecast_calls": 0,
        "gradient_updates": 0,
        "feature_passes": 0,
        "feature_compute_seconds": 0.0,
        "teacher_forecast_calls": 0,
        "teacher_compute_seconds": 0.0,
        "teacher_episode_ids": [],
    }
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            saved["identity_sha256"] != file_sha256(identity_path)
            or saved["forecaster_digest"] != forecaster_digest
        ):
            raise ValueError("checkpoint belongs to another protocol or forecasting model")
        state = saved["state"]
        for name in models:
            models[name].load_state_dict(saved["models"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])

    def checkpoint():
        atomic_torch_save(
            checkpoint_path,
            {
                "identity_sha256": file_sha256(identity_path),
                "forecaster_digest": forecaster_digest,
                "state": state,
                "models": {name: cpu_state(model) for name, model in models.items()},
                "optimizers": {
                    name: optimizer.state_dict() for name, optimizer in optimizers.items()
                },
            },
        )

    def forecast(context):
        state["forecast_calls"] += 1
        if args.model == "timesfm2p5":
            return timesfm_median(forecaster, context, horizon, targets)
        return chronos_median(pipeline, context, horizon, targets)

    teacher_cache = {}
    teacher_root = output / "teacher-cache"
    if args.teacher_weight:
        teacher_root.mkdir(exist_ok=True)
    teacher_identity = file_sha256(identity_path)
    training_set = set(training_ids)
    teacher_seen = set(state["teacher_episode_ids"])

    def prediction_teacher(episode_id, candidates, context):
        if not args.teacher_weight or bool(torch.isfinite(context).all()):
            return None
        if episode_id not in training_set:
            raise ValueError("prediction teachers must not use validation episodes")
        if episode_id not in teacher_seen:
            teacher_seen.add(episode_id)
            state["teacher_episode_ids"].append(episode_id)
        if episode_id not in teacher_cache:
            path = teacher_root / (hashlib.sha256(episode_id.encode()).hexdigest()[:24] + ".pt")
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != teacher_identity
                    or saved["episode_id"] != episode_id
                    or saved["forecaster_digest"] != forecaster_digest
                ):
                    raise ValueError("cached teacher belongs to a different training protocol")
                value = saved["prediction"]
            else:
                torch.cuda.synchronize()
                beginning = monotonic()
                with torch.no_grad():
                    native_context = (
                        candidates[actions.index("locf")]
                        if bool((~torch.isfinite(context).any(0)).any())
                        else context
                    )
                    point = torch.stack(
                        [forecast(values) for values in [*candidates, native_context]]
                    )
                    value = point.quantile(0.5, dim=0).float().cpu()
                torch.cuda.synchronize()
                state["teacher_compute_seconds"] += monotonic() - beginning
                state["teacher_forecast_calls"] += len(actions) + 1
                atomic_torch_save(
                    path,
                    {
                        "identity_sha256": teacher_identity,
                        "episode_id": episode_id,
                        "forecaster_digest": forecaster_digest,
                        "prediction": value,
                    },
                )
            if value.shape != (horizon, len(targets)) or not bool(torch.isfinite(value).all()):
                raise ValueError("cached teacher must be a finite [H,K] prediction")
            teacher_cache[episode_id] = value
        return teacher_cache[episode_id].to("cuda")

    feature_cache = {}

    def predictor_features(episode_id, candidates, context):
        if not args.forecaster_features or bool(torch.isfinite(context).all()):
            return None
        if episode_id not in feature_cache:
            torch.cuda.synchronize()
            beginning = monotonic()
            with torch.no_grad():
                count, length, dimensions = candidates.shape
                flattened = candidates.transpose(1, 2).reshape(count * dimensions, length)
                patches, _, _ = forecaster._prepare_patched_context(flattened)
                embedding = forecaster.input_patch_embedding(patches)
                feature_cache[episode_id] = (
                    embedding.reshape(count, dimensions, embedding.shape[1], -1).float().cpu()
                )
            torch.cuda.synchronize()
            state["feature_compute_seconds"] += monotonic() - beginning
            state["feature_passes"] += 1
        return feature_cache[episode_id].to("cuda")

    def compose(name, candidates, context, record, patches):
        if name == "adaptive":
            return models[name](
                candidates,
                context,
                record["period"],
                forecaster_patches=patches if args.forecaster_features else None,
                targets=targets if args.target_conditioned else None,
            )
        return models[name](candidates, context, record["period"])

    def errors(prediction, truth):
        residual = prediction - truth
        mae, mse = residual.abs().mean(), residual.square().mean()
        objective = 0.5 * (mae / normalizers["mae"] + mse / normalizers["mse"])
        if not bool(torch.isfinite(objective)):
            raise ValueError("nonfinite forecast objective")
        return mae, mse, objective

    def evaluate(epoch):
        rows = []
        for model in models.values():
            model.eval()
        with torch.no_grad():
            for episode_id in validation_ids:
                candidates, context, truth = load_episode(episode_id)
                record = records[episode_id]
                patches = predictor_features(episode_id, candidates, context)
                for name in models:
                    composition = compose(name, candidates, context, record, patches)
                    mae, mse, _ = errors(forecast(composition.values), truth)
                    diagnostics = {
                        "mean_action_weights": None,
                        "weight_entropy": None,
                        "prior_top1_fraction": None,
                    }
                    if composition.block_count:
                        valid_ids = composition.block_ids[composition.block_ids >= 0]
                        lengths = torch.bincount(valid_ids, minlength=composition.block_count).to(
                            composition.weights.dtype
                        )
                        mass = lengths / lengths.sum()
                        mean_weights = composition.weights @ mass
                        entropy = -(
                            composition.weights * composition.weights.clamp_min(1e-12).log()
                        ).sum(0)
                        diagnostics = {
                            "mean_action_weights": json.dumps(mean_weights.cpu().tolist()),
                            "weight_entropy": float(
                                (entropy * mass).sum() / math.log(len(actions))
                            ),
                            "prior_top1_fraction": float(
                                (
                                    (
                                        composition.weights.argmax(0) == actions.index(prior_action)
                                    ).to(mass.dtype)
                                    * mass
                                ).sum()
                            ),
                        }
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "origin_id": record["origin_id"],
                            "family_id": record["family_id"],
                            "dataset_id": record["dataset_id"],
                            "model_id": args.model,
                            "method": name,
                            "epoch": epoch,
                            "mae": float(mae),
                            "mse": float(mse),
                            "block_count": composition.block_count,
                            **diagnostics,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.to_parquet(output / f"epoch-{epoch:02d}-validation.parquet", index=False)
        summary = []
        for name in models:
            metrics = family_errors(frame[frame.method == name]).mean().to_dict()
            objective = 0.5 * sum(
                metrics[metric] / normalizers[metric] for metric in ("mae", "mse")
            )
            summary.append(
                {"method": name, "epoch": epoch, **metrics, "joint_objective": objective}
            )
            if name not in state["best"] or objective < state["best"][name]["joint_objective"]:
                state["best"][name] = summary[-1]
                atomic_torch_save(
                    output / f"best-{name}.pt",
                    {
                        "model": cpu_state(models[name]),
                        "summary": summary[-1],
                        "identity_sha256": file_sha256(identity_path),
                    },
                )
        state["epoch_summaries"].extend(summary)
        pd.DataFrame(state["epoch_summaries"]).to_csv(output / "epoch_summary.csv", index=False)
        print(json.dumps({"phase": "validation", "epoch": epoch, "results": summary}), flush=True)

    if not state["initial_evaluation_done"]:
        evaluate(0)
        state["initial_evaluation_done"] = True
        checkpoint()
    started = monotonic()
    while state["epoch"] <= args.epochs:
        epoch = state["epoch"]
        order = np.random.default_rng(args.seed + epoch * 1001).permutation(len(training_ids))
        for model in models.values():
            model.train()
        for index in range(state["next_index"], len(order)):
            episode_id = training_ids[int(order[index])]
            try:
                candidates, context, truth = load_episode(episode_id)
                patches = predictor_features(episode_id, candidates, context)
                teacher = prediction_teacher(episode_id, candidates, context)
                torch.cuda.synchronize()
                beginning = monotonic()
                for name, model in models.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    composition = compose(name, candidates, context, records[episode_id], patches)
                    if composition.block_count:
                        prediction = forecast(composition.values)
                        objective = training_objective(
                            prediction,
                            truth,
                            normalizers,
                            teacher=teacher,
                            teacher_weight=args.teacher_weight,
                        )
                        if not bool(torch.isfinite(objective)):
                            raise ValueError("nonfinite blended training objective")
                        (objective + args.regularization * composition.penalty).backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 5.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        state["gradient_updates"] += 1
                torch.cuda.synchronize()
                state["training_compute_seconds"] += monotonic() - beginning
            except Exception as error:
                _write_json(
                    output / "failure.json",
                    {
                        "epoch": epoch,
                        "index": index,
                        "episode_id": episode_id,
                        "error": f"{type(error).__name__}: {error}",
                        "resume": "last complete checkpoint; incomplete batch is not committed",
                    },
                )
                raise
            state["next_index"] = index + 1
            if (index + 1) % 20 == 0 or index + 1 == len(order):
                checkpoint()
                progress = {
                    "phase": "training",
                    "epoch": epoch,
                    "completed": index + 1,
                    "total": len(order),
                    "training_compute_seconds": state["training_compute_seconds"],
                    "gradient_updates": state["gradient_updates"],
                    "teacher_forecast_calls": state["teacher_forecast_calls"],
                    "teacher_compute_seconds": state["teacher_compute_seconds"],
                }
                _write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        evaluate(epoch)
        state["epoch"], state["next_index"] = epoch + 1, 0
        checkpoint()
    final_digest = parameter_digest(forecaster)
    if forecaster_digest != final_digest or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in forecaster.parameters()
    ):
        raise ValueError("frozen forecasting parameters changed")
    final_rows = []
    for name, best in state["best"].items():
        frame = pd.read_parquet(output / f"epoch-{best['epoch']:02d}-validation.parquet")
        final_rows.append(frame[frame.method == name])
    pd.concat(final_rows).to_parquet(output / "selected_validation.parquet", index=False)
    teacher_records = (
        collect_teacher_records(
            teacher_root, teacher_seen, teacher_identity, forecaster_digest, (horizon, len(targets))
        )
        if args.teacher_weight
        else []
    )
    if args.teacher_weight:
        _write_json(
            output / "teacher_manifest.json",
            {
                "identity_sha256": teacher_identity,
                "forecaster_digest": forecaster_digest,
                "records": teacher_records,
                "logical_forecast_calls": len(teacher_records) * (len(actions) + 1),
            },
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "best_epochs": state["best"],
            "training_compute_seconds": state["training_compute_seconds"],
            "elapsed_seconds_this_execution": monotonic() - started,
            "forecast_calls": state["forecast_calls"],
            "full_backbone_passes_per_forecast": 2 if args.model == "timesfm2p5" else 1,
            "source_prior_label_queries": len(training_ids) * len(actions),
            "cost_scope": "checkpointed optimization, validation and teacher calls; teacher time is separate from optimization time; source-prior and teacher-cache logical queries are separately reported; imputer preparation and discarded interrupted work require separate end-to-end accounting",
            "gradient_updates": state["gradient_updates"],
            "feature_passes": state["feature_passes"],
            "feature_compute_seconds": state["feature_compute_seconds"],
            "teacher_weight": args.teacher_weight,
            "teacher_checkpointed_forecast_calls": state["teacher_forecast_calls"],
            "teacher_compute_seconds": state["teacher_compute_seconds"],
            "teacher_cached_episodes": len(teacher_records),
            "teacher_logical_forecast_calls": len(teacher_records) * (len(actions) + 1),
            "teacher_manifest_sha256": file_sha256(output / "teacher_manifest.json")
            if args.teacher_weight
            else None,
            "parameter_digest_unchanged": True,
            "forecaster_parameter_sha256": forecaster_digest,
            "sdk_model_sha256": file_sha256(Path(inspect.getfile(type(forecaster)))),
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "parameter_counts": {
                name: sum(parameter.numel() for parameter in model.parameters())
                for name, model in models.items()
            },
            "interpretation": "validation selects checkpoints and is development evidence; all future labels remain outside the deployed composer; each composed context needs one full forecasting pass; optional frozen input-patch embedding computation and imputer cost are additional",
        },
    )


if __name__ == "__main__":
    main()
