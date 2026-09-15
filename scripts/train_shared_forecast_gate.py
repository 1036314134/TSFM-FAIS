"""Compare matched small forecast gates using cached source-only teacher supervision."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402

from tsfm_fais.routing.forecast_gate import (  # noqa: E402
    SharedForecastGate,
    compose_forecasts,
    gate_objective,
    teacher_quadratics,
)
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402

SETTINGS = {
    "epochs": 25,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "gradient_norm": 1.0,
    "seeds": [5101, 5102, 5103],
    "hidden": 16,
}


def fit_gate(features, gram, alignment, weights, *, kind, seed):
    torch.manual_seed(seed)
    model = SharedForecastGate(hidden=SETTINGS["hidden"])
    model.fit_normalization(features, weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    tensors = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features, gram, alignment, weights)
    ]
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(SETTINGS["epochs"]):
        order = torch.randperm(len(features), generator=generator)
        total, count = 0.0, 0
        for indices in order.split(SETTINGS["batch_size"]):
            x, g, b, family = [value[indices] for value in tensors]
            probability = model(x)
            loss = (gate_objective(probability, g, b, kind) * family).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite shared-gate training objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite shared-gate gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        history.append(
            {"epoch": epoch + 1, "mean_training_batch_relative_objective": total / count}
        )
    model.eval()
    return model, history


def predict_weights(model, features):
    with torch.no_grad():
        result = (
            torch.cat(
                [
                    model(batch)
                    for batch in torch.as_tensor(features, dtype=torch.float32).split(512)
                ]
            )
            .numpy()
            .astype(float)
        )
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("invalid shared-gate mixture weights")
    return result / result.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned-root", "accuracy-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--objective", choices=("ensemble", "member"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.objective
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed shared-gate studies")
    prep = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if prep["status"] != "completed" or prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("the audited source feature and forecast banks must match")
    for name, sha in prep["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError("source feature definitions changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "settings": SETTINGS,
        "objective": args.objective,
        "protocol_sha256": file_sha256(args.protocol),
        "aligned_manifest_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/forecast_gate.py",
                "src/tsfm_fais/routing/forecast_projection.py",
                "src/tsfm_fais/routing/utility.py",
                "scripts/aligned_portfolio_io.py",
            )
        },
        "source_outcome_supervision": False,
        "current_query_budget": 7,
        "primary_seed_rule": "average the three prespecified weight vectors",
        "trainable_parameters_per_seed": 1096,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial gate study identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folds, results = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        if info["feature_names"][:33] != list(FORECAST_FEATURES):
            raise ValueError("the candidate feature order changed")
        features = arrays["features"][:, :7, :33]
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("the source forecast bank changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][:, :7])
        for family in sorted(decisions.family_id.unique()):
            key = hashlib.sha256((model_id + "|" + family).encode()).hexdigest()[:20]
            directory = output / "folds" / key
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "manifest.json"
            train_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "train") & (decisions.family_id.to_numpy() != family)
            )
            eval_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "validation")
                & (decisions.family_id.to_numpy() == family)
            )
            training, evaluation = decisions.iloc[train_ids], decisions.iloc[eval_ids]
            if set(training.origin_id) & set(evaluation.origin_id) or family in set(
                training.family_id
            ):
                raise ValueError("the held-family boundary was violated")
            if marker.exists():
                fold = json.loads(marker.read_text(encoding="utf-8"))
                if fold["identity_sha256"] != identity_sha:
                    raise ValueError("a completed gate fold has another identity")
                for row in fold["files"]:
                    if file_sha256(output / row["path"]) != row["sha256"]:
                        raise ValueError("a completed gate fold artifact changed")
            else:
                started = monotonic()
                sample_weights = _family_weights(training)
                seed_weights, files, timings = [], [], []
                train_ids_sha = hashlib.sha256(train_ids.tobytes()).hexdigest()
                for seed in SETTINGS["seeds"]:
                    path = directory / f"seed_{seed}.pt"
                    if path.exists():
                        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            checkpoint["identity_sha256"] != identity_sha
                            or checkpoint["train_ids_sha256"] != train_ids_sha
                            or checkpoint["seed"] != seed
                        ):
                            raise ValueError("a partial seed checkpoint changed identity")
                        model = SharedForecastGate(hidden=SETTINGS["hidden"])
                        model.load_state_dict(checkpoint["state_dict"])
                        model.eval()
                    else:
                        seed_start = monotonic()
                        model, history = fit_gate(
                            features[train_ids],
                            gram[train_ids],
                            alignment[train_ids],
                            sample_weights,
                            kind=args.objective,
                            seed=seed,
                        )
                        checkpoint = {
                            "identity_sha256": identity_sha,
                            "train_ids_sha256": train_ids_sha,
                            "seed": seed,
                            "state_dict": model.state_dict(),
                            "training_history": history,
                            "training_seconds": monotonic() - seed_start,
                        }
                        temporary = path.with_suffix(".tmp")
                        torch.save(checkpoint, temporary)
                        temporary.replace(path)
                    if sum(parameter.numel() for parameter in model.parameters()) != 1096:
                        raise ValueError("the shared-gate capacity changed")
                    seed_weights.append(predict_weights(model, features[eval_ids]))
                    timings.append(checkpoint["training_seconds"])
                    files.append(
                        {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
                    )
                probability = sample_weights / sample_weights.sum()
                mean_gram = np.einsum("n,nab->ab", probability, gram[train_ids])
                mean_alignment = np.einsum("n,na->a", probability, alignment[train_ids])
                fixed, certificate, _ = simplex_quadratic_weights(
                    mean_gram[None], mean_alignment[None]
                )
                fixed_single = int((mean_gram.diagonal() - 2 * mean_alignment).argmin())
                averaged = np.mean(seed_weights, axis=0)
                predictions = {
                    args.objective + "_gate": compose_forecasts(vectors[eval_ids], averaged),
                    "source_fixed_convex": compose_forecasts(
                        vectors[eval_ids], np.repeat(fixed, len(eval_ids), axis=0)
                    ),
                    "source_fixed_single": vectors[eval_ids, fixed_single],
                    "forecast_mean_guarded": vectors[eval_ids].mean(axis=1),
                    "forecast_median_guarded": np.median(vectors[eval_ids], axis=1),
                }
                for seed, weights in zip(SETTINGS["seeds"], seed_weights, strict=True):
                    predictions[f"{args.objective}_seed{seed}"] = compose_forecasts(
                        vectors[eval_ids], weights
                    )
                pred_path = directory / "predictions.npz"
                _save_npz(
                    pred_path,
                    decision_indices=eval_ids,
                    mean_weights=averaged,
                    seed_weights=np.stack(seed_weights),
                    fixed_weights=fixed,
                    fixed_single=np.asarray(fixed_single),
                    point=np.stack(list(predictions.values())),
                    methods=np.asarray(list(predictions)),
                    identity_sha256=np.asarray(identity_sha),
                )
                # All learned parameters and output choices are saved before validation futures are read.
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("validation outcome bank changed")
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                rows = []
                for name, point in predictions.items():
                    rows.append(
                        evaluation.assign(
                            model_id=model_id,
                            method=name,
                            mae=abs(point - truth).mean(axis=1),
                            mse=((point - truth) ** 2).mean(axis=1),
                        )
                    )
                score_path = directory / "scores.parquet"
                pd.concat(rows, ignore_index=True).to_parquet(score_path, index=False)
                files.extend(
                    {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
                    for path in (pred_path, score_path)
                )
                fold = {
                    "status": "completed",
                    "identity_sha256": identity_sha,
                    "model_id": model_id,
                    "held_family": family,
                    "training_origins": sorted(training.origin_id.unique()),
                    "training_families": sorted(training.family_id.unique()),
                    "train_ids_sha256": train_ids_sha,
                    "files": files,
                    "seed_training_seconds": timings,
                    "fixed_simplex_optimality_gap": float(certificate[0]),
                    "seconds": monotonic() - started,
                    "scores_path": str(score_path.relative_to(output)),
                    "predictions_path": str(pred_path.relative_to(output)),
                }
                _write_json(marker, fold)
            results.append(pd.read_parquet(output / fold["scores_path"]))
            folds.append({"path": str(marker.relative_to(output)), "sha256": file_sha256(marker)})
            _write_json(
                output / "progress.json",
                {"status": "fitting", "completed_folds": len(folds), "total_folds": 30},
            )
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "family": family,
                        "objective": args.objective,
                        "completed_folds": len(folds),
                        "seconds": fold["seconds"],
                    }
                ),
                flush=True,
            )
    scores = pd.concat(results, ignore_index=True)
    keys = [
        "model_id",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    episodes = scores.groupby(keys)[["mae", "mse"]].mean().reset_index()
    for _, group in episodes.groupby(["model_id", "method"]):
        if len(group) != 1872 or group.source_episode_id.nunique() != 1872:
            raise ValueError("shared-gate development coverage changed")
    episodes.to_parquet(output / "episode_results.parquet", index=False)
    family = (
        episodes.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    )
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": folds,
            "new_forecaster_calls": 0,
            "summary": summary.to_dict("records"),
            "limits": "source-development experiment only; previously examined confirmation cohorts are not reused as fresh validation",
        },
    )


if __name__ == "__main__":
    main()
