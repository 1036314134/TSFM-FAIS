"""Compare fixed-size gates over point summaries and frozen projected head inputs."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, load_source_inputs, read_json
from train_shared_forecast_gate import SETTINGS, predict_weights

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts, gate_objective
from tsfm_fais.routing.forecast_projection import (
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

CONDITIONS = ("point_teacher", "latent_teacher", "point_future", "latent_future")


def condition_features(features, condition):
    result = np.array(features, dtype=np.float32, copy=True, order="C")
    if result.ndim != 3 or result.shape[1:] != (7, 97):
        raise ValueError("the common 97-dimensional gate layout changed")
    if condition.startswith("point_"):
        result[:, :, 33:] = 0
    return result


def fit(features, gram, alignment, weights, seed):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97, hidden=16)
    model.fit_normalization(features, weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
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
        total = 0.0
        for indices in order.split(SETTINGS["batch_size"]):
            x, g, b, weight = [value[indices] for value in tensors]
            loss = (gate_objective(model(x), g, b, "ensemble") * weight).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite source-gate loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite source-gate gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "relative_training_loss": total / len(features)})
    return model.eval().requires_grad_(False), history, initial


def aggregate(scores):
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
    families = (
        episodes.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    )
    summary = families.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    return episodes, families, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--accuracy-root", type=Path, default=ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R7_LATENT_SOURCE_PROTOCOL.md"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed source representation experiment")
    collected = read_json(args.input_root / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    truth_path = args.accuracy_root / "truth_z.npy"
    if (
        collected["status"] != "completed"
        or file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]
    ):
        raise ValueError("source collection or original future labels are incomplete")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_module_sha256": file_sha256(ROOT / "scripts/latent_source_inputs.py"),
        "model_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        "input_manifest_sha256": file_sha256(args.input_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "settings": SETTINGS,
        "conditions": list(CONDITIONS),
        "primary": "latent_future",
        "parameters_per_seed": 2120,
        "new_forecaster_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial source-learning definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, initial_parameters = [], [], {}
    for model_id in ("chronos2", "timesfm2p5"):
        source, frame, arrays = load_source_inputs(args.input_root, model_id)
        vectors, features = arrays["vectors"], arrays["features"]
        _, _, _, gram = forecast_geometry(vectors)
        teacher_alignment = projection_targets(vectors, arrays["teacher"])["raw_projection"]
        source_train = np.flatnonzero(frame.split.to_numpy() == "train")
        training_truth = decision_truth(
            frame.iloc[source_train], np.load(truth_path, mmap_mode="r")
        )
        future_alignment = np.full_like(teacher_alignment, np.nan)
        future_alignment[source_train] = projection_targets(vectors[source_train], training_truth)[
            "raw_projection"
        ]
        for family in [None, *sorted(frame.family_id.unique())]:
            train = (
                source_train
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            validation = (
                np.array([], dtype=int)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            training, evaluation = frame.iloc[train], frame.iloc[validation]
            if family in set(training.family_id) or set(training.origin_id) & set(
                evaluation.origin_id
            ):
                raise ValueError("held-family or temporal separation failed")
            weights = _family_weights(training)
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            directory = output / model_id / ("full_source" if family is None else family)
            directory.mkdir(parents=True, exist_ok=True)
            predictions, saved_weights, controls = {}, {}, {}
            for label, alignment in (("teacher", teacher_alignment), ("future", future_alignment)):
                probability = weights / weights.sum()
                g = np.einsum("n,nab->ab", probability, gram[train])
                b = np.einsum("n,na->a", probability, alignment[train])
                fixed, gap, _ = simplex_quadratic_weights(g[None], b[None])
                single = int((g.diagonal() - 2 * b).argmin())
                controls[label] = {
                    "weights": fixed[0].tolist(),
                    "single_index": single,
                    "optimality_gap": float(gap[0]),
                }
                if len(validation):
                    predictions[f"fixed_{label}"] = compose_forecasts(
                        vectors[validation], np.broadcast_to(fixed[0], (len(validation), 7))
                    )
                    predictions[f"single_{label}"] = vectors[validation, single]
            for condition in CONDITIONS:
                values = condition_features(features, condition)
                alignment = teacher_alignment if condition.endswith("teacher") else future_alignment
                condition_weights = []
                for seed in SETTINGS["seeds"]:
                    path = directory / f"{condition}_{seed}.pt"
                    if not path.exists():
                        model, history, initial = fit(
                            values[train], gram[train], alignment[train], weights, seed
                        )
                        temporary = path.with_suffix(".tmp")
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "identity_sha256": identity_sha,
                                "train_ids_sha256": train_sha,
                                "seed": seed,
                                "model_id": model_id,
                                "condition": condition,
                                "held_family": family,
                                "initial_parameter_sha256": initial,
                                "training_history": history,
                                "training_origins": sorted(training.origin_id.unique()),
                                "training_families": sorted(training.family_id.unique()),
                            },
                            temporary,
                        )
                        temporary.replace(path)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != identity_sha
                        or saved["train_ids_sha256"] != train_sha
                        or saved["condition"] != condition
                        or saved["seed"] != seed
                    ):
                        raise ValueError("a partial learned gate belongs to another experiment")
                    initial_parameters.setdefault(str(seed), saved["initial_parameter_sha256"])
                    if initial_parameters[str(seed)] != saved["initial_parameter_sha256"]:
                        raise ValueError("matched-condition initial parameters differ")
                    if len(validation):
                        model = SharedForecastGate(features=97).eval().requires_grad_(False)
                        model.load_state_dict(saved["state_dict"])
                        probability = predict_weights(model, values[validation])
                        np.testing.assert_array_equal(
                            probability, replay_network(saved["state_dict"], values[validation])
                        )
                        condition_weights.append(probability)
                        predictions[f"{condition}_seed{seed}"] = compose_forecasts(
                            vectors[validation], probability
                        )
                    checkpoints.append(
                        {
                            "model_id": model_id,
                            "condition": condition,
                            "held_family": family,
                            "seed": seed,
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                        }
                    )
                if len(validation):
                    saved_weights[condition] = np.stack(condition_weights)
                    predictions[condition] = compose_forecasts(
                        vectors[validation], np.mean(condition_weights, axis=0)
                    )
            _write_json(directory / "controls.json", controls)
            if len(validation):
                predictions["forecast_median_guarded"] = np.median(vectors[validation], axis=1)
                predictions["forecast_mean_guarded"] = vectors[validation].mean(1)
                path = directory / "predictions.npz"
                _save_npz(
                    path,
                    validation_indices=validation,
                    methods=np.asarray(list(predictions)),
                    point=np.stack(list(predictions.values())),
                    **saved_weights,
                )
                folds.append(
                    {
                        "model_id": model_id,
                        "held_family": family,
                        "train_ids_sha256": train_sha,
                        "prediction_path": str(path.relative_to(output)),
                        "prediction_sha256": file_sha256(path),
                        "controls_path": str((directory / "controls.json").relative_to(output)),
                        "controls_sha256": file_sha256(directory / "controls.json"),
                    }
                )
            print(
                f"{model_id} {family or 'full_source'}: four matched conditions saved", flush=True
            )
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "validation_outcomes_read": False,
        },
    )
    scores = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, _ = load_source_inputs(args.input_root, model_id)
        for fold in (row for row in folds if row["model_id"] == model_id):
            path = output / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a frozen validation prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                evaluation = frame.iloc[saved["validation_indices"]]
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    scores.append(
                        evaluation.assign(
                            model_id=model_id,
                            method=method,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(scores, ignore_index=True)
    episodes, families, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if len(checkpoints) != 384 or len(folds) != 30:
        raise ValueError("the registered factorial study is incomplete")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "initial_parameters": initial_parameters,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "single-seed source development; independent audit required before interpreting the four conditions",
        },
    )


if __name__ == "__main__":
    main()
