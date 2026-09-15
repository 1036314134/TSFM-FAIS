"""Compare existing full-source gates with audited family-held predictions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, load_source_inputs, read_json
from train_latent_source_gates import CONDITIONS, condition_features
from train_shared_forecast_gate import predict_weights

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def score(frame, points, truth, teacher, model_id, scope):
    return [
        frame.assign(
            model_id=model_id,
            scope=scope,
            method=method,
            mae=np.abs(point - truth).mean(1),
            mse=np.square(point - truth).mean(1),
            teacher_mae=np.abs(point - teacher).mean(1),
            teacher_mse=np.square(point - teacher).mean(1),
        )
        for method, point in points.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "input-root": "artifacts/iclr27-r7/latent-source-v001",
        "study-root": "artifacts/iclr27-r7/latent-source-gates-v001",
        "audit-root": "artifacts/iclr27-r7/latent-source-audit-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "protocol": "docs/iclr2027/R7_LATENT_GENERALIZATION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed generalization diagnostic")
    study, audit = [
        read_json(root / "manifest.json") for root in (args.study_root, args.audit_root)
    ]
    if (
        study["status"] != "completed"
        or audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.study_root / "manifest.json")
        or study["identity"]["input_manifest_sha256"]
        != file_sha256(args.input_root / "manifest.json")
        or audit["verified_checkpoints"] != 384
    ):
        raise ValueError("the completed source study and audit do not match")
    truth_path = args.accuracy_root / "truth_z.npy"
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        file_sha256(args.accuracy_root / "manifest.json") != study["identity"]["accuracy_sha256"]
        or file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]
    ):
        raise ValueError("the source outcome definitions changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "audit_sha256": file_sha256(args.audit_root / "manifest.json"),
        "modules": {
            name: file_sha256(ROOT / "scripts" / name)
            for name in (
                "latent_source_inputs.py",
                "train_latent_source_gates.py",
                "audit_shared_forecast_gate.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("a partial diagnostic has different definitions")
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, prediction_files = [], []
    maximum_difference, maximum_gap, checked_models = 0.0, 0.0, 0
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_source_inputs(args.input_root, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        if set(frame.iloc[training].origin_id) & set(frame.iloc[validation].origin_id):
            raise ValueError("training and validation histories overlap")
        predictions = {}
        vectors, teacher = arrays["vectors"], arrays["teacher"]
        for condition in CONDITIONS:
            features = condition_features(arrays["features"], condition)
            seed_weights = []
            for seed in (5101, 5102, 5103):
                entry = next(
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id
                    and row["condition"] == condition
                    and row["seed"] == seed
                    and row["held_family"] is None
                )
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a full-source checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["train_ids_sha256"] != hashlib.sha256(training.tobytes()).hexdigest()
                    or saved["identity_sha256"] != study["identity_sha256"]
                    or set(saved["training_origins"]) != set(frame.iloc[training].origin_id)
                    or set(saved["training_families"]) != set(frame.iloc[training].family_id)
                ):
                    raise ValueError("the full-source training population changed")
                model = SharedForecastGate(features=97).eval().requires_grad_(False)
                model.load_state_dict(saved["state_dict"])
                probability = predict_weights(model, features)
                np.testing.assert_array_equal(
                    probability, replay_network(saved["state_dict"], features)
                )
                seed_weights.append(probability)
                predictions[f"{condition}_seed{seed}"] = compose_forecasts(vectors, probability)
                checked_models += 1
            predictions[condition] = compose_forecasts(vectors, np.mean(seed_weights, axis=0))
            for name, probability in [
                (condition, np.mean(seed_weights, axis=0)),
                *[
                    (f"{condition}_seed{seed}", weight)
                    for seed, weight in zip((5101, 5102, 5103), seed_weights, strict=True)
                ],
            ]:
                probability = probability / probability.sum(1, keepdims=True)
                independent = (vectors * probability[:, :, None]).sum(1)
                maximum_difference = max(
                    maximum_difference, float(np.abs(independent - predictions[name]).max())
                )
                np.testing.assert_allclose(independent, predictions[name], atol=1e-12, rtol=1e-12)
        controls_path = args.study_root / model_id / "full_source" / "controls.json"
        controls = read_json(controls_path)
        training_truth = decision_truth(frame.iloc[training], np.load(truth_path, mmap_mode="r"))
        _, _, _, gram = forecast_geometry(vectors[training])
        weight = _family_weights(frame.iloc[training])
        weight = weight / weight.sum()
        g = np.einsum("n,nab->ab", weight, gram)
        for label, target in (("teacher", teacher[training]), ("future", training_truth)):
            alignment = projection_targets(vectors[training], target)["raw_projection"]
            b = np.einsum("n,na->a", weight, alignment)
            fixed = np.asarray(controls[label]["weights"])
            gradient = 2 * (g @ fixed - b)
            gap = float(
                (gradient @ fixed - gradient.min()) / max(abs(g).max(), abs(b).max(), 1e-12)
            )
            if gap > 1e-7 or (fixed < 0).any() or abs(fixed.sum() - 1) > 1e-10:
                raise ValueError("the full-source fixed control fails optimality")
            maximum_gap = max(maximum_gap, gap)
            single = int((g.diagonal() - 2 * b).argmin())
            if single != controls[label]["single_index"]:
                raise ValueError("the full-source single candidate changed")
            predictions[f"fixed_{label}"] = compose_forecasts(
                vectors, np.broadcast_to(fixed, (len(frame), 7))
            )
            predictions[f"single_{label}"] = vectors[:, single]
        predictions["forecast_mean_guarded"] = vectors.mean(1)
        predictions["forecast_median_guarded"] = np.median(vectors, axis=1)
        path = output / f"{model_id}_full_source_predictions.npz"
        _save_npz(
            path, methods=np.asarray(list(predictions)), point=np.stack(list(predictions.values()))
        )
        prediction_files.append(
            {"model_id": model_id, "path": path.name, "sha256": file_sha256(path)}
        )
        _write_json(
            output / f"{model_id}_prediction_freeze.json",
            {
                "predictions": prediction_files[-1],
                "validation_outcomes_read_in_this_model": False,
                "controls_sha256": file_sha256(controls_path),
            },
        )
        truth = decision_truth(frame, np.load(truth_path, mmap_mode="r"))
        for scope, indices in (("source_train", training), ("temporal_validation", validation)):
            records.extend(
                score(
                    frame.iloc[indices],
                    {name: point[indices] for name, point in predictions.items()},
                    truth[indices],
                    teacher[indices],
                    model_id,
                    scope,
                )
            )
        del predictions
        old_scores = pd.read_parquet(args.study_root / "decision_scores.parquet")
        old_scores = old_scores[old_scores.model_id == model_id].set_index(["method", "episode_id"])
        for fold in (row for row in study["folds"] if row["model_id"] == model_id):
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a family-held prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                indices = saved["validation_indices"]
                points = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            current = score(
                frame.iloc[indices],
                points,
                truth[indices],
                teacher[indices],
                model_id,
                "family_held_validation",
            )
            for values in current:
                previous = old_scores.loc[
                    pd.MultiIndex.from_frame(values[["method", "episode_id"]])
                ]
                np.testing.assert_array_equal(previous.mae.to_numpy(), values.mae.to_numpy())
                np.testing.assert_array_equal(previous.mse.to_numpy(), values.mse.to_numpy())
            records.extend(current)
        print(f"{model_id}: source and both validation scopes verified", flush=True)
    scores = pd.concat(records, ignore_index=True)
    metrics = ["mae", "mse", "teacher_mae", "teacher_mse"]
    keys = [
        "model_id",
        "scope",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    episodes = scores.groupby(keys)[metrics].mean().reset_index()
    families = (
        episodes.groupby(["model_id", "scope", "method", "family_id"])[metrics].mean().reset_index()
    )
    summary = families.groupby(["model_id", "scope", "method"])[metrics].mean().reset_index()
    reference = pd.read_csv(args.audit_root / "summary.csv", float_precision="round_trip")
    repeated = summary[summary.scope == "family_held_validation"][reference.columns].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(repeated, reference, check_exact=True)
    if checked_models != 24 or scores.groupby(["model_id", "scope"]).method.nunique().ne(22).any():
        raise ValueError("the prespecified diagnostic is incomplete")
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "verified_checkpoints": checked_models,
            "decision_scores": len(scores),
            "maximum_prediction_difference": maximum_difference,
            "maximum_fixed_optimality_gap": maximum_gap,
            "prediction_files": prediction_files,
            "new_forecaster_calls": 0,
            "new_model_fits": 0,
            "limits": "development diagnostic; training error is descriptive; source populations differ across validation scopes",
        },
    )


if __name__ == "__main__":
    main()
