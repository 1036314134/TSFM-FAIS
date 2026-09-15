"""Replay matched geometry features, normalization, models and source validation scores."""

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
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from geometry_forecast_gate import geometry_features  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "aligned-root", "accuracy-root", "reference-study", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed geometry audits")
    study, prep, accuracy = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    if (
        study["status"] != "completed"
        or len(study["checkpoints"]) != 192
        or len(study["source_models"]) != 12
        or len(study["folds"]) != 60
    ):
        raise ValueError("complete both matched geometry conditions")
    if (
        study["identity"]["aligned_sha256"] != file_sha256(args.aligned_root / "manifest.json")
        or study["identity"]["accuracy_sha256"] != file_sha256(args.accuracy_root / "manifest.json")
        or study["identity"]["geometry_module_sha256"]
        != file_sha256(ROOT / "scripts/geometry_forecast_gate.py")
    ):
        raise ValueError("the geometry study inputs or feature definition changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("the source validation labels changed")
    torch.set_num_threads(1)
    results, count, maximum_feature_difference, maximum_prediction_difference = [], 0, 0.0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("source prediction vectors changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        centered = vectors - np.median(vectors, axis=1)[:, None]
        direct_gram = (centered @ centered.transpose(0, 2, 1)) / vectors.shape[2]
        feature_sets = {
            mode: geometry_features(arrays["features"][:, :7, :33], vectors, mode=mode)
            for mode in ("full", "diagonal")
        }
        for mode, features in feature_sets.items():
            expected_gram = direct_gram if mode == "full" else direct_gram * np.eye(7)[None]
            expected = np.concatenate(
                [
                    arrays["features"][:, :7, :33],
                    np.broadcast_to(np.eye(7), (len(vectors), 7, 7)),
                    np.sign(expected_gram) * np.log1p(abs(expected_gram)),
                ],
                axis=2,
            ).astype(np.float32)
            maximum_feature_difference = max(
                maximum_feature_difference, float(abs(features - expected).max())
            )
            np.testing.assert_allclose(features, expected, rtol=1e-7, atol=1e-7)
        for family in [*sorted(decisions.family_id.unique()), None]:
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train")
                & ((decisions.family_id.to_numpy() != family) if family is not None else True)
            )
            validation = (
                np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                if family is not None
                else np.array([], dtype=int)
            )
            training, evaluation = decisions.iloc[train], decisions.iloc[validation]
            if set(training.origin_id) & set(evaluation.origin_id) or (
                family is not None and family in set(training.family_id)
            ):
                raise ValueError("source temporal or family separation changed")
            family_weights = _family_weights(training)
            shared_normalization = {}
            for mode, features in feature_sets.items():
                values = features[train].astype(float)
                denominator = family_weights.sum() * 7
                mean = (values * family_weights[:, None, None]).sum((0, 1)) / denominator
                variance = ((values - mean) ** 2 * family_weights[:, None, None]).sum(
                    (0, 1)
                ) / denominator
                scale = np.maximum(np.sqrt(variance), 1e-6)
                entries = [
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_family"] == family
                    and row["mode"] == mode
                ]
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("matched seeds are incomplete")
                seed_weights = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"] or entry["actions"] != info["actions"]:
                        raise ValueError("a geometry checkpoint or candidate identity changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != study["identity_sha256"]
                        or saved["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                    ):
                        raise ValueError("a geometry checkpoint has different fitting rows")
                    if set(saved["training_origins"]) != set(training.origin_id) or set(
                        saved["training_families"]
                    ) != set(training.family_id):
                        raise ValueError("a saved model's training population changed")
                    if (
                        saved["initial_parameter_sha256"]
                        != study["initial_parameter_sha256"][str(entry["seed"])]
                    ):
                        raise ValueError("matched initialization changed")
                    state = saved["state_dict"]
                    model = SharedForecastGate(features=47)
                    model.load_state_dict(state)
                    if sum(value.numel() for value in model.parameters()) != 1320:
                        raise ValueError("matched model capacity changed")
                    for name, expected in (("feature_mean", mean), ("feature_scale", scale)):
                        np.testing.assert_array_equal(
                            state[name].numpy().reshape(-1), expected.astype(np.float32)
                        )
                        key = (entry["seed"], name)
                        if mode == "full":
                            shared_normalization[key] = state[name][:, :, :40]
                        elif not torch.equal(shared_normalization[key], state[name][:, :, :40]):
                            raise ValueError("matched base or identity normalization changed")
                    if len(validation):
                        seed_weights.append(replay_network(state, features[validation]))
                    count += 1
                if family is None:
                    continue
                fold = next(
                    row
                    for row in study["folds"]
                    if row["model_id"] == model_id
                    and row["family_id"] == family
                    and row["mode"] == mode
                )
                path = args.study_root / fold["prediction_path"]
                if file_sha256(path) != fold["prediction_sha256"]:
                    raise ValueError("saved geometry predictions changed")
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    np.testing.assert_array_equal(saved["seed_weights"], np.stack(seed_weights))
                    average = np.mean(seed_weights, axis=0)
                    np.testing.assert_array_equal(saved["mean_weights"], average)
                    predictions = {
                        f"geometry_{mode}_gate": compose_forecasts(vectors[validation], average),
                        "forecast_median_guarded": np.median(vectors[validation], axis=1),
                    }
                    for seed, weights in zip((5101, 5102, 5103), seed_weights, strict=True):
                        predictions[f"geometry_{mode}_seed{seed}"] = compose_forecasts(
                            vectors[validation], weights
                        )
                    if saved["methods"].tolist() != list(predictions):
                        raise ValueError("source output definitions changed")
                    maximum_prediction_difference = max(
                        maximum_prediction_difference,
                        float(abs(np.stack(list(predictions.values())) - saved["point"]).max()),
                    )
                    np.testing.assert_array_equal(
                        np.stack(list(predictions.values())), saved["point"]
                    )
                score_path = args.study_root / fold["scores_path"]
                if file_sha256(score_path) != fold["scores_sha256"]:
                    raise ValueError("source score records changed")
                recorded = pd.read_parquet(score_path)
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                for name, point in predictions.items():
                    new = evaluation.assign(
                        model_id=model_id,
                        mode=mode,
                        method=name,
                        mae=abs(point - truth).mean(1),
                        mse=((point - truth) ** 2).mean(1),
                    )
                    old = (
                        recorded[recorded.method == name]
                        .set_index("episode_id")
                        .loc[new.episode_id]
                    )
                    np.testing.assert_array_equal(old.mae.to_numpy(), new.mae.to_numpy())
                    np.testing.assert_array_equal(old.mse.to_numpy(), new.mse.to_numpy())
                    results.append(new)
    frame = pd.concat(results, ignore_index=True)
    keys = [
        "model_id",
        "mode",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    episodes = frame.groupby(keys)[["mae", "mse"]].mean().reset_index()
    families = (
        episodes.groupby(["model_id", "mode", "method", "family_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary = families.groupby(["model_id", "mode", "method"])[["mae", "mse"]].mean().reset_index()
    expected = pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(summary, expected, check_exact=True)
    original = pd.read_csv(
        args.reference_study / "family_metrics.csv", float_precision="round_trip"
    )
    comparisons = []
    for model_id in ("chronos2", "timesfm2p5"):
        current = families[
            (families.model_id == model_id) & (families.method == "geometry_full_gate")
        ].set_index("family_id")[["mae", "mse"]]
        for name, baseline in (
            (
                "diagonal",
                families[
                    (families.model_id == model_id) & (families.method == "geometry_diagonal_gate")
                ].set_index("family_id"),
            ),
            (
                "original_33",
                original[
                    (original.model_id == model_id) & (original.method == "ensemble_gate")
                ].set_index("family_id"),
            ),
        ):
            baseline = baseline.loc[current.index, ["mae", "mse"]]
            delta = current - baseline
            comparisons.append(
                {
                    "model_id": model_id,
                    "comparator": name,
                    "primary_mae": float(current.mae.mean()),
                    "primary_mse": float(current.mse.mean()),
                    "baseline_mae": float(baseline.mae.mean()),
                    "baseline_mse": float(baseline.mse.mean()),
                    "strict_joint_family_wins": int(((delta.mae < 0) & (delta.mse < 0)).sum()),
                    "strict_joint_family_losses": int(((delta.mae > 0) & (delta.mse > 0)).sum()),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_checkpoints": count,
            "verified_decision_scores": len(frame),
            "maximum_feature_difference": maximum_feature_difference,
            "maximum_prediction_difference": maximum_prediction_difference,
            "comparisons": comparisons,
            "limits": "matched source validation only; target transfer remains required; no new independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
