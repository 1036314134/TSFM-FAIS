"""Independently replay grouped fits, observed losses and native-source results."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_shared_forecast_gate import replay_network  # noqa: E402
from native_source_transfer_io import (  # noqa: E402
    checked_source_bindings,
    fold_indices,
    input_arguments,
    load_inputs,
    read_json,
    summarize_scores,
)

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def direct_errors(point, truth, observed, *, joint):
    targets = 2 if joint else 1
    prediction = point.reshape(len(point), -1, targets)
    labels, mask = truth.reshape(prediction.shape), observed.reshape(prediction.shape)
    mae, mse = [], []
    for index in range(len(point)):
        absolute, squared = [], []
        for slot in range(targets):
            keep = mask[index, :, slot]
            residual = prediction[index, keep, slot] - labels[index, keep, slot]
            if len(residual) < 48 or not np.isfinite(residual).all():
                raise ValueError("a held-group score lacks enough original observations")
            absolute.append(abs(residual).mean())
            squared.append((residual**2).mean())
        mae.append(np.mean(absolute))
        mse.append(np.mean(squared))
    return np.asarray(mae), np.asarray(mse)


def independent_family_weights(frame):
    counts = frame.groupby(["family_id", "dataset_id"]).size().to_dict()
    datasets = frame.groupby("family_id").dataset_id.nunique().to_dict()
    weights = np.array(
        [
            1.0 / (datasets[row.family_id] * counts[(row.family_id, row.dataset_id)])
            for row in frame.itertuples()
        ]
    )
    return weights / weights.mean()


def fixed_optimality(points, truth, observed, sample_weights, weights, *, joint):
    targets = 2 if joint else 1
    shaped_mask = observed.reshape(len(observed), -1, targets)
    coefficient = (shaped_mask / shaped_mask.sum(1)[:, None] / targets).reshape(observed.shape)
    weights = np.asarray(weights, float)
    if (weights < 0).any() or abs(weights.sum() - 1) > 1e-10:
        raise ValueError("invalid source-fitted convex weights")
    point = np.sum(points * weights[None, :, None], axis=1)
    residual = np.where(observed, point - truth, 0.0)
    changes = points - np.median(points, axis=1)[:, None]
    probability = sample_weights / sample_weights.sum()
    gradient = 2 * np.einsum("n,naq,nq,nq->a", probability, changes, residual, coefficient)
    raw_gap = float(gradient @ weights - gradient.min())
    gram = np.einsum("n,naq,nbq,nq->ab", probability, changes, changes, coefficient)
    target_residual = np.where(observed, truth - np.median(points, axis=1), 0.0)
    alignment = np.einsum("n,naq,nq,nq->a", probability, changes, target_residual, coefficient)
    scale = max(float(abs(gram).max()), float(abs(alignment).max()), 1e-12)
    if raw_gap / scale > 1e-7:
        raise ValueError("the fixed control fails direct observed-residual optimality")
    return max(raw_gap / scale, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--prior-study-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed native-source audit")
    study = read_json(args.study_root / "manifest.json")
    identity = study["identity"]
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if study["status"] != "completed" or study["checkpoints"] != 48 or len(study["folds"]) != 16:
        raise ValueError("complete the registered grouped study before auditing")
    if (
        freeze["identity_sha256"] != study["identity_sha256"]
        or freeze["models"] != study["models"]
        or freeze["folds"] != study["folds"]
    ):
        raise ValueError("the saved pre-scoring predictions differ from the completed study")
    if identity["source_bindings"] != checked_source_bindings(args):
        raise ValueError("an original data or comparison manifest changed")
    for field, name in (
        ("script_sha256", "scripts/run_native_source_transfer.py"),
        ("io_sha256", "scripts/native_source_transfer_io.py"),
        ("trainer_sha256", "scripts/train_shared_forecast_gate.py"),
        ("model_sha256", "src/tsfm_fais/routing/forecast_gate.py"),
        ("protocol_sha256", "docs/iclr2027/R6_NATIVE_SOURCE_TRANSFER_PLAN.md"),
    ):
        if file_sha256(ROOT / name) != identity[field]:
            raise ValueError("a fitting definition changed after registration")
    torch.set_num_threads(1)
    prior_checkpoints_verified = 0
    all_scores, checkpoints, prediction_difference, metric_difference, maximum_gap = (
        [],
        0,
        0.0,
        0.0,
        0.0,
    )
    stored_scores = pd.read_parquet(args.study_root / "decision_scores.parquet")
    coverage = []
    for info in study["models"]:
        model_id = info["model_id"]
        frame, rebuilt, actions = load_inputs(args, model_id)
        if actions != info["actions"]:
            raise ValueError("source and target candidate ordering changed")
        for kind in ("frame", "data"):
            if file_sha256(args.study_root / info[kind + "_path"]) != info[kind + "_sha256"]:
                raise ValueError("the training input cache changed")
        pd.testing.assert_frame_equal(
            frame, pd.read_parquet(args.study_root / info["frame_path"]), check_exact=True
        )
        with np.load(args.study_root / info["data_path"], allow_pickle=False) as saved:
            for name, values in rebuilt.items():
                np.testing.assert_array_equal(saved[name], values)
        if args.prior_study_root:
            prior_identity = read_json(args.prior_study_root / "identity.json")
            for name in (
                "io_sha256",
                "trainer_sha256",
                "model_sha256",
                "protocol_sha256",
                "source_bindings",
                "settings",
            ):
                if prior_identity[name] != identity[name]:
                    raise ValueError("the corrected control changed a fitting definition")
            pd.testing.assert_frame_equal(
                frame,
                pd.read_parquet(args.prior_study_root / model_id / "decisions.parquet"),
                check_exact=True,
            )
            with np.load(
                args.prior_study_root / model_id / "inputs.npz", allow_pickle=False
            ) as saved:
                for name, values in rebuilt.items():
                    np.testing.assert_array_equal(saved[name], values)
        data = rebuilt
        covered = []
        for entry in (row for row in study["folds"] if row["model_id"] == model_id):
            train, evaluation = fold_indices(frame, entry["held_group"])
            covered.extend(evaluation.tolist())
            training = frame.iloc[train]
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            sample_weights = independent_family_weights(training)
            values = data["features"][train].astype(float)
            mean = (values * sample_weights[:, None, None]).sum((0, 1)) / (sample_weights.sum() * 7)
            variance = ((values - mean) ** 2 * sample_weights[:, None, None]).sum((0, 1)) / (
                sample_weights.sum() * 7
            )
            scale = np.maximum(np.sqrt(variance), 1e-6)
            if entry["train_ids_sha256"] != train_sha:
                raise ValueError("a fit includes rows from a different partition")
            seed_weights = []
            if [row["seed"] for row in entry["checkpoints"]] != [5101, 5102, 5103]:
                raise ValueError("a grouped fit lost a predeclared seed")
            for checkpoint in entry["checkpoints"]:
                path = args.study_root / checkpoint["path"]
                if file_sha256(path) != checkpoint["sha256"]:
                    raise ValueError("a grouped checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != study["identity_sha256"]
                    or saved["input_sha256"] != info["data_sha256"]
                    or saved["train_ids_sha256"] != train_sha
                    or saved["model_id"] != model_id
                    or saved["held_group"] != entry["held_group"]
                    or saved["seed"] != checkpoint["seed"]
                    or set(saved["training_origins"]) != set(training.origin_id)
                    or set(saved["training_families"]) != set(training.family_id)
                ):
                    raise ValueError("checkpoint metadata violates whole-group separation")
                state = saved["state_dict"]
                if args.prior_study_root:
                    prior_path = args.prior_study_root / checkpoint["path"]
                    if prior_path.exists():
                        previous = torch.load(prior_path, map_location="cpu", weights_only=True)
                        if previous["train_ids_sha256"] != train_sha:
                            raise ValueError("the prior fit used other training rows")
                        for name, tensor in state.items():
                            if not torch.equal(tensor, previous["state_dict"][name]):
                                raise ValueError("the baseline correction altered a fitted model")
                        prior_checkpoints_verified += 1
                parameters = sum(
                    value.numel()
                    for name, value in state.items()
                    if name not in ("feature_mean", "feature_scale")
                )
                if parameters != 1096 or len(saved["training_history"]) != 25:
                    raise ValueError("the registered capacity or epoch budget changed")
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().ravel(), mean.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().ravel(), scale.astype(np.float32)
                )
                seed_weights.append(replay_network(state, data["features"][evaluation]))
                checkpoints += 1
            bank_path = args.study_root / entry["prediction_path"]
            if file_sha256(bank_path) != entry["prediction_sha256"]:
                raise ValueError("a held-group prediction bank changed")
            with np.load(bank_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["evaluation_indices"], evaluation)
                np.testing.assert_array_equal(saved["seed_weights"], np.stack(seed_weights))
                methods, predictions, fixed = (
                    saved["methods"].tolist(),
                    saved["points"],
                    saved["fixed_weights"],
                )
            native_points = data["points"][evaluation]
            probabilities = [
                np.mean(seed_weights, axis=0),
                np.broadcast_to(fixed, (len(evaluation), 7)),
            ]
            for position, probability in enumerate(probabilities):
                probability = probability / probability.sum(1, keepdims=True)
                reconstructed = (native_points * probability[:, :, None]).sum(1)
                prediction_difference = max(
                    prediction_difference, float(abs(reconstructed - predictions[position]).max())
                )
                np.testing.assert_allclose(
                    reconstructed, predictions[position], rtol=1e-12, atol=1e-12
                )
            maximum_gap = max(
                maximum_gap,
                fixed_optimality(
                    data["points"][train],
                    data["truth"][train],
                    data["observed"][train],
                    sample_weights,
                    fixed,
                    joint=model_id == "chronos2",
                ),
            )
            for method, prediction in zip(methods, predictions, strict=True):
                mae, mse = direct_errors(
                    prediction,
                    data["truth"][evaluation],
                    data["observed"][evaluation],
                    joint=model_id == "chronos2",
                )
                current = frame.iloc[evaluation].assign(
                    model_id=model_id, method=method, mae=mae, mse=mse
                )
                previous = (
                    stored_scores[
                        (stored_scores.model_id == model_id) & (stored_scores.method == method)
                    ]
                    .set_index("episode_id")
                    .loc[current.episode_id]
                )
                for metric, values in (("mae", mae), ("mse", mse)):
                    metric_difference = max(
                        metric_difference, float(abs(previous[metric].to_numpy() - values).max())
                    )
                    np.testing.assert_allclose(
                        previous[metric].to_numpy(), values, rtol=1e-12, atol=1e-12
                    )
                all_scores.append(current)
        expected = np.flatnonzero(frame.cohort.to_numpy() != "source")
        np.testing.assert_array_equal(np.sort(covered), expected)
        coverage.append(
            {
                "model_id": model_id,
                "source_origins": frame[frame.cohort == "source"].origin_id.nunique(),
                "native_origins": frame.iloc[expected].origin_id.nunique(),
                "native_families": frame.iloc[expected].family_id.nunique(),
                "native_groups": frame.iloc[expected].holdout_group.nunique(),
                "native_decisions": len(expected),
            }
        )
    if checkpoints != 48:
        raise ValueError("grouped checkpoint coverage is incomplete")
    if args.prior_study_root and prior_checkpoints_verified != 24:
        raise ValueError("the earlier 24 Chronos fits were not completely replayed")
    scores = pd.concat(all_scores, ignore_index=True)
    episodes, families, summary, group_summary = summarize_scores(scores)
    for filename, computed in (
        ("summary.csv", summary),
        ("family_metrics.csv", families),
        ("group_summary.csv", group_summary),
    ):
        pd.testing.assert_frame_equal(
            computed,
            pd.read_csv(args.study_root / filename, float_precision="round_trip"),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    group_summary.to_csv(output / "group_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_checkpoints": checkpoints,
            "prior_checkpoints_replayed": prior_checkpoints_verified,
            "verified_scores": len(scores),
            "maximum_prediction_difference": prediction_difference,
            "maximum_metric_difference": metric_difference,
            "maximum_direct_optimality_gap": maximum_gap,
            "coverage": coverage,
            "limits": "retrospective whole-group validation; augmentation changes sample count, domains and missingness jointly; all prior outcomes retained",
        },
    )


if __name__ == "__main__":
    main()
