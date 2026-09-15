"""Independently replay source-future gates, fixed combinations and matched comparisons."""

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

from tsfm_fais.routing.forecast_gate import compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "aligned-root", "accuracy-root", "teacher-study", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source-objective audits")
    study = read_json(args.study_root / "manifest.json")
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    if (
        study["status"] != "completed"
        or study["verified_checkpoints"] != 90
        or len(study["folds"]) != 30
    ):
        raise ValueError("complete all matched source folds")
    if (
        study["identity"]["aligned_sha256"] != file_sha256(args.aligned_root / "manifest.json")
        or study["identity"]["accuracy_sha256"] != file_sha256(args.accuracy_root / "manifest.json")
        or study["identity"]["teacher_study_sha256"]
        != file_sha256(args.teacher_study / "manifest.json")
    ):
        raise ValueError("the matched source inputs changed")
    for entry in study["full_source_parity"]:
        path = args.study_root / entry["path"]
        if (
            file_sha256(path) != entry["sha256"]
            or read_json(path)["maximum_parameter_difference"] != 0
        ):
            raise ValueError("full-source baseline parity failed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("the source truth bank changed")
    truth_bank = np.load(truth_path, mmap_mode="r")
    records, checkpoints, maximum_gap, maximum_prediction_gap = [], 0, 0.0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        if info["feature_names"][:33] != list(FORECAST_FEATURES):
            raise ValueError("the matched feature specification changed")
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("the source candidate bank changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        for fold in [row for row in study["folds"] if row["model_id"] == model_id]:
            family = fold["held_family"]
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train") & (decisions.family_id.to_numpy() != family)
            )
            validation = np.flatnonzero(
                (decisions.split.to_numpy() == "validation")
                & (decisions.family_id.to_numpy() == family)
            )
            training, evaluation = decisions.iloc[train], decisions.iloc[validation]
            if (
                fold["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                or set(fold["training_origins"]) != set(training.origin_id)
                or set(training.origin_id) & set(evaluation.origin_id)
            ):
                raise ValueError("fold training or validation identity changed")
            family_weights = _family_weights(training)
            values = arrays["features"][train, :7, :33].astype(float)
            denominator = family_weights.sum() * 7
            mean = (values * family_weights[:, None, None]).sum((0, 1)) / denominator
            variance = ((values - mean) ** 2 * family_weights[:, None, None]).sum(
                (0, 1)
            ) / denominator
            scale = np.maximum(np.sqrt(variance), 1e-6)
            reproduced_weights = []
            if [entry["seed"] for entry in fold["checkpoints"]] != [5101, 5102, 5103]:
                raise ValueError("a matched fold lost its prespecified seeds")
            for entry in fold["checkpoints"]:
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != study["identity_sha256"]
                    or saved["train_ids_sha256"] != fold["train_ids_sha256"]
                ):
                    raise ValueError("a checkpoint belongs to another fit")
                if set(saved["training_origins"]) != set(training.origin_id) or set(
                    saved["training_families"]
                ) != set(training.family_id):
                    raise ValueError("a checkpoint's source population changed")
                state = saved["state_dict"]
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().reshape(-1), mean.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().reshape(-1), scale.astype(np.float32)
                )
                reproduced_weights.append(
                    replay_network(
                        state, np.ascontiguousarray(arrays["features"][validation, :7, :33])
                    )
                )
                checkpoints += 1
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("saved fold predictions changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                np.testing.assert_array_equal(saved["seed_weights"], np.stack(reproduced_weights))
                averaged = np.mean(reproduced_weights, axis=0)
                np.testing.assert_array_equal(saved["mean_weights"], averaged)
                fixed = saved["fixed_weights"]
                predictions = {
                    "source_future_gate": compose_forecasts(vectors[validation], averaged),
                    "future_source_fixed_convex": compose_forecasts(
                        vectors[validation], np.repeat(fixed, len(validation), axis=0)
                    ),
                    "forecast_median_guarded": np.median(vectors[validation], axis=1),
                }
                for seed, weight in zip((5101, 5102, 5103), reproduced_weights, strict=True):
                    predictions[f"future_seed{seed}"] = compose_forecasts(
                        vectors[validation], weight
                    )
                if saved["methods"].tolist() != list(predictions):
                    raise ValueError("a saved fold changed its output methods")
                reproduced = np.stack(list(predictions.values()))
                maximum_prediction_gap = max(
                    maximum_prediction_gap, float(abs(reproduced - saved["point"]).max())
                )
                np.testing.assert_array_equal(reproduced, saved["point"])
            if fixed.min() < 0 or abs(fixed.sum() - 1) > 1e-10:
                raise ValueError("fixed source weights are outside the simplex")
            train_future = decision_truth(training, truth_bank)
            estimate = compose_forecasts(vectors[train], np.repeat(fixed, len(train), axis=0))
            gradient_rows = 2 * np.mean(vectors[train] * (estimate - train_future)[:, None], axis=2)
            gradient = np.einsum("n,na->a", family_weights / family_weights.sum(), gradient_rows)
            gap = float((gradient @ fixed[0] - gradient.min()) / max(1.0, abs(gradient).max()))
            maximum_gap = max(maximum_gap, gap)
            if gap > 1e-7:
                raise ValueError("the fixed future-loss mixture failed direct residual optimality")
            truth = decision_truth(evaluation, truth_bank)
            score_path = args.study_root / fold["scores_path"]
            if file_sha256(score_path) != fold["scores_sha256"]:
                raise ValueError("saved source evaluation scores changed")
            recorded = pd.read_parquet(score_path)
            for method, point in predictions.items():
                calculated = evaluation.assign(
                    model_id=model_id,
                    method=method,
                    mae=abs(point - truth).mean(1),
                    mse=((point - truth) ** 2).mean(1),
                )
                previous = (
                    recorded[recorded.method == method]
                    .set_index("episode_id")
                    .loc[calculated.episode_id]
                )
                np.testing.assert_array_equal(previous.mae.to_numpy(), calculated.mae.to_numpy())
                np.testing.assert_array_equal(previous.mse.to_numpy(), calculated.mse.to_numpy())
                records.append(calculated)
    scores = pd.concat(records, ignore_index=True)
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
    expected = pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(summary, expected, check_exact=True)
    teacher = pd.read_csv(args.teacher_study / "family_metrics.csv", float_precision="round_trip")
    comparisons = []
    for model_id in ("chronos2", "timesfm2p5"):
        actual = families[
            (families.model_id == model_id) & (families.method == "source_future_gate")
        ].set_index("family_id")[["mae", "mse"]]
        baseline = (
            teacher[(teacher.model_id == model_id) & (teacher.method == "ensemble_gate")]
            .set_index("family_id")
            .loc[actual.index, ["mae", "mse"]]
        )
        difference = actual - baseline
        comparisons.append(
            {
                "model_id": model_id,
                "families": len(actual),
                "future_mae": float(actual.mae.mean()),
                "future_mse": float(actual.mse.mean()),
                "teacher_mae": float(baseline.mae.mean()),
                "teacher_mse": float(baseline.mse.mean()),
                "mae_relative_percent": float(100 * (actual.mae.mean() / baseline.mae.mean() - 1)),
                "mse_relative_percent": float(100 * (actual.mse.mean() / baseline.mse.mean() - 1)),
                "strict_joint_family_wins": int(
                    ((difference.mae < 0) & (difference.mse < 0)).sum()
                ),
                "strict_joint_family_losses": int(
                    ((difference.mae > 0) & (difference.mse > 0)).sum()
                ),
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
            "verified_checkpoints": checkpoints,
            "verified_decision_scores": len(scores),
            "maximum_prediction_difference": maximum_prediction_gap,
            "maximum_fixed_mixture_optimality_gap": maximum_gap,
            "matched_objective_comparisons": comparisons,
            "limits": "15-family source validation; no population significance or new R6 confirmation claim; source-static controls retained",
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
