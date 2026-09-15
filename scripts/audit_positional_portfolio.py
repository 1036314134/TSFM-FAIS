"""Replay positional source models, normalization and bounded prediction scores."""

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
from positional_forecast_portfolio import (  # noqa: E402
    PositionalPortfolio,
    position_inputs,
    replay_offsets,
)
from positional_portfolio_io import target_nodes  # noqa: E402

from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replay_points(state, inputs, indices, mode):
    points = []
    for selection in np.array_split(indices, max(1, (len(indices) + 255) // 256)):
        corrections = replay_offsets(
            state, inputs["context"][selection], inputs["local"][selection], mode
        )
        points.append(
            np.clip(
                inputs["median"][selection] + corrections,
                inputs["lower"][selection],
                inputs["upper"][selection],
            )
        )
    return np.concatenate(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "aligned-root", "accuracy-root", "teacher-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed positional audits")
    study, prep, accuracy = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    teachers = read_json(args.teacher_root / "manifest.json")
    if (
        study["status"] != "completed"
        or len(study["checkpoints"]) != 192
        or len(study["source_models"]) != 12
    ):
        raise ValueError("complete both positional source conditions")
    for key, path in (
        ("aligned_sha256", args.aligned_root / "manifest.json"),
        ("accuracy_sha256", args.accuracy_root / "manifest.json"),
        ("teacher_sha256", args.teacher_root / "manifest.json"),
        ("model_module_sha256", ROOT / "scripts/positional_forecast_portfolio.py"),
        ("io_module_sha256", ROOT / "scripts/positional_portfolio_io.py"),
    ):
        if file_sha256(path) != study["identity"][key]:
            raise ValueError("positional data or implementation changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("source validation labels changed")
    torch.set_num_threads(1)
    records, verified, feature_difference, prediction_difference = [], 0, 0.0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        info, original, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        point_path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("source prediction coordinates changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        original_vectors = decision_vectors(original, bank)
        decisions, base, points = target_nodes(
            original, arrays["features"][:, :7, :33], original_vectors, joint=model_id == "chronos2"
        )
        inputs = position_inputs(base, points)
        median, lower, upper = np.median(points, axis=1), points.min(1), points.max(1)
        relative = np.moveaxis(
            (points - median[:, None]) / np.maximum(upper[:, None] - lower[:, None], 1e-6), 1, 2
        )
        difference = np.concatenate(
            [np.zeros_like(relative[:, :1]), relative[:, 1:] - relative[:, :-1]], axis=1
        )
        step = np.broadcast_to(
            (np.arange(points.shape[2]) / (points.shape[2] - 1))[None, :, None],
            (len(points), points.shape[2], 1),
        )
        expected_local = np.concatenate(
            [relative, difference, np.log1p(upper - lower)[:, :, None], step], axis=2
        ).astype(np.float32)
        expected_context = np.concatenate(
            [
                np.asarray(base, float).mean(1),
                np.concatenate(
                    [relative, difference, np.log1p(upper - lower)[:, :, None], step], axis=2
                ).mean(1),
            ],
            axis=1,
        ).astype(np.float32)
        feature_difference = max(
            feature_difference,
            float(abs(inputs["local"] - expected_local).max()),
            float(abs(inputs["context"] - expected_context).max()),
        )
        np.testing.assert_allclose(inputs["local"], expected_local, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(inputs["context"], expected_context, rtol=1e-7, atol=1e-7)
        for name, expected in (("median", median), ("lower", lower), ("upper", upper)):
            np.testing.assert_array_equal(inputs[name], expected)
        teacher_entry = next(row for row in teachers["models"] if row["model_id"] == model_id)
        if (
            file_sha256(args.teacher_root / teacher_entry["teacher_file"])
            != teacher_entry["teacher_sha256"]
        ):
            raise ValueError("source teachers changed")
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
            if family in set(training.family_id) or set(training.origin_id) & set(
                evaluation.origin_id
            ):
                raise ValueError("family or time separation changed")
            weights = _family_weights(training)
            context, local = (
                inputs["context"][train].astype(float),
                inputs["local"][train].astype(float),
            )
            cm = (context * weights[:, None]).sum(0) / weights.sum()
            cv = ((context - cm) ** 2 * weights[:, None]).sum(0) / weights.sum()
            lm = (local * weights[:, None, None]).sum((0, 1)) / (weights.sum() * local.shape[1])
            lv = ((local - lm) ** 2 * weights[:, None, None]).sum((0, 1)) / (
                weights.sum() * local.shape[1]
            )
            normalization = {
                "context_mean": cm,
                "context_scale": np.maximum(np.sqrt(cv), 1e-6),
                "local_mean": lm,
                "local_scale": np.maximum(np.sqrt(lv), 1e-6),
            }
            for mode in ("local", "pooled"):
                entries = [
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_family"] == family
                    and row["mode"] == mode
                ]
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("matched seed coverage changed")
                seed_points = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a positional checkpoint changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != study["identity_sha256"]
                        or saved["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                    ):
                        raise ValueError("a positional checkpoint used different source rows")
                    if set(saved["training_origins"]) != set(training.origin_id) or set(
                        saved["training_families"]
                    ) != set(training.family_id):
                        raise ValueError(
                            "positional training metadata differs from the actual source population"
                        )
                    if (
                        saved["initial_parameter_sha256"]
                        != study["initial_parameter_sha256"][str(entry["seed"])]
                    ):
                        raise ValueError("matched initialization changed")
                    state = saved["state_dict"]
                    model = PositionalPortfolio(mode)
                    model.load_state_dict(state)
                    if sum(value.numel() for value in model.parameters()) != 609:
                        raise ValueError("the positional model capacity changed")
                    for name, expected in normalization.items():
                        np.testing.assert_array_equal(
                            state[name].numpy().reshape(-1), expected.astype(np.float32)
                        )
                    if len(validation):
                        prediction = replay_points(state, inputs, validation, mode)
                        if np.any(prediction < lower[validation]) or np.any(
                            prediction > upper[validation]
                        ):
                            raise ValueError("a saved model left its forecast envelope")
                        seed_points.append(prediction)
                    verified += 1
                if family is None:
                    continue
                fold = next(
                    row
                    for row in study["folds"]
                    if row["model_id"] == model_id
                    and row["held_family"] == family
                    and row["mode"] == mode
                )
                path = args.study_root / fold["prediction_path"]
                if file_sha256(path) != fold["prediction_sha256"]:
                    raise ValueError("saved positional predictions changed")
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    np.testing.assert_array_equal(saved["seed_points"], np.stack(seed_points))
                    reference = median[validation]
                    averaged = np.clip(
                        reference + np.mean(np.stack(seed_points) - reference[None], axis=0),
                        lower[validation],
                        upper[validation],
                    )
                    predictions = {
                        f"position_{mode}": averaged,
                        "forecast_median_guarded": reference,
                    }
                    for seed, prediction in zip((5101, 5102, 5103), seed_points, strict=True):
                        predictions[f"position_{mode}_seed{seed}"] = prediction
                    if list(predictions) != saved["methods"].tolist():
                        raise ValueError("positional forecast definitions changed")
                    prediction_difference = max(
                        prediction_difference,
                        float(abs(np.stack(list(predictions.values())) - saved["point"]).max()),
                    )
                    np.testing.assert_array_equal(
                        np.stack(list(predictions.values())), saved["point"]
                    )
                score_path = args.study_root / fold["scores_path"]
                if file_sha256(score_path) != fold["scores_sha256"]:
                    raise ValueError("source score records changed")
                saved_scores = pd.read_parquet(score_path)
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                for name, prediction in predictions.items():
                    current = evaluation.assign(
                        model_id=model_id,
                        mode=mode,
                        method=name,
                        mae=abs(prediction - truth).mean(1),
                        mse=((prediction - truth) ** 2).mean(1),
                    )
                    previous = (
                        saved_scores[saved_scores.method == name]
                        .set_index("episode_id")
                        .loc[current.episode_id]
                    )
                    np.testing.assert_array_equal(previous.mae.to_numpy(), current.mae.to_numpy())
                    np.testing.assert_array_equal(previous.mse.to_numpy(), current.mse.to_numpy())
                    records.append(current)
    if verified != 192:
        raise ValueError("positional source audit is incomplete")
    frame = pd.concat(records, ignore_index=True)
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
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
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
            "verified_checkpoints": verified,
            "verified_decision_scores": len(frame),
            "maximum_feature_difference": feature_difference,
            "maximum_prediction_difference": prediction_difference,
            "normalization_and_envelopes_verified": True,
            "limits": "source audit only; both old target cohorts remain previously used data",
        },
    )


if __name__ == "__main__":
    main()
