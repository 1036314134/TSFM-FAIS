"""Independently replay actual-future positional models and their source validation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_metric_source_gates import direct_control
from audit_positional_portfolio import replay_points
from latent_source_inputs import ROOT, read_json
from position_objective_inputs import arguments, average_positions, checked_sources, load_inputs
from train_latent_source_gates import aggregate

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = arguments(__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=ROOT / "artifacts/iclr27-r17/position-objectives-v001"
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed position-objective audits")
    reference = checked_sources(args)
    study = read_json(args.study_root / "manifest.json")
    for key, path in (
        ("script_sha256", ROOT / "scripts/train_position_objectives.py"),
        ("loss_module_sha256", ROOT / "scripts/position_objective.py"),
        ("input_module_sha256", ROOT / "scripts/position_objective_inputs.py"),
        ("model_module_sha256", ROOT / "scripts/positional_forecast_portfolio.py"),
        ("reference_sha256", args.reference_study / "manifest.json"),
        ("reference_audit_sha256", args.reference_audit / "manifest.json"),
        ("protocol_sha256", args.protocol),
    ):
        if study["identity"][key] != file_sha256(path):
            raise ValueError("positional objective definitions changed")
    if (
        study["status"] != "completed"
        or study["identity"]["settings"] != reference["identity"]["settings"]
    ):
        raise ValueError("the positional study or original budget changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    for key in ("checkpoints", "folds", "compatibility"):
        if freeze[key] != study[key]:
            raise ValueError("a fitted model or prediction changed after freeze")
    for entry in study["compatibility"]:
        path = args.study_root / entry["path"]
        original = args.reference_study / entry["original_path"]
        if (
            file_sha256(path) != entry["sha256"]
            or file_sha256(original) != entry["original_sha256"]
        ):
            raise ValueError("an original model replay changed")
        replay = torch.load(path, map_location="cpu", weights_only=True)
        old = torch.load(original, map_location="cpu", weights_only=True)
        renamed = [
            {"epoch": row["epoch"], "mean_training_loss": row["mean_training_teacher_mse"]}
            for row in old["training_history"]
        ]
        if (
            replay["history"] != renamed
            or replay["initial"] != old["initial_parameter_sha256"]
            or replay["index_sha256"] != old["train_ids_sha256"]
        ):
            raise ValueError("the original teacher training no longer replays exactly")
        for name, value in old["state_dict"].items():
            torch.testing.assert_close(value, replay["state_dict"][name], rtol=0, atol=0)
    torch.set_num_threads(1)
    rows, verified, maximum_difference, maximum_gap = [], 0, 0.0, 0.0
    for fold in study["folds"]:
        model_id = fold["model_id"]
        frame, inputs, points, labels, actions = load_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        if (
            not np.isnan(labels["future"][validation]).all()
            or not np.isnan(labels["teacher"][validation]).all()
        ):
            raise ValueError("validation labels were exposed to source fitting")
        weight = _family_weights(frame.iloc[training])
        context, local = (
            inputs["context"][training].astype(float),
            inputs["local"][training].astype(float),
        )
        cm = (context * weight[:, None]).sum(0) / weight.sum()
        cv = ((context - cm) ** 2 * weight[:, None]).sum(0) / weight.sum()
        lm = (local * weight[:, None, None]).sum((0, 1)) / (weight.sum() * local.shape[1])
        lv = ((local - lm) ** 2 * weight[:, None, None]).sum((0, 1)) / (
            weight.sum() * local.shape[1]
        )
        normalization = {
            "context_mean": cm,
            "context_scale": np.maximum(np.sqrt(cv), 1e-6),
            "local_mean": lm,
            "local_scale": np.maximum(np.sqrt(lv), 1e-6),
        }
        expected = {action: points[validation, i] for i, action in enumerate(actions)}
        expected["forecast_median_guarded"] = inputs["median"][validation]
        expected["forecast_mean_guarded"] = points[validation].mean(1)
        for mode in ("local", "pooled"):
            originals = sorted(
                [
                    row
                    for row in reference["source_models"]
                    if row["model_id"] == model_id and row["mode"] == mode
                ],
                key=lambda row: row["seed"],
            )
            old_predictions = []
            for entry in originals:
                path = args.reference_study / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an original source seed changed")
                state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
                prediction = replay_points(state, inputs, validation, mode)
                expected[f"position17_{mode}_teacher_seed{entry['seed']}"] = prediction
                old_predictions.append(prediction)
            expected[f"position17_{mode}_teacher"] = average_positions(
                old_predictions, inputs, validation
            )
            for objective in ("mse", "joint"):
                group = f"position17_{mode}_{objective}_future"
                entries = sorted(
                    [
                        row
                        for row in study["checkpoints"]
                        if row["model_id"] == model_id
                        and row["mode"] == mode
                        and row["objective"] == objective
                    ],
                    key=lambda row: row["seed"],
                )
                if [entry["seed"] for entry in entries] != [5101, 5102, 5103]:
                    raise ValueError("the matched positional seed population changed")
                predictions = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a trained positional objective model changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    metadata = {
                        key: value for key, value in entry.items() if key not in ("path", "sha256")
                    }
                    if (
                        saved["metadata"] != metadata
                        or metadata["train_indices_sha256"]
                        != hashlib.sha256(training.tobytes()).hexdigest()
                    ):
                        raise ValueError("the actual fitted source rows changed")
                    if metadata["training_origins"] != sorted(
                        frame.iloc[training].origin_id.unique()
                    ) or metadata["training_families"] != sorted(
                        frame.iloc[training].family_id.unique()
                    ):
                        raise ValueError("a positional source population changed")
                    if (
                        saved["initial_parameter_sha256"]
                        != reference["initial_parameter_sha256"][str(entry["seed"])]
                    ):
                        raise ValueError("the matched positional initialization changed")
                    if len(saved["history"]) != 25:
                        raise ValueError("the positional training budget changed")
                    state = saved["state_dict"]
                    for name, value in normalization.items():
                        np.testing.assert_array_equal(
                            state[name].numpy().ravel(), value.astype(np.float32)
                        )
                    if (
                        sum(
                            value.numel()
                            for name, value in state.items()
                            if name not in normalization
                        )
                        != 609
                    ):
                        raise ValueError("the positional model capacity changed")
                    prediction = replay_points(state, inputs, validation, mode)
                    if np.any(prediction < inputs["lower"][validation]) or np.any(
                        prediction > inputs["upper"][validation]
                    ):
                        raise ValueError("an actual-future model left its forecast bounds")
                    predictions.append(prediction)
                    expected[f"{group}_seed{entry['seed']}"] = prediction
                    verified += 1
                expected[group] = average_positions(predictions, inputs, validation)
        for entry in fold["controls"]:
            path = args.study_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a positional matched fixed control changed")
            control = read_json(path)
            probability = np.asarray(control["weights"])
            selected, target = points[training], labels["future"][training]
            objective = entry["objective"]
            if objective == "joint":
                value, gradient = direct_control(selected, target, weight, probability, "joint")
                singles = [
                    direct_control(selected, target, weight, np.eye(7)[i], "joint")[0]
                    for i in range(7)
                ]
            else:
                distribution = weight / weight.sum()
                prediction = (selected * probability[None, :, None]).sum(1)
                error = prediction - target
                value = float(distribution @ (error**2).mean(1))
                gradient = (
                    2 * np.einsum("naq,nq,n->a", selected, error, distribution) / selected.shape[2]
                )
                gradient -= (
                    2
                    * np.einsum("nq,nq,n->", np.median(selected, axis=1), error, distribution)
                    / selected.shape[2]
                )
                singles = np.einsum(
                    "n,na->a", distribution, ((selected - target[:, None]) ** 2).mean(2)
                )
            gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
            if gap > 1e-7 or probability.min() < 0 or abs(probability.sum() - 1) > 1e-10:
                raise ValueError("a positional fixed control fails direct optimality")
            maximum_gap = max(maximum_gap, gap)
            np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                singles, control["single_objectives"], rtol=1e-12, atol=1e-12
            )
            if int(np.argmin(singles)) != control["single_index"]:
                raise ValueError("a matched fixed single candidate changed")
            expected[f"position17_fixed_{objective}"] = (
                points[validation] * probability[None, :, None]
            ).sum(1)
            expected[f"position17_single_{objective}"] = points[validation, control["single_index"]]
        path = args.study_root / fold["prediction_path"]
        if file_sha256(path) != fold["prediction_sha256"]:
            raise ValueError("a frozen positional validation bank changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        if set(predictions) != set(expected) or len(predictions) != 37:
            raise ValueError("the positional baseline panel changed")
        for name, value in expected.items():
            maximum_difference = max(
                maximum_difference, float(abs(value - predictions[name]).max())
            )
            np.testing.assert_array_equal(value, predictions[name])
        _, _, _, held_labels, _ = load_inputs(args, model_id, validation=True)
        for method, point in predictions.items():
            error = point - held_labels["future"][validation]
            rows.append(
                frame.iloc[validation].assign(
                    method=method, mae=abs(error).mean(1), mse=(error**2).mean(1)
                )
            )
        print(
            f"{model_id}: positional objective models, controls and validation audited", flush=True
        )
    scores = pd.concat(rows, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    previous = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        previous.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    if verified != 24 or len(study["compatibility"]) != 4 or len(scores) != 277056:
        raise ValueError("the positional objective audit is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_models": verified,
            "verified_original_replays": 4,
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_difference,
            "maximum_fixed_optimality_gap": maximum_gap,
            "new_forecaster_calls": 0,
            "limits": "full-source temporal validation; retrospective target transfer still required",
        },
    )


if __name__ == "__main__":
    main()
