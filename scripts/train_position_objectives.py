"""Fit matched local/pooled forecasts with actual-future MSE and joint objectives."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from position_objective import fit_position_loss
from position_objective_inputs import (
    arguments,
    average_positions,
    checked_sources,
    fixed_control,
    load_inputs,
)
from positional_forecast_portfolio import PositionalPortfolio, predict_position
from train_latent_source_gates import aggregate

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed position-objective studies")
    reference = checked_sources(args)
    settings = reference["identity"]["settings"]
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "loss_module_sha256": file_sha256(ROOT / "scripts/position_objective.py"),
        "input_module_sha256": file_sha256(ROOT / "scripts/position_objective_inputs.py"),
        "model_module_sha256": file_sha256(ROOT / "scripts/positional_forecast_portfolio.py"),
        "reference_sha256": file_sha256(args.reference_study / "manifest.json"),
        "reference_audit_sha256": file_sha256(args.reference_audit / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "settings": settings,
        "primary": "position17_local_joint_future",
        "scope": "full-source temporal validation",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial positional objective definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, replays = [], [], []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, inputs, points, labels, actions = load_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        train_inputs = {key: value[training] for key, value in inputs.items()}
        weights = _family_weights(frame.iloc[training])
        index_sha = hashlib.sha256(training.tobytes()).hexdigest()
        root = output / model_id
        root.mkdir(exist_ok=True)
        predictions = {action: points[validation, i] for i, action in enumerate(actions)}
        predictions["forecast_median_guarded"] = inputs["median"][validation]
        predictions["forecast_mean_guarded"] = points[validation].mean(1)
        controls = []
        for mode in ("local", "pooled"):
            originals = sorted(
                [
                    row
                    for row in reference["source_models"]
                    if row["model_id"] == model_id and row["mode"] == mode
                ],
                key=lambda row: row["seed"],
            )
            if [row["seed"] for row in originals] != [5101, 5102, 5103]:
                raise ValueError("an original source seed is missing")
            old_predictions = []
            for entry in originals:
                path = args.reference_study / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an original positional model changed")
                old = torch.load(path, map_location="cpu", weights_only=True)
                model = PositionalPortfolio(mode).eval().requires_grad_(False)
                model.load_state_dict(old["state_dict"])
                prediction = predict_position(model, inputs, validation)
                old_predictions.append(prediction)
                predictions[f"position17_{mode}_teacher_seed{entry['seed']}"] = prediction
                if entry["seed"] == 5101:
                    replay_path = root / f"replay_{mode}_5101.pt"
                    if not replay_path.exists():
                        model, history, initial = fit_position_loss(
                            train_inputs,
                            labels["teacher"][training],
                            weights,
                            mode=mode,
                            seed=5101,
                            settings=settings,
                            objective="mse",
                        )
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "history": history,
                                "initial": initial,
                                "index_sha256": index_sha,
                            },
                            replay_path,
                        )
                    current = torch.load(replay_path, map_location="cpu", weights_only=True)
                    renamed = [
                        {
                            "epoch": row["epoch"],
                            "mean_training_loss": row["mean_training_teacher_mse"],
                        }
                        for row in old["training_history"]
                    ]
                    if (
                        current["history"] != renamed
                        or current["initial"] != old["initial_parameter_sha256"]
                        or current["index_sha256"] != old["train_ids_sha256"]
                    ):
                        raise ValueError("the original positional fit changed")
                    for key, value in old["state_dict"].items():
                        torch.testing.assert_close(
                            value, current["state_dict"][key], rtol=0, atol=0
                        )
                    replays.append(
                        {
                            "model_id": model_id,
                            "mode": mode,
                            "path": str(replay_path.relative_to(output)),
                            "sha256": file_sha256(replay_path),
                            "original_path": entry["path"],
                            "original_sha256": entry["sha256"],
                        }
                    )
            predictions[f"position17_{mode}_teacher"] = average_positions(
                old_predictions, inputs, validation
            )
            for objective in ("mse", "joint"):
                group = f"position17_{mode}_{objective}_future"
                seed_predictions = []
                for seed in (5101, 5102, 5103):
                    path = root / f"{mode}_{objective}_{seed}.pt"
                    metadata = {
                        "identity_sha256": identity_sha,
                        "model_id": model_id,
                        "mode": mode,
                        "objective": objective,
                        "seed": seed,
                        "train_indices_sha256": index_sha,
                        "training_origins": sorted(frame.iloc[training].origin_id.unique()),
                        "training_families": sorted(frame.iloc[training].family_id.unique()),
                    }
                    if not path.exists():
                        model, history, initial = fit_position_loss(
                            train_inputs,
                            labels["future"][training],
                            weights,
                            mode=mode,
                            seed=seed,
                            settings=settings,
                            objective=objective,
                        )
                        temporary = path.with_suffix(".tmp")
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "history": history,
                                "initial_parameter_sha256": initial,
                                "metadata": metadata,
                            },
                            temporary,
                        )
                        temporary.replace(path)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["metadata"] != metadata:
                        raise ValueError("a resumed actual-future model changed")
                    model = PositionalPortfolio(mode).eval().requires_grad_(False)
                    model.load_state_dict(saved["state_dict"])
                    prediction = predict_position(model, inputs, validation)
                    if np.any(prediction < inputs["lower"][validation]) or np.any(
                        prediction > inputs["upper"][validation]
                    ):
                        raise ValueError("a positional forecast left its registered bounds")
                    seed_predictions.append(prediction)
                    predictions[f"{group}_seed{seed}"] = prediction
                    checkpoints.append(
                        {
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            **metadata,
                        }
                    )
                predictions[group] = average_positions(seed_predictions, inputs, validation)
                print(f"{model_id} {mode} {objective}: three source seeds complete", flush=True)
        for objective in ("mse", "joint"):
            path = root / f"fixed_{objective}.json"
            if not path.exists():
                _write_json(
                    path,
                    {
                        **fixed_control(frame, points, labels["future"], training, objective),
                        "identity_sha256": identity_sha,
                        "train_indices_sha256": index_sha,
                    },
                )
            control = read_json(path)
            if (
                control["identity_sha256"] != identity_sha
                or control["train_indices_sha256"] != index_sha
            ):
                raise ValueError("a resumed fixed control changed")
            probability = np.asarray(control["weights"])
            predictions[f"position17_fixed_{objective}"] = (
                points[validation] * probability[None, :, None]
            ).sum(1)
            predictions[f"position17_single_{objective}"] = points[
                validation, control["single_index"]
            ]
            controls.append(
                {
                    "objective": objective,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
        if len(predictions) != 37:
            raise ValueError("the registered positional comparison methods changed")
        path = root / "predictions.npz"
        _save_npz(
            path,
            validation_indices=validation,
            methods=np.asarray(list(predictions)),
            point=np.stack(list(predictions.values())),
        )
        folds.append(
            {
                "model_id": model_id,
                "actions": actions,
                "controls": controls,
                "prediction_path": str(path.relative_to(output)),
                "prediction_sha256": file_sha256(path),
            }
        )
    if len(checkpoints) != 24 or len(replays) != 4:
        raise ValueError("the positional model population is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": replays,
            "validation_targets_passed_to_fit": False,
        },
    )
    rows = []
    for fold in folds:
        frame, _, _, labels, _ = load_inputs(args, fold["model_id"], validation=True)
        with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
            indices = saved["validation_indices"]
            for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                error = point - labels["future"][indices]
                rows.append(
                    frame.iloc[indices].assign(
                        method=method, mae=abs(error).mean(1), mse=(error**2).mean(1)
                    )
                )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 277056:
        raise ValueError("the positional validation population changed")
    episodes, families, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": replays,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "full-source temporal validation; independent audit and retrospective real-data transfer required",
        },
    )


if __name__ == "__main__":
    main()
