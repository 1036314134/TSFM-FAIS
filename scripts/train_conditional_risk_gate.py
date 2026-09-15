"""Compare conditional and factual supervision with matching source and update budgets."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401 - load Arrow before Torch on Windows.
import torch
from conditional_future import expected_risks
from conditional_risk_gate import CONDITIONS, fit_conditional_fixed, fit_conditional_gate
from conditional_training_inputs import (
    aggregate_panels,
    arguments,
    checked_sources,
    load_training_inputs,
)
from latent_source_inputs import ROOT, read_json
from pool_gate_model import pool_probability

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditional-risk training")
    reference = checked_sources(args)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/conditional_risk_gate.py"),
        "input_module_sha256": file_sha256(ROOT / "scripts/conditional_training_inputs.py"),
        "original_loss_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
        "original_model_sha256": file_sha256(ROOT / "scripts/pool_gate_model.py"),
        "risk_reference_sha256": file_sha256(ROOT / "scripts/conditional_future.py"),
        "reference_sha256": file_sha256(args.reference_root / "manifest.json"),
        "reference_audit_sha256": file_sha256(args.reference_audit / "manifest.json"),
        "synthetic_sha256": file_sha256(args.synthetic_root / "manifest.json"),
        "synthetic_audit_sha256": file_sha256(args.synthetic_audit / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "conditions": list(CONDITIONS),
        "primary": "conditional_expected",
        "parameters": 2121,
        "scope": "full-source fitting with temporal and synthetic validation; not leave-family-out",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial conditional-risk training definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, replays = [], [], []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, data, actions = load_training_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        original = training[~data["simulated"][training]]
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        updates = 25 * int(np.ceil(len(training) / 128))
        if updates != (775 if model_id == "chronos2" else 1550):
            raise ValueError("the mixed-population update budget changed")
        root = output / model_id
        root.mkdir(exist_ok=True)
        originals = [
            entry
            for entry in reference["checkpoints"]
            if entry["model_id"] == model_id and entry["held_family"] is None
        ]
        original_seed = next(entry for entry in originals if entry["seed"] == 5101)
        original_path = args.reference_root / original_seed["path"]
        if file_sha256(original_path) != original_seed["sha256"]:
            raise ValueError("an original source model changed")
        old = torch.load(original_path, map_location="cpu", weights_only=True)
        replay_path = root / "original_replay_5101.pt"
        if not replay_path.exists():
            replay = fit_conditional_gate(
                frame, data, original, 5101, 25 * int(np.ceil(len(original) / 128)), False
            )
            temporary = replay_path.with_suffix(".tmp")
            torch.save(replay, temporary)
            temporary.replace(replay_path)
        replay = torch.load(replay_path, map_location="cpu", weights_only=True)
        for key in (
            "initial_parameter_sha256",
            "train_indices_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if replay[key] != old[key]:
                raise ValueError(f"original R12 training changed: {key}")
        for name, value in old["state_dict"].items():
            torch.testing.assert_close(value, replay["state_dict"][name], rtol=0, atol=0)
        replays.append(
            {
                "model_id": model_id,
                "path": str(replay_path.relative_to(output)),
                "sha256": file_sha256(replay_path),
                "original_path": original_seed["path"],
                "original_sha256": original_seed["sha256"],
            }
        )
        predictions = {
            name: data["points"][validation, index] for index, name in enumerate(actions)
        }
        predictions["pool8_mean"] = data["points"][validation].mean(1)
        predictions["pool8_median"] = np.median(data["points"][validation], axis=1)
        source_fold = next(
            entry
            for entry in reference["folds"]
            if entry["model_id"] == model_id and entry["held_family"] is None
        )
        path = args.reference_root / source_fold["control_path"]
        if file_sha256(path) != source_fold["control_sha256"]:
            raise ValueError("an original fixed control changed")
        source_control = read_json(path)
        predictions["pool8_fixed_joint_future"] = compose_forecasts(
            data["points"][validation],
            np.broadcast_to(source_control["weights"], (len(validation), 8)),
        )
        predictions["pool8_single_joint_future"] = data["points"][
            validation, source_control["single_index"]
        ]
        original_probabilities = []
        for entry in sorted(originals, key=lambda value: value["seed"]):
            path = args.reference_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("an original seed model changed")
            old = torch.load(path, map_location="cpu", weights_only=True)
            probability = pool_probability(old["state_dict"], data["features"][validation])
            original_probabilities.append(probability)
            predictions[f"pool8_joint_future_seed{entry['seed']}"] = compose_forecasts(
                data["points"][validation], probability
            )
        predictions["pool8_joint_future"] = compose_forecasts(
            data["points"][validation], np.mean(original_probabilities, axis=0)
        )
        controls, fits = [], []
        for condition in CONDITIONS:
            indices = original if condition == "source_steps_matched" else training
            expected = condition == "conditional_expected"
            probabilities = []
            for seed in (5101, 5102, 5103):
                path = root / f"{condition}_{seed}.pt"
                metadata = {
                    "identity_sha256": identity_sha,
                    "model_id": model_id,
                    "condition": condition,
                    "seed": seed,
                    "held_family": None,
                    "updates": updates,
                }
                if not path.exists():
                    fitted = {
                        **fit_conditional_gate(frame, data, indices, seed, updates, expected),
                        "metadata": metadata,
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(fitted, temporary)
                    temporary.replace(path)
                fitted = torch.load(path, map_location="cpu", weights_only=True)
                if fitted["metadata"] != metadata:
                    raise ValueError("a resumed conditional model changed")
                checkpoints.append(
                    {"path": str(path.relative_to(output)), "sha256": file_sha256(path), **metadata}
                )
                probability = pool_probability(fitted["state_dict"], data["features"][validation])
                probabilities.append(probability)
                predictions[f"{condition}_seed{seed}"] = compose_forecasts(
                    data["points"][validation], probability
                )
            predictions[condition] = compose_forecasts(
                data["points"][validation], np.mean(probabilities, axis=0)
            )
            fits.append({"condition": condition, "indices": indices.tolist(), "updates": updates})
            if condition != "source_steps_matched":
                path = root / f"fixed_{condition}.json"
                if not path.exists():
                    control = fit_conditional_fixed(frame, data, indices, expected)
                    _write_json(
                        path,
                        {
                            **control,
                            "identity_sha256": identity_sha,
                            "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        },
                    )
                control = read_json(path)
                if (
                    control["identity_sha256"] != identity_sha
                    or control["train_indices_sha256"]
                    != hashlib.sha256(indices.tobytes()).hexdigest()
                ):
                    raise ValueError("a resumed conditional control changed")
                controls.append(
                    {
                        "condition": condition,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                predictions[f"fixed_{condition}"] = compose_forecasts(
                    data["points"][validation],
                    np.broadcast_to(control["weights"], (len(validation), 8)),
                )
                predictions[f"single_{condition}"] = data["points"][
                    validation, control["single_index"]
                ]
            print(f"{model_id} {condition}: three matched seeds complete", flush=True)
        if len(predictions) != 32:
            raise ValueError("the prespecified 32-method panel changed")
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
                "fits": fits,
                "controls": controls,
                "original_controls": source_fold,
                "prediction_path": str(path.relative_to(output)),
                "prediction_sha256": file_sha256(path),
            }
        )
    if len(checkpoints) != 18 or len(replays) != 2:
        raise ValueError("the matched model population is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": replays,
            "validation_targets_passed_to_fit": False,
        },
    )
    rows = []
    for fold in folds:
        frame, data, _ = load_training_inputs(args, fold["model_id"], validation=True)
        with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
            indices = saved["validation_indices"]
            evaluation = frame.iloc[indices].copy()
            evaluation["panel"] = np.where(
                data["simulated"][indices], "known_process", "source_temporal"
            )
            simulated = data["simulated"][indices]
            for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                error = point - data["truth"][indices]
                mae, mse = np.full(len(indices), np.nan), np.full(len(indices), np.nan)
                mae[simulated], mse[simulated] = expected_risks(
                    point[simulated],
                    data["mean"][indices][simulated],
                    data["variance"][indices][simulated],
                )
                rows.append(
                    evaluation.assign(
                        method=method,
                        mae=abs(error).mean(1),
                        mse=(error**2).mean(1),
                        expected_mae=mae,
                        expected_mse=mse,
                    )
                )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 112896:
        raise ValueError("the registered validation score population changed")
    episodes, families, summary = aggregate_panels(scores)
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
            "limits": "full-source temporal and known-process validation; real-data transfer remains required",
        },
    )


if __name__ == "__main__":
    main()
