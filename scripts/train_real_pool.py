"""Compare real-missing source augmentation with a matched-update synthetic source."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from latent_source_inputs import ROOT, read_json
from masked_pool_gate import fit_observed_fixed, fit_observed_gate
from native_source_transfer_io import observed_errors, summarize_scores
from pool_gate_model import pool_probability
from real_pool_inputs import arguments, load_real_inputs

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = arguments(__doc__)
    parser.add_argument(
        "--prepared-root", type=Path, default=ROOT / "artifacts/iclr27-r13/real-inputs-v001"
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed real-source study")
    prepared = read_json(args.prepared_root / "manifest.json")
    original = read_json(args.pool_study / "manifest.json")
    if prepared["status"] != "completed" or prepared["identity"][
        "source_study_sha256"
    ] != file_sha256(args.pool_study / "manifest.json"):
        raise ValueError("prepared real-source data and original models disagree")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/masked_pool_gate.py"),
        "input_module_sha256": file_sha256(ROOT / "scripts/real_pool_inputs.py"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "original_study_sha256": file_sha256(args.pool_study / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "primary": "real_augmented",
        "native_labels_used_for_other_groups": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial real-source definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, compatibility = [], [], []
    for model_id in ("chronos2", "timesfm2p5"):
        info, frame, data, references = load_real_inputs(args.prepared_root, model_id)
        source = np.flatnonzero(frame.cohort.to_numpy() == "source")
        path = output / "compatibility" / f"{model_id}_5101.pt"
        if not path.exists():
            saved = fit_observed_gate(
                frame, data, source, 5101, int(np.ceil(len(source) / 128)) * 25
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        old_entry = next(
            row
            for row in original["checkpoints"]
            if row["model_id"] == model_id and row["held_family"] is None and row["seed"] == 5101
        )
        old = torch.load(args.pool_study / old_entry["path"], map_location="cpu", weights_only=True)
        for name in (
            "initial_parameter_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if saved[name] != old[name]:
                raise ValueError(
                    "full-observation training did not reproduce the original source model"
                )
        mapped = frame.iloc[source].source_position.to_numpy(np.int64)
        if hashlib.sha256(mapped.tobytes()).hexdigest() != old["train_indices_sha256"]:
            raise ValueError("source index compaction changed the original training rows")
        for name in old["state_dict"]:
            torch.testing.assert_close(
                saved["state_dict"][name], old["state_dict"][name], rtol=0, atol=0
            )
        compatibility.append(
            {
                "model_id": model_id,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "original_path": old_entry["path"],
                "original_sha256": old_entry["sha256"],
            }
        )
        groups = sorted(frame[frame.cohort != "source"].holdout_group.unique())
        for group in groups:
            evaluation = np.flatnonzero(
                (frame.cohort.to_numpy() != "source") & (frame.holdout_group.to_numpy() == group)
            )
            augmented = np.flatnonzero(
                (frame.cohort.to_numpy() == "source") | (frame.holdout_group.to_numpy() != group)
            )
            if set(frame.iloc[augmented].family_id) & set(frame.iloc[evaluation].family_id) or set(
                frame.iloc[augmented].origin_id
            ) & set(frame.iloc[evaluation].origin_id):
                raise ValueError("an evaluation group entered source fitting")
            updates = int(np.ceil(len(augmented) / 128)) * 25
            directory = output / model_id / group
            directory.mkdir(parents=True, exist_ok=True)
            predictions = {name: value[evaluation] for name, value in references.items()}
            for regime, indices in (
                ("real_augmented", augmented),
                ("source_steps_matched", source),
            ):
                probabilities = []
                for seed in (5101, 5102, 5103):
                    metadata = {
                        "identity_sha256": identity_sha,
                        "model_id": model_id,
                        "held_group": group,
                        "regime": regime,
                        "seed": seed,
                        "updates": updates,
                    }
                    path = directory / f"{regime}_{seed}.pt"
                    if not path.exists():
                        fitted = {
                            **fit_observed_gate(frame, data, indices, seed, updates),
                            "metadata": metadata,
                        }
                        temporary = path.with_suffix(".tmp")
                        torch.save(fitted, temporary)
                        temporary.replace(path)
                    fitted = torch.load(path, map_location="cpu", weights_only=True)
                    if fitted["metadata"] != metadata:
                        raise ValueError("a resumed real-source model changed")
                    probability = pool_probability(
                        fitted["state_dict"], data["features"][evaluation]
                    )
                    probabilities.append(probability)
                    predictions[f"{regime}_seed{seed}"] = compose_forecasts(
                        data["points"][evaluation], probability
                    )
                    checkpoints.append(
                        {
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            **metadata,
                        }
                    )
                predictions[regime] = compose_forecasts(
                    data["points"][evaluation], np.mean(probabilities, axis=0)
                )
            control_path = directory / "fixed.json"
            if not control_path.exists():
                control = fit_observed_fixed(frame, data, augmented)
                _write_json(
                    control_path,
                    {
                        **control,
                        "identity_sha256": identity_sha,
                        "train_indices_sha256": hashlib.sha256(augmented.tobytes()).hexdigest(),
                    },
                )
            control = read_json(control_path)
            if (
                control["identity_sha256"] != identity_sha
                or control["train_indices_sha256"]
                != hashlib.sha256(augmented.tobytes()).hexdigest()
            ):
                raise ValueError("a resumed mixed-source fixed control changed")
            predictions["real_augmented_fixed"] = compose_forecasts(
                data["points"][evaluation],
                np.broadcast_to(control["weights"], (len(evaluation), 8)),
            )
            predictions["real_augmented_single"] = data["points"][
                evaluation, control["single_index"]
            ]
            path = directory / "predictions.npz"
            _save_npz(
                path,
                methods=np.asarray(list(predictions)),
                point=np.stack(list(predictions.values())),
                evaluation_indices=evaluation,
            )
            folds.append(
                {
                    "model_id": model_id,
                    "held_group": group,
                    "augmented_indices": augmented.tolist(),
                    "source_indices": source.tolist(),
                    "evaluation_indices": evaluation.tolist(),
                    "updates": updates,
                    "prediction_path": str(path.relative_to(output)),
                    "prediction_sha256": file_sha256(path),
                    "control_path": str(control_path.relative_to(output)),
                    "control_sha256": file_sha256(control_path),
                }
            )
            print(f"{model_id} {group}: same-update real-source comparison saved", flush=True)
    if len(checkpoints) != 96 or len(folds) != 16:
        raise ValueError("the grouped real-source comparison is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": compatibility,
            "held_group_scoring_started": False,
            "native_labels_read_for_other_groups": True,
        },
    )
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, data, _ = load_real_inputs(args.prepared_root, model_id)
        for fold in (row for row in folds if row["model_id"] == model_id):
            with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
                indices = saved["evaluation_indices"]
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    mae, mse = observed_errors(
                        point,
                        data["truth"][indices],
                        data["observed"][indices],
                        joint=model_id == "chronos2",
                    )
                    rows.append(frame.iloc[indices].assign(method=method, mae=mae, mse=mse))
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 21480 or scores.groupby("model_id").method.nunique().ne(20).any():
        raise ValueError("the grouped real-source scoring population changed")
    episodes, families, summary, groups = summarize_scores(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    groups.to_csv(output / "group_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": compatibility,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "new_imputer_fits": 0,
            "limits": "retrospective grouped native-source development; held group excluded from its fit; independent audit required",
        },
    )


if __name__ == "__main__":
    main()
