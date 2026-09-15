"""Compare an eight-candidate learned and fixed portfolio on unchanged source histories."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from latent_source_inputs import ROOT, load_source_inputs, read_json
from metric_source_gate import fit_fixed_metric
from pool_gate_inputs import load_pool_inputs
from pool_gate_model import fit_pool_gate, pool_probability
from train_latent_source_gates import aggregate, condition_features

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "pool-root": "artifacts/iclr27-r12/motm-pool-inputs-v001",
        "motm-root": "artifacts/iclr27-r12/source-motm-v001",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "metric-root": "artifacts/iclr27-r10/metric-source-v002",
        "metric-audit": "artifacts/iclr27-r10/metric-source-audit-v002",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "protocol": "docs/iclr2027/R12_MOTM_POOL_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def checked_sources(args):
    metric = read_json(args.metric_root / "manifest.json")
    audit = read_json(args.metric_audit / "manifest.json")
    pool = read_json(args.pool_root / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.metric_root / "manifest.json")
        or pool["status"] != "completed"
        or pool["identity"]["base_sha256"] != metric["identity"]["source_sha256"]
    ):
        raise ValueError("source and metric references do not match")
    reference_path = ROOT / "artifacts/iclr27-r7/latent-source-gates-v001/manifest.json"
    reference = read_json(reference_path)
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        file_sha256(reference_path) != metric["identity"]["reference_sha256"]
        or file_sha256(args.accuracy_root / "manifest.json")
        != reference["identity"]["accuracy_sha256"]
        or file_sha256(args.accuracy_root / "truth_z.npy")
        != accuracy["prediction_arrays"]["truth_z.npy"]
    ):
        raise ValueError("source outcome provenance changed")
    return metric


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed eight-candidate study")
    metric = checked_sources(args)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_module_sha256": file_sha256(ROOT / "scripts/pool_gate_model.py"),
        "input_module_sha256": file_sha256(ROOT / "scripts/pool_gate_inputs.py"),
        "loss_module_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
        "pool_sha256": file_sha256(args.pool_root / "manifest.json"),
        "metric_sha256": file_sha256(args.metric_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "primary": "pool8_joint_future",
        "candidate_count": 8,
        "parameters_per_model": 2121,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial eight-candidate training definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, compatibility, initializations = [], [], [], {}
    for model_id in ("chronos2", "timesfm2p5"):
        pool_manifest, frame, arrays = load_pool_inputs(args.pool_root, model_id)
        _, old_frame, old_arrays = load_source_inputs(args.base_root, model_id)
        for name in old_frame.columns:
            np.testing.assert_array_equal(frame[name], old_frame[name])
        features, points = arrays["features"], arrays["vectors"]
        with np.load(
            args.pool_root / model_id / pool_manifest["episodes"][0]["path"], allow_pickle=False
        ) as saved:
            actions = saved["actions"].tolist()
        motm_index = actions.index("motm_reference")
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        truth = np.full((len(frame), points.shape[-1]), np.nan)
        truth[training] = decision_truth(
            frame.iloc[training], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        _, _, _, gram = forecast_geometry(points)
        alignment = np.full(points.shape[:2], np.nan)
        alignment[training] = projection_targets(points[training], truth[training])[
            "raw_projection"
        ]
        old_points = old_arrays["vectors"]
        _, _, _, old_gram = forecast_geometry(old_points)
        old_alignment = np.full(old_points.shape[:2], np.nan)
        old_alignment[training] = projection_targets(old_points[training], truth[training])[
            "raw_projection"
        ]
        replay_path = output / "compatibility" / f"{model_id}_5101.pt"
        if not replay_path.exists():
            reproduced = fit_pool_gate(
                frame,
                condition_features(old_arrays["features"], "point_future"),
                old_points,
                truth,
                old_gram,
                old_alignment,
                training,
                5101,
                "joint",
            )
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = replay_path.with_suffix(".tmp")
            torch.save(reproduced, temporary)
            temporary.replace(replay_path)
        reproduced = torch.load(replay_path, map_location="cpu", weights_only=True)
        old_entry = next(
            row
            for row in metric["checkpoints"]
            if row["model_id"] == model_id
            and row["held_family"] is None
            and row["condition"] == "joint_future"
            and row["seed"] == 5101
        )
        original = torch.load(
            args.metric_root / old_entry["path"], map_location="cpu", weights_only=True
        )
        for name in (
            "initial_parameter_sha256",
            "train_indices_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if reproduced[name] != original[name]:
                raise ValueError("the generalized seven-candidate fit differs from R10")
        for name in original["state_dict"]:
            torch.testing.assert_close(
                reproduced["state_dict"][name], original["state_dict"][name], rtol=0, atol=0
            )
        compatibility.append(
            {
                "model_id": model_id,
                "path": str(replay_path.relative_to(output)),
                "sha256": file_sha256(replay_path),
                "original_path": old_entry["path"],
                "original_sha256": old_entry["sha256"],
            }
        )
        for family in [None, *sorted(frame.family_id.unique())]:
            indices = (
                training
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            validation = (
                np.array([], np.int64)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            if family in set(frame.iloc[indices].family_id) or set(
                frame.iloc[indices].origin_id
            ) & set(frame.iloc[validation].origin_id):
                raise ValueError("a validation history or family entered pool training")
            directory = output / model_id / (family or "full_source")
            directory.mkdir(parents=True, exist_ok=True)
            predictions = {}
            if len(validation):
                old = next(
                    row
                    for row in metric["folds"]
                    if row["model_id"] == model_id and row["held_family"] == family
                )
                path = args.metric_root / old["prediction_path"]
                if file_sha256(path) != old["prediction_sha256"]:
                    raise ValueError("an original metric prediction changed")
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    predictions.update(zip(saved["methods"].tolist(), saved["point"], strict=True))
            control_path = directory / "fixed_joint_future.json"
            if not control_path.exists():
                control = fit_fixed_metric(
                    points[indices], truth[indices], _family_weights(frame.iloc[indices]), "joint"
                )
                _write_json(
                    control_path,
                    {
                        **control,
                        "identity_sha256": identity_sha,
                        "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    },
                )
            control = read_json(control_path)
            if (
                control["identity_sha256"] != identity_sha
                or control["train_indices_sha256"] != hashlib.sha256(indices.tobytes()).hexdigest()
            ):
                raise ValueError("a resumed pool fixed control changed")
            probabilities = []
            for seed in (5101, 5102, 5103):
                path = directory / f"pool8_joint_future_{seed}.pt"
                metadata = {
                    "identity_sha256": identity_sha,
                    "model_id": model_id,
                    "held_family": family,
                    "seed": seed,
                    "candidate_count": 8,
                }
                if not path.exists():
                    fitted = {
                        **fit_pool_gate(
                            frame, features, points, truth, gram, alignment, indices, seed, "joint"
                        ),
                        "metadata": metadata,
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(fitted, temporary)
                    temporary.replace(path)
                fitted = torch.load(path, map_location="cpu", weights_only=True)
                if fitted["metadata"] != metadata:
                    raise ValueError("a resumed eight-candidate model changed")
                initializations.setdefault(str(seed), fitted["initial_parameter_sha256"])
                if initializations[str(seed)] != fitted["initial_parameter_sha256"]:
                    raise ValueError("matched pool initializations differ")
                checkpoints.append(
                    {"path": str(path.relative_to(output)), "sha256": file_sha256(path), **metadata}
                )
                if len(validation):
                    probability = pool_probability(fitted["state_dict"], features[validation])
                    probabilities.append(probability)
                    predictions[f"pool8_joint_future_seed{seed}"] = compose_forecasts(
                        points[validation], probability
                    )
            fold = {
                "model_id": model_id,
                "held_family": family,
                "train_indices": indices.tolist(),
                "control_path": str(control_path.relative_to(output)),
                "control_sha256": file_sha256(control_path),
                "actions": actions,
            }
            if len(validation):
                predictions["pool8_joint_future"] = compose_forecasts(
                    points[validation], np.mean(probabilities, axis=0)
                )
                predictions["pool8_fixed_joint_future"] = compose_forecasts(
                    points[validation], np.broadcast_to(control["weights"], (len(validation), 8))
                )
                predictions["pool8_single_joint_future"] = points[
                    validation, control["single_index"]
                ]
                predictions["pool8_mean"] = points[validation].mean(1)
                predictions["pool8_median"] = np.median(points[validation], axis=1)
                predictions["pool8_motm_reference"] = points[validation, motm_index]
                path = directory / "predictions.npz"
                _save_npz(
                    path,
                    validation_indices=validation,
                    methods=np.asarray(list(predictions)),
                    point=np.stack(list(predictions.values())),
                )
                fold.update(
                    prediction_path=str(path.relative_to(output)),
                    prediction_sha256=file_sha256(path),
                )
            folds.append(fold)
            print(
                f"{model_id} {family or 'full_source'}: eight-candidate source comparison saved",
                flush=True,
            )
    if len(checkpoints) != 96 or len(folds) != 32 or len(compatibility) != 2:
        raise ValueError("the registered pool study is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "compatibility": compatibility,
            "validation_future_arrays_read": False,
        },
    )
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, _ = load_pool_inputs(args.pool_root, model_id)
        for fold in (
            row for row in folds if row["model_id"] == model_id and row["held_family"] is not None
        ):
            with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
                evaluation = frame.iloc[saved["validation_indices"]]
                target = decision_truth(
                    evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
                )
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    rows.append(
                        evaluation.assign(
                            method=method,
                            mae=abs(point - target).mean(1),
                            mse=((point - target) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 154440 or scores.groupby("model_id").method.nunique().ne(55).any():
        raise ValueError("the eight-candidate comparison coverage changed")
    episodes, families, summary = aggregate(scores)
    old = pd.read_csv(args.metric_audit / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(old.method.unique())].reset_index(drop=True),
        old,
        check_exact=True,
    )
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
            "compatibility": compatibility,
            "initializations": initializations,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "matched eight-candidate source development; audit required; same observation and forecasting budget within each pool",
        },
    )


if __name__ == "__main__":
    main()
