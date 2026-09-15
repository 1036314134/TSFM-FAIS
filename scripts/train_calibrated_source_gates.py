"""Fit source-calibrated forecast gates without selecting on outer outcomes."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from calibrated_source_gate import (
    EPOCHS,
    FEATURES,
    LAMBDAS,
    calibration_metrics,
    choose_configuration,
    fit_snapshots,
    fixed_weights,
    inputs_for,
    split_origins,
)
from latent_source_inputs import ROOT, load_source_inputs, read_json
from train_latent_source_gates import aggregate
from train_shared_forecast_gate import SETTINGS, predict_weights

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "input-root": "artifacts/iclr27-r7/latent-source-v001",
        "reference-root": "artifacts/iclr27-r7/latent-source-gates-v001",
        "reference-audit": "artifacts/iclr27-r7/latent-source-audit-v001",
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "protocol": "docs/iclr2027/R8_CALIBRATED_GATE_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def load_references(args):
    reference = read_json(args.reference_root / "manifest.json")
    audit = read_json(args.reference_audit / "manifest.json")
    collected = read_json(args.input_root / "manifest.json")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        audit["status"] != "completed"
        or reference["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.reference_root / "manifest.json")
        or reference["identity"]["input_manifest_sha256"]
        != file_sha256(args.input_root / "manifest.json")
        or collected["identity"]["source_manifest_sha256"]
        != file_sha256(args.source_root / "episodes_manifest.json")
        or reference["identity"]["accuracy_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or accuracy["prediction_arrays"]["truth_z.npy"]
        != file_sha256(args.accuracy_root / "truth_z.npy")
    ):
        raise ValueError("source references are incomplete or changed")
    source = read_json(args.source_root / "episodes_manifest.json")
    positions = {row["origin_id"]: row["origin"] for row in source["episodes"]}
    return reference, positions


def probability_from_state(state, features):
    model = SharedForecastGate(features=97).eval().requires_grad_(False)
    model.load_state_dict(state)
    probability = predict_weights(model, features)
    np.testing.assert_array_equal(probability, replay_network(state, features))
    return probability


def obtain_checkpoint(path, metadata, factory):
    if not path.exists():
        saved = {**factory(), "metadata": metadata}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["metadata"] != metadata:
        raise ValueError("a partial checkpoint belongs to another calibrated experiment")
    return saved


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed calibrated source experiment")
    reference, positions = load_references(args)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/calibrated_source_gate.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "input_sha256": file_sha256(args.input_root / "manifest.json"),
        "reference_sha256": file_sha256(args.reference_root / "manifest.json"),
        "reference_audit_sha256": file_sha256(args.reference_audit / "manifest.json"),
        "settings": SETTINGS,
        "epochs": list(EPOCHS),
        "strengths": list(LAMBDAS),
        "features": list(FEATURES),
        "primary": "point_calibrated",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("the calibrated source definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_source_inputs(args.input_root, model_id)
        source_train = np.flatnonzero(frame.split.to_numpy() == "train")
        vectors = arrays["vectors"]
        _, _, _, gram = forecast_geometry(vectors)
        source_truth = np.full((len(frame), vectors.shape[-1]), np.nan)
        source_truth[source_train] = decision_truth(
            frame.iloc[source_train], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        alignment = np.full(vectors.shape[:2], np.nan)
        alignment[source_train] = projection_targets(
            vectors[source_train], source_truth[source_train]
        )["raw_projection"]
        for family in [None, *sorted(frame.family_id.unique())]:
            allowed = (
                source_train
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
            inner, calibration, purged = split_origins(frame, allowed, positions)
            if family in set(frame.iloc[allowed].family_id) or set(
                frame.iloc[allowed].origin_id
            ) & set(frame.iloc[validation].origin_id):
                raise ValueError("an outer validation family or history entered training")
            inner_anchor, inner_gap = fixed_weights(frame, gram, alignment, inner)
            outer_anchor, outer_gap = fixed_weights(frame, gram, alignment, allowed)
            directory = output / model_id / (family or "full_source")
            directory.mkdir(parents=True, exist_ok=True)
            reference_metrics = calibration_metrics(
                frame.iloc[calibration],
                compose_forecasts(
                    vectors[calibration], np.broadcast_to(inner_anchor, (len(calibration), 7))
                ),
                source_truth[calibration],
            )
            candidate_records, choices, calibration_weights = [], {}, {}
            for feature in FEATURES:
                features = inputs_for(arrays, feature)
                feature_records = []
                for strength in LAMBDAS:
                    probabilities = {epoch: [] for epoch in EPOCHS}
                    for seed in SETTINGS["seeds"]:
                        path = directory / "inner" / f"{feature}_r{int(strength)}_s{seed}.pt"
                        metadata = {
                            "identity_sha256": identity_sha,
                            "model_id": model_id,
                            "held_family": family,
                            "feature": feature,
                            "stage": "inner",
                            "strength": strength,
                            "seed": seed,
                            "epochs": list(EPOCHS),
                        }
                        saved = obtain_checkpoint(
                            path,
                            metadata,
                            partial(
                                fit_snapshots,
                                features,
                                gram,
                                alignment,
                                frame,
                                inner,
                                inner_anchor,
                                strength,
                                seed,
                                EPOCHS,
                            ),
                        )
                        if (
                            saved["initial_parameter_sha256"]
                            != reference["initial_parameters"][str(seed)]
                        ):
                            raise ValueError("matched initialization changed")
                        for epoch in EPOCHS:
                            probabilities[epoch].append(
                                probability_from_state(
                                    saved["states"][str(epoch)], features[calibration]
                                )
                            )
                        checkpoints.append(
                            {
                                "path": str(path.relative_to(output)),
                                "sha256": file_sha256(path),
                                **metadata,
                            }
                        )
                    averaged = np.stack([np.mean(probabilities[epoch], axis=0) for epoch in EPOCHS])
                    calibration_weights[f"{feature}_r{int(strength)}"] = averaged
                    for epoch, probability in zip(EPOCHS, averaged, strict=True):
                        metrics = calibration_metrics(
                            frame.iloc[calibration],
                            compose_forecasts(vectors[calibration], probability),
                            source_truth[calibration],
                        )
                        feature_records.append(
                            {"feature": feature, "strength": strength, "epoch": epoch, **metrics}
                        )
                candidate_records.extend(feature_records)
                choices[feature + "_calibrated"] = choose_configuration(
                    feature_records, reference_metrics
                )
                choices[feature + "_early_stop"] = choose_configuration(
                    feature_records, reference_metrics, early_only=True
                )
            selection = {
                "identity_sha256": identity_sha,
                "model_id": model_id,
                "held_family": family,
                "allowed": allowed.tolist(),
                "inner": inner.tolist(),
                "calibration": calibration.tolist(),
                "purged": purged.tolist(),
                "inner_anchor": inner_anchor.tolist(),
                "outer_anchor": outer_anchor.tolist(),
                "inner_optimality_gap": inner_gap,
                "outer_optimality_gap": outer_gap,
                "reference_metrics": reference_metrics,
                "candidates": candidate_records,
                "choices": choices,
            }
            selection_path = directory / "selection.json"
            _write_json(selection_path, selection)
            calibration_path = directory / "calibration_weights.npz"
            _save_npz(calibration_path, **calibration_weights)
            predictions = {}
            if len(validation):
                old_fold = next(
                    row
                    for row in reference["folds"]
                    if row["model_id"] == model_id and row["held_family"] == family
                )
                old_path = args.reference_root / old_fold["prediction_path"]
                if file_sha256(old_path) != old_fold["prediction_sha256"]:
                    raise ValueError("a reference prediction changed")
                with np.load(old_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    predictions.update(zip(saved["methods"].tolist(), saved["point"], strict=True))
                np.testing.assert_allclose(
                    compose_forecasts(
                        vectors[validation], np.broadcast_to(outer_anchor, (len(validation), 7))
                    ),
                    predictions["fixed_future"],
                    atol=1e-12,
                    rtol=1e-12,
                )
            fitted = {}
            for method, choice in choices.items():
                if choice["kind"] == "fixed":
                    probabilities = [np.broadcast_to(outer_anchor, (len(validation), 7))] * 3
                else:
                    feature = method.split("_")[0]
                    features = inputs_for(arrays, feature)
                    strength, epoch = choice["strength"], choice["epoch"]
                    key = (feature, strength, epoch)
                    if key not in fitted:
                        fitted[key] = []
                        for seed in SETTINGS["seeds"]:
                            path = (
                                directory
                                / "outer"
                                / f"{feature}_r{int(strength)}_e{epoch}_s{seed}.pt"
                            )
                            metadata = {
                                "identity_sha256": identity_sha,
                                "model_id": model_id,
                                "held_family": family,
                                "feature": feature,
                                "stage": "outer",
                                "strength": strength,
                                "seed": seed,
                                "epochs": [epoch],
                            }
                            saved = obtain_checkpoint(
                                path,
                                metadata,
                                partial(
                                    fit_snapshots,
                                    features,
                                    gram,
                                    alignment,
                                    frame,
                                    allowed,
                                    outer_anchor,
                                    strength,
                                    seed,
                                    (epoch,),
                                ),
                            )
                            checkpoints.append(
                                {
                                    "path": str(path.relative_to(output)),
                                    "sha256": file_sha256(path),
                                    **metadata,
                                }
                            )
                            if len(validation):
                                fitted[key].append(
                                    probability_from_state(
                                        saved["states"][str(epoch)], features[validation]
                                    )
                                )
                    probabilities = fitted[key]
                if len(validation):
                    for seed, probability in zip(SETTINGS["seeds"], probabilities, strict=True):
                        predictions[f"{method}_seed{seed}"] = compose_forecasts(
                            vectors[validation], probability
                        )
                    predictions[method] = compose_forecasts(
                        vectors[validation], np.mean(probabilities, axis=0)
                    )
            record = {
                "model_id": model_id,
                "held_family": family,
                "selection_path": str(selection_path.relative_to(output)),
                "selection_sha256": file_sha256(selection_path),
                "calibration_path": str(calibration_path.relative_to(output)),
                "calibration_sha256": file_sha256(calibration_path),
            }
            if len(validation):
                path = directory / "predictions.npz"
                _save_npz(
                    path,
                    validation_indices=validation,
                    methods=np.asarray(list(predictions)),
                    point=np.stack(list(predictions.values())),
                )
                record.update(
                    prediction_path=str(path.relative_to(output)),
                    prediction_sha256=file_sha256(path),
                )
            folds.append(record)
            print(
                f"{model_id} {family or 'full_source'}: calibrated choices and predictions saved",
                flush=True,
            )
    if len(folds) != 32 or sum(row["stage"] == "inner" for row in checkpoints) != 384:
        raise ValueError("the registered nested experiment is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "outer_validation_outcomes_read": False,
        },
    )
    all_scores = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, _ = load_source_inputs(args.input_root, model_id)
        for fold in (
            row for row in folds if row["model_id"] == model_id and row["held_family"] is not None
        ):
            with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
                evaluation = frame.iloc[saved["validation_indices"]]
                truth = decision_truth(
                    evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
                )
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    all_scores.append(
                        evaluation.assign(
                            model_id=model_id,
                            method=method,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(all_scores, ignore_index=True)
    if len(scores) != 106704 or scores.groupby("model_id").method.nunique().ne(38).any():
        raise ValueError("the original and calibrated evaluation methods are incomplete")
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
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "source development with nested temporal calibration; independent audit required; outer outcomes already used in earlier development",
        },
    )


if __name__ == "__main__":
    main()
