"""Fit matched MAE and joint objectives on the unchanged R7 source population."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from latent_source_inputs import ROOT, load_source_inputs, read_json
from metric_source_gate import CONDITIONS, EPSILON, fit_fixed_metric, metric_objective
from train_calibrated_source_gates import arguments, load_references, probability_from_state
from train_latent_source_gates import aggregate, condition_features
from train_shared_forecast_gate import SETTINGS

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def fit_metric_gate(frame, features, points, target, gram, alignment, indices, seed, kind):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97)
    weights = _family_weights(frame.iloc[indices])
    model.fit_normalization(features[indices], weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    x, g, b, w = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features[indices], gram[indices], alignment[indices], weights)
    ]
    p, y = (
        torch.tensor(points[indices], dtype=torch.float64),
        torch.tensor(target[indices], dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(seed + 100000)
    history = []
    for epoch in range(25):
        total = 0.0
        for batch in torch.randperm(len(indices), generator=generator).split(128):
            loss = (
                metric_objective(model(x[batch]), p[batch], y[batch], g[batch], b[batch], kind)
                * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite matched metric objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite matched metric gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch + 1, "relative_training_loss": total / len(indices)})
    return {
        "state_dict": model.state_dict(),
        "initial_parameter_sha256": initial,
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest(),
        "training_origins": sorted(frame.iloc[indices].origin_id.unique()),
        "training_families": sorted(frame.iloc[indices].family_id.unique()),
        "history": history,
    }


def main():
    parser = arguments(__doc__)
    parser.set_defaults(protocol=ROOT / "docs/iclr2027/R10_METRIC_OBJECTIVE_PROTOCOL.md")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed metric-objective study")
    reference, _ = load_references(args)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "source_sha256": file_sha256(args.input_root / "manifest.json"),
        "reference_sha256": file_sha256(args.reference_root / "manifest.json"),
        "reference_audit_sha256": file_sha256(args.reference_audit / "manifest.json"),
        "settings": SETTINGS,
        "epsilon": EPSILON,
        "conditions": list(CONDITIONS),
        "primary": "joint_future",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial metric-objective definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_source_inputs(args.input_root, model_id)
        features = condition_features(arrays["features"], "point_future")
        points = arrays["vectors"]
        _, _, _, gram = forecast_geometry(points)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        targets = {"teacher": arrays["teacher"], "future": np.full_like(arrays["teacher"], np.nan)}
        targets["future"][training] = decision_truth(
            frame.iloc[training], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        alignments = {label: np.full(points.shape[:2], np.nan) for label in targets}
        for label, target in targets.items():
            alignments[label][training] = projection_targets(points[training], target[training])[
                "raw_projection"
            ]
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
                raise ValueError("an outer family or history entered fitting")
            directory = output / model_id / (family or "full_source")
            directory.mkdir(parents=True, exist_ok=True)
            predictions, controls = {}, {}
            if len(validation):
                old = next(
                    row
                    for row in reference["folds"]
                    if row["model_id"] == model_id and row["held_family"] == family
                )
                path = args.reference_root / old["prediction_path"]
                if file_sha256(path) != old["prediction_sha256"]:
                    raise ValueError("the original prediction bank changed")
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["validation_indices"], validation)
                    predictions.update(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for condition in CONDITIONS:
                kind, label = condition.split("_")
                control_path = directory / f"fixed_{condition}.json"
                if not control_path.exists():
                    control = fit_fixed_metric(
                        points[indices],
                        targets[label][indices],
                        _family_weights(frame.iloc[indices]),
                        kind,
                    )
                    _write_json(
                        control_path,
                        {
                            **control,
                            "identity_sha256": identity_sha,
                            "condition": condition,
                            "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                        },
                    )
                controls[condition] = read_json(control_path)
                if (
                    controls[condition]["identity_sha256"] != identity_sha
                    or controls[condition]["train_indices_sha256"]
                    != hashlib.sha256(indices.tobytes()).hexdigest()
                ):
                    raise ValueError("a resumed fixed control has different definitions")
                if len(validation):
                    predictions[f"fixed_{condition}"] = compose_forecasts(
                        points[validation],
                        np.broadcast_to(controls[condition]["weights"], (len(validation), 7)),
                    )
                    predictions[f"single_{condition}"] = points[
                        validation, controls[condition]["single_index"]
                    ]
                probabilities = []
                for seed in (5101, 5102, 5103):
                    metadata = {
                        "identity_sha256": identity_sha,
                        "model_id": model_id,
                        "held_family": family,
                        "condition": condition,
                        "seed": seed,
                    }
                    path = directory / f"{condition}_{seed}.pt"
                    if not path.exists():
                        saved = {
                            **fit_metric_gate(
                                frame,
                                features,
                                points,
                                targets[label],
                                gram,
                                alignments[label],
                                indices,
                                seed,
                                kind,
                            ),
                            "metadata": metadata,
                        }
                        temporary = path.with_suffix(".tmp")
                        torch.save(saved, temporary)
                        temporary.replace(path)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["metadata"] != metadata
                        or saved["initial_parameter_sha256"]
                        != reference["initial_parameters"][str(seed)]
                    ):
                        raise ValueError("a resumed model or initialization changed")
                    checkpoints.append(
                        {
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            **metadata,
                        }
                    )
                    if len(validation):
                        probability = probability_from_state(
                            saved["state_dict"], features[validation]
                        )
                        probabilities.append(probability)
                        predictions[f"{condition}_seed{seed}"] = compose_forecasts(
                            points[validation], probability
                        )
                if len(validation):
                    predictions[condition] = compose_forecasts(
                        points[validation], np.mean(probabilities, axis=0)
                    )
            record = {
                "model_id": model_id,
                "held_family": family,
                "train_indices": indices.tolist(),
                "controls": {
                    condition: {
                        "path": str((directory / f"fixed_{condition}.json").relative_to(output)),
                        "sha256": file_sha256(directory / f"fixed_{condition}.json"),
                    }
                    for condition in CONDITIONS
                },
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
                f"{model_id} {family or 'full_source'}: four metric objectives complete", flush=True
            )
    if len(checkpoints) != 384 or len(folds) != 32:
        raise ValueError("the metric-objective experiment is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "outer_validation_outcomes_read": False,
        },
    )
    rows = []
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
                    rows.append(
                        evaluation.assign(
                            model_id=model_id,
                            method=method,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 129168 or scores.groupby("model_id").method.nunique().ne(46).any():
        raise ValueError("the original and matched metric comparisons are incomplete")
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
            "limits": "same-source metric objective comparison; audit required before interpretation",
        },
    )


if __name__ == "__main__":
    main()
