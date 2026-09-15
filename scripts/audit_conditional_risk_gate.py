"""Audit matched conditional-risk fits, numerical certificates and validation outcomes."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_conditional_risk import direct_risks
from audit_shared_forecast_gate import replay_network
from conditional_training_inputs import (
    aggregate_panels,
    arguments,
    checked_sources,
    load_training_inputs,
)
from latent_source_inputs import ROOT, read_json
from metric_source_gate import EPSILON
from scipy.special import erf

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_fixed(
    points, target, variance, simulated, row_weights, probability, expected, certificate
):
    weights = np.array(row_weights, float, copy=True)
    weights /= weights.sum()
    prediction = (points * probability[None, :, None]).sum(1)
    error = prediction - target
    magnitude = np.sqrt(error**2 + EPSILON**2)
    slope = error / magnitude
    risk_variance = np.zeros_like(error)
    if expected:
        sigma = np.sqrt(variance[simulated])
        z = error[simulated] / sigma
        slope[simulated] = erf(z / np.sqrt(2))
        magnitude[simulated] = (
            sigma * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + error[simulated] * slope[simulated]
        )
        risk_variance[simulated] = variance[simulated]
    else:
        magnitude[simulated], slope[simulated] = abs(error[simulated]), np.sign(error[simulated])
        if certificate is not None:
            ties = np.asarray(certificate["tie_indices"], int)
            values = np.asarray(certificate["tie_subgradient"], float)
            if len(ties) != len(values) or (abs(values) > 1 + 1e-12).any():
                raise ValueError("an absolute-loss certificate is infeasible")
            if len(ties):
                if (abs(error.ravel()[ties]) > 1e-10).any() or not np.repeat(
                    simulated, points.shape[2]
                )[ties].all():
                    raise ValueError("a nonzero residual received a zero-residual subgradient")
                slope.ravel()[ties] = values
    derivative = slope / 2 + error
    gradient = np.einsum("naq,nq,n->a", points, derivative, weights) / points.shape[2]
    gradient -= (
        np.einsum("nq,nq,n->", np.median(points, axis=1), derivative, weights) / points.shape[2]
    )
    value = float(np.sum(weights * np.mean((magnitude + error**2 + risk_variance) / 2, axis=1)))
    return value, gradient


def compose(points, probability):
    probability = probability / probability.sum(1, keepdims=True)
    return (points * probability[:, :, None]).sum(1)


def main():
    parser = arguments(__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=ROOT / "artifacts/iclr27-r16/conditional-training-v001"
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditional training audits")
    reference = checked_sources(args)
    study = read_json(args.study_root / "manifest.json")
    bindings = {
        "script_sha256": ROOT / "scripts/train_conditional_risk_gate.py",
        "module_sha256": ROOT / "scripts/conditional_risk_gate.py",
        "input_module_sha256": ROOT / "scripts/conditional_training_inputs.py",
        "original_loss_sha256": ROOT / "scripts/metric_source_gate.py",
        "original_model_sha256": ROOT / "scripts/pool_gate_model.py",
        "risk_reference_sha256": ROOT / "scripts/conditional_future.py",
        "reference_sha256": args.reference_root / "manifest.json",
        "reference_audit_sha256": args.reference_audit / "manifest.json",
        "synthetic_sha256": args.synthetic_root / "manifest.json",
        "synthetic_audit_sha256": args.synthetic_audit / "manifest.json",
        "protocol_sha256": args.protocol,
    }
    if study["status"] != "completed" or any(
        study["identity"][key] != file_sha256(path) for key, path in bindings.items()
    ):
        raise ValueError("conditional training definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    for name in ("checkpoints", "folds", "compatibility"):
        if freeze[name] != study[name]:
            raise ValueError("models or predictions changed after validation freeze")
    for entry in study["compatibility"]:
        path = args.study_root / entry["path"]
        original = args.reference_root / entry["original_path"]
        if (
            file_sha256(path) != entry["sha256"]
            or file_sha256(original) != entry["original_sha256"]
        ):
            raise ValueError("a source model replay changed")
        current = torch.load(path, map_location="cpu", weights_only=True)
        old = torch.load(original, map_location="cpu", weights_only=True)
        for key in (
            "initial_parameter_sha256",
            "train_indices_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if current[key] != old[key]:
                raise ValueError("an original R12 source fit no longer replays")
        for key, value in old["state_dict"].items():
            torch.testing.assert_close(value, current["state_dict"][key], rtol=0, atol=0)
    rows, verified, maximum_delta, maximum_gap = [], 0, 0.0, 0.0
    torch.set_num_threads(1)
    for fold in study["folds"]:
        model_id = fold["model_id"]
        frame, data, actions = load_training_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        original = training[~data["simulated"][training]]
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        updates = 25 * int(np.ceil(len(training) / 128))
        expected_predictions = {
            name: data["points"][validation, i] for i, name in enumerate(actions)
        }
        expected_predictions["pool8_mean"] = data["points"][validation].mean(1)
        expected_predictions["pool8_median"] = np.median(data["points"][validation], axis=1)
        source_fold = fold["original_controls"]
        source_path = args.reference_root / source_fold["control_path"]
        if file_sha256(source_path) != source_fold["control_sha256"]:
            raise ValueError("an original fixed control changed")
        control = read_json(source_path)
        source_weights = np.asarray(control["weights"])
        value, gradient = direct_fixed(
            data["points"][original],
            data["truth"][original],
            data["variance"][original],
            data["simulated"][original],
            _family_weights(frame.iloc[original]),
            source_weights,
            False,
            None,
        )
        gap = float((gradient @ source_weights - gradient.min()) / max(abs(gradient).max(), 1.0))
        if gap > 1e-7:
            raise ValueError("an original fixed control fails optimality")
        np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
        maximum_gap = max(maximum_gap, gap)
        expected_predictions["pool8_fixed_joint_future"] = compose(
            data["points"][validation], np.broadcast_to(source_weights, (len(validation), 8))
        )
        expected_predictions["pool8_single_joint_future"] = data["points"][
            validation, control["single_index"]
        ]
        probabilities = []
        for entry in sorted(
            (
                row
                for row in reference["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] is None
            ),
            key=lambda row: row["seed"],
        ):
            path = args.reference_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("an original seed model changed")
            state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
            probability = replay_network(state, data["features"][validation])
            probabilities.append(probability)
            expected_predictions[f"pool8_joint_future_seed{entry['seed']}"] = compose(
                data["points"][validation], probability
            )
        expected_predictions["pool8_joint_future"] = compose(
            data["points"][validation], np.mean(probabilities, axis=0)
        )
        initializations, mixed_statistics = {}, {}
        for fit in fold["fits"]:
            condition = fit["condition"]
            indices = original if condition == "source_steps_matched" else training
            np.testing.assert_array_equal(indices, fit["indices"])
            if fit["updates"] != updates or set(frame.iloc[indices].origin_id) & set(
                frame.iloc[validation].origin_id
            ):
                raise ValueError("a validation history entered a fit or the budget changed")
            weight = _family_weights(frame.iloc[indices])
            values = data["features"][indices].astype(float)
            mean = (values * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 8)
            variance = ((values - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                weight.sum() * 8
            )
            entries = sorted(
                (
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id and row["condition"] == condition
                ),
                key=lambda row: row["seed"],
            )
            if [entry["seed"] for entry in entries] != [5101, 5102, 5103]:
                raise ValueError("a matched seed is missing")
            probabilities = []
            for entry in entries:
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a learned conditional model changed")
                fitted = torch.load(path, map_location="cpu", weights_only=True)
                if fitted["metadata"] != {
                    name: value for name, value in entry.items() if name not in ("path", "sha256")
                }:
                    raise ValueError("conditional model metadata changed")
                if (
                    fitted["updates"] != updates
                    or fitted["train_indices_sha256"]
                    != hashlib.sha256(indices.tobytes()).hexdigest()
                ):
                    raise ValueError("conditional model indices or update count changed")
                if fitted["training_origins"] != sorted(
                    frame.iloc[indices].origin_id.unique()
                ) or fitted["training_families"] != sorted(frame.iloc[indices].family_id.unique()):
                    raise ValueError("conditional model training population changed")
                initializations.setdefault(entry["seed"], fitted["initial_parameter_sha256"])
                if initializations[entry["seed"]] != fitted["initial_parameter_sha256"]:
                    raise ValueError("matched initialization changed")
                state = fitted["state_dict"]
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().ravel(), mean.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().ravel(),
                    np.maximum(np.sqrt(variance), 1e-6).astype(np.float32),
                )
                if (
                    sum(
                        value.numel()
                        for name, value in state.items()
                        if name not in ("feature_mean", "feature_scale")
                    )
                    != 2121
                ):
                    raise ValueError("matched model capacity changed")
                if condition != "source_steps_matched":
                    for name in ("feature_mean", "feature_scale"):
                        key = entry["seed"], name
                        if key in mixed_statistics:
                            torch.testing.assert_close(
                                mixed_statistics[key], state[name], rtol=0, atol=0
                            )
                        else:
                            mixed_statistics[key] = state[name]
                probability = replay_network(state, data["features"][validation])
                probabilities.append(probability)
                expected_predictions[f"{condition}_seed{entry['seed']}"] = compose(
                    data["points"][validation], probability
                )
                verified += 1
            expected_predictions[condition] = compose(
                data["points"][validation], np.mean(probabilities, axis=0)
            )
        for entry in fold["controls"]:
            path = args.study_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a conditional fixed control changed")
            control = read_json(path)
            probability = np.asarray(control["weights"])
            expected = entry["condition"] == "conditional_expected"
            target = data["truth"][training].copy()
            simulated = data["simulated"][training]
            if expected:
                target[simulated] = data["mean"][training][simulated]
            inputs = (
                data["points"][training],
                target,
                data["variance"][training],
                simulated,
                _family_weights(frame.iloc[training]),
            )
            value, gradient = direct_fixed(*inputs, probability, expected, control["certificate"])
            gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
            if gap > 1e-7 or probability.min() < 0 or abs(probability.sum() - 1) > 1e-10:
                raise ValueError("a conditional fixed certificate fails independent optimality")
            maximum_gap = max(maximum_gap, gap)
            np.testing.assert_allclose(value, control["objective"], rtol=1e-10, atol=1e-10)
            single_values = [
                direct_fixed(*inputs, np.eye(8)[i], expected, None)[0] for i in range(8)
            ]
            if int(np.argmin(single_values)) != control["single_index"]:
                raise ValueError("a fixed single candidate changed")
            np.testing.assert_allclose(
                single_values, control["single_objectives"], rtol=1e-10, atol=1e-10
            )
            expected_predictions[f"fixed_{entry['condition']}"] = compose(
                data["points"][validation], np.broadcast_to(probability, (len(validation), 8))
            )
            expected_predictions[f"single_{entry['condition']}"] = data["points"][
                validation, control["single_index"]
            ]
        path = args.study_root / fold["prediction_path"]
        if file_sha256(path) != fold["prediction_sha256"]:
            raise ValueError("a frozen prediction file changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        if set(predictions) != set(expected_predictions) or len(predictions) != 32:
            raise ValueError("the matched baseline panel is incomplete")
        for method, value in expected_predictions.items():
            maximum_delta = max(maximum_delta, float(abs(value - predictions[method]).max()))
            np.testing.assert_allclose(value, predictions[method], rtol=1e-12, atol=1e-12)
        _, evaluation_data, _ = load_training_inputs(args, model_id, validation=True)
        evaluation = frame.iloc[validation].copy()
        simulated = data["simulated"][validation]
        evaluation["panel"] = np.where(simulated, "known_process", "source_temporal")
        for method, point in predictions.items():
            error = point - evaluation_data["truth"][validation]
            mae, mse = np.full(len(validation), np.nan), np.full(len(validation), np.nan)
            mae[simulated], mse[simulated] = direct_risks(
                point[simulated],
                evaluation_data["mean"][validation][simulated],
                evaluation_data["variance"][validation][simulated],
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
        print(f"{model_id}: all models, controls and held-out predictions verified", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    previous = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        previous.sort_values(keys).reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    episodes, families, summary = aggregate_panels(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if verified != 18 or len(study["compatibility"]) != 2 or len(scores) != 112896:
        raise ValueError("the conditional-risk audit coverage changed")
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
            "verified_original_replays": 2,
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "new_forecaster_calls": 0,
            "limits": "full-source temporal and known-process validation; not independent real-data confirmation",
        },
    )


if __name__ == "__main__":
    main()
