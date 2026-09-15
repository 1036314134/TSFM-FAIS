"""Reconstruct fixed metric controls, source populations and exact validation scores."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, load_source_inputs, read_json
from metric_source_gate import CONDITIONS, EPSILON
from train_calibrated_source_gates import arguments, load_references
from train_latent_source_gates import aggregate, condition_features

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_control(points, target, row_weights, probability, kind):
    weights = np.array(row_weights, dtype=float, copy=True)
    weights /= weights.sum()
    prediction = (points * probability[None, :, None]).sum(1)
    error = prediction - target
    magnitude = np.sqrt(error**2 + EPSILON**2)
    alpha, beta = (1.0, 0.0) if kind == "mae" else (0.5, 0.5)
    derivative = alpha * error / magnitude + 2 * beta * error
    gradient = np.einsum("naq,nq,n->a", points, derivative, weights) / points.shape[2]
    gradient -= (
        np.einsum("nq,nq,n->", np.median(points, axis=1), derivative, weights) / points.shape[2]
    )
    value = float(np.sum(weights * np.mean(alpha * magnitude + beta * error**2, axis=1)))
    return value, gradient


def compare_previous_models(previous_root, current_root, identity):
    previous_identity = read_json(previous_root / "identity.json")
    unchanged = {
        name: value
        for name, value in identity.items()
        if name not in ("module_sha256", "protocol_sha256")
    }
    if unchanged != {
        name: value
        for name, value in previous_identity.items()
        if name not in ("module_sha256", "protocol_sha256")
    }:
        raise ValueError("the prior attempt differs beyond the fixed-solver precision repair")
    count = 0
    for path in sorted(previous_root.rglob("*.pt")):
        current = current_root / path.relative_to(previous_root)
        old = torch.load(path, map_location="cpu", weights_only=True)
        new = torch.load(current, map_location="cpu", weights_only=True)
        for name in (
            "initial_parameter_sha256",
            "train_indices_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if new[name] != old[name]:
                raise ValueError(f"prior model fitting changed: {path.name}, {name}")
        if {
            name: value for name, value in new["metadata"].items() if name != "identity_sha256"
        } != {name: value for name, value in old["metadata"].items() if name != "identity_sha256"}:
            raise ValueError("a prior model changed condition, family or seed")
        if new["state_dict"].keys() != old["state_dict"].keys():
            raise ValueError("a prior model changed architecture")
        for name in old["state_dict"]:
            torch.testing.assert_close(
                new["state_dict"][name], old["state_dict"][name], rtol=0, atol=0
            )
        count += 1
    if not count:
        raise ValueError("no prior completed models were available for comparison")
    return count


def main():
    parser = arguments(__doc__)
    parser.set_defaults(protocol=ROOT / "docs/iclr2027/R10_METRIC_OBJECTIVE_PROTOCOL.md")
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed metric-objective audit")
    reference, _ = load_references(args)
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"]
        != file_sha256(ROOT / "scripts/train_metric_source_gates.py")
        or study["identity"]["module_sha256"] != file_sha256(ROOT / "scripts/metric_source_gate.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
        or study["identity"]["source_sha256"] != file_sha256(args.input_root / "manifest.json")
        or study["identity"]["reference_sha256"]
        != file_sha256(args.reference_root / "manifest.json")
    ):
        raise ValueError("the completed objective experiment has changed definitions")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["outer_validation_outcomes_read"]
    ):
        raise ValueError("the outer prediction freeze changed")
    torch.set_num_threads(1)
    verified, maximum_delta, maximum_gap = 0, 0.0, 0.0
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_source_inputs(args.input_root, model_id)
        points = arrays["vectors"]
        features = condition_features(arrays["features"], "point_future")
        train = np.flatnonzero(frame.split.to_numpy() == "train")
        targets = {"teacher": arrays["teacher"], "future": np.full_like(arrays["teacher"], np.nan)}
        targets["future"][train] = decision_truth(
            frame.iloc[train], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        for fold in (part for part in study["folds"] if part["model_id"] == model_id):
            family = fold["held_family"]
            indices = (
                train
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            np.testing.assert_array_equal(indices, fold["train_indices"])
            validation = (
                np.array([], np.int64)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            weight = _family_weights(frame.iloc[indices])
            values = features[indices].astype(float)
            mean = (values * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 7)
            variance = ((values - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                weight.sum() * 7
            )
            expected = {}
            for condition in CONDITIONS:
                kind, label = condition.split("_")
                control_entry = fold["controls"][condition]
                control_path = args.study_root / control_entry["path"]
                if file_sha256(control_path) != control_entry["sha256"]:
                    raise ValueError("a matched fixed metric control changed")
                control = read_json(control_path)
                probability = np.asarray(control["weights"])
                value, gradient = direct_control(
                    points[indices], targets[label][indices], weight, probability, kind
                )
                gap = float(
                    (gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0)
                )
                if gap > 1e-7 or (probability < 0).any() or abs(probability.sum() - 1) > 1e-10:
                    raise ValueError("a matched fixed control fails independent optimality")
                maximum_gap = max(maximum_gap, gap)
                np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
                single_values = [
                    direct_control(
                        points[indices], targets[label][indices], weight, np.eye(7)[index], kind
                    )[0]
                    for index in range(7)
                ]
                np.testing.assert_allclose(
                    single_values, control["single_objectives"], rtol=1e-12, atol=1e-12
                )
                if (
                    int(np.argmin(single_values)) != control["single_index"]
                    or control["condition"] != condition
                    or control["train_indices_sha256"]
                    != hashlib.sha256(indices.tobytes()).hexdigest()
                ):
                    raise ValueError("the matched single control or source population changed")
                if len(validation):
                    expected[f"fixed_{condition}"] = (
                        points[validation] * probability[None, :, None]
                    ).sum(1)
                    expected[f"single_{condition}"] = points[validation, control["single_index"]]
                entries = [
                    entry
                    for entry in study["checkpoints"]
                    if entry["model_id"] == model_id
                    and entry["held_family"] == family
                    and entry["condition"] == condition
                ]
                if [entry["seed"] for entry in entries] != [5101, 5102, 5103]:
                    raise ValueError("a metric condition lost a source seed")
                seed_weights = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a metric source checkpoint changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["metadata"]
                        != {
                            name: value
                            for name, value in entry.items()
                            if name not in ("path", "sha256")
                        }
                        or saved["initial_parameter_sha256"]
                        != reference["initial_parameters"][str(entry["seed"])]
                        or saved["train_indices_sha256"]
                        != hashlib.sha256(indices.tobytes()).hexdigest()
                        or set(saved["training_origins"]) != set(frame.iloc[indices].origin_id)
                        or set(saved["training_families"]) != set(frame.iloc[indices].family_id)
                        or len(saved["history"]) != 25
                    ):
                        raise ValueError(
                            "model population, initialization or training schedule changed"
                        )
                    state = saved["state_dict"]
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
                        != 2120
                    ):
                        raise ValueError("matched gate capacity changed")
                    if len(validation):
                        seed_weights.append(replay_network(state, features[validation]))
                    verified += 1
                if len(validation):
                    for name, probability in [
                        (condition, np.mean(seed_weights, axis=0)),
                        *[
                            (f"{condition}_seed{seed}", probability)
                            for seed, probability in zip(
                                (5101, 5102, 5103), seed_weights, strict=True
                            )
                        ],
                    ]:
                        probability = probability / probability.sum(1, keepdims=True)
                        expected[name] = (points[validation] * probability[:, :, None]).sum(1)
            if not len(validation):
                continue
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("an outer prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for method, point in expected.items():
                maximum_delta = max(maximum_delta, float(abs(point - predictions[method]).max()))
                np.testing.assert_allclose(point, predictions[method], rtol=1e-12, atol=1e-12)
            old = next(
                part
                for part in reference["folds"]
                if part["model_id"] == model_id and part["held_family"] == family
            )
            with np.load(args.reference_root / old["prediction_path"], allow_pickle=False) as saved:
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    np.testing.assert_array_equal(point, predictions[method])
            evaluation = frame.iloc[validation]
            truth = decision_truth(
                evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
            )
            for method, point in predictions.items():
                rows.append(
                    evaluation.assign(
                        model_id=model_id,
                        method=method,
                        mae=abs(point - truth).mean(1),
                        mse=((point - truth) ** 2).mean(1),
                    )
                )
        print(f"{model_id}: metric controls and all source models audited", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    original = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        original.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    old = pd.read_csv(args.reference_audit / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(old.method.unique())].reset_index(drop=True),
        old,
        check_exact=True,
    )
    if verified != 384 or len(scores) != 129168:
        raise ValueError("the metric-objective audit is incomplete")
    previous_models = (
        compare_previous_models(args.previous_attempt, args.study_root, study["identity"])
        if args.previous_attempt
        else 0
    )
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
            "verified_checkpoints": verified,
            "previous_attempt_models_exactly_replayed": previous_models,
            "verified_fixed_controls": 128,
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "matched source objective comparison; no new independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
