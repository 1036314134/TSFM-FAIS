"""Replay the interval-only source comparison and all preserved point controls."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from audit_source_quantiles import read_json  # noqa: E402
from interval_gate_inputs import HAS_INDEX, WIDTH_INDEX, load_interval_inputs  # noqa: E402
from train_interval_forecast_gate import add_arguments, aggregate  # noqa: E402

from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed interval gate audit")
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or len(study["checkpoints"]) != 96
        or len(study["folds"]) != 30
    ):
        raise ValueError("complete the registered interval-input study")
    identity = study["identity"]
    for name, path in (
        ("script_sha256", ROOT / "scripts/train_interval_forecast_gate.py"),
        ("input_module_sha256", ROOT / "scripts/interval_gate_inputs.py"),
        ("trainer_sha256", ROOT / "scripts/train_shared_forecast_gate.py"),
        ("model_sha256", ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        ("protocol_sha256", args.protocol),
        ("aligned_sha256", args.aligned_root / "manifest.json"),
        ("accuracy_sha256", args.accuracy_root / "manifest.json"),
        ("quantile_audit_sha256", args.quantile_root / "manifest.json"),
        ("original_study_sha256", args.original_study / "manifest.json"),
        ("original_bundle_sha256", args.original_bundle / "manifest.json"),
    ):
        if file_sha256(path) != identity[name]:
            raise ValueError("a source fitting definition or input changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    for name in ("identity_sha256", "folds", "checkpoints", "models"):
        if freeze[name] != study[name]:
            raise ValueError("saved predictions differ from the pre-scoring freeze")
    if len(study["point_only_parity"]) != 2:
        raise ValueError("both point-only controls must pass exact fitting replay")
    for row in study["point_only_parity"]:
        path = args.study_root / row["path"]
        result = read_json(path)
        if (
            file_sha256(path) != row["sha256"]
            or result["status"] != "passed"
            or result["maximum_parameter_difference"] != 0
            or result["identity_sha256"] != study["identity_sha256"]
        ):
            raise ValueError("an original-control parity check is incomplete")
    original = read_json(args.original_study / "manifest.json")
    original_folds = {}
    for row in original["folds"]:
        path = args.original_study / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("an original control fold changed")
        record = read_json(path)
        original_folds[(record["model_id"], record["held_family"])] = record
    truth_path = args.accuracy_root / "truth_z.npy"
    if (
        file_sha256(truth_path)
        != read_json(args.accuracy_root / "manifest.json")["prediction_arrays"][truth_path.name]
    ):
        raise ValueError("validation future labels changed")
    scores = pd.read_parquet(args.study_root / "decision_scores.parquet")
    torch.set_num_threads(1)
    verified, max_prediction_difference, all_scores, widths_checked = 0, 0.0, [], 0
    keep = [index for index in range(33) if index not in (HAS_INDEX, WIDTH_INDEX)]
    for model_id in ("chronos2", "timesfm2p5"):
        _, decisions, _, base, features, vectors, _ = load_interval_inputs(args, model_id)
        model_info = next(row for row in study["models"] if row["model_id"] == model_id)
        path = args.study_root / model_info["features_path"]
        if file_sha256(path) != model_info["features_sha256"]:
            raise ValueError("fitted interval features changed")
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), features)
        np.testing.assert_array_equal(features[:, :, keep], base[:, :, keep])
        np.testing.assert_array_equal(
            features[:, :, HAS_INDEX], np.ones(features.shape[:2], np.float32)
        )
        joint_width = np.load(
            args.quantile_root / f"{model_id}_width_joint.npy", allow_pickle=False
        )
        target_width = np.load(
            args.quantile_root / f"{model_id}_width_by_target.npy", allow_pickle=False
        )
        for index, row in enumerate(decisions.itertuples(index=False)):
            expected = (
                joint_width[row.episode_index]
                if row.target_slot == -1
                else target_width[row.episode_index, :, row.target_slot]
            )
            np.testing.assert_array_equal(
                features[index, :, WIDTH_INDEX], expected.astype(np.float32)
            )
            widths_checked += 7
        for family in [None, *sorted(decisions.family_id.unique())]:
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train")
                & ((decisions.family_id.to_numpy() != family) if family is not None else True)
            )
            evaluation = (
                np.array([], dtype=int)
                if family is None
                else np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
            )
            training = decisions.iloc[train]
            if family in set(training.family_id) or set(training.origin_id) & set(
                decisions.iloc[evaluation].origin_id
            ):
                raise ValueError("held-family source separation failed")
            weight = _family_weights(training)
            values = features[train].astype(float)
            mean = (values * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 7)
            variance = ((values - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                weight.sum() * 7
            )
            scale = np.maximum(np.sqrt(variance), 1e-6)
            selected = [
                row
                for row in study["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] == family
            ]
            if [row["seed"] for row in selected] != [5101, 5102, 5103]:
                raise ValueError("the prespecified source seed coverage changed")
            seed_weights = []
            for row in selected:
                path = args.study_root / row["path"]
                if file_sha256(path) != row["sha256"]:
                    raise ValueError("a source interval checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != study["identity_sha256"]
                    or saved["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                    or saved["model_id"] != model_id
                    or saved["held_family"] != family
                    or saved["seed"] != row["seed"]
                    or set(saved["training_origins"]) != set(training.origin_id)
                    or set(saved["training_families"]) != set(training.family_id)
                ):
                    raise ValueError("a source checkpoint belongs to another fitting population")
                state = saved["state_dict"]
                count = sum(
                    value.numel()
                    for name, value in state.items()
                    if name not in ("feature_mean", "feature_scale")
                )
                if count != 1096 or len(saved["training_history"]) != 25:
                    raise ValueError("capacity or epoch budget changed")
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().ravel(), mean.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().ravel(), scale.astype(np.float32)
                )
                if len(evaluation):
                    seed_weights.append(replay_network(state, features[evaluation]))
                verified += 1
            if family is None:
                continue
            fold = next(
                row
                for row in study["folds"]
                if row["model_id"] == model_id and row["held_family"] == family
            )
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a saved interval prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["decision_indices"], evaluation)
                np.testing.assert_array_equal(saved["seed_weights"], np.stack(seed_weights))
                methods, points = saved["methods"].tolist(), saved["point"]
            old_fold = original_folds[(model_id, family)]
            old_path = args.original_study / old_fold["predictions_path"]
            if file_sha256(old_path) != fold["original_prediction_sha256"]:
                raise ValueError("an original point-only comparison changed")
            with np.load(old_path, allow_pickle=False) as saved:
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    np.testing.assert_array_equal(points[methods.index(method)], point)
            expected_weights = [
                ("interval_gate", np.mean(seed_weights, axis=0)),
                *[
                    (f"interval_seed{seed}", weight)
                    for seed, weight in zip((5101, 5102, 5103), seed_weights, strict=True)
                ],
            ]
            for method, probability in expected_weights:
                probability = probability / probability.sum(1, keepdims=True)
                predicted = (vectors[evaluation] * probability[:, :, None]).sum(1)
                previous = points[methods.index(method)]
                max_prediction_difference = max(
                    max_prediction_difference, float(abs(predicted - previous).max())
                )
                np.testing.assert_allclose(predicted, previous, rtol=1e-12, atol=1e-12)
            frame = decisions.iloc[evaluation]
            truth = decision_truth(frame, np.load(truth_path, mmap_mode="r"))
            for method, point in zip(methods, points, strict=True):
                current = frame.assign(
                    model_id=model_id,
                    method=method,
                    mae=abs(point - truth).mean(1),
                    mse=((point - truth) ** 2).mean(1),
                )
                previous = (
                    scores[(scores.model_id == model_id) & (scores.method == method)]
                    .set_index("episode_id")
                    .loc[current.episode_id]
                )
                np.testing.assert_array_equal(previous.mae.to_numpy(), current.mae.to_numpy())
                np.testing.assert_array_equal(previous.mse.to_numpy(), current.mse.to_numpy())
                all_scores.append(current)
    if verified != 96:
        raise ValueError("interval-source checkpoint coverage is incomplete")
    scored = pd.concat(all_scores, ignore_index=True)
    episodes, family, summary = aggregate(scored)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_checkpoints": verified,
            "verified_scores": len(scored),
            "candidate_interval_fields_checked": widths_checked,
            "maximum_prediction_difference": max_prediction_difference,
            "old_controls_preserved_exactly": True,
            "point_only_parity_checks": 2,
            "limits": "source validation only; no independent target evidence is added",
        },
    )


if __name__ == "__main__":
    main()
