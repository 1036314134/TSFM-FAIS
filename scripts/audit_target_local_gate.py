"""Reconstruct target-local features and matched-budget source gate predictions."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_metric_source_gates import direct_control
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, load_source_inputs, read_json
from target_local_inputs import load_target_local
from train_latent_source_gates import aggregate
from train_target_local_gate import arguments, checked_reference

from tsfm_fais.routing.forecast_response import FORECAST_FEATURES
from tsfm_fais.routing.utility import _family_weights, response_features, sequence_features
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_target_features(
    context, candidates, actions, coverage, points, mean, scale, period, joint
):
    names = sorted([*actions, "guarded_direct"])
    locf = candidates[actions.index("locf")]
    empty = ~np.isfinite(context).any(0)
    pool = np.median(points[[names.index(name) for name in actions]], axis=0)
    reference = points[names.index("locf")]
    values = []
    for slot in (0, 1):
        target_rows = []
        for action in names:
            direct = action == "guarded_direct"
            completed = locf if direct else candidates[actions.index(action)]
            result = sequence_features(
                context - mean,
                completed - mean,
                locf - mean,
                scale,
                [slot],
                period=period,
                native_coverage=1.0 if direct else float(coverage[actions.index(action)]),
            )
            result.update(
                {
                    "static.direct_missing": float(direct),
                    "static.empty_channel_fraction": float(empty.mean()),
                    "static.empty_target_fraction": float(empty[slot]),
                    "static.fallback_target_fraction": float(empty.any() if joint else empty[slot])
                    if direct
                    else 0.0,
                }
            )
            last = (locf[-1, [slot]] - mean[[slot]]) / scale[[slot]]
            result.update(
                response_features(
                    points[names.index(action)][:, [slot]],
                    reference[:, [slot]],
                    pool[:, [slot]],
                    last,
                    np.ones(1),
                    None,
                )
            )
            target_rows.append([result[name] for name in FORECAST_FEATURES])
        values.append(target_rows)
    return np.asarray(values, np.float32)


def audit_inputs(args, frame, arrays):
    source = read_json(args.source_root / "episodes_manifest.json")
    _, original, c_arrays = load_source_inputs(args.base_root, "chronos2")
    _, t_frame, t_arrays = load_source_inputs(args.base_root, "timesfm2p5")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    for index, base in enumerate(original.itertuples(index=False)):
        record = source["episodes"][base.episode_index]
        path = args.source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("an original source input changed")
        current = frame.iloc[2 * index : 2 * index + 2]
        if (
            current.base_position.tolist() != [index, index]
            or current.source_episode_id.tolist() != [record["episode_id"]] * 2
            or current.target_slot.tolist() != [0, 1]
        ):
            raise ValueError("a target decision has the wrong source or target")
        with np.load(path, allow_pickle=False) as raw:
            context, candidates, actions, coverage = (
                raw["context"],
                raw["candidate_values"],
                raw["candidate_ids"].tolist(),
                raw["native_coverage"],
            )
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        points = c_arrays["vectors"][index].reshape(7, 96, 2)
        np.testing.assert_array_equal(arrays["vectors"][2 * index], points[:, :, 0])
        np.testing.assert_array_equal(arrays["vectors"][2 * index + 1], points[:, :, 1])
        np.testing.assert_array_equal(
            arrays["broadcast_features"][2 * index : 2 * index + 2],
            np.repeat(c_arrays["features"][index : index + 1, :, :33], 2, axis=0),
        )
        expected = direct_target_features(
            context, candidates, actions, coverage, points, mean, scale, record["period"], True
        )
        np.testing.assert_array_equal(
            expected, arrays["target_features"][2 * index : 2 * index + 2]
        )
        if (
            t_frame.iloc[2 * index : 2 * index + 2].source_episode_id.tolist()
            != [record["episode_id"]] * 2
        ):
            raise ValueError("the original TimesFM source mapping changed")
        t_points = np.stack(
            [t_arrays["vectors"][2 * index], t_arrays["vectors"][2 * index + 1]], axis=-1
        )
        t_expected = direct_target_features(
            context, candidates, actions, coverage, t_points, mean, scale, record["period"], False
        )
        np.testing.assert_array_equal(
            t_expected, t_arrays["features"][2 * index : 2 * index + 2, :, :33]
        )
    return len(frame)


def main():
    parser = arguments(__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed target-local audit")
    prep, frame, arrays = load_target_local(args.prepared_root)
    metric, reference = checked_reference(args, prep)
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"]
        != file_sha256(ROOT / "scripts/train_target_local_gate.py")
        or study["identity"]["module_sha256"]
        != file_sha256(ROOT / "scripts/target_local_inputs.py")
        or study["identity"]["fit_module_sha256"]
        != file_sha256(ROOT / "scripts/train_metric_source_gates.py")
        or study["identity"]["loss_module_sha256"]
        != file_sha256(ROOT / "scripts/metric_source_gate.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("target-local training definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["validation_future_arrays_read"]
    ):
        raise ValueError("the target-local prediction freeze changed")
    torch.set_num_threads(1)
    checked_inputs = audit_inputs(args, frame, arrays)
    training = np.flatnonzero(frame.split.to_numpy() == "train")
    points = arrays["vectors"]
    truth = np.full((len(frame), 96), np.nan)
    truth[training] = decision_truth(
        frame.iloc[training], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    )
    verified, maximum_delta, maximum_gap = 0, 0.0, 0.0
    records = []
    for fold in study["folds"]:
        family = fold["held_family"]
        indices = (
            training
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
                (frame.split.to_numpy() == "validation") & (frame.family_id.to_numpy() == family)
            )
        )
        expected = {}
        weight = _family_weights(frame.iloc[indices])
        for mode in ("broadcast", "target"):
            features = np.ascontiguousarray(
                np.pad(arrays[mode + "_features"], ((0, 0), (0, 0), (0, 64)))
            )
            values = features[indices].astype(float)
            mean = (values * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 7)
            variance = ((values - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                weight.sum() * 7
            )
            entries = [
                row
                for row in study["checkpoints"]
                if row["held_family"] == family and row["mode"] == mode
            ]
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("a target-feature condition lost a seed")
            seeds = []
            for entry in entries:
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a target-local checkpoint changed")
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
                    raise ValueError("target-local population or training schedule changed")
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
                    raise ValueError("matched target-local capacity changed")
                if len(validation):
                    probability = replay_network(state, features[validation])
                    if mode == "broadcast":
                        np.testing.assert_array_equal(probability[::2], probability[1::2])
                    seeds.append(probability)
                verified += 1
            if len(validation):
                for name, probability in [
                    (f"scope_{mode}", np.mean(seeds, axis=0)),
                    *[
                        (f"scope_{mode}_seed{seed}", probability)
                        for seed, probability in zip((5101, 5102, 5103), seeds, strict=True)
                    ],
                ]:
                    probability = probability / probability.sum(1, keepdims=True)
                    expected[name] = (points[validation] * probability[:, :, None]).sum(1)
        fixed = np.zeros((len(validation), 96))
        single = fixed.copy()
        for slot in (0, 1):
            selected = indices[frame.iloc[indices].target_slot.to_numpy() == slot]
            entry = fold["controls"][str(slot)]
            path = args.study_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a target-specific fixed control changed")
            control = read_json(path)
            probability = np.asarray(control["weights"])
            value, gradient = direct_control(
                points[selected],
                truth[selected],
                _family_weights(frame.iloc[selected]),
                probability,
                "joint",
            )
            gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
            if (
                gap > 1e-7
                or (probability < 0).any()
                or abs(probability.sum() - 1) > 1e-10
                or control["train_indices_sha256"] != hashlib.sha256(selected.tobytes()).hexdigest()
            ):
                raise ValueError("target-specific fixed optimality or population changed")
            maximum_gap = max(maximum_gap, gap)
            np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
            singles = [
                direct_control(
                    points[selected],
                    truth[selected],
                    _family_weights(frame.iloc[selected]),
                    np.eye(7)[candidate],
                    "joint",
                )[0]
                for candidate in range(7)
            ]
            if int(np.argmin(singles)) != control["single_index"]:
                raise ValueError("the target-specific single control changed")
            if len(validation):
                mask = frame.iloc[validation].target_slot.to_numpy() == slot
                fixed[mask] = (points[validation[mask]] * probability[None, :, None]).sum(1)
                single[mask] = points[validation[mask], control["single_index"]]
        if not len(validation):
            continue
        expected["scope_fixed_by_target"] = fixed
        expected["scope_single_by_target"] = single
        path = args.study_root / fold["prediction_path"]
        if file_sha256(path) != fold["prediction_sha256"]:
            raise ValueError("a target-local prediction changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        for method, point in expected.items():
            maximum_delta = max(maximum_delta, float(abs(point - predictions[method]).max()))
            np.testing.assert_allclose(point, predictions[method], rtol=1e-12, atol=1e-12)
        old = next(
            row
            for row in metric["folds"]
            if row["model_id"] == "chronos2" and row["held_family"] == family
        )
        with np.load(args.metric_root / old["prediction_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(
                frame.iloc[validation].base_position, np.repeat(saved["validation_indices"], 2)
            )
            for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                np.testing.assert_array_equal(
                    point.reshape(len(point), 96, 2).transpose(0, 2, 1).reshape(-1, 96),
                    predictions[method],
                )
        future = decision_truth(
            frame.iloc[validation], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        for method, point in predictions.items():
            records.append(
                frame.iloc[validation].assign(
                    method=method,
                    mae=abs(point - future).mean(1),
                    mse=((point - future) ** 2).mean(1),
                )
            )
    scores = pd.concat(records, ignore_index=True)
    keys = ["method", "episode_id"]
    saved = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    old = pd.read_csv(args.metric_audit / "summary.csv", float_precision="round_trip")
    old = old[old.model_id == "chronos2"].set_index(["model_id", "method"])
    current = summary.set_index(["model_id", "method"]).loc[old.index]
    baseline_delta = float(
        abs(current[["mae", "mse"]].to_numpy() - old[["mae", "mse"]].to_numpy()).max()
    )
    np.testing.assert_allclose(current[["mae", "mse"]], old[["mae", "mse"]], rtol=1e-12, atol=1e-12)
    if verified != 96 or len(scores) != 104832:
        raise ValueError("the target-local audit is incomplete")
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
            "verified_source_decisions": checked_inputs,
            "verified_checkpoints": verified,
            "verified_scores": len(scores),
            "verified_target_fixed_controls": 32,
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "maximum_baseline_aggregation_difference": baseline_delta,
            "limits": "matched Chronos target features; TimesFM input semantics independently unchanged; reused source validation",
        },
    )


if __name__ == "__main__":
    main()
