"""Audit MoTM provenance, pool representations, matched source fitting and scores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_metric_source_gates import direct_control
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, load_source_inputs, read_json
from pool_gate_inputs import load_pool_inputs
from train_latent_source_gates import aggregate
from train_motm_pool import arguments, checked_sources

from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES
from tsfm_fais.routing.utility import _family_weights, response_features, sequence_features
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_features(context, candidates, actions, coverage, points, mean, scale, period, joint):
    names = sorted([*actions, "guarded_direct"])
    empty = ~np.isfinite(context).any(0)
    locf = candidates[actions.index("locf")]
    reference = points[names.index("locf")]
    pool = np.median(points[[names.index(name) for name in actions]], axis=0)
    rows = []
    for targets in ([0, 1],) if joint else ([0], [1]):
        current = []
        for action in names:
            direct = action == "guarded_direct"
            values = locf if direct else candidates[actions.index(action)]
            fields = sequence_features(
                context - mean,
                values - mean,
                locf - mean,
                scale,
                targets,
                period=period,
                native_coverage=1.0 if direct else float(coverage[actions.index(action)]),
            )
            fields.update(
                {
                    "static.direct_missing": float(direct),
                    "static.empty_channel_fraction": float(empty.mean()),
                    "static.empty_target_fraction": float(empty[targets].mean()),
                    "static.fallback_target_fraction": float(
                        empty.any() if joint else empty[targets].mean()
                    )
                    if direct
                    else 0.0,
                }
            )
            last = (locf[-1, targets] - mean[targets]) / scale[targets]
            fields.update(
                response_features(
                    points[names.index(action)][:, targets],
                    reference[:, targets],
                    pool[:, targets],
                    last,
                    np.ones(len(targets)),
                    None,
                )
            )
            current.append([fields[name] for name in FORECAST_FEATURES])
        rows.append(current)
    return np.asarray(rows, np.float32)


def audit_pool_inputs(args, model_id, frame, arrays):
    source = read_json(args.source_root / "episodes_manifest.json")
    prepared = read_json(args.motm_root / "manifest.json")
    pool = read_json(args.pool_root / "manifest.json")
    if (
        prepared["status"] != "completed"
        or not prepared["pretrained_networks_unchanged"]
        or pool["identity"]["motm_sha256"] != file_sha256(args.motm_root / "manifest.json")
        or prepared["identity"]["source_sha256"]
        != file_sha256(args.source_root / "episodes_manifest.json")
    ):
        raise ValueError("the source imputation provenance changed")
    if prepared["identity"]["module_sha256"] != file_sha256(
        ROOT / "src/tsfm_fais/imputers/motm.py"
    ):
        raise ValueError("the verified MoTM implementation changed")
    motm_rows = {row["episode_id"]: row for row in prepared["episodes"]}
    _, old_frame, old_arrays = load_source_inputs(args.base_root, model_id)
    model = read_json(args.pool_root / model_id / "manifest.json")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    queries = {}
    observed_count = 0
    for position, record in enumerate(model["episodes"]):
        row = source["episodes"][record["source_index"]]
        raw_path = args.source_root / row["path"]
        if file_sha256(raw_path) != row["sha256"]:
            raise ValueError("an original source input changed")
        extra_record = motm_rows[row["episode_id"]]
        extra_path = args.motm_root / extra_record["path"]
        if file_sha256(extra_path) != extra_record["sha256"]:
            raise ValueError("a source MoTM completion changed")
        with (
            np.load(raw_path, allow_pickle=False) as raw,
            np.load(extra_path, allow_pickle=False) as extra,
            np.load(args.pool_root / model_id / record["path"], allow_pickle=False) as saved,
        ):
            context, candidates, actions, coverage = (
                raw["context"],
                raw["candidate_values"],
                raw["candidate_ids"].tolist(),
                raw["native_coverage"],
            )
            completed = extra["values"]
            missing = ~np.isfinite(context)
            available = np.isfinite(context).any(0)
            diagnostics = json.loads(str(extra["diagnostics"]))
            np.testing.assert_array_equal(completed[~missing], context[~missing])
            observed_count += int((~missing).sum())
            if (
                not np.isfinite(completed).all()
                or diagnostics["fallback_columns"] != np.flatnonzero(~available).tolist()
                or diagnostics["fitted_variables"] != (int(available.sum()) if missing.any() else 0)
            ):
                raise ValueError("MoTM coverage or local fitting population changed")
            np.testing.assert_array_equal(
                completed[:, ~available], candidates[actions.index("locf")][:, ~available]
            )
            expected_coverage = (
                float(missing[:, available].sum() / missing.sum()) if missing.any() else 1.0
            )
            np.testing.assert_equal(float(extra["native_coverage"]), expected_coverage)
            effective = np.asarray(
                completed if model_id == "chronos2" else completed[:, :2], np.float32
            ).copy(order="C")
            key = hashlib.sha256(str(effective.shape).encode() + effective.tobytes()).hexdigest()
            if key != str(saved["motm_query"]):
                raise ValueError("a MoTM forecast query has the wrong input")
            borrowed = bool(saved["query_borrowed"])
            token = (borrowed, key)
            if token not in queries:
                query_path = (
                    (args.base_root if borrowed else args.pool_root)
                    / model_id
                    / "queries"
                    / f"{key}.npz"
                )
                with np.load(query_path, allow_pickle=False) as queried:
                    np.testing.assert_array_equal(queried["effective_input"], effective)
                    if str(queried["parameter_sha256"]) != model["parameter_sha256"]:
                        raise ValueError("a MoTM forecast uses changed forecasting parameters")
                    queries[token] = queried["point"].copy()
            indices = (
                np.array([position])
                if model_id == "chronos2"
                else np.array([2 * position, 2 * position + 1])
            )
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            extra_point = (queries[token] - mean[:2]) / scale[:2]
            old_points = (
                old_arrays["vectors"][indices][0].reshape(7, 96, 2)
                if model_id == "chronos2"
                else np.stack(old_arrays["vectors"][indices], axis=-1)
            )
            old_names = sorted([*actions, "guarded_direct"])
            names = sorted([*old_names, "motm_reference"])
            if saved["actions"].tolist() != names:
                raise ValueError("the eight-candidate ordering changed")
            points = np.stack(
                [
                    extra_point if name == "motm_reference" else old_points[old_names.index(name)]
                    for name in names
                ]
            )
            rebuilt = (
                points[None].reshape(1, 8, 192)
                if model_id == "chronos2"
                else np.stack([points[:, :, 0], points[:, :, 1]])
            )
            np.testing.assert_array_equal(rebuilt, arrays["vectors"][indices])
            expected = direct_features(
                context,
                np.concatenate([candidates, completed[None]]),
                [*actions, "motm_reference"],
                np.r_[coverage, expected_coverage],
                points,
                mean,
                scale,
                row["period"],
                model_id == "chronos2",
            )
            np.testing.assert_array_equal(expected, arrays["features"][indices, :, :33])
            if np.any(arrays["features"][indices, :, 33:]):
                raise ValueError("the pool gate received undeclared latent features")
            for column in old_frame.columns:
                np.testing.assert_array_equal(
                    frame.iloc[indices][column], old_frame.iloc[indices][column]
                )
    return len(queries), observed_count


def main():
    parser = arguments(__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed pool audit")
    metric = checked_sources(args)
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"] != file_sha256(ROOT / "scripts/train_motm_pool.py")
        or study["identity"]["model_module_sha256"]
        != file_sha256(ROOT / "scripts/pool_gate_model.py")
        or study["identity"]["input_module_sha256"]
        != file_sha256(ROOT / "scripts/pool_gate_inputs.py")
        or study["identity"]["loss_module_sha256"]
        != file_sha256(ROOT / "scripts/metric_source_gate.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("the pool fitting definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["validation_future_arrays_read"]
    ):
        raise ValueError("the pool prediction freeze changed")
    torch.set_num_threads(1)
    prepared = read_json(args.motm_root / "manifest.json")
    reference_root = ROOT / "artifacts/iclr27-r5/motm-reference-v001"
    runtime_root = ROOT / "artifacts/iclr27-r5/motm-runtime-v001"
    if (
        file_sha256(reference_root / "manifest.json") != prepared["identity"]["reference_sha256"]
        or file_sha256(runtime_root / "manifest.json") != prepared["identity"]["runtime_sha256"]
    ):
        raise ValueError("the MoTM reference or runtime changed")
    reference_imputer = MOTMReference(
        reference_root, runtime_root, device="cpu", ridge=0.5, batch_size=32
    )
    if reference_imputer.initial_digest != prepared["network_parameter_sha256"]:
        raise ValueError("the source imputer parameter digest differs from the published weights")
    reference_imputer.verify_frozen()
    del reference_imputer
    for entry in study["compatibility"]:
        path = args.study_root / entry["path"]
        reference = args.metric_root / entry["original_path"]
        if (
            file_sha256(path) != entry["sha256"]
            or file_sha256(reference) != entry["original_sha256"]
        ):
            raise ValueError("a seven-candidate replay changed")
        current = torch.load(path, map_location="cpu", weights_only=True)
        old = torch.load(reference, map_location="cpu", weights_only=True)
        for name in (
            "initial_parameter_sha256",
            "train_indices_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if current[name] != old[name]:
                raise ValueError("the generalized fitting routine changed the original training")
        for name in old["state_dict"]:
            torch.testing.assert_close(
                current["state_dict"][name], old["state_dict"][name], rtol=0, atol=0
            )
    verified, query_count, observed_count, maximum_delta, maximum_gap = 0, 0, 0, 0.0, 0.0
    rows = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_pool_inputs(args.pool_root, model_id)
        queries, observed = audit_pool_inputs(args, model_id, frame, arrays)
        query_count += queries
        observed_count += observed
        features, points = arrays["features"], arrays["vectors"]
        train = np.flatnonzero(frame.split.to_numpy() == "train")
        truth = np.full((len(frame), points.shape[-1]), np.nan)
        truth[train] = decision_truth(
            frame.iloc[train], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
        )
        for fold in (row for row in study["folds"] if row["model_id"] == model_id):
            family = fold["held_family"]
            indices = (
                train
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
            np.testing.assert_array_equal(indices, fold["train_indices"])
            weight = _family_weights(frame.iloc[indices])
            values = features[indices].astype(float)
            mean = (values * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 8)
            variance = ((values - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                weight.sum() * 8
            )
            entries = [
                row
                for row in study["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] == family
            ]
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("a pool condition lost a seed")
            seeds = []
            for entry in entries:
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an eight-candidate checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["metadata"]
                    != {
                        name: value
                        for name, value in entry.items()
                        if name not in ("path", "sha256")
                    }
                    or saved["train_indices_sha256"]
                    != hashlib.sha256(indices.tobytes()).hexdigest()
                    or set(saved["training_origins"]) != set(frame.iloc[indices].origin_id)
                    or set(saved["training_families"]) != set(frame.iloc[indices].family_id)
                    or saved["initial_parameter_sha256"]
                    != study["initializations"][str(entry["seed"])]
                    or len(saved["history"]) != 25
                ):
                    raise ValueError("pool population, initialization or fitting schedule changed")
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
                    != 2121
                ):
                    raise ValueError("eight-candidate model capacity changed")
                if len(validation):
                    seeds.append(replay_network(state, features[validation]))
                verified += 1
            path = args.study_root / fold["control_path"]
            if file_sha256(path) != fold["control_sha256"]:
                raise ValueError("a matched pool fixed control changed")
            control = read_json(path)
            probability = np.asarray(control["weights"])
            value, gradient = direct_control(
                points[indices], truth[indices], weight, probability, "joint"
            )
            gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
            if gap > 1e-7 or (probability < 0).any() or abs(probability.sum() - 1) > 1e-10:
                raise ValueError("pool fixed optimality failed")
            maximum_gap = max(maximum_gap, gap)
            np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
            single_values = [
                direct_control(
                    points[indices], truth[indices], weight, np.eye(8)[candidate], "joint"
                )[0]
                for candidate in range(8)
            ]
            if int(np.argmin(single_values)) != control["single_index"]:
                raise ValueError("pool fixed single selection changed")
            if not len(validation):
                continue
            expected = {
                "pool8_fixed_joint_future": (points[validation] * probability[None, :, None]).sum(
                    1
                ),
                "pool8_single_joint_future": points[validation, control["single_index"]],
                "pool8_mean": points[validation].mean(1),
                "pool8_median": np.median(points[validation], axis=1),
                "pool8_motm_reference": points[validation, fold["actions"].index("motm_reference")],
            }
            for method, probability in [
                ("pool8_joint_future", np.mean(seeds, axis=0)),
                *[
                    (f"pool8_joint_future_seed{seed}", probability)
                    for seed, probability in zip((5101, 5102, 5103), seeds, strict=True)
                ],
            ]:
                probability = probability / probability.sum(1, keepdims=True)
                expected[method] = (points[validation] * probability[:, :, None]).sum(1)
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a pool validation prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for method, point in expected.items():
                maximum_delta = max(maximum_delta, float(abs(point - predictions[method]).max()))
                np.testing.assert_allclose(point, predictions[method], rtol=1e-12, atol=1e-12)
            old = next(
                row
                for row in metric["folds"]
                if row["model_id"] == model_id and row["held_family"] == family
            )
            with np.load(args.metric_root / old["prediction_path"], allow_pickle=False) as saved:
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    np.testing.assert_array_equal(point, predictions[method])
            evaluation = frame.iloc[validation]
            future = decision_truth(
                evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
            )
            for method, point in predictions.items():
                rows.append(
                    evaluation.assign(
                        method=method,
                        mae=abs(point - future).mean(1),
                        mse=((point - future) ** 2).mean(1),
                    )
                )
        print(f"{model_id}: eight-candidate inputs, controls and models audited", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    previous = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        previous.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    old = pd.read_csv(args.metric_audit / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(old.method.unique())].reset_index(drop=True),
        old,
        check_exact=True,
    )
    if verified != 96 or len(scores) != 154440 or len(study["compatibility"]) != 2:
        raise ValueError("the pool study audit is incomplete")
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
            "verified_original_model_replays": 2,
            "verified_scores": len(scores),
            "verified_motm_forecast_inputs": query_count,
            "observed_coordinates_checked_across_models": observed_count,
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "matched eight-candidate source development; no new independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
