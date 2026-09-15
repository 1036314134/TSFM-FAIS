"""Independently reconstruct supplemental inputs, training budgets and source scores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, read_json
from r6_policy_inputs import decision_inputs
from replay_preforecast_student import assemble_selected_context
from source_expansion_inputs import load_expanded_inputs
from train_latent_source_gates import aggregate
from train_source_expansion import arguments

from tsfm_fais.data import MaskingSpec, load_dataset, load_manifest, mask_time_series, stable_seed
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256, load_utility_config


def audit_raw_inputs(args):
    inventory, prepared = read_json(args.inventory), read_json(args.prepared_root / "manifest.json")
    if prepared["status"] != "completed" or prepared["identity"]["inventory_sha256"] != file_sha256(
        args.inventory
    ):
        raise ValueError("the raw supplemental preparation changed")
    config = load_utility_config(ROOT / "configs/iclr27-r3/development_expanded.yaml")
    catalog = load_manifest(config.data_manifest)
    observed, checked = 0, 0
    for item_record in inventory["items"]:
        rows = [
            row
            for row in prepared["episodes"]
            if row["dataset_id"] == item_record["dataset_id"]
            and row["item_id"] == item_record["item_id"]
        ]
        if not rows:
            if item_record["added_train"]:
                raise ValueError("a registered source item was omitted")
            continue
        for name, digest in item_record["sources"].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError("a raw data source changed")
        spec = catalog.get(item_record["dataset_id"])
        item = next(
            item
            for item in load_dataset(spec)[: config.max_items]
            if item.item_id == item_record["item_id"]
        )
        if set(row["origin"] for row in rows) != set(item_record["added_train"]) or len(
            rows
        ) != 18 * len(item_record["added_train"]):
            raise ValueError("the added origin population changed")
        all_origins = sorted([*item_record["old_train"], *item_record["added_train"]])
        if len(all_origins) != len(set(all_origins)) or np.any(np.diff(all_origins) < 192):
            raise ValueError("training window intervals overlap")
        prefix = item.values[: item_record["prefix_end"]]
        for mechanism in config.mechanisms:
            for rate in config.missing_rates:
                masked = mask_time_series(
                    item.values,
                    MaskingSpec(mechanism, rate, config.block_lengths),
                    stable_seed(
                        config.protocol_id,
                        spec.dataset_id,
                        item.item_id,
                        "train",
                        mechanism,
                        rate,
                        6101,
                    ),
                    calibration_values=prefix,
                )
                for row in (
                    part
                    for part in rows
                    if part["mechanism"] == mechanism and part["missing_rate"] == rate
                ):
                    path = args.prepared_root / row["path"]
                    if (
                        file_sha256(path) != row["sha256"]
                        or row["split"] != "train"
                        or row["mask_seed"] != 6101
                    ):
                        raise ValueError("a source input changed or entered the wrong split")
                    origin = row["origin"]
                    if origin - 96 < item_record["prefix_end"] or origin + 96 > int(
                        len(item.values) * config.temporal_boundary
                    ):
                        raise ValueError("a supplemental window crossed its temporal boundary")
                    with np.load(path, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(
                            saved["context"], masked.values[origin - 96 : origin]
                        )
                        np.testing.assert_array_equal(
                            saved["clean_context"], item.values[origin - 96 : origin]
                        )
                        np.testing.assert_array_equal(
                            saved["future"], item.values[origin : origin + 96]
                        )
                        if (
                            saved["candidate_ids"].tolist() != list(config.candidate_ids)
                            or not np.isfinite(saved["candidate_values"]).all()
                        ):
                            raise ValueError("candidate identities or finite coverage changed")
                        mask = np.isfinite(saved["context"])
                        for candidate in saved["candidate_values"]:
                            np.testing.assert_array_equal(candidate[mask], saved["context"][mask])
                        observed += int(mask.sum())
                    checked += 1
    if checked != 2826:
        raise ValueError("the supplemental raw-input audit is incomplete")
    return prepared, observed


def audit_forecast_inputs(args, prepared):
    supplement = read_json(args.supplement_root / "manifest.json")
    if supplement["status"] != "completed" or supplement["identity"][
        "prepared_sha256"
    ] != file_sha256(args.prepared_root / "manifest.json"):
        raise ValueError("the supplemental prediction collection changed")
    standards = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    original = {row["episode_id"]: row for row in prepared["episodes"]}
    queries = {}
    for model_id in ("chronos2", "timesfm2p5"):
        joint = model_id == "chronos2"
        model = read_json(args.supplement_root / model_id / "manifest.json")

        def query_point(values, key, borrowed, model_id=model_id, joint=joint, model=model):
            effective = np.asarray(values if joint else values[:, :2], np.float32).copy(order="C")
            effective[np.isnan(effective)] = np.nan
            if (
                hashlib.sha256(str(effective.shape).encode() + effective.tobytes()).hexdigest()
                != key
            ):
                raise ValueError("a forecast cache key does not match the effective input")
            token = (model_id, bool(borrowed), key)
            if token not in queries:
                root = args.base_root if borrowed else args.supplement_root
                with np.load(
                    root / model_id / "queries" / f"{key}.npz", allow_pickle=False
                ) as saved:
                    np.testing.assert_array_equal(saved["effective_input"], effective)
                    if (
                        str(saved["parameter_sha256"]) != model["parameter_sha256"]
                        or saved["point"].shape != (96, 2)
                        or not np.isfinite(saved["point"]).all()
                    ):
                        raise ValueError("a cached point forecast is incomplete or changed")
                    queries[token] = saved["point"].copy()
            return queries[token]

        for row in model["episodes"]:
            record = original[row["episode_id"]]
            with (
                np.load(args.prepared_root / record["path"], allow_pickle=False) as raw,
                np.load(args.supplement_root / model_id / row["path"], allow_pickle=False) as saved,
            ):
                context, candidates, actions = (
                    raw["context"],
                    raw["candidate_values"],
                    raw["candidate_ids"].tolist(),
                )
                order = saved["actions"].tolist()
                if order != sorted([*actions, "guarded_direct"]):
                    raise ValueError("the candidate prediction order changed")
                scaler = standards[(record["dataset_id"], record["item_id"])]
                mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
                points = {}
                for action, key, borrowed in zip(
                    order, saved["query_keys"].tolist(), saved["query_reused"].tolist(), strict=True
                ):
                    effective = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if joint else [action, action],
                        [0, 1],
                        joint=joint,
                    )
                    points[action] = (query_point(effective, key, borrowed) - mean[:2]) / scale[:2]
                teacher = (
                    query_point(
                        raw["clean_context"],
                        str(saved["teacher_key"]),
                        bool(saved["teacher_reused"]),
                    )
                    - mean[:2]
                ) / scale[:2]
                truth = (raw["future"][:, :2] - mean[:2]) / scale[:2]
                inputs = decision_inputs(
                    context,
                    candidates,
                    actions,
                    raw["native_coverage"],
                    np.stack([points[action] for action in [*actions, "guarded_direct"]]),
                    mean,
                    scale,
                    joint=joint,
                    period=record["period"],
                    metadata={**record, "episode_index": -1, "model_id": model_id},
                )
                np.testing.assert_array_equal(inputs["gate_features"], saved["features"])
                np.testing.assert_array_equal(inputs["vectors"], saved["vectors"])
                for name, values in (("truth", truth), ("teacher", teacher)):
                    target = np.stack(
                        [
                            values.reshape(-1) if slot == -1 else values[:, slot]
                            for slot in inputs["decisions"].target_slot
                        ]
                    )
                    np.testing.assert_array_equal(target, saved[name])
                expected = inputs["decisions"].assign(
                    source_episode_id=record["episode_id"], episode_index=-1
                )
                pd.testing.assert_frame_equal(
                    expected, pd.DataFrame(json.loads(str(saved["decisions"]))), check_dtype=False
                )
        print(f"{model_id}: supplemental prediction inputs independently reconstructed", flush=True)
    return len(queries)


def main():
    parser = arguments(__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed source learning-curve audit")
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"]
        != file_sha256(ROOT / "scripts/train_source_expansion.py")
        or study["identity"]["input_module_sha256"]
        != file_sha256(ROOT / "scripts/source_expansion_inputs.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("source learning definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["outer_validation_outcomes_read"]
    ):
        raise ValueError("the outer prediction freeze changed")
    torch.set_num_threads(1)
    prepared, observed = audit_raw_inputs(args)
    query_count = audit_forecast_inputs(args, prepared)
    reference = read_json(args.reference_root / "manifest.json")
    verified, maximum_delta, maximum_gap = 0, 0.0, 0.0
    all_scores = []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, arrays = load_expanded_inputs(
            args.base_root, args.supplement_root, args.accuracy_root, model_id
        )
        vectors, features = arrays["vectors"], arrays["features"]
        if np.any(features[:, :, 33:]):
            raise ValueError("the point-only study received latent inputs")
        _, _, _, gram = forecast_geometry(vectors)
        train = np.flatnonzero(frame.split.to_numpy() == "train")
        labels = {
            "teacher": projection_targets(vectors, arrays["teacher"])["raw_projection"],
            "future": np.full(vectors.shape[:2], np.nan),
        }
        labels["future"][train] = projection_targets(vectors[train], arrays["truth"][train])[
            "raw_projection"
        ]
        for fold in (row for row in study["folds"] if row["model_id"] == model_id):
            family = fold["held_family"]
            allowed = (
                train
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            original = allowed[frame.iloc[allowed].source_population.to_numpy() == "original"]
            np.testing.assert_array_equal(fold["allowed"], allowed)
            np.testing.assert_array_equal(fold["original"], original)
            updates = ((len(allowed) + 127) // 128) * 25
            if updates != fold["updates"]:
                raise ValueError("the matched update budget changed")
            validation = (
                np.array([], np.int64)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            states = {}
            for entry in (
                row
                for row in study["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] == family
            ):
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a source-learning checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                indices = allowed if entry["regime"] == "expanded" else original
                if (
                    saved["metadata"]
                    != {
                        name: value
                        for name, value in entry.items()
                        if name not in ("path", "sha256")
                    }
                    or saved["updates"] != updates
                    or saved["epochs_started"]
                    != (updates + ((len(indices) + 127) // 128) - 1)
                    // ((len(indices) + 127) // 128)
                    or saved["train_indices_sha256"]
                    != hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()
                    or set(saved["training_origins"]) != set(frame.iloc[indices].origin_id)
                    or set(saved["training_families"]) != set(frame.iloc[indices].family_id)
                    or saved["initial_parameter_sha256"]
                    != reference["initial_parameters"][str(entry["seed"])]
                ):
                    raise ValueError("source population, initialization or update schedule changed")
                weights = _family_weights(frame.iloc[indices])
                values = features[indices].astype(float)
                mean = (values * weights[:, None, None]).sum((0, 1)) / (weights.sum() * 7)
                variance = ((values - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / (
                    weights.sum() * 7
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
                    raise ValueError("source gate capacity changed")
                states[(entry["regime"], entry["label"], entry["seed"])] = state
                verified += 1
            control_path = args.study_root / fold["controls_path"]
            if file_sha256(control_path) != fold["controls_sha256"]:
                raise ValueError("an expanded fixed control changed")
            controls = read_json(control_path)
            expected = {}
            for label, alignment in labels.items():
                weights = _family_weights(frame.iloc[allowed])
                weights /= weights.sum()
                g = np.einsum("n,nab->ab", weights, gram[allowed])
                b = np.einsum("n,na->a", weights, alignment[allowed])
                fixed = np.asarray(controls[label]["weights"])
                gradient = 2 * (g @ fixed - b)
                gap = float(
                    (gradient @ fixed - gradient.min()) / max(abs(g).max(), abs(b).max(), 1e-12)
                )
                single = int((g.diagonal() - 2 * b).argmin())
                if (
                    gap > 1e-7
                    or (fixed < 0).any()
                    or abs(fixed.sum() - 1) > 1e-10
                    or single != controls[label]["single_index"]
                ):
                    raise ValueError("an expanded fixed control is not optimal")
                maximum_gap = max(maximum_gap, gap)
                if len(validation):
                    expected[f"expanded_fixed_{label}"] = (
                        vectors[validation] * fixed[None, :, None]
                    ).sum(1)
                    expected[f"expanded_single_{label}"] = vectors[validation, single]
                    for regime in ("expanded", "steps_matched"):
                        probabilities = [
                            replay_network(states[(regime, label, seed)], features[validation])
                            for seed in (5101, 5102, 5103)
                        ]
                        for name, probability in [
                            (f"{regime}_{label}", np.mean(probabilities, axis=0)),
                            *[
                                (f"{regime}_{label}_seed{seed}", weight)
                                for seed, weight in zip(
                                    (5101, 5102, 5103), probabilities, strict=True
                                )
                            ],
                        ]:
                            probability = probability / probability.sum(1, keepdims=True)
                            expected[name] = (vectors[validation] * probability[:, :, None]).sum(1)
            if not len(validation):
                continue
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("an outer prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                points = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for method, point in expected.items():
                maximum_delta = max(maximum_delta, float(abs(point - points[method]).max()))
                np.testing.assert_allclose(point, points[method], rtol=1e-12, atol=1e-12)
            old_fold = next(
                row
                for row in reference["folds"]
                if row["model_id"] == model_id and row["held_family"] == family
            )
            with np.load(
                args.reference_root / old_fold["prediction_path"], allow_pickle=False
            ) as saved:
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    np.testing.assert_array_equal(point, points[method])
            truth = decision_truth(
                frame.iloc[validation], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
            )
            for method, point in points.items():
                all_scores.append(
                    frame.iloc[validation].assign(
                        model_id=model_id,
                        method=method,
                        mae=abs(point - truth).mean(1),
                        mse=((point - truth) ** 2).mean(1),
                    )
                )
        print(f"{model_id}: 192 source learning-curve fits and predictions checked", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
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
    old = pd.read_csv(args.reference_audit / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(old.method.unique())].reset_index(drop=True),
        old,
        check_exact=True,
    )
    if verified != 384 or len(scores) != 117936:
        raise ValueError("the source expansion audit is incomplete")
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
            "verified_scores": len(scores),
            "verified_supplemental_inputs": 2826,
            "observed_coordinates_verified": observed,
            "verified_query_inputs": query_count,
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "source development learning curve; no new independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
