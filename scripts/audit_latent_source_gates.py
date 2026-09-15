"""Verify projected source representations, matched fits and downstream errors."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import (
    ROOT,
    load_source_inputs,
    project_heads,
    projection_matrix,
    read_json,
)
from r6_policy_inputs import decision_inputs
from replay_preforecast_student import assemble_selected_context
from train_latent_source_gates import CONDITIONS, aggregate, condition_features

from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument(
        "--accuracy-root", type=Path, default=ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed representation audits")
    study = read_json(args.study_root / "manifest.json")
    collected = read_json(args.input_root / "manifest.json")
    if (
        study["status"] != "completed"
        or len(study["checkpoints"]) != 384
        or len(study["folds"]) != 30
    ):
        raise ValueError("complete all four source conditions before analysis")
    if file_sha256(args.input_root / "manifest.json") != study["identity"]["input_manifest_sha256"]:
        raise ValueError("the source representation input changed")
    for field, path in (
        ("script_sha256", ROOT / "scripts/train_latent_source_gates.py"),
        ("input_module_sha256", ROOT / "scripts/latent_source_inputs.py"),
        ("model_sha256", ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        ("protocol_sha256", ROOT / "docs/iclr2027/R7_LATENT_SOURCE_PROTOCOL.md"),
    ):
        if file_sha256(path) != study["identity"][field]:
            raise ValueError("a source-learning definition changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if any(freeze[key] != study[key] for key in ("identity_sha256", "checkpoints", "folds")):
        raise ValueError("validation outputs differ from the pre-scoring freeze")
    accuracy = read_json(args.accuracy_root / "manifest.json")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("original future labels changed")
    standard_path = ROOT / "artifacts/iclr27-r4/accuracy-development-v002/standardizers.json"
    if file_sha256(standard_path) != collected["identity"]["standardizers_sha256"]:
        raise ValueError("prefix standardization changed")
    scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(standard_path)}
    source_root = ROOT / "artifacts/iclr27-r3/development-expanded-v001"
    original_source = read_json(source_root / "episodes_manifest.json")
    if (
        file_sha256(source_root / "episodes_manifest.json")
        != collected["identity"]["source_manifest_sha256"]
    ):
        raise ValueError("the original source input population changed")
    stored_scores = pd.read_parquet(args.study_root / "decision_scores.parquet")
    torch.set_num_threads(1)
    verified, checked_queries, prediction_delta, all_scores, maximum_gap = 0, 0, 0.0, [], 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        source, frame, arrays = load_source_inputs(args.input_root, model_id)
        root = args.input_root / model_id
        matrix = projection_matrix(model_id)
        np.testing.assert_array_equal(matrix, np.load(root / "projection.npy", allow_pickle=False))
        if file_sha256(root / "projection.npy") != source["projection_sha256"]:
            raise ValueError("the data-independent projection changed")
        for path in sorted((root / "queries").glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["parameter_sha256"]) != source["parameter_sha256"]
                    or str(saved["identity_sha256"]) != collected["identity_sha256"]
                ):
                    raise ValueError("a query is bound to another predictor")
                effective = saved["effective_input"]
                expected_key = hashlib.sha256(
                    str(effective.shape).encode() + effective.tobytes()
                ).hexdigest()
                if path.stem != expected_key:
                    raise ValueError("effective input identity changed")
                heads = [saved[f"head_{index}"] for index in range(int(saved["head_count"]))]
                np.testing.assert_array_equal(
                    project_heads(model_id, heads, matrix), saved["projected"]
                )
                if saved["point"].shape != (96, 2) or not np.isfinite(saved["point"]).all():
                    raise ValueError("a captured forecast is incomplete")
                checked_queries += 1
        for entry in source["episodes"]:
            with np.load(root / entry["path"], allow_pickle=False) as saved:
                import json

                decisions = pd.DataFrame(json.loads(str(saved["decisions"])))
                features, vectors, teacher = saved["features"], saved["vectors"], saved["teacher"]
                actions, keys, teacher_key = (
                    saved["actions"].tolist(),
                    saved["query_keys"].tolist(),
                    str(saved["teacher_key"]),
                )
            projected, points = [], []
            for key in keys:
                with np.load(root / "queries" / f"{key}.npz", allow_pickle=False) as saved:
                    projected.append(saved["projected"])
                    points.append(saved["point"])
            latent = np.stack(projected).transpose(1, 0, 2)
            np.testing.assert_array_equal(features[:, :, 33:65], latent)
            np.testing.assert_array_equal(
                features[:, :, 65:],
                latent - latent[:, actions.index("locf") : actions.index("locf") + 1],
            )
            with np.load(root / "queries" / f"{teacher_key}.npz", allow_pickle=False) as saved:
                teacher_point = saved["point"]
            row = decisions.iloc[0]
            scaler = scalers[(row.dataset_id, row.item_id)]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            normalized = (np.stack(points) - mean) / scale
            teacher_z = (teacher_point - mean) / scale
            original = original_source["episodes"][entry["source_index"]]
            original_path = source_root / original["path"]
            if original["mask_seed"] != 6101 or file_sha256(original_path) != original["sha256"]:
                raise ValueError("a selected original history or mask changed")
            with np.load(original_path, allow_pickle=False) as saved:
                context, candidates, clean = (
                    saved["context"],
                    saved["candidate_values"],
                    saved["clean_context"],
                )
                original_actions, coverage = (
                    saved["candidate_ids"].tolist(),
                    saved["native_coverage"],
                )
            native_order = [*original_actions, "guarded_direct"]
            for action, key in zip(actions, keys, strict=True):
                current = assemble_selected_context(
                    context,
                    candidates,
                    original_actions,
                    [action] if model_id == "chronos2" else [action, action],
                    [0, 1],
                    joint=model_id == "chronos2",
                )
                effective = np.asarray(
                    current if model_id == "chronos2" else current[:, :2], dtype=np.float32
                ).copy(order="C")
                effective[np.isnan(effective)] = np.nan
                with np.load(root / "queries" / f"{key}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["effective_input"], effective)
            effective_teacher = np.asarray(
                clean if model_id == "chronos2" else clean[:, :2], dtype=np.float32
            )
            with np.load(root / "queries" / f"{teacher_key}.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["effective_input"], effective_teacher)
            rebuilt = decision_inputs(
                context,
                candidates,
                original_actions,
                coverage,
                normalized[[actions.index(name) for name in native_order]],
                np.asarray(scaler["mean"]),
                np.asarray(scaler["scale"]),
                joint=model_id == "chronos2",
                period=original["period"],
                metadata={**original, "episode_index": entry["source_index"], "model_id": model_id},
            )
            np.testing.assert_array_equal(rebuilt["gate_features"], features[:, :, :33])
            for index, slot in enumerate(decisions.target_slot):
                expected = normalized.reshape(7, -1) if slot == -1 else normalized[:, :, slot]
                target = teacher_z.reshape(-1) if slot == -1 else teacher_z[:, slot]
                np.testing.assert_array_equal(vectors[index], expected)
                np.testing.assert_array_equal(teacher[index], target)
        _, _, _, gram = forecast_geometry(arrays["vectors"])
        teacher_alignment = projection_targets(arrays["vectors"], arrays["teacher"])[
            "raw_projection"
        ]
        full_train = np.flatnonzero(frame.split.to_numpy() == "train")
        actual_train = decision_truth(frame.iloc[full_train], np.load(truth_path, mmap_mode="r"))
        actual_alignment = np.full_like(teacher_alignment, np.nan)
        actual_alignment[full_train] = projection_targets(
            arrays["vectors"][full_train], actual_train
        )["raw_projection"]
        for family in [None, *sorted(frame.family_id.unique())]:
            train = (
                full_train
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            validation = (
                np.array([], dtype=int)
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "validation")
                    & (frame.family_id.to_numpy() == family)
                )
            )
            training, evaluation = frame.iloc[train], frame.iloc[validation]
            if family in set(training.family_id) or set(training.origin_id) & set(
                evaluation.origin_id
            ):
                raise ValueError("a held source family entered fitting")
            weights = _family_weights(training)
            probabilities = {}
            for condition in CONDITIONS:
                inputs = condition_features(arrays["features"], condition)
                np.testing.assert_array_equal(inputs[:, :, :33], arrays["features"][:, :, :33])
                if condition.startswith("point_") and np.any(inputs[:, :, 33:] != 0):
                    raise ValueError("the point control received latent information")
                values = inputs[train].astype(float)
                mean = (values * weights[:, None, None]).sum((0, 1)) / (weights.sum() * 7)
                variance = ((values - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / (
                    weights.sum() * 7
                )
                entries = [
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_family"] == family
                    and row["condition"] == condition
                ]
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("matched seeds changed")
                seeds = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a learned source checkpoint changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != study["identity_sha256"]
                        or saved["train_ids_sha256"] != hashlib.sha256(train.tobytes()).hexdigest()
                        or saved["initial_parameter_sha256"]
                        != study["initial_parameters"][str(entry["seed"])]
                    ):
                        raise ValueError("source fitting inputs or initialization changed")
                    if set(saved["training_origins"]) != set(training.origin_id) or set(
                        saved["training_families"]
                    ) != set(training.family_id):
                        raise ValueError("a source fit used different histories")
                    state = saved["state_dict"]
                    if (
                        sum(
                            value.numel()
                            for name, value in state.items()
                            if name not in ("feature_mean", "feature_scale")
                        )
                        != 2120
                        or len(saved["training_history"]) != 25
                    ):
                        raise ValueError("the common model capacity or training budget changed")
                    np.testing.assert_array_equal(
                        state["feature_mean"].numpy().ravel(), mean.astype(np.float32)
                    )
                    np.testing.assert_array_equal(
                        state["feature_scale"].numpy().ravel(),
                        np.maximum(np.sqrt(variance), 1e-6).astype(np.float32),
                    )
                    if len(validation):
                        seeds.append(replay_network(state, inputs[validation]))
                    verified += 1
                probabilities[condition] = seeds
            if not len(validation):
                continue
            fold = next(
                row
                for row in study["folds"]
                if row["model_id"] == model_id and row["held_family"] == family
            )
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("saved source validation predictions changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                points = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
                for condition in CONDITIONS:
                    np.testing.assert_array_equal(
                        saved[condition], np.stack(probabilities[condition])
                    )
            for condition, seeds in probabilities.items():
                for name, probability in [
                    (condition, np.mean(seeds, axis=0)),
                    *[
                        (f"{condition}_seed{seed}", value)
                        for seed, value in zip((5101, 5102, 5103), seeds, strict=True)
                    ],
                ]:
                    probability = probability / probability.sum(1, keepdims=True)
                    expected = (arrays["vectors"][validation] * probability[:, :, None]).sum(1)
                    prediction_delta = max(
                        prediction_delta, float(abs(expected - points[name]).max())
                    )
                    np.testing.assert_allclose(expected, points[name], rtol=1e-12, atol=1e-12)
            controls_path = args.study_root / fold["controls_path"]
            if file_sha256(controls_path) != fold["controls_sha256"]:
                raise ValueError("matched fixed controls changed")
            controls = read_json(controls_path)
            for label, alignment in (("teacher", teacher_alignment), ("future", actual_alignment)):
                probability = weights / weights.sum()
                g = np.einsum("n,nab->ab", probability, gram[train])
                b = np.einsum("n,na->a", probability, alignment[train])
                weight = np.asarray(controls[label]["weights"])
                gradient = 2 * (g @ weight - b)
                gap = (gradient @ weight - gradient.min()) / max(
                    float(abs(g).max()), float(abs(b).max()), 1e-12
                )
                if gap > 1e-7 or (weight < 0).any() or abs(weight.sum() - 1) > 1e-10:
                    raise ValueError("a fixed mixture fails optimality")
                maximum_gap = max(maximum_gap, float(max(gap, 0)))
                expected = np.sum(arrays["vectors"][validation] * weight[None, :, None], axis=1)
                np.testing.assert_allclose(
                    expected, points[f"fixed_{label}"], rtol=1e-12, atol=1e-12
                )
                single = int((g.diagonal() - 2 * b).argmin())
                np.testing.assert_array_equal(
                    arrays["vectors"][validation, single], points[f"single_{label}"]
                )
            truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
            for method, point in points.items():
                current = evaluation.assign(
                    model_id=model_id,
                    method=method,
                    mae=abs(point - truth).mean(1),
                    mse=((point - truth) ** 2).mean(1),
                )
                previous = (
                    stored_scores[
                        (stored_scores.model_id == model_id) & (stored_scores.method == method)
                    ]
                    .set_index("episode_id")
                    .loc[current.episode_id]
                )
                np.testing.assert_array_equal(previous.mae.to_numpy(), current.mae.to_numpy())
                np.testing.assert_array_equal(previous.mse.to_numpy(), current.mse.to_numpy())
                all_scores.append(current)
    scores = pd.concat(all_scores, ignore_index=True)
    episodes, families, summary = aggregate(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    if verified != 384:
        raise ValueError("the factorial source audit is incomplete")
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
            "verified_cached_queries": checked_queries,
            "maximum_prediction_difference": prediction_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "source development at one mask seed; no target transfer or acceptance claim",
        },
    )


if __name__ == "__main__":
    main()
