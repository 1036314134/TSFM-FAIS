"""Audit held-group exclusion, observed-label objectives and matched update budgets."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_shared_forecast_gate import replay_network
from latent_source_inputs import ROOT, read_json
from metric_source_gate import EPSILON
from native_source_transfer_io import observed_errors, observed_weights, summarize_scores
from real_pool_inputs import arguments, build_model_inputs, load_real_inputs, reference_check

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_observed_control(points, truth, observed, weights, probability, joint):
    row = np.asarray(weights, float)
    row = row / row.sum()
    coordinate = observed_weights(observed, joint=joint, minimum=48)
    prediction = (points * probability[None, :, None]).sum(1)
    error = np.where(observed, prediction - truth, 0.0)
    magnitude = np.sqrt(error**2 + EPSILON**2)
    coefficient = (error + 0.5 * error / magnitude) * coordinate * row[:, None]
    gradient = np.einsum("naq,nq->a", points, coefficient) - np.sum(
        np.median(points, axis=1) * coefficient
    )
    value = float(np.sum(row[:, None] * coordinate * (error**2 + magnitude) / 2))
    return value, gradient


def main():
    parser = arguments(__doc__)
    parser.add_argument(
        "--prepared-root", type=Path, default=ROOT / "artifacts/iclr27-r13/real-inputs-v001"
    )
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed real-source audit")
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"] != file_sha256(ROOT / "scripts/train_real_pool.py")
        or study["identity"]["module_sha256"] != file_sha256(ROOT / "scripts/masked_pool_gate.py")
        or study["identity"]["input_module_sha256"]
        != file_sha256(ROOT / "scripts/real_pool_inputs.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("real-source study definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["held_group_scoring_started"]
    ):
        raise ValueError("held-group predictions were not frozen as registered")
    torch.set_num_threads(1)
    for entry in study["compatibility"]:
        path = args.study_root / entry["path"]
        old_path = args.pool_study / entry["original_path"]
        if (
            file_sha256(path) != entry["sha256"]
            or file_sha256(old_path) != entry["original_sha256"]
        ):
            raise ValueError("an original source replay changed")
        current = torch.load(path, map_location="cpu", weights_only=True)
        old = torch.load(old_path, map_location="cpu", weights_only=True)
        for name in (
            "initial_parameter_sha256",
            "training_origins",
            "training_families",
            "history",
        ):
            if current[name] != old[name]:
                raise ValueError("the complete-observation source fit changed")
        for name in old["state_dict"]:
            torch.testing.assert_close(
                current["state_dict"][name], old["state_dict"][name], rtol=0, atol=0
            )
    verified, maximum_delta, maximum_gap = 0, 0.0, 0.0
    records = []
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, data, references = load_real_inputs(args.prepared_root, model_id)
        rebuilt, values, old_references, _ = build_model_inputs(args, model_id)
        pd.testing.assert_frame_equal(frame, rebuilt, check_exact=True)
        for name in data:
            np.testing.assert_array_equal(data[name], values[name])
        for name in references:
            np.testing.assert_array_equal(references[name], old_references[name])
        reference_check(args, frame, data, references, model_id)
        del rebuilt, values, old_references
        coordinates = observed_weights(data["observed"], joint=model_id == "chronos2", minimum=48)
        np.testing.assert_array_equal(coordinates, data["coordinates"])
        median = np.median(data["points"], axis=1)
        delta = data["points"] - median[:, None]
        residual = np.where(data["observed"], data["truth"] - median, 0.0)
        direct_g = np.einsum("naq,nbq,nq->nab", delta, delta, coordinates)
        direct_b = np.einsum("naq,nq,nq->na", delta, residual, coordinates)
        np.testing.assert_allclose(direct_g, data["gram"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(direct_b, data["alignment"], rtol=1e-12, atol=1e-12)
        source = np.flatnonzero(frame.cohort.to_numpy() == "source")
        for fold in (row for row in study["folds"] if row["model_id"] == model_id):
            group = fold["held_group"]
            evaluation = np.flatnonzero(
                (frame.cohort.to_numpy() != "source") & (frame.holdout_group.to_numpy() == group)
            )
            augmented = np.flatnonzero(
                (frame.cohort.to_numpy() == "source") | (frame.holdout_group.to_numpy() != group)
            )
            for name, indices in (
                ("evaluation_indices", evaluation),
                ("augmented_indices", augmented),
                ("source_indices", source),
            ):
                np.testing.assert_array_equal(fold[name], indices)
            if set(frame.iloc[augmented].family_id) & set(frame.iloc[evaluation].family_id) or set(
                frame.iloc[augmented].origin_id
            ) & set(frame.iloc[evaluation].origin_id):
                raise ValueError("held-group data entered normalization or fitting")
            updates = ((len(augmented) + 127) // 128) * 25
            if updates != fold["updates"]:
                raise ValueError("the matched update budget changed")
            expected = {name: value[evaluation] for name, value in references.items()}
            for regime, indices in (
                ("real_augmented", augmented),
                ("source_steps_matched", source),
            ):
                weights = _family_weights(frame.iloc[indices])
                features = data["features"][indices].astype(float)
                mean = (features * weights[:, None, None]).sum((0, 1)) / (weights.sum() * 8)
                variance = ((features - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / (
                    weights.sum() * 8
                )
                entries = [
                    row
                    for row in study["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_group"] == group
                    and row["regime"] == regime
                ]
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("a real-source condition lost a seed")
                seeds = []
                for entry in entries:
                    path = args.study_root / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a grouped source model changed")
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
                        or saved["updates"] != updates
                        or len(saved["history"])
                        != (updates + ((len(indices) + 127) // 128) - 1)
                        // ((len(indices) + 127) // 128)
                    ):
                        raise ValueError("the group population or update schedule changed")
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
                        raise ValueError("the matched source capacity changed")
                    seeds.append(replay_network(state, data["features"][evaluation]))
                    verified += 1
                for name, probability in [
                    (regime, np.mean(seeds, axis=0)),
                    *[
                        (f"{regime}_seed{seed}", probability)
                        for seed, probability in zip((5101, 5102, 5103), seeds, strict=True)
                    ],
                ]:
                    probability = probability / probability.sum(1, keepdims=True)
                    expected[name] = (data["points"][evaluation] * probability[:, :, None]).sum(1)
            path = args.study_root / fold["control_path"]
            if file_sha256(path) != fold["control_sha256"]:
                raise ValueError("the mixed-source fixed control changed")
            control = read_json(path)
            probability = np.asarray(control["weights"])
            weight = _family_weights(frame.iloc[augmented])
            value, gradient = direct_observed_control(
                data["points"][augmented],
                data["truth"][augmented],
                data["observed"][augmented],
                weight,
                probability,
                model_id == "chronos2",
            )
            gap = float((gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0))
            if gap > 1e-7 or (probability < 0).any() or abs(probability.sum() - 1) > 1e-10:
                raise ValueError("the mixed-source fixed optimality check failed")
            maximum_gap = max(maximum_gap, gap)
            np.testing.assert_allclose(value, control["objective"], rtol=1e-12, atol=1e-12)
            singles = [
                direct_observed_control(
                    data["points"][augmented],
                    data["truth"][augmented],
                    data["observed"][augmented],
                    weight,
                    np.eye(8)[index],
                    model_id == "chronos2",
                )[0]
                for index in range(8)
            ]
            if int(np.argmin(singles)) != control["single_index"]:
                raise ValueError("the mixed-source single control changed")
            expected["real_augmented_fixed"] = (
                data["points"][evaluation] * probability[None, :, None]
            ).sum(1)
            expected["real_augmented_single"] = data["points"][evaluation, control["single_index"]]
            path = args.study_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a held-group prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["evaluation_indices"], evaluation)
                predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for method, point in expected.items():
                maximum_delta = max(maximum_delta, float(abs(point - predictions[method]).max()))
                np.testing.assert_allclose(point, predictions[method], rtol=1e-12, atol=1e-12)
            for method, point in predictions.items():
                mae, mse = observed_errors(
                    point,
                    data["truth"][evaluation],
                    data["observed"][evaluation],
                    joint=model_id == "chronos2",
                )
                records.append(frame.iloc[evaluation].assign(method=method, mae=mae, mse=mse))
        print(
            f"{model_id}: real-source group exclusion and results independently checked", flush=True
        )
    scores = pd.concat(records, ignore_index=True)
    keys = ["model_id", "method", "episode_id"]
    previous = pd.read_parquet(args.study_root / "decision_scores.parquet")
    pd.testing.assert_frame_equal(
        scores.sort_values(keys).reset_index(drop=True),
        previous.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )
    episodes, families, summary, groups = summarize_scores(scores)
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(args.study_root / "summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        groups,
        pd.read_csv(args.study_root / "group_summary.csv", float_precision="round_trip"),
        check_exact=True,
    )
    if verified != 96 or len(scores) != 21480:
        raise ValueError("the real-source audit is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    groups.to_csv(output / "group_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_checkpoints": verified,
            "verified_original_source_replays": 2,
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "retrospective grouped development; native labels are source labels only for other groups",
        },
    )


if __name__ == "__main__":
    main()
