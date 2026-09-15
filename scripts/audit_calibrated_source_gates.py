"""Reconstruct nested selections and downstream scores from saved source gates."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from audit_shared_forecast_gate import replay_network
from calibrated_source_gate import EPOCHS, FEATURES, LAMBDAS, inputs_for
from latent_source_inputs import ROOT, load_source_inputs, read_json
from train_calibrated_source_gates import arguments, load_references
from train_latent_source_gates import aggregate

from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_metrics(frame, point, truth):
    values = frame[["family_id", "source_episode_id"]].copy()
    values["mae"] = np.mean(np.abs(point - truth), axis=1)
    values["mse"] = np.mean(np.square(point - truth), axis=1)
    values = values.groupby(["family_id", "source_episode_id"])[["mae", "mse"]].mean()
    return values.groupby("family_id").mean().mean().to_dict()


def weighted_point(vectors, probability):
    probability = probability / probability.sum(1, keepdims=True)
    return np.sum(vectors * probability[:, :, None], axis=1)


def audit_anchor(frame, indices, gram, alignment, anchor):
    weights = _family_weights(frame.iloc[indices])
    weights /= weights.sum()
    g = np.einsum("n,nab->ab", weights, gram[indices])
    b = np.einsum("n,na->a", weights, alignment[indices])
    gradient = 2 * (g @ anchor - b)
    gap = float((gradient @ anchor - gradient.min()) / max(abs(g).max(), abs(b).max(), 1e-12))
    if np.any(anchor < 0) or abs(anchor.sum() - 1) > 1e-10 or gap > 1e-7:
        raise ValueError("a fixed anchor fails its source optimality check")
    return max(gap, 0)


def main():
    parser = arguments(__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed calibrated source audit")
    reference, positions = load_references(args)
    study = read_json(args.study_root / "manifest.json")
    if (
        study["status"] != "completed"
        or study["identity"]["script_sha256"]
        != file_sha256(ROOT / "scripts/train_calibrated_source_gates.py")
        or study["identity"]["module_sha256"]
        != file_sha256(ROOT / "scripts/calibrated_source_gate.py")
        or study["identity"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("the completed training definitions changed")
    freeze = read_json(args.study_root / "prediction_freeze.json")
    if (
        freeze["checkpoints"] != study["checkpoints"]
        or freeze["folds"] != study["folds"]
        or freeze["outer_validation_outcomes_read"]
    ):
        raise ValueError("the prediction freeze does not match the completed experiment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_scores, choices_log = [], []
    verified, maximum_delta, maximum_gap = 0, 0.0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_source_inputs(args.input_root, model_id)
        vectors = arrays["vectors"]
        _, _, _, gram = forecast_geometry(vectors)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        truth = decision_truth(frame, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r"))
        alignment = np.full(vectors.shape[:2], np.nan)
        alignment[training] = projection_targets(vectors[training], truth[training])[
            "raw_projection"
        ]
        for fold in (row for row in study["folds"] if row["model_id"] == model_id):
            family = fold["held_family"]
            allowed = (
                training
                if family is None
                else np.flatnonzero(
                    (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
                )
            )
            path = args.study_root / fold["selection_path"]
            if file_sha256(path) != fold["selection_sha256"]:
                raise ValueError("a saved calibration choice changed")
            selected = read_json(path)
            np.testing.assert_array_equal(selected["allowed"], allowed)
            # Rebuild time groups independently of the training split helper.
            inner_ids, calibration_ids, purged_ids = set(), set(), set()
            groups = frame.iloc[allowed].drop_duplicates("origin_id")
            for _, group in groups.groupby(["family_id", "dataset_id", "item_id"]):
                origins = sorted(group.origin_id, key=lambda name: (positions[name], name))
                if len(origins) < 4:
                    inner_ids.update(origins)
                    continue
                count = (len(origins) + 3) // 4
                calibration_ids.update(origins[-count:])
                boundary = positions[origins[-count]]
                for origin in origins[:-count]:
                    (inner_ids if positions[origin] + 96 <= boundary else purged_ids).add(origin)
            for name, ids in (
                ("inner", inner_ids),
                ("calibration", calibration_ids),
                ("purged", purged_ids),
            ):
                np.testing.assert_array_equal(
                    selected[name], [i for i in allowed if frame.iloc[i].origin_id in ids]
                )
            inner, calibration = np.asarray(selected["inner"]), np.asarray(selected["calibration"])
            inner_anchor, outer_anchor = (
                np.asarray(selected["inner_anchor"]),
                np.asarray(selected["outer_anchor"]),
            )
            for indices, anchor in ((inner, inner_anchor), (allowed, outer_anchor)):
                maximum_gap = max(
                    maximum_gap, audit_anchor(frame, indices, gram, alignment, anchor)
                )
            calibration_path = args.study_root / fold["calibration_path"]
            if file_sha256(calibration_path) != fold["calibration_sha256"]:
                raise ValueError("saved calibration probabilities changed")
            with np.load(calibration_path, allow_pickle=False) as saved:
                calibration_weights = {name: saved[name] for name in saved.files}
            loaded = {}
            for entry in (
                row
                for row in study["checkpoints"]
                if row["model_id"] == model_id and row["held_family"] == family
            ):
                path = args.study_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a calibrated checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                metadata = {
                    key: value for key, value in entry.items() if key not in ("path", "sha256")
                }
                if saved["metadata"] != metadata:
                    raise ValueError("checkpoint metadata changed")
                indices = inner if entry["stage"] == "inner" else allowed
                anchor = inner_anchor if entry["stage"] == "inner" else outer_anchor
                if (
                    saved["train_indices_sha256"]
                    != hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()
                    or set(saved["training_origins"]) != set(frame.iloc[indices].origin_id)
                    or set(saved["training_families"]) != set(frame.iloc[indices].family_id)
                    or saved["initial_parameter_sha256"]
                    != reference["initial_parameters"][str(entry["seed"])]
                    or len(saved["history"]) != max(entry["epochs"])
                    or set(saved["states"]) != set(map(str, entry["epochs"]))
                    or saved["strength"] != entry["strength"]
                    or saved["seed"] != entry["seed"]
                ):
                    raise ValueError("fitting population, initialization or schedule changed")
                np.testing.assert_array_equal(saved["anchor"], anchor)
                features = inputs_for(arrays, entry["feature"])[indices].astype(float)
                weight = _family_weights(frame.iloc[indices])
                mean = (features * weight[:, None, None]).sum((0, 1)) / (weight.sum() * 7)
                variance = ((features - mean) ** 2 * weight[:, None, None]).sum((0, 1)) / (
                    weight.sum() * 7
                )
                for state in saved["states"].values():
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
                            for key, value in state.items()
                            if key not in ("feature_mean", "feature_scale")
                        )
                        != 2120
                    ):
                        raise ValueError("gate capacity changed")
                loaded[
                    (
                        entry["stage"],
                        entry["feature"],
                        entry["strength"],
                        entry["seed"],
                        tuple(entry["epochs"]),
                    )
                ] = saved
                verified += 1
            baseline = direct_metrics(
                frame.iloc[calibration],
                weighted_point(
                    vectors[calibration], np.broadcast_to(inner_anchor, (len(calibration), 7))
                ),
                truth[calibration],
            )
            np.testing.assert_allclose(
                list(baseline.values()),
                list(selected["reference_metrics"].values()),
                rtol=1e-12,
                atol=1e-12,
            )
            rebuilt_choices = {}
            for feature in FEATURES:
                rebuilt = []
                for strength in LAMBDAS:
                    for epoch_index, epoch in enumerate(EPOCHS):
                        probabilities = [
                            replay_network(
                                loaded[("inner", feature, strength, seed, EPOCHS)]["states"][
                                    str(epoch)
                                ],
                                inputs_for(arrays, feature)[calibration],
                            )
                            for seed in (5101, 5102, 5103)
                        ]
                        probability = np.mean(probabilities, axis=0)
                        np.testing.assert_array_equal(
                            probability,
                            calibration_weights[f"{feature}_r{int(strength)}"][epoch_index],
                        )
                        metrics = direct_metrics(
                            frame.iloc[calibration],
                            weighted_point(vectors[calibration], probability),
                            truth[calibration],
                        )
                        reported = next(
                            row
                            for row in selected["candidates"]
                            if row["feature"] == feature
                            and row["strength"] == strength
                            and row["epoch"] == epoch
                        )
                        np.testing.assert_allclose(
                            [metrics["mae"], metrics["mse"]],
                            [reported["mae"], reported["mse"]],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                        # Use the saved, independently verified scores for exact tie handling.
                        rebuilt.append(reported)
                for suffix in ("calibrated", "early_stop"):
                    eligible = [
                        row
                        for row in rebuilt
                        if (suffix != "early_stop" or row["strength"] == 0)
                        and row["mae"] < selected["reference_metrics"]["mae"]
                        and row["mse"] < selected["reference_metrics"]["mse"]
                    ]
                    if eligible:
                        best = min(
                            eligible,
                            key=lambda row: (
                                0.5
                                * (
                                    row["mae"] / selected["reference_metrics"]["mae"]
                                    + row["mse"] / selected["reference_metrics"]["mse"]
                                ),
                                -row["strength"],
                                row["epoch"],
                            ),
                        )
                        choice = {
                            "kind": "gate",
                            "strength": best["strength"],
                            "epoch": best["epoch"],
                        }
                    else:
                        choice = {"kind": "fixed", "strength": None, "epoch": None}
                    method = feature + "_" + suffix
                    if choice != selected["choices"][method]:
                        raise ValueError("the fixed calibration selection rule was not followed")
                    rebuilt_choices[method] = choice
                    choices_log.append(
                        {"model_id": model_id, "held_family": family, "method": method, **choice}
                    )
            if family is None:
                continue
            validation = np.flatnonzero(
                (frame.split.to_numpy() == "validation") & (frame.family_id.to_numpy() == family)
            )
            prediction_path = args.study_root / fold["prediction_path"]
            if file_sha256(prediction_path) != fold["prediction_sha256"]:
                raise ValueError("a frozen outer prediction changed")
            with np.load(prediction_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["validation_indices"], validation)
                points = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            for method, choice in rebuilt_choices.items():
                feature = method.split("_")[0]
                seeds = (
                    [np.broadcast_to(outer_anchor, (len(validation), 7))] * 3
                    if choice["kind"] == "fixed"
                    else [
                        replay_network(
                            loaded[
                                ("outer", feature, choice["strength"], seed, (choice["epoch"],))
                            ]["states"][str(choice["epoch"])],
                            inputs_for(arrays, feature)[validation],
                        )
                        for seed in (5101, 5102, 5103)
                    ]
                )
                for name, weight in [
                    (method, np.mean(seeds, axis=0)),
                    *[
                        (f"{method}_seed{seed}", probability)
                        for seed, probability in zip((5101, 5102, 5103), seeds, strict=True)
                    ],
                ]:
                    expected = weighted_point(vectors[validation], weight)
                    maximum_delta = max(maximum_delta, float(abs(expected - points[name]).max()))
                    np.testing.assert_allclose(expected, points[name], rtol=1e-12, atol=1e-12)
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
            for method, point in points.items():
                all_scores.append(
                    frame.iloc[validation].assign(
                        model_id=model_id,
                        method=method,
                        mae=abs(point - truth[validation]).mean(1),
                        mse=((point - truth[validation]) ** 2).mean(1),
                    )
                )
        print(
            f"{model_id}: nested choices, states and outer predictions independently verified",
            flush=True,
        )
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
    old_summary = pd.read_csv(args.reference_audit / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(old_summary.method.unique())].reset_index(drop=True),
        old_summary,
        check_exact=True,
    )
    if verified != len(study["checkpoints"]) or len(scores) != 106704 or len(choices_log) != 128:
        raise ValueError("the nested source audit is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(choices_log).to_csv(output / "choices.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(args.study_root / "manifest.json"),
            "verified_checkpoints": verified,
            "verified_choices": len(choices_log),
            "verified_scores": len(scores),
            "maximum_prediction_difference": maximum_delta,
            "maximum_fixed_optimality_gap": maximum_gap,
            "limits": "nested source development only; calibration eligibility does not guarantee target improvement",
        },
    )


if __name__ == "__main__":
    main()
