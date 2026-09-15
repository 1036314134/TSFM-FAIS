"""Complete matched source-family validation for the existing actual-future gate."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth, decision_vectors, load_prepared_model  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from train_shared_forecast_gate import SETTINGS, fit_gate, predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_state(features, weights):
    values = np.asarray(features, float)
    denominator = weights.sum() * 7
    mean = (values * weights[:, None, None]).sum((0, 1)) / denominator
    variance = ((values - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / denominator
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "accuracy-root",
        "teacher-study",
        "future-control",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source-label comparisons")
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    teacher_study, control = (
        read_json(args.teacher_study / "manifest.json"),
        read_json(args.future_control / "manifest.json"),
    )
    if any(row["status"] != "completed" for row in (prep, teacher_study, control)):
        raise ValueError("complete the existing matched studies first")
    if (
        control["identity"]["settings"] != SETTINGS
        or teacher_study["identity"]["settings"] != SETTINGS
    ):
        raise ValueError("the fitting schedule changed")
    if file_sha256(ROOT / "scripts/train_shared_forecast_gate.py") != teacher_study["identity"][
        "script_sha256"
    ] or prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("the trainer or cached source data changed")
    for name, digest in teacher_study["identity"]["runtime_source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("a frozen source-learning dependency changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("the source outcome array changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "teacher_study_sha256": file_sha256(args.teacher_study / "manifest.json"),
        "future_control_sha256": file_sha256(args.future_control / "manifest.json"),
        "settings": SETTINGS,
        "objective": "actual source future ensemble MSE",
        "family_weights_unchanged": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial source-label study identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folds, all_scores, parity, full_controls = [], [], [], {}
    verified_checkpoints, max_replay_delta = 0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("source candidate forecasts changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        features = arrays["features"][:, :7, :33]
        _, _, _, gram = forecast_geometry(vectors)
        full_train = np.flatnonzero(decisions.split.to_numpy() == "train")
        full_frame = decisions.iloc[full_train]
        if (
            full_frame.origin_id.nunique(),
            full_frame.family_id.nunique(),
            full_frame.source_episode_id.nunique(),
        ) != (165, 15, 5940):
            raise ValueError("the original source training population changed")
        full_truth = decision_truth(full_frame, np.load(truth_path, mmap_mode="r"))
        alignment = np.full((len(decisions), 7), np.nan)
        alignment[full_train] = projection_targets(vectors[full_train], full_truth)[
            "raw_projection"
        ]
        full_weights = _family_weights(full_frame)
        parity_path = output / model_id / "full_source_parity.json"
        original = next(
            row for row in control["models"] if row["model_id"] == model_id and row["seed"] == 5101
        )
        original_path = args.future_control / original["path"]
        if (
            file_sha256(original_path) != original["sha256"]
            or original["actions"] != info["actions"]
        ):
            raise ValueError("the existing source-future model changed")
        if not parity_path.exists():
            model, _ = fit_gate(
                features[full_train],
                gram[full_train],
                alignment[full_train],
                full_weights,
                kind="ensemble",
                seed=5101,
            )
            previous = torch.load(original_path, map_location="cpu", weights_only=True)
            if previous["train_ids_sha256"] != hashlib.sha256(full_train.tobytes()).hexdigest():
                raise ValueError("full-source parity uses a different training population")
            for name, value in model.state_dict().items():
                if not torch.equal(value, previous["state_dict"][name]):
                    raise ValueError(f"the actual-future control did not reproduce exactly: {name}")
            _write_json(
                parity_path,
                {
                    "status": "passed",
                    "identity_sha256": identity_sha,
                    "maximum_parameter_difference": 0,
                    "original_checkpoint_sha256": original["sha256"],
                },
            )
        if read_json(parity_path)["identity_sha256"] != identity_sha:
            raise ValueError("full-source parity belongs to another execution")
        parity.append(
            {
                "model_id": model_id,
                "path": str(parity_path.relative_to(output)),
                "sha256": file_sha256(parity_path),
            }
        )
        probability = full_weights / full_weights.sum()
        fixed, gap, _ = simplex_quadratic_weights(
            np.einsum("n,nab->ab", probability, gram[full_train])[None],
            np.einsum("n,na->a", probability, alignment[full_train])[None],
        )
        full_controls[model_id] = {
            "actions": info["actions"],
            "convex_weights": fixed[0].tolist(),
            "optimality_gap": float(gap[0]),
        }
        for family in sorted(decisions.family_id.unique()):
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train") & (decisions.family_id.to_numpy() != family)
            )
            validation = np.flatnonzero(
                (decisions.split.to_numpy() == "validation")
                & (decisions.family_id.to_numpy() == family)
            )
            training, evaluation = decisions.iloc[train], decisions.iloc[validation]
            if (
                family in set(training.family_id)
                or set(training.origin_id) & set(evaluation.origin_id)
                or not np.isfinite(alignment[train]).all()
            ):
                raise ValueError("a fitted fold violated the family, time or label boundary")
            weights = _family_weights(training)
            expected_mean, expected_scale = normalized_state(features[train], weights)
            key = hashlib.sha256((model_id + "|" + family).encode()).hexdigest()[:20]
            directory = output / "folds" / key
            directory.mkdir(parents=True, exist_ok=True)
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            seed_weights, checkpoints = [], []
            for seed in SETTINGS["seeds"]:
                path = directory / f"seed_{seed}.pt"
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved["identity_sha256"] != identity_sha
                        or saved["train_ids_sha256"] != train_sha
                        or saved["seed"] != seed
                    ):
                        raise ValueError("a saved fold model changed identity")
                    model = SharedForecastGate()
                    model.load_state_dict(saved["state_dict"])
                    model.eval()
                else:
                    model, history = fit_gate(
                        features[train],
                        gram[train],
                        alignment[train],
                        weights,
                        kind="ensemble",
                        seed=seed,
                    )
                    saved = {
                        "state_dict": model.state_dict(),
                        "identity_sha256": identity_sha,
                        "train_ids_sha256": train_sha,
                        "seed": seed,
                        "training_history": history,
                        "training_origins": sorted(training.origin_id.unique()),
                        "training_families": sorted(training.family_id.unique()),
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                state = saved["state_dict"]
                np.testing.assert_array_equal(
                    state["feature_mean"].numpy().reshape(-1), expected_mean
                )
                np.testing.assert_array_equal(
                    state["feature_scale"].numpy().reshape(-1), expected_scale
                )
                teacher_path = args.teacher_study / "folds" / key / f"seed_{seed}.pt"
                teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
                if teacher["train_ids_sha256"] != train_sha:
                    raise ValueError(
                        "matched teacher and future folds use different training decisions"
                    )
                for name in ("feature_mean", "feature_scale"):
                    if not torch.equal(state[name], teacher["state_dict"][name]):
                        raise ValueError("matched objectives use different feature normalization")
                x = np.ascontiguousarray(features[validation])
                predicted = predict_weights(model, x)
                np.testing.assert_array_equal(predicted, replay_network(state, x))
                seed_weights.append(predicted)
                checkpoints.append(
                    {
                        "seed": seed,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                verified_checkpoints += 1
            probability = weights / weights.sum()
            fixed, gap, _ = simplex_quadratic_weights(
                np.einsum("n,nab->ab", probability, gram[train])[None],
                np.einsum("n,na->a", probability, alignment[train])[None],
            )
            average = np.mean(seed_weights, axis=0)
            predictions = {
                "source_future_gate": compose_forecasts(vectors[validation], average),
                "future_source_fixed_convex": compose_forecasts(
                    vectors[validation], np.repeat(fixed, len(validation), axis=0)
                ),
                "forecast_median_guarded": np.median(vectors[validation], axis=1),
            }
            for seed, seed_weight in zip(SETTINGS["seeds"], seed_weights, strict=True):
                predictions[f"future_seed{seed}"] = compose_forecasts(
                    vectors[validation], seed_weight
                )
            direct = np.einsum("na,naq->nq", average, vectors[validation])
            max_replay_delta = max(
                max_replay_delta, float(abs(direct - predictions["source_future_gate"]).max())
            )
            np.testing.assert_allclose(
                direct, predictions["source_future_gate"], rtol=1e-12, atol=1e-12
            )
            prediction_path = directory / "predictions.npz"
            _save_npz(
                prediction_path,
                point=np.stack(list(predictions.values())),
                methods=np.asarray(list(predictions)),
                seed_weights=np.stack(seed_weights),
                mean_weights=average,
                fixed_weights=fixed,
                validation_indices=validation,
                identity_sha256=np.asarray(identity_sha),
            )
            # Validation outcomes enter only after the fold's models and predictions are saved.
            truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
            scores = pd.concat(
                [
                    evaluation.assign(
                        model_id=model_id,
                        method=name,
                        mae=abs(point - truth).mean(1),
                        mse=((point - truth) ** 2).mean(1),
                    )
                    for name, point in predictions.items()
                ],
                ignore_index=True,
            )
            score_path = directory / "scores.parquet"
            scores.to_parquet(score_path, index=False)
            all_scores.append(scores)
            fold = {
                "model_id": model_id,
                "held_family": family,
                "checkpoints": checkpoints,
                "train_ids_sha256": train_sha,
                "training_origins": sorted(training.origin_id.unique()),
                "prediction_path": str(prediction_path.relative_to(output)),
                "prediction_sha256": file_sha256(prediction_path),
                "scores_path": str(score_path.relative_to(output)),
                "scores_sha256": file_sha256(score_path),
                "fixed_optimality_gap": float(gap[0]),
            }
            _write_json(
                directory / "manifest.json",
                {"status": "completed", "identity_sha256": identity_sha, **fold},
            )
            folds.append(fold)
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "held_family": family,
                        "completed_folds": len(folds),
                        "total_folds": 30,
                    }
                ),
                flush=True,
            )
    if len(folds) != 30 or verified_checkpoints != 90 or len(parity) != 2:
        raise ValueError("matched objective coverage changed")
    scores = pd.concat(all_scores, ignore_index=True)
    keys = [
        "model_id",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    episodes = scores.groupby(keys)[["mae", "mse"]].mean().reset_index()
    for _, group in episodes.groupby(["model_id", "method"]):
        if group.source_episode_id.nunique() != 1872:
            raise ValueError("source evaluation coverage changed")
    families = (
        episodes.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    )
    summary = families.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    original = pd.read_csv(args.teacher_study / "summary.csv", float_precision="round_trip")
    actual = summary[summary.method == "forecast_median_guarded"].set_index("model_id")[
        ["mae", "mse"]
    ]
    expected = (
        original[original.method == "forecast_median_guarded"]
        .set_index("model_id")
        .loc[actual.index, ["mae", "mse"]]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    episodes.to_parquet(output / "episode_results.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(
        [original, summary[summary.method != "forecast_median_guarded"]], ignore_index=True
    ).to_csv(output / "comparison_summary.csv", index=False)
    _write_json(output / "full_source_fixed_controls.json", full_controls)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": folds,
            "verified_checkpoints": verified_checkpoints,
            "full_source_parity": parity,
            "maximum_prediction_reconstruction_difference": max_replay_delta,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_imputer_fits": 0,
            "limits": "matched source-family validation only; existing R6 results remain used; a separate saved-output audit follows",
        },
    )


if __name__ == "__main__":
    main()
