"""Train matched local and pooled position predictors using existing complete-history teachers."""

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
from positional_forecast_portfolio import (  # noqa: E402
    PositionalPortfolio,
    fit_position,
    position_inputs,
    predict_position,
)
from positional_portfolio_io import target_nodes  # noqa: E402

from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402

SETTINGS = {
    "epochs": 25,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "gradient_norm": 1.0,
    "seeds": [5101, 5102, 5103],
    "hidden": 8,
    "context_features": 49,
    "local_features": 16,
    "parameters_per_seed": 609,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "accuracy-root",
        "teacher-root",
        "reference-study",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed positional source studies")
    prep, accuracy, teachers = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
        read_json(args.teacher_root / "manifest.json"),
    )
    if prep["status"] != "completed" or teachers["status"] != "completed":
        raise ValueError("complete the source feature and teacher preparation")
    if prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ) or prep["identity"]["teacher_manifest_sha256"] != file_sha256(
        args.teacher_root / "manifest.json"
    ):
        raise ValueError("the original teacher/forecast bindings changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_module_sha256": file_sha256(ROOT / "scripts/positional_forecast_portfolio.py"),
        "io_module_sha256": file_sha256(ROOT / "scripts/positional_portfolio_io.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "teacher_sha256": file_sha256(args.teacher_root / "manifest.json"),
        "settings": SETTINGS,
        "primary": "local",
        "matched_control": "pooled",
        "seed_aggregation": "mean of the three bounded point predictions",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial positional study changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, source_models, folds, all_scores, initial_hashes = [], [], [], [], {}
    max_label_difference = 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        info, original_decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        point_path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("source candidate forecasts changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        original_vectors = decision_vectors(original_decisions, bank)
        teacher_entry = next(row for row in teachers["models"] if row["model_id"] == model_id)
        teacher_path = args.teacher_root / teacher_entry["teacher_file"]
        if file_sha256(teacher_path) != teacher_entry["teacher_sha256"]:
            raise ValueError("source coordinate teacher predictions changed")
        teacher_bank = np.load(teacher_path, mmap_mode="r")
        old_train = np.flatnonzero(original_decisions.split.to_numpy() == "train")
        old_teacher = decision_truth(original_decisions.iloc[old_train], teacher_bank)
        old_points = original_vectors[old_train]
        reference = np.median(old_points, axis=1)
        relative = ((old_points - old_teacher[:, None]) ** 2).mean(2) - (
            (reference - old_teacher) ** 2
        ).mean(1)[:, None]
        max_label_difference = max(
            max_label_difference, float(abs(relative - arrays["direct_risk"][old_train, :7]).max())
        )
        np.testing.assert_allclose(
            relative, arrays["direct_risk"][old_train, :7], rtol=1e-10, atol=1e-10
        )
        decisions, base, points = target_nodes(
            original_decisions,
            arrays["features"][:, :7, :33],
            original_vectors,
            joint=model_id == "chronos2",
        )
        inputs = position_inputs(base, points)
        for family in [*sorted(decisions.family_id.unique()), None]:
            train = np.flatnonzero(
                (decisions.split.to_numpy() == "train")
                & ((decisions.family_id.to_numpy() != family) if family is not None else True)
            )
            validation = (
                np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                if family is not None
                else np.array([], dtype=int)
            )
            training, evaluation = decisions.iloc[train], decisions.iloc[validation]
            if set(training.origin_id) & set(evaluation.origin_id) or (
                family is not None and family in set(training.family_id)
            ):
                raise ValueError("positional training violates the family or temporal boundary")
            if family is None and (
                training.origin_id.nunique(),
                training.family_id.nunique(),
                training.source_episode_id.nunique(),
            ) != (165, 15, 5940):
                raise ValueError("the original source population changed")
            local_inputs = {name: value[train] for name, value in inputs.items()}
            target = decision_truth(training, teacher_bank)
            weights = _family_weights(training)
            key = (
                "source"
                if family is None
                else hashlib.sha256((model_id + "|" + family).encode()).hexdigest()[:20]
            )
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            norm_reference = {}
            for mode in ("local", "pooled"):
                directory = output / model_id / key / mode
                directory.mkdir(parents=True, exist_ok=True)
                seed_predictions, entries = [], []
                for seed in SETTINGS["seeds"]:
                    path = directory / f"seed_{seed}.pt"
                    if path.exists():
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            saved["identity_sha256"] != identity_sha
                            or saved["train_ids_sha256"] != train_sha
                            or saved["seed"] != seed
                        ):
                            raise ValueError("a saved positional model changed identity")
                        model = PositionalPortfolio(mode).eval().requires_grad_(False)
                        model.load_state_dict(saved["state_dict"])
                    else:
                        model, history, initial_sha = fit_position(
                            local_inputs, target, weights, mode=mode, seed=seed, settings=SETTINGS
                        )
                        saved = {
                            "state_dict": model.state_dict(),
                            "identity_sha256": identity_sha,
                            "train_ids_sha256": train_sha,
                            "seed": seed,
                            "mode": mode,
                            "model_id": model_id,
                            "initial_parameter_sha256": initial_sha,
                            "training_origins": sorted(training.origin_id.unique()),
                            "training_families": sorted(training.family_id.unique()),
                            "training_history": history,
                        }
                        temporary = path.with_suffix(".tmp")
                        torch.save(saved, temporary)
                        temporary.replace(path)
                    if saved["initial_parameter_sha256"] != initial_hashes.setdefault(
                        seed, saved["initial_parameter_sha256"]
                    ):
                        raise ValueError("matched initialization changed")
                    for name in ("context_mean", "context_scale", "local_mean", "local_scale"):
                        if mode == "local":
                            norm_reference[(seed, name)] = saved["state_dict"][name]
                        elif not torch.equal(
                            norm_reference[(seed, name)], saved["state_dict"][name]
                        ):
                            raise ValueError("matched local and pooled input normalization differ")
                    row = {
                        "model_id": model_id,
                        "held_family": family,
                        "mode": mode,
                        "seed": seed,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "actions": info["actions"],
                    }
                    checkpoints.append(row)
                    entries.append(row)
                    if len(validation):
                        predicted = predict_position(model, inputs, validation)
                        if np.any(predicted < inputs["lower"][validation]) or np.any(
                            predicted > inputs["upper"][validation]
                        ):
                            raise ValueError("a positional forecast left its candidate envelope")
                        seed_predictions.append(predicted)
                    print(
                        json.dumps(
                            {
                                "model": model_id,
                                "held_family": family,
                                "mode": mode,
                                "completed_models": len(checkpoints),
                                "total_models": 192,
                            }
                        ),
                        flush=True,
                    )
                if family is None:
                    source_models.extend(entries)
                    continue
                median = inputs["median"][validation]
                averaged = np.clip(
                    median + np.mean(np.stack(seed_predictions) - median[None], axis=0),
                    inputs["lower"][validation],
                    inputs["upper"][validation],
                )
                predictions = {f"position_{mode}": averaged, "forecast_median_guarded": median}
                for seed, prediction in zip(SETTINGS["seeds"], seed_predictions, strict=True):
                    predictions[f"position_{mode}_seed{seed}"] = prediction
                path = directory / "predictions.npz"
                _save_npz(
                    path,
                    point=np.stack(list(predictions.values())),
                    methods=np.asarray(list(predictions)),
                    validation_indices=validation,
                    seed_points=np.stack(seed_predictions),
                    identity_sha256=np.asarray(identity_sha),
                )
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("source validation futures changed")
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                scores = pd.concat(
                    [
                        evaluation.assign(
                            model_id=model_id,
                            mode=mode,
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
                folds.append(
                    {
                        "model_id": model_id,
                        "held_family": family,
                        "mode": mode,
                        "checkpoints": entries,
                        "prediction_path": str(path.relative_to(output)),
                        "prediction_sha256": file_sha256(path),
                        "scores_path": str(score_path.relative_to(output)),
                        "scores_sha256": file_sha256(score_path),
                    }
                )
    if len(checkpoints) != 192 or len(source_models) != 12 or len(folds) != 60:
        raise ValueError("matched positional study coverage changed")
    frame = pd.concat(all_scores, ignore_index=True)
    keys = [
        "model_id",
        "mode",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    episodes = frame.groupby(keys)[["mae", "mse"]].mean().reset_index()
    for _, group in episodes.groupby(["model_id", "mode", "method"]):
        if group.source_episode_id.nunique() != 1872:
            raise ValueError("source validation coverage changed")
    family = (
        episodes.groupby(["model_id", "mode", "method", "family_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary = family.groupby(["model_id", "mode", "method"])[["mae", "mse"]].mean().reset_index()
    old = pd.read_csv(args.reference_study / "summary.csv", float_precision="round_trip")
    for mode in ("local", "pooled"):
        actual = summary[
            (summary["mode"] == mode) & (summary.method == "forecast_median_guarded")
        ].set_index("model_id")[["mae", "mse"]]
        reference = (
            old[old.method == "forecast_median_guarded"]
            .set_index("model_id")
            .loc[actual.index, ["mae", "mse"]]
        )
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    episodes.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "source_models": source_models,
            "folds": folds,
            "initial_parameter_sha256": initial_hashes,
            "maximum_original_teacher_risk_difference": max_label_difference,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "target_cohort_features_read": False,
            "limits": "positional source study complete; independent model/feature audit and used-cohort transfer remain required",
        },
    )


if __name__ == "__main__":
    main()
