"""Train matched full- and diagonal-geometry gates using existing teacher labels."""

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
from geometry_forecast_gate import fit_geometry_gate, geometry_features  # noqa: E402
from train_shared_forecast_gate import SETTINGS, predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import (  # noqa: E402
    SharedForecastGate,
    compose_forecasts,
    teacher_quadratics,
)
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "accuracy-root",
        "reference-study",
        "reference-bundle",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed geometry studies")
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    reference, bundle = (
        read_json(args.reference_study / "manifest.json"),
        read_json(args.reference_bundle / "manifest.json"),
    )
    if any(row["status"] != "completed" for row in (prep, reference, bundle)):
        raise ValueError("complete the original source studies")
    if (
        prep["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or reference["identity"]["settings"] != SETTINGS
    ):
        raise ValueError("source inputs or fitting settings changed")
    for name, digest in reference["identity"]["runtime_source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the original source-gate runtime changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "geometry_module_sha256": file_sha256(ROOT / "scripts/geometry_forecast_gate.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "reference_study_sha256": file_sha256(args.reference_study / "manifest.json"),
        "reference_bundle_sha256": file_sha256(args.reference_bundle / "manifest.json"),
        "settings": SETTINGS,
        "feature_count": 47,
        "parameters_per_seed": 1320,
        "primary_mode": "full",
        "matched_mode": "diagonal",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("a partial geometry study changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint_records, folds, scores_all, source_models, initial_hashes = [], [], [], [], {}
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("source candidate predictions changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][:, :7])
        feature_sets = {
            mode: geometry_features(arrays["features"][:, :7, :33], vectors, mode=mode)
            for mode in ("full", "diagonal")
        }
        np.testing.assert_array_equal(
            feature_sets["full"][:, :, :40], feature_sets["diagonal"][:, :, :40]
        )
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
                raise ValueError("the family or temporal fitting boundary changed")
            if family is None and (
                training.origin_id.nunique(),
                training.family_id.nunique(),
                training.source_episode_id.nunique(),
            ) != (165, 15, 5940):
                raise ValueError("the full source fitting population changed")
            weights = _family_weights(training)
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            key = (
                "source"
                if family is None
                else hashlib.sha256((model_id + "|" + family).encode()).hexdigest()[:20]
            )
            old_models = {}
            for seed in SETTINGS["seeds"]:
                if family is None:
                    entry = next(
                        row
                        for row in bundle["models"]
                        if row["model_id"] == model_id
                        and row["objective"] == "ensemble"
                        and row["seed"] == seed
                    )
                    path = args.reference_bundle / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("an original full-source gate changed")
                else:
                    path = args.reference_study / "folds" / key / f"seed_{seed}.pt"
                old = torch.load(path, map_location="cpu", weights_only=True)
                if old["train_ids_sha256"] != train_sha:
                    raise ValueError("geometry and original gates use different source histories")
                old_models[seed] = old
            matched_states = {}
            for mode, features in feature_sets.items():
                directory = output / model_id / key / mode
                directory.mkdir(parents=True, exist_ok=True)
                seed_weights, local_checkpoints = [], []
                for seed in SETTINGS["seeds"]:
                    path = directory / f"seed_{seed}.pt"
                    if path.exists():
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            saved["identity_sha256"] != identity_sha
                            or saved["train_ids_sha256"] != train_sha
                            or saved["seed"] != seed
                        ):
                            raise ValueError("a geometry checkpoint changed fitting identity")
                        model = SharedForecastGate(features=47)
                        model.load_state_dict(saved["state_dict"])
                        model.eval().requires_grad_(False)
                    else:
                        model, history, initial_sha = fit_geometry_gate(
                            features[train], gram[train], alignment[train], weights, seed=seed
                        )
                        saved = {
                            "state_dict": model.state_dict(),
                            "identity_sha256": identity_sha,
                            "train_ids_sha256": train_sha,
                            "model_id": model_id,
                            "mode": mode,
                            "seed": seed,
                            "training_history": history,
                            "initial_parameter_sha256": initial_sha,
                            "training_origins": sorted(training.origin_id.unique()),
                            "training_families": sorted(training.family_id.unique()),
                        }
                        temporary = path.with_suffix(".tmp")
                        torch.save(saved, temporary)
                        temporary.replace(path)
                    expected_initial = initial_hashes.setdefault(
                        seed, saved["initial_parameter_sha256"]
                    )
                    if saved["initial_parameter_sha256"] != expected_initial:
                        raise ValueError("matched gates did not share initialization")
                    for name in ("feature_mean", "feature_scale"):
                        if not torch.equal(
                            saved["state_dict"][name][:, :, :33],
                            old_models[seed]["state_dict"][name],
                        ):
                            raise ValueError("the original 33 feature statistics changed")
                        key_state = (seed, name)
                        if mode == "full":
                            matched_states[key_state] = saved["state_dict"][name][:, :, :40]
                        elif not torch.equal(
                            saved["state_dict"][name][:, :, :40], matched_states[key_state]
                        ):
                            raise ValueError(
                                "matched identity and base-feature normalization differ"
                            )
                    entry = {
                        "model_id": model_id,
                        "held_family": family,
                        "mode": mode,
                        "seed": seed,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "actions": info["actions"],
                    }
                    checkpoint_records.append(entry)
                    local_checkpoints.append(entry)
                    if len(validation):
                        predicted = predict_weights(model, features[validation])
                        np.testing.assert_array_equal(
                            predicted, replay_network(saved["state_dict"], features[validation])
                        )
                        seed_weights.append(predicted)
                if family is None:
                    source_models.extend(local_checkpoints)
                    continue
                mean_weights = np.mean(seed_weights, axis=0)
                methods = {
                    f"geometry_{mode}_gate": compose_forecasts(vectors[validation], mean_weights),
                    "forecast_median_guarded": np.median(vectors[validation], axis=1),
                }
                for seed, probability in zip(SETTINGS["seeds"], seed_weights, strict=True):
                    methods[f"geometry_{mode}_seed{seed}"] = compose_forecasts(
                        vectors[validation], probability
                    )
                prediction_path = directory / "predictions.npz"
                _save_npz(
                    prediction_path,
                    point=np.stack(list(methods.values())),
                    methods=np.asarray(list(methods)),
                    validation_indices=validation,
                    seed_weights=np.stack(seed_weights),
                    mean_weights=mean_weights,
                    identity_sha256=np.asarray(identity_sha),
                )
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("source validation outcomes changed")
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                scores = pd.concat(
                    [
                        evaluation.assign(
                            model_id=model_id,
                            method=name,
                            mode=mode,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                        for name, point in methods.items()
                    ],
                    ignore_index=True,
                )
                score_path = directory / "scores.parquet"
                scores.to_parquet(score_path, index=False)
                scores_all.append(scores)
                folds.append(
                    {
                        "model_id": model_id,
                        "family_id": family,
                        "mode": mode,
                        "checkpoints": local_checkpoints,
                        "prediction_path": str(prediction_path.relative_to(output)),
                        "prediction_sha256": file_sha256(prediction_path),
                        "scores_path": str(score_path.relative_to(output)),
                        "scores_sha256": file_sha256(score_path),
                    }
                )
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "held_family": family,
                        "completed_checkpoints": len(checkpoint_records),
                        "total_checkpoints": 192,
                    }
                ),
                flush=True,
            )
    if len(checkpoint_records) != 192 or len(source_models) != 12 or len(folds) != 60:
        raise ValueError("matched geometry study coverage changed")
    scores = pd.concat(scores_all, ignore_index=True)
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
    episodes = scores.groupby(keys)[["mae", "mse"]].mean().reset_index()
    for _, group in episodes.groupby(["model_id", "mode", "method"]):
        if group.source_episode_id.nunique() != 1872:
            raise ValueError("geometry validation coverage changed")
    family = (
        episodes.groupby(["model_id", "mode", "method", "family_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary = family.groupby(["model_id", "mode", "method"])[["mae", "mse"]].mean().reset_index()
    original = pd.read_csv(args.reference_study / "summary.csv", float_precision="round_trip")
    for mode in ("full", "diagonal"):
        current = summary[
            (summary["mode"] == mode) & (summary.method == "forecast_median_guarded")
        ].set_index("model_id")[["mae", "mse"]]
        expected = (
            original[original.method == "forecast_median_guarded"]
            .set_index("model_id")
            .loc[current.index, ["mae", "mse"]]
        )
        np.testing.assert_allclose(current, expected, rtol=1e-12, atol=1e-12)
    episodes.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoint_records,
            "source_models": source_models,
            "folds": folds,
            "initial_parameter_sha256": initial_hashes,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "target_cohort_features_read": False,
            "limits": "matched source study; independent saved-model audit and used-cohort transfer still required",
        },
    )


if __name__ == "__main__":
    main()
