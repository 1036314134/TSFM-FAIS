"""Fit a single origin-weighted ensemble gate with fixed family-based normalization."""

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
from origin_weighted_gate import fit_weighted_gate  # noqa: E402
from train_shared_forecast_gate import SETTINGS, predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import (  # noqa: E402
    SharedForecastGate,
    compose_forecasts,
    teacher_quadratics,
)
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "accuracy-root",
        "original-study",
        "original-audit",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed weighting studies")
    prep, accuracy = (
        read_json(args.aligned_root / "manifest.json"),
        read_json(args.accuracy_root / "manifest.json"),
    )
    original, audit = (
        read_json(args.original_study / "manifest.json"),
        read_json(args.original_audit / "manifest.json"),
    )
    if any(row["status"] != "completed" for row in (prep, original, audit)):
        raise ValueError("complete the original source study and audit")
    if (
        prep["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or original["identity"]["settings"] != SETTINGS
        or original["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
    ):
        raise ValueError("source records or training settings changed")
    for name, digest in original["identity"]["runtime_source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the original gate runtime changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "trainer_sha256": file_sha256(ROOT / "scripts/origin_weighted_gate.py"),
        "original_study_sha256": file_sha256(args.original_study / "manifest.json"),
        "original_audit_sha256": file_sha256(args.original_audit / "manifest.json"),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "settings": SETTINGS,
        "normalization": "unchanged family weights",
        "loss_weights": "equal original histories",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial weighting study changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results, checkpoints, controls, parity = [], [], {}, []
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model_id)
        point_path = args.accuracy_root / f"{model_id}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("source candidate predictions changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model_id].index(name) for name in info["actions"]]
        ]
        vectors = decision_vectors(decisions, bank)
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][:, :7])
        features = arrays["features"][:, :7, :33]
        families = sorted(decisions.family_id.unique())
        for held_family in [*families, None]:
            train_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "train")
                & (
                    (decisions.family_id.to_numpy() != held_family)
                    if held_family is not None
                    else True
                )
            )
            eval_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "validation")
                & (
                    (decisions.family_id.to_numpy() == held_family)
                    if held_family is not None
                    else False
                )
            )
            training, evaluation = decisions.iloc[train_ids], decisions.iloc[eval_ids]
            if set(training.origin_id) & set(evaluation.origin_id) or (
                held_family is not None and held_family in set(training.family_id)
            ):
                raise ValueError("the family or temporal holdout boundary changed")
            counts = training.groupby("origin_id").size()
            if set(counts) != {36 if model_id == "chronos2" else 72}:
                raise ValueError("equal decision rows no longer imply equal original histories")
            if held_family is None and (
                training.origin_id.nunique(),
                training.family_id.nunique(),
                training.source_episode_id.nunique(),
            ) != (165, 15, 5940):
                raise ValueError("full source training population changed")
            norm_weights = _family_weights(training)
            unit_weights = np.ones(len(training))
            norm_model = SharedForecastGate()
            norm_model.fit_normalization(features[train_ids], norm_weights)
            key = (
                "source"
                if held_family is None
                else hashlib.sha256((model_id + "|" + held_family).encode()).hexdigest()[:20]
            )
            directory = output / model_id / key
            directory.mkdir(parents=True, exist_ok=True)
            train_sha = hashlib.sha256(train_ids.tobytes()).hexdigest()
            if held_family == families[0]:
                parity_path = output / model_id / "baseline_parity.json"
                if not parity_path.exists():
                    recovered, _ = fit_weighted_gate(
                        features[train_ids],
                        gram[train_ids],
                        alignment[train_ids],
                        norm_weights,
                        norm_weights,
                        seed=5101,
                    )
                    old_path = args.original_study / "folds" / key / "seed_5101.pt"
                    old = torch.load(old_path, map_location="cpu", weights_only=True)
                    if old["train_ids_sha256"] != train_sha:
                        raise ValueError("baseline parity used different training decisions")
                    for name, tensor in recovered.state_dict().items():
                        if not torch.equal(tensor, old["state_dict"][name]):
                            raise ValueError(
                                f"the new trainer did not exactly reproduce the original baseline: {name}"
                            )
                    _write_json(
                        parity_path,
                        {
                            "status": "passed",
                            "original_checkpoint_sha256": file_sha256(old_path),
                            "maximum_parameter_difference": 0,
                            "identity_sha256": identity_sha,
                            "held_family": held_family,
                        },
                    )
                if read_json(parity_path)["identity_sha256"] != identity_sha:
                    raise ValueError("a parity check used another trainer identity")
                parity.append(
                    {
                        "model_id": model_id,
                        "path": str(parity_path.relative_to(output)),
                        "sha256": file_sha256(parity_path),
                    }
                )
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
                        raise ValueError("a saved weighting model changed identity")
                    model = SharedForecastGate()
                    model.load_state_dict(saved["state_dict"])
                    model.eval()
                else:
                    model, history = fit_weighted_gate(
                        features[train_ids],
                        gram[train_ids],
                        alignment[train_ids],
                        norm_weights,
                        unit_weights,
                        seed=seed,
                    )
                    saved = {
                        "identity_sha256": identity_sha,
                        "train_ids_sha256": train_sha,
                        "seed": seed,
                        "model_id": model_id,
                        "held_family": held_family,
                        "state_dict": model.state_dict(),
                        "training_history": history,
                        "training_origins": sorted(training.origin_id.unique()),
                        "training_families": sorted(training.family_id.unique()),
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                for name in ("feature_mean", "feature_scale"):
                    if not torch.equal(saved["state_dict"][name], norm_model.state_dict()[name]):
                        raise ValueError(
                            "the loss-weighting intervention changed feature normalization"
                        )
                if len(eval_ids):
                    predicted = predict_weights(model, np.ascontiguousarray(features[eval_ids]))
                    np.testing.assert_array_equal(
                        predicted,
                        replay_network(
                            saved["state_dict"], np.ascontiguousarray(features[eval_ids])
                        ),
                    )
                    seed_weights.append(predicted)
                row = {
                    "model_id": model_id,
                    "held_family": held_family,
                    "seed": seed,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
                local_checkpoints.append(row)
                checkpoints.append(row)
            fixed, gap, _ = simplex_quadratic_weights(
                gram[train_ids].mean(0)[None], alignment[train_ids].mean(0)[None]
            )
            mass = (
                training.assign(family_weight=norm_weights)
                .groupby("family_id")
                .agg(
                    origins=("origin_id", "nunique"),
                    family_mass=("family_weight", "sum"),
                    rows=("origin_id", "size"),
                )
            )
            mass["original_fraction"] = mass.family_mass / mass.family_mass.sum()
            mass["origin_equal_fraction"] = mass.rows / mass.rows.sum()
            mass.to_csv(directory / "training_mass.csv")
            if held_family is None:
                controls[model_id] = {
                    "actions": info["actions"],
                    "convex_weights": fixed[0].tolist(),
                    "optimality_gap": float(gap[0]),
                    "models": local_checkpoints,
                }
            else:
                mean_weights = np.mean(seed_weights, axis=0)
                predictions = {
                    "origin_weighted_gate": compose_forecasts(vectors[eval_ids], mean_weights),
                    "origin_weighted_fixed": compose_forecasts(
                        vectors[eval_ids], np.repeat(fixed, len(eval_ids), axis=0)
                    ),
                    "forecast_median_guarded": np.median(vectors[eval_ids], axis=1),
                }
                _save_npz(
                    directory / "predictions.npz",
                    point=np.stack(list(predictions.values())),
                    methods=np.asarray(list(predictions)),
                    decision_indices=eval_ids,
                    seed_weights=np.stack(seed_weights),
                    mean_weights=mean_weights,
                )
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("validation futures changed")
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
                scores.to_parquet(directory / "scores.parquet", index=False)
                results.append(scores)
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "held_family": held_family,
                        "completed_checkpoints": len(checkpoints),
                        "total_checkpoints": 96,
                    }
                ),
                flush=True,
            )
    if len(checkpoints) != 96 or len(parity) != 2:
        raise ValueError("weighting study coverage changed")
    scores = pd.concat(results, ignore_index=True)
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
            raise ValueError("source development coverage changed")
    family = (
        episodes.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    )
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    old = pd.read_csv(args.original_study / "summary.csv", float_precision="round_trip")
    actual = summary[summary.method == "forecast_median_guarded"].set_index("model_id")[
        ["mae", "mse"]
    ]
    expected = (
        old[old.method == "forecast_median_guarded"]
        .set_index("model_id")
        .loc[actual.index, ["mae", "mse"]]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    episodes.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "controls.json", controls)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "parity_checks": parity,
            "controls_sha256": file_sha256(output / "controls.json"),
            "new_forecaster_calls": 0,
            "target_evaluation_features_read": False,
            "limits": "single loss-weighting intervention; R6 transfer and independent input/metric replay remain pending",
        },
    )


if __name__ == "__main__":
    main()
