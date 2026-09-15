"""Run the prespecified source comparison that restores only interval inputs."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_truth  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from audit_source_quantiles import read_json  # noqa: E402
from interval_gate_inputs import load_interval_inputs  # noqa: E402
from train_shared_forecast_gate import SETTINGS, fit_gate, predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import (  # noqa: E402
    SharedForecastGate,
    compose_forecasts,
    teacher_quadratics,
)
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def add_arguments(parser):
    for name, path in {
        "aligned-root": "artifacts/iclr27-r5/aligned-portfolio-prepared-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "quantile-root": "artifacts/iclr27-r6/source-quantile-audit-v001",
        "original-study": "artifacts/iclr27-r5/shared-forecast-gate-v001/ensemble",
        "original-bundle": "artifacts/iclr27-r5/shared-gate-source-bundle-v001",
        "protocol": "docs/iclr2027/R6_INTERVAL_GATE_PLAN.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)


def aggregate(scores):
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
    family = (
        episodes.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    )
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    return episodes, family, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed interval-input study")
    old, bundle = (
        read_json(args.original_study / "manifest.json"),
        read_json(args.original_bundle / "manifest.json"),
    )
    if (
        old["status"] != "completed"
        or bundle["status"] != "completed"
        or old["identity"]["settings"] != SETTINGS
        or bundle["identity"]["settings"] != SETTINGS
    ):
        raise ValueError("complete original source controls with the same fitting schedule")
    for name, digest in old["identity"]["runtime_source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the original source runtime changed")
    if bundle["identity"]["trainer_sha256"] != file_sha256(
        ROOT / "scripts/train_shared_forecast_gate.py"
    ):
        raise ValueError("the original gate trainer changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_module_sha256": file_sha256(ROOT / "scripts/interval_gate_inputs.py"),
        "trainer_sha256": file_sha256(ROOT / "scripts/train_shared_forecast_gate.py"),
        "model_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "aligned_sha256": file_sha256(args.aligned_root / "manifest.json"),
        "accuracy_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "quantile_audit_sha256": file_sha256(args.quantile_root / "manifest.json"),
        "original_study_sha256": file_sha256(args.original_study / "manifest.json"),
        "original_bundle_sha256": file_sha256(args.original_bundle / "manifest.json"),
        "settings": SETTINGS,
        "objective": "unchanged source complete-history teacher ensemble MSE",
        "changed_fields": ["response.has_quantiles", "response.interval_width"],
        "quantile_crossings": "retained without reordering or deleting cases",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial interval source identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    old_folds = {}
    for entry in old["folds"]:
        path = args.original_study / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("an original source fold changed")
        fold = read_json(path)
        old_folds[(fold["model_id"], fold["held_family"])] = fold
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds, model_records, parity = [], [], [], []
    for model_id in ("chronos2", "timesfm2p5"):
        info, decisions, arrays, base, features, vectors, width_delta = load_interval_inputs(
            args, model_id
        )
        gram, alignment = teacher_quadratics(vectors, arrays["direct_risk"][:, :7])
        full_train = np.flatnonzero(decisions.split.to_numpy() == "train")
        training = decisions.iloc[full_train]
        if (
            training.origin_id.nunique(),
            training.family_id.nunique(),
            training.source_episode_id.nunique(),
        ) != (165, 15, 5940):
            raise ValueError("the original source population changed")
        directory = output / model_id
        directory.mkdir(exist_ok=True)
        feature_path = directory / "features.npy"
        if feature_path.exists():
            np.testing.assert_array_equal(np.load(feature_path, allow_pickle=False), features)
        else:
            np.save(feature_path, features, allow_pickle=False)
        model_records.append(
            {
                "model_id": model_id,
                "actions": info["actions"],
                "features_path": str(feature_path.relative_to(output)),
                "features_sha256": file_sha256(feature_path),
                "maximum_width_reconstruction_difference": width_delta,
            }
        )
        original = next(
            row
            for row in bundle["models"]
            if row["model_id"] == model_id
            and row["objective"] == "ensemble"
            and row["seed"] == 5101
        )
        original_path = args.original_bundle / original["path"]
        if file_sha256(original_path) != original["sha256"]:
            raise ValueError("an original full-source control changed")
        original_state = torch.load(original_path, map_location="cpu", weights_only=True)
        parity_path = directory / "point_only_parity.json"
        if not parity_path.exists():
            model, _ = fit_gate(
                base[full_train],
                gram[full_train],
                alignment[full_train],
                _family_weights(training),
                kind="ensemble",
                seed=5101,
            )
            if (
                original_state["train_ids_sha256"]
                != hashlib.sha256(full_train.tobytes()).hexdigest()
            ):
                raise ValueError("point-only parity uses a different source population")
            for name, value in model.state_dict().items():
                if not torch.equal(value, original_state["state_dict"][name]):
                    raise ValueError(f"point-only fitting failed exact replay: {name}")
            _write_json(
                parity_path,
                {
                    "status": "passed",
                    "identity_sha256": identity_sha,
                    "original_checkpoint_sha256": original["sha256"],
                    "maximum_parameter_difference": 0,
                },
            )
        if read_json(parity_path)["identity_sha256"] != identity_sha:
            raise ValueError("point-only parity belongs to a different study")
        parity.append(
            {
                "model_id": model_id,
                "path": str(parity_path.relative_to(output)),
                "sha256": file_sha256(parity_path),
            }
        )
        for family in [None, *sorted(decisions.family_id.unique())]:
            train = (
                full_train
                if family is None
                else np.flatnonzero(
                    (decisions.split.to_numpy() == "train")
                    & (decisions.family_id.to_numpy() != family)
                )
            )
            evaluation = (
                np.array([], dtype=int)
                if family is None
                else np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
            )
            frame = decisions.iloc[train]
            if family in set(frame.family_id) or set(frame.origin_id) & set(
                decisions.iloc[evaluation].origin_id
            ):
                raise ValueError("a held family or validation origin entered fitting")
            folder = directory / ("full_source" if family is None else family)
            folder.mkdir(exist_ok=True)
            seed_weights, checkpoint_rows = [], []
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            for seed in SETTINGS["seeds"]:
                path = folder / f"seed_{seed}.pt"
                if not path.exists():
                    model, history = fit_gate(
                        features[train],
                        gram[train],
                        alignment[train],
                        _family_weights(frame),
                        kind="ensemble",
                        seed=seed,
                    )
                    model.eval().requires_grad_(False)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "identity_sha256": identity_sha,
                            "train_ids_sha256": train_sha,
                            "seed": seed,
                            "model_id": model_id,
                            "held_family": family,
                            "training_origins": sorted(frame.origin_id.unique()),
                            "training_families": sorted(frame.family_id.unique()),
                            "training_history": history,
                        },
                        path,
                    )
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != identity_sha
                    or saved["train_ids_sha256"] != train_sha
                    or saved["seed"] != seed
                ):
                    raise ValueError("a partial interval checkpoint changed identity")
                model = SharedForecastGate().eval().requires_grad_(False)
                model.load_state_dict(saved["state_dict"])
                if len(evaluation):
                    probability = predict_weights(model, features[evaluation])
                    np.testing.assert_array_equal(
                        probability, replay_network(saved["state_dict"], features[evaluation])
                    )
                    seed_weights.append(probability)
                checkpoint_rows.append(
                    {
                        "model_id": model_id,
                        "held_family": family,
                        "seed": seed,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
            checkpoints.extend(checkpoint_rows)
            if family is None:
                continue
            old_fold = old_folds[(model_id, family)]
            old_path = args.original_study / old_fold["predictions_path"]
            expected = next(
                row["sha256"]
                for row in old_fold["files"]
                if row["path"] == old_fold["predictions_path"]
            )
            if file_sha256(old_path) != expected:
                raise ValueError("original point-only comparison predictions changed")
            with np.load(old_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["decision_indices"], evaluation)
                predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
            np.testing.assert_array_equal(
                predictions["forecast_median_guarded"], np.median(vectors[evaluation], axis=1)
            )
            predictions["interval_gate"] = compose_forecasts(
                vectors[evaluation], np.mean(seed_weights, axis=0)
            )
            for seed, weights in zip(SETTINGS["seeds"], seed_weights, strict=True):
                predictions[f"interval_seed{seed}"] = compose_forecasts(
                    vectors[evaluation], weights
                )
            path = folder / "predictions.npz"
            _save_npz(
                path,
                decision_indices=evaluation,
                methods=np.asarray(list(predictions)),
                point=np.stack(list(predictions.values())),
                seed_weights=np.stack(seed_weights),
                identity_sha256=np.asarray(identity_sha),
            )
            folds.append(
                {
                    "model_id": model_id,
                    "held_family": family,
                    "train_ids_sha256": train_sha,
                    "prediction_path": str(path.relative_to(output)),
                    "prediction_sha256": file_sha256(path),
                    "original_prediction_sha256": expected,
                }
            )
            print(f"{model_id} {family}: interval fits and fixed comparisons saved", flush=True)
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "folds": folds,
            "checkpoints": checkpoints,
            "models": model_records,
            "new_validation_scoring_started": False,
        },
    )
    accuracy = read_json(args.accuracy_root / "manifest.json")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("original validation future coordinates changed")
    scores = []
    for model_id in ("chronos2", "timesfm2p5"):
        from aligned_portfolio_io import load_prepared_model

        _, decisions, _ = load_prepared_model(
            args.aligned_root, read_json(args.aligned_root / "manifest.json"), model_id
        )
        for fold in (row for row in folds if row["model_id"] == model_id):
            path = output / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a saved validation prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                frame = decisions.iloc[saved["decision_indices"]]
                truth = decision_truth(frame, np.load(truth_path, mmap_mode="r"))
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    scores.append(
                        frame.assign(
                            model_id=model_id,
                            method=method,
                            mae=abs(point - truth).mean(1),
                            mse=((point - truth) ** 2).mean(1),
                        )
                    )
    scores = pd.concat(scores, ignore_index=True)
    episodes, family, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    family.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    previous = pd.read_csv(args.original_study / "summary.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(
        summary[summary.method.isin(previous.method)].reset_index(drop=True),
        previous,
        check_exact=True,
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": model_records,
            "point_only_parity": parity,
            "checkpoints": checkpoints,
            "folds": folds,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "new_fits": len(checkpoints) + 2,
            "limits": "source comparison only; original controls preserved; independent audit required before interpretation or target collection",
        },
    )


if __name__ == "__main__":
    main()
