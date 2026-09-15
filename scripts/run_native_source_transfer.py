"""Fit the fixed shared gate with other groups' original-missing histories."""

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
from native_source_transfer_io import (  # noqa: E402
    checked_source_bindings,
    fold_indices,
    input_arguments,
    load_inputs,
    masked_geometry,
    native_bank_path,
    observed_errors,
    read_json,
    summarize_scores,
)
from train_shared_forecast_gate import SETTINGS, fit_gate, predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights  # noqa: E402
from tsfm_fais.routing.utility import _family_weights  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def scores_for(frame, model, predictions, truth, observed):
    records = []
    for method, point in predictions.items():
        mae, mse = observed_errors(point, truth, observed, joint=model == "chronos2")
        records.append(frame.assign(model_id=model, method=method, mae=mae, mse=mse))
    return pd.concat(records, ignore_index=True)


def source_references(args, model_id, actions, data, frame):
    control = read_json(args.future_control / "manifest.json")
    if control["status"] != "completed" or control["identity"]["settings"] != SETTINGS:
        raise ValueError("the original future-supervised control changed")
    selected = [row for row in control["models"] if row["model_id"] == model_id]
    if [row["seed"] for row in selected] != SETTINGS["seeds"]:
        raise ValueError("source control seeds changed")
    weights = []
    for entry in selected:
        path = args.future_control / entry["path"]
        if file_sha256(path) != entry["sha256"] or entry["actions"] != actions:
            raise ValueError("a source control checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = SharedForecastGate().eval().requires_grad_(False)
        model.load_state_dict(saved["state_dict"])
        probability = predict_weights(model, data["features"])
        np.testing.assert_array_equal(
            probability, replay_network(saved["state_dict"], data["features"])
        )
        weights.append(probability)
    fixed = read_json(args.future_cv / "full_source_fixed_controls.json")[model_id]
    if fixed["actions"] != actions:
        raise ValueError("source fixed candidate ordering changed")
    result = {
        "source_future_gate": compose_forecasts(data["points"], np.mean(weights, axis=0)),
        "future_source_fixed_convex": compose_forecasts(
            data["points"], np.broadcast_to(fixed["convex_weights"], (len(data["points"]), 7))
        ),
        "forecast_median_guarded": np.median(data["points"], axis=1),
        "forecast_median_with_motm": data["median8"],
    }
    result.update({name: data["points"][:, index] for index, name in enumerate(actions)})
    # Preserve the audited controls' original batching and prediction values.
    for cohort in ("legacy_native", "r6"):
        indices = np.flatnonzero(frame.cohort.to_numpy() == cohort)
        with np.load(native_bank_path(args, model_id, cohort), allow_pickle=False) as saved:
            cached = saved["point_z"][:, saved["methods"].tolist().index("source_future_gate")]
        result["source_future_gate"][indices] = decision_truth(frame.iloc[indices], cached)
    return result


def check_reference_scores(args, frame, data, model, predictions):
    original = pd.read_csv(
        args.comparison_root / "comparison_summary.csv", float_precision="round_trip"
    )
    records = []
    subsets = (
        ("legacy_native", "naturally_missing", (frame.cohort == "legacy_native").to_numpy()),
        ("r6", "new_native_missing", (frame.family_id == "beijing_multisite").to_numpy()),
        ("r6", "time_grid_gap_missing", (frame.family_id == "bike_sharing").to_numpy()),
    )
    methods = [
        "source_future_gate",
        "future_source_fixed_convex",
        "forecast_median_guarded",
        "forecast_median_with_motm",
    ]
    for cohort, panel, keep in subsets:
        scored = scores_for(
            frame[keep],
            model,
            {name: predictions[name][keep] for name in methods},
            data["truth"][keep],
            data["observed"][keep],
        )
        summary = summarize_scores(scored)[2].set_index("method")
        expected = (
            original[
                (original.cohort == cohort)
                & (original.panel == panel)
                & (original.model_id == model)
                & (original.horizon == 96)
            ]
            .set_index("method")
            .loc[methods]
        )
        np.testing.assert_allclose(
            summary.loc[methods, ["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-10, atol=1e-10
        )
        records.append(
            {
                "cohort": cohort,
                "panel": panel,
                "model_id": model,
                "methods": methods,
                "maximum_difference": float(
                    abs(
                        summary.loc[methods, ["mae", "mse"]].to_numpy()
                        - expected[["mae", "mse"]].to_numpy()
                    ).max()
                ),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R6_NATIVE_SOURCE_TRANSFER_PLAN.md"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed native-source study")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "io_sha256": file_sha256(ROOT / "scripts/native_source_transfer_io.py"),
        "trainer_sha256": file_sha256(ROOT / "scripts/train_shared_forecast_gate.py"),
        "model_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "source_bindings": checked_source_bindings(args),
        "fixed_source_weights_sha256": file_sha256(
            args.future_cv / "full_source_fixed_controls.json"
        ),
        "settings": SETTINGS,
        "native_origins": 358,
        "native_families": 9,
        "held_groups": 8,
        "objective": "observed-target equal-weight standardized future MSE",
        "source_control_predictions": "original audited native-cohort predictions; preserve original batching",
        "interpretation": "retrospective grouped development; all held-group families excluded from fitting",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial study identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    entries, model_inputs, reference_checks = [], [], []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, data, actions = load_inputs(args, model_id)
        directory = output / model_id
        directory.mkdir(exist_ok=True)
        frame_path, data_path = directory / "decisions.parquet", directory / "inputs.npz"
        if data_path.exists():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(frame_path), check_exact=True)
            with np.load(data_path, allow_pickle=False) as saved:
                for name, values in data.items():
                    np.testing.assert_array_equal(saved[name], values)
        else:
            frame.to_parquet(frame_path, index=False)
            _save_npz(data_path, **data)
        input_sha = file_sha256(data_path)
        model_inputs.append(
            {
                "model_id": model_id,
                "actions": actions,
                "data_path": str(data_path.relative_to(output)),
                "data_sha256": input_sha,
                "frame_path": str(frame_path.relative_to(output)),
                "frame_sha256": file_sha256(frame_path),
            }
        )
        references = source_references(args, model_id, actions, data, frame)
        reference_checks.extend(check_reference_scores(args, frame, data, model_id, references))
        _write_json(output / "reference_checks.json", reference_checks)
        gram, alignment = masked_geometry(
            data["points"], data["truth"], data["observed"], joint=model_id == "chronos2"
        )
        groups = sorted(frame.loc[frame.cohort != "source", "holdout_group"].unique())
        for group in groups:
            train, evaluation = fold_indices(frame, group)
            training = frame.iloc[train]
            sample_weights = _family_weights(training)
            train_sha = hashlib.sha256(train.tobytes()).hexdigest()
            fold_root = directory / group
            fold_root.mkdir(exist_ok=True)
            checkpoints, seed_weights = [], []
            for seed in SETTINGS["seeds"]:
                checkpoint = fold_root / f"seed_{seed}.pt"
                if not checkpoint.exists():
                    model, history = fit_gate(
                        data["features"][train],
                        gram[train],
                        alignment[train],
                        sample_weights,
                        kind="ensemble",
                        seed=seed,
                    )
                    model.eval().requires_grad_(False)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "identity_sha256": identity_sha,
                            "input_sha256": input_sha,
                            "train_ids_sha256": train_sha,
                            "training_origins": sorted(training.origin_id.unique()),
                            "training_families": sorted(training.family_id.unique()),
                            "held_group": group,
                            "model_id": model_id,
                            "seed": seed,
                            "training_history": history,
                        },
                        checkpoint,
                    )
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != identity_sha
                    or saved["input_sha256"] != input_sha
                    or saved["train_ids_sha256"] != train_sha
                    or saved["seed"] != seed
                ):
                    raise ValueError("a partial checkpoint belongs to another fit")
                model = SharedForecastGate().eval().requires_grad_(False)
                model.load_state_dict(saved["state_dict"])
                probability = predict_weights(model, data["features"][evaluation])
                np.testing.assert_array_equal(
                    probability, replay_network(saved["state_dict"], data["features"][evaluation])
                )
                seed_weights.append(probability)
                checkpoints.append(
                    {
                        "path": str(checkpoint.relative_to(output)),
                        "sha256": file_sha256(checkpoint),
                        "seed": seed,
                    }
                )
            probability = sample_weights / sample_weights.sum()
            fixed, gap, _ = simplex_quadratic_weights(
                np.einsum("n,nab->ab", probability, gram[train])[None],
                np.einsum("n,na->a", probability, alignment[train])[None],
            )
            forecasts = {
                "augmented_native_gate": compose_forecasts(
                    data["points"][evaluation], np.mean(seed_weights, axis=0)
                ),
                "augmented_native_fixed": compose_forecasts(
                    data["points"][evaluation], np.broadcast_to(fixed[0], (len(evaluation), 7))
                ),
                **{name: point[evaluation] for name, point in references.items()},
            }
            bank_path = fold_root / "predictions.npz"
            _save_npz(
                bank_path,
                points=np.stack(list(forecasts.values())),
                methods=np.asarray(list(forecasts)),
                evaluation_indices=evaluation,
                seed_weights=np.stack(seed_weights),
                fixed_weights=fixed[0],
            )
            entries.append(
                {
                    "model_id": model_id,
                    "held_group": group,
                    "checkpoints": checkpoints,
                    "prediction_path": str(bank_path.relative_to(output)),
                    "prediction_sha256": file_sha256(bank_path),
                    "train_ids_sha256": train_sha,
                    "training_origins": training.origin_id.nunique(),
                    "training_families": training.family_id.nunique(),
                    "evaluation_origins": frame.iloc[evaluation].origin_id.nunique(),
                    "fixed_optimality_gap": float(gap[0]),
                }
            )
            print(f"{model_id} {group}: three fits and fixed control saved", flush=True)
    _write_json(
        output / "prediction_freeze.json",
        {
            "identity_sha256": identity_sha,
            "models": model_inputs,
            "folds": entries,
            "held_group_scoring_started": False,
        },
    )
    scores = []
    for info in model_inputs:
        frame = pd.read_parquet(output / info["frame_path"])
        with np.load(output / info["data_path"], allow_pickle=False) as saved:
            truth, observed = saved["truth"], saved["observed"]
        for entry in (row for row in entries if row["model_id"] == info["model_id"]):
            path = output / entry["prediction_path"]
            if file_sha256(path) != entry["prediction_sha256"]:
                raise ValueError("a frozen held-group prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                indices = saved["evaluation_indices"]
                predictions = dict(zip(saved["methods"].tolist(), saved["points"], strict=True))
            scores.append(
                scores_for(
                    frame.iloc[indices],
                    info["model_id"],
                    predictions,
                    truth[indices],
                    observed[indices],
                )
            )
    scores = pd.concat(scores, ignore_index=True)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes, families, summary, group_summary = summarize_scores(scores)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    group_summary.to_csv(output / "group_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": model_inputs,
            "folds": entries,
            "checkpoints": sum(len(row["checkpoints"]) for row in entries),
            "score_rows": len(scores),
            "reference_checks_sha256": file_sha256(output / "reference_checks.json"),
            "new_forecaster_calls": 0,
            "independent_audit": "required before interpreting new scores",
        },
    )


if __name__ == "__main__":
    main()
