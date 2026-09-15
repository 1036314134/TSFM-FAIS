"""Compare unit, full-range and robust disagreement scales at fixed source budgets."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from disagreement_scale import SCALES, ScaledPortfolio, fit_scaled, forecast_scale, predict_scaled
from latent_source_inputs import ROOT, read_json
from position_objective_inputs import arguments, average_positions, checked_sources, load_inputs
from train_latent_source_gates import aggregate

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = arguments(__doc__)
    parser.set_defaults(protocol=ROOT / "docs/iclr2027/R18_DISAGREEMENT_SCALE_PROTOCOL.md")
    parser.add_argument(
        "--previous-study",
        type=Path,
        default=ROOT / "artifacts/iclr27-r17/position-objectives-v001",
    )
    parser.add_argument(
        "--previous-audit",
        type=Path,
        default=ROOT / "artifacts/iclr27-r17/position-objectives-audit-v002",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed scale studies")
    source = checked_sources(args)
    previous, audit = (
        read_json(args.previous_study / "manifest.json"),
        read_json(args.previous_audit / "manifest.json"),
    )
    if audit["status"] != "completed" or audit["study_sha256"] != file_sha256(
        args.previous_study / "manifest.json"
    ):
        raise ValueError("complete the previous positional audit first")
    if previous["identity"]["reference_sha256"] != file_sha256(
        args.reference_study / "manifest.json"
    ):
        raise ValueError("the source population differs from R17")
    settings = source["identity"]["settings"]
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "scale_module_sha256": file_sha256(ROOT / "scripts/disagreement_scale.py"),
        "input_module_sha256": file_sha256(ROOT / "scripts/position_objective_inputs.py"),
        "loss_module_sha256": file_sha256(ROOT / "scripts/position_objective.py"),
        "model_module_sha256": file_sha256(ROOT / "scripts/positional_forecast_portfolio.py"),
        "previous_sha256": file_sha256(args.previous_study / "manifest.json"),
        "previous_audit_sha256": file_sha256(args.previous_audit / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "settings": settings,
        "primary": "scale18_mad_local",
        "scale_kinds": list(SCALES),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial scale-study definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints, folds = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, inputs, points, labels, actions = load_inputs(args, model_id)
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        validation = np.flatnonzero(frame.split.to_numpy() == "validation")
        train_inputs = {name: value[training] for name, value in inputs.items()}
        weight = _family_weights(frame.iloc[training])
        index_sha = hashlib.sha256(training.tobytes()).hexdigest()
        old_fold = next(row for row in previous["folds"] if row["model_id"] == model_id)
        path = args.previous_study / old_fold["prediction_path"]
        if file_sha256(path) != old_fold["prediction_sha256"]:
            raise ValueError("an original prediction bank changed")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["validation_indices"], validation)
            predictions = dict(zip(saved["methods"].tolist(), saved["point"], strict=True))
        root = output / model_id
        root.mkdir(exist_ok=True)
        for scale_kind in SCALES:
            scale = forecast_scale(points, scale_kind)
            for mode in ("local", "pooled"):
                group = f"scale18_{scale_kind}_{mode}"
                seed_predictions = []
                for seed in (5101, 5102, 5103):
                    path = root / f"{scale_kind}_{mode}_{seed}.pt"
                    metadata = {
                        "identity_sha256": identity_sha,
                        "model_id": model_id,
                        "scale_kind": scale_kind,
                        "mode": mode,
                        "seed": seed,
                        "train_indices_sha256": index_sha,
                        "training_origins": sorted(frame.iloc[training].origin_id.unique()),
                        "training_families": sorted(frame.iloc[training].family_id.unique()),
                    }
                    if not path.exists():
                        model, history, initial = fit_scaled(
                            train_inputs,
                            scale[training],
                            labels["future"][training],
                            weight,
                            mode=mode,
                            seed=seed,
                            settings=settings,
                        )
                        temporary = path.with_suffix(".tmp")
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "history": history,
                                "initial_parameter_sha256": initial,
                                "metadata": metadata,
                            },
                            temporary,
                        )
                        temporary.replace(path)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["metadata"] != metadata:
                        raise ValueError("a resumed scale model changed")
                    if (
                        saved["initial_parameter_sha256"]
                        != source["initial_parameter_sha256"][str(seed)]
                    ):
                        raise ValueError("scaling changed source initialization")
                    model = ScaledPortfolio(mode).eval().requires_grad_(False)
                    model.load_state_dict(saved["state_dict"])
                    point = predict_scaled(model, inputs, scale, validation)
                    if np.any(point < inputs["lower"][validation]) or np.any(
                        point > inputs["upper"][validation]
                    ):
                        raise ValueError("a scaled prediction left its bounds")
                    predictions[f"{group}_seed{seed}"] = point
                    seed_predictions.append(point)
                    checkpoints.append(
                        {
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            **metadata,
                        }
                    )
                predictions[group] = average_positions(seed_predictions, inputs, validation)
                print(
                    f"{model_id} {scale_kind} {mode}: three matched source seeds complete",
                    flush=True,
                )
        if len(predictions) != 61:
            raise ValueError("the registered scale-study methods changed")
        path = root / "predictions.npz"
        _save_npz(
            path,
            validation_indices=validation,
            methods=np.asarray(list(predictions)),
            point=np.stack(list(predictions.values())),
        )
        folds.append(
            {
                "model_id": model_id,
                "actions": actions,
                "prediction_path": str(path.relative_to(output)),
                "prediction_sha256": file_sha256(path),
                "previous_prediction_path": old_fold["prediction_path"],
                "previous_prediction_sha256": old_fold["prediction_sha256"],
            }
        )
    if len(checkpoints) != 36:
        raise ValueError("the scale-model population is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {"checkpoints": checkpoints, "folds": folds, "validation_targets_passed_to_fit": False},
    )
    rows = []
    for fold in folds:
        frame, _, _, labels, _ = load_inputs(args, fold["model_id"], validation=True)
        with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
            indices = saved["validation_indices"]
            for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                error = point - labels["future"][indices]
                rows.append(
                    frame.iloc[indices].assign(
                        model_id=fold["model_id"],
                        method=method,
                        mae=abs(error).mean(1),
                        mse=(error**2).mean(1),
                    )
                )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 456768:
        raise ValueError("the scale-study scoring population changed")
    episodes, families, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "limits": "full-source scale study; independent audit and retrospective transfer required",
        },
    )


if __name__ == "__main__":
    main()
