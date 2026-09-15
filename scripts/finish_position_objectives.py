"""Finish reporting for already frozen R17 models without repeating any fitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json
from position_objective_inputs import arguments, checked_sources, load_inputs
from train_latent_source_gates import aggregate

from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed positional reports")
    checked_sources(args)
    identity = read_json(output / "identity.json")
    for name, path in (
        ("script_sha256", ROOT / "scripts/train_position_objectives.py"),
        ("loss_module_sha256", ROOT / "scripts/position_objective.py"),
        ("input_module_sha256", ROOT / "scripts/position_objective_inputs.py"),
        ("model_module_sha256", ROOT / "scripts/positional_forecast_portfolio.py"),
        ("protocol_sha256", args.protocol),
    ):
        if identity[name] != file_sha256(path):
            raise ValueError("a frozen training definition changed")
    freeze = read_json(output / "prediction_freeze.json")
    if (
        len(freeze["checkpoints"]) != 24
        or len(freeze["compatibility"]) != 4
        or len(freeze["folds"]) != 2
    ):
        raise ValueError("complete all registered fits before recovering reports")
    preserved = [*freeze["checkpoints"], *freeze["compatibility"]]
    for fold in freeze["folds"]:
        preserved.extend(fold["controls"])
        preserved.append({"path": fold["prediction_path"], "sha256": fold["prediction_sha256"]})
    for entry in preserved:
        if file_sha256(output / entry["path"]) != entry["sha256"]:
            raise ValueError("an existing model, control or prediction changed")
    rows = []
    for fold in freeze["folds"]:
        frame, _, _, labels, _ = load_inputs(args, fold["model_id"], validation=True)
        with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
            indices = saved["validation_indices"]
            if len(saved["methods"]) != 37 or len(indices) != 3744:
                raise ValueError("the frozen method or validation population changed")
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
    if len(scores) != 277056 or set(scores.model_id) != {"chronos2", "timesfm2p5"}:
        raise ValueError("the reporting population is incomplete")
    episodes, families, summary = aggregate(scores)
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    recovery = {
        "script_sha256": file_sha256(Path(__file__)),
        "note_sha256": file_sha256(ROOT / "docs/iclr2027/R17_REPORTING_RECOVERY_NOTE.md"),
        "preserved_files": [{"path": row["path"], "sha256": row["sha256"]} for row in preserved],
        "new_fits": 0,
        "new_predictions": 0,
        "restored_field": "model_id",
    }
    _write_json(output / "reporting_recovery.json", recovery)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": file_sha256(output / "identity.json"),
            "checkpoints": freeze["checkpoints"],
            "folds": freeze["folds"],
            "compatibility": freeze["compatibility"],
            "score_rows": len(scores),
            "new_forecaster_calls": 0,
            "reporting_recovery": recovery,
            "limits": "recovered full-source temporal report; no retraining; independent audit and target transfer required",
        },
    )


if __name__ == "__main__":
    main()
