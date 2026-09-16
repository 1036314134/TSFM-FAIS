"""Freeze the single-factor MAE-trained repair before source/native future scoring."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from evaluate_patch_repair import summary_tables
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook
from mae_repair_inputs import evaluation_cases, load_evaluation_case
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--readout-only", action="store_true")
    args = parser.parse_args()
    training_root, output = args.training_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MAE-repair results")
    trained = read_json(training_root / "manifest.json")
    if (
        trained["status"] != "completed"
        or len(trained["models"]) != 1
        or bool(trained["identity"]["smoke_only"]) != args.smoke
    ):
        raise ValueError("finish the matching single source-trained model first")
    record = trained["models"][0]
    checkpoint = training_root / record["checkpoint_path"]
    if file_sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("the frozen MAE checkpoint changed")
    cases = evaluation_cases(args.smoke)
    identity = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            training_root / "manifest.json",
            ROOT / "scripts/mae_repair_inputs.py",
            ROOT / "scripts/learned_patch_repair.py",
            ROOT / "docs/iclr2027/R26_MAE_REPAIR_PROTOCOL.md",
            ROOT / "artifacts/iclr27-r24/repair-evaluation-v001/predictions_frozen.json",
            ROOT / "artifacts/iclr27-r25/long-forecasts-v001/manifest.json",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial MAE readout definitions changed")
    _write_json(output / "identity.json", identity)
    freeze_path = output / "predictions_frozen.json"
    if not args.readout_only:
        if freeze_path.exists():
            raise ValueError("preserve frozen MAE-repair predictions")
        torch.set_num_threads(1)
        started = perf_counter()
        runner, adapter, backbone, digest, _ = make_forecaster(
            "chronos2",
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        repair = PatchRepair(record["width"], record["patch_size"], record["rank"]).to("cuda")
        repair.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        repair.eval().requires_grad_(False)
        predictions = []
        for row in cases:
            data = load_evaluation_case(row)
            if file_sha256(Path(row["old_prediction_path"])) != row["old_prediction_sha256"]:
                raise ValueError("an old fixed prediction control changed")
            spec = ForecastSpec(
                "chronos2",
                "joint_multivariate",
                96,
                context_length=row["context_length"],
                target_indices=[0, 1],
            )
            with RepairHook(backbone, "chronos2", np.isfinite(data["context"]), repair, "fraction"):
                point = runner.predict_missing(data["base"][None], spec).point[0]
            with np.load(row["old_prediction_path"], allow_pickle=False) as old:
                methods = {
                    name: value
                    for name, value in zip(old["methods"].tolist(), old["points"], strict=True)
                }
            methods["mae_repair"] = (point - data["mean"][:2]) / data["scale"][:2]
            if len(methods) != (20 if row["panel"] == "source_validation" else 19):
                raise ValueError("the registered comparison population changed")
            path = output / "predictions" / f"{row['case_id']}.npz"
            _save_npz(
                path, methods=np.asarray(list(methods)), points=np.stack(list(methods.values()))
            )
            predictions.append(
                {
                    "case_id": row["case_id"],
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            if len(predictions) % 100 == 0:
                print(f"frozen MAE-repair cases: {len(predictions)}", flush=True)
        if parameter_digest(backbone) != digest:
            raise ValueError("the frozen forecasting backbone changed")
        _write_json(
            freeze_path,
            {
                "status": "completed",
                "identity": identity,
                "predictions": predictions,
                "forecast_calls": len(cases),
                "wall_seconds": perf_counter() - started,
                "evaluation_future_read": False,
            },
        )
        return
    frozen = read_json(freeze_path)
    if frozen["status"] != "completed" or frozen["identity"] != identity:
        raise ValueError("freeze all matching forecasts before the readout")
    metadata = {r["case_id"]: r for r in cases}
    scores, targets = [], []
    for entry in frozen["predictions"]:
        row = metadata[entry["case_id"]]
        if (
            file_sha256(output / entry["path"]) != entry["sha256"]
            or file_sha256(Path(row["original_path"])) != row["original_sha256"]
        ):
            raise ValueError("a frozen prediction or original outcome changed")
        data = load_evaluation_case(row)
        with np.load(row["original_path"], allow_pickle=False) as original:
            truth = original["future"][:96, :2]
            observed = np.isfinite(truth)
            if "future_observed" in original.files:
                np.testing.assert_array_equal(observed, original["future_observed"][:96, :2])
        counts = observed.sum(0)
        if (counts < 48).any():
            raise ValueError("the observed target support changed")
        truth_z = (truth - data["mean"][:2]) / data["scale"][:2]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            delta = np.where(observed[None], saved["points"] - truth_z[None], 0.0)
            mae, mse = abs(delta).sum(1) / counts, (delta**2).sum(1) / counts
            for index, method in enumerate(saved["methods"].tolist()):
                record = {
                    name: row[name]
                    for name in (
                        "case_id",
                        "panel",
                        "context_length",
                        "group_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "origin_id",
                    )
                }
                record.update(
                    model_id="chronos2",
                    method=method,
                    mae=float(mae[index].mean()),
                    mse=float(mse[index].mean()),
                )
                scores.append(record)
                for slot in (0, 1):
                    targets.append(
                        {
                            **record,
                            "target_slot": slot,
                            "observed_count": int(counts[slot]),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if not args.smoke and (len(frame) != 22999 or len(targets) != 45998):
        raise ValueError("full MAE-repair score coverage changed")
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    for name, table in summary_tables(frame).items():
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "predictions": frozen["predictions"],
            "score_rows": len(frame),
            "target_score_rows": len(targets),
            "primary": "mae_repair",
            "primary_panel": "native_development",
            "primary_context_length": 192,
            "primary_metric": "mae",
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
