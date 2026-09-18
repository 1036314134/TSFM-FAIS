"""Train forecast and imputation objectives on the same historical calibration cases."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import (
    LEARNED,
    PARENT,
    ROOT,
    SEED,
    evaluation_rows,
    evaluation_sources,
    fixed_mae_fit,
    load_npz,
    mixed_context,
    new_gate,
    normalized_features,
    observable_features,
    read_json,
    smooth_mae,
    tensor_inputs,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    inputs, forecast, output = (
        p.resolve() for p in (args.input_root, args.forecast_root, args.output_root)
    )
    if (output / "manifest.json").exists() or (output / "initial_state.json").exists():
        raise ValueError("preserve completed or partial gate training; diagnose before restarting")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecast / "manifest.json")
    if prepared["smoke"] != args.smoke or fm["smoke"] != args.smoke:
        raise ValueError("smoke and formal inputs cannot mix")
    if fm["input_root"] != str(inputs) or fm["status"] != "completed":
        raise ValueError("complete the corresponding calibration forecast bank first")
    entries = {r["case_id"]: r for r in fm["cases"]}
    rows = prepared["cases"]
    if set(entries) != {r["case_id"] for r in rows} or (not args.smoke and len(rows) != 288):
        raise ValueError("source calibration population is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        str(p): file_sha256(p)
        for p in (
            inputs / "manifest.json",
            forecast / "manifest.json",
            Path(__file__),
            ROOT / "scripts/forecast_calibration_core.py",
            ROOT / "docs/iclr2027/R31_FORECAST_CALIBRATION_PROTOCOL.md",
        )
    }
    _write_json(output / "initial_state.json", {"identity": identity, "status": "started"})
    torch.set_num_threads(1)
    started = perf_counter()
    data = [load_npz(inputs / r["path"], r["sha256"]) for r in rows]
    labels = [load_npz(inputs / r["label_path"], r["label_sha256"]) for r in rows]
    features = np.stack([d["features"] for d in data])
    mean, scale = features.mean((0, 1)), features.std((0, 1), ddof=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    _save_npz(output / "feature_scaler.npz", mean=mean, scale=scale, values=features)
    batches, bank, truth, methods = [], [], [], None
    for row, d, label in zip(rows, data, labels, strict=True):
        if row["origin"] - 192 < row["prefix_end"] or row["origin"] + 24 > row["calibration_end"]:
            raise ValueError("a source calibration history crosses a split boundary")
        future = (label["future"] - d["mean"][:2]) / d["scale"][:2]
        imputation = np.where(
            label["hidden"], (label["original_history"] - d["mean"][:2]) / d["scale"][:2], np.nan
        )
        batches.append(
            {
                "inputs": tensor_inputs(d),
                "features": normalized_features(d["features"], mean, scale),
                "future": torch.tensor(future, dtype=torch.float32, device="cuda"),
                "imputation": torch.tensor(imputation, dtype=torch.float32, device="cuda"),
            }
        )
        entry = entries[row["case_id"]]
        f = load_npz(forecast / entry["path"], entry["sha256"])
        names = [n for n in f["methods"].tolist() if "_source_" not in n]
        if len(names) != 25 or (methods is not None and names != methods):
            raise ValueError("the 25 registered base outputs changed")
        methods = names
        bank.append(f["points"][[f["methods"].tolist().index(n) for n in names]])
        truth.append(future)
    bank, truth = np.stack(bank), np.stack(truth)
    fixed = {"methods": methods, "global": [], "stations": {}}
    for target in (0, 1):
        fixed["global"].append(fixed_mae_fit(bank[:, :, :, target], truth[:, :, target]))
    for station in sorted({r["station"] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r["station"] == station]
        fixed["stations"][station] = [
            fixed_mae_fit(bank[selected, :, :, t], truth[selected, :, t]) for t in (0, 1)
        ]
    _write_json(output / "fixed_output.json", fixed)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    backbone.eval().requires_grad_(False)
    epochs = 1 if args.smoke else 3
    random = np.random.default_rng(SEED)
    order = np.stack([random.permutation(len(rows)) for _ in range(epochs)])
    _save_npz(
        output / "source_order.npz",
        indices=order,
        case_ids=np.asarray([r["case_id"] for r in rows]),
    )
    learned, trace = [], []
    total_steps = len(rows) * epochs
    torch.cuda.reset_peak_memory_stats()
    for name in LEARNED:
        gate = new_gate(name)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=0.01, weight_decay=0.001)
        model_started = perf_counter()
        for step, index in enumerate(order.ravel()):
            batch = batches[index]
            optimizer.zero_grad(set_to_none=True)
            if step == total_steps - 1:
                before = {
                    "model": copy.deepcopy(gate.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "case_id": rows[index]["case_id"],
                    "step": step,
                }
                torch.save(before, output / f"{name}-before-last.pt")
            alpha = gate(batch["features"])
            context = mixed_context(batch["inputs"], alpha)
            prediction = (
                context[:, :2]
                if name == "imputation_gate"
                else chronos_median(pipeline, context, 24, [0, 1])
            )
            objective = batch["imputation"] if name == "imputation_gate" else batch["future"]
            loss = smooth_mae(prediction, objective)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0, error_if_nonfinite=True)
            if not bool(torch.isfinite(loss)) or any(p.grad is None for p in gate.parameters()):
                raise ValueError("gate optimization did not produce finite, complete gradients")
            optimizer.step()
            trace.append(
                {
                    "method": name,
                    "step": step,
                    "case_id": rows[index]["case_id"],
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "alpha": alpha.detach().cpu().tolist(),
                }
            )
            if (step + 1) % 48 == 0 or step + 1 == total_steps:
                _write_json(
                    output / "progress.json",
                    {"method": name, "steps": step + 1, "steps_per_method": total_steps},
                )
                print(
                    json.dumps({"method": name, "step": step + 1, "total": total_steps}), flush=True
                )
        checkpoint = output / f"{name}.pt"
        torch.save({"model": gate.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        learned.append(
            {
                "method": name,
                "path": checkpoint.name,
                "sha256": file_sha256(checkpoint),
                "before_last_sha256": file_sha256(output / f"{name}-before-last.pt"),
                "parameter_count": sum(p.numel() for p in gate.parameters()),
                "steps": total_steps,
                "seconds": perf_counter() - model_started,
            }
        )
    _write_json(output / "training_trace.json", trace)
    if parameter_digest(backbone) != digest or any(
        p.grad is not None for p in backbone.parameters()
    ):
        raise ValueError("the forecasting backbone was not frozen")
    # These features use only arrived evaluation histories and the original full prefix.
    eval_rows = evaluation_rows()
    if args.smoke:
        selected_stations = {r["station"] for r in rows}
        eval_rows = [r for r in eval_rows if r["station"] in selected_stations][:4]
    regressions = evaluation_sources()
    evaluation = []
    for row in eval_rows:
        d = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        values = observable_features(d, regressions[row["station"]])
        path = output / "evaluation-features" / f"{row['case_id']}.npz"
        _save_npz(path, features=values)
        evaluation.append(
            {
                **row,
                "feature_path": str(path.relative_to(output)),
                "feature_sha256": file_sha256(path),
            }
        )
    regression_files = []
    for station, fit in regressions.items():
        path = output / "evaluation-regressions" / f"{station}.json"
        fit.save(path)
        regression_files.append(
            {"station": station, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "identity": identity,
            "input_root": str(inputs),
            "source_forecast_root": str(forecast),
            "parameter_sha256": digest,
            "epochs": epochs,
            "seed": SEED,
            "source_cases": len(rows),
            "learned": learned,
            "evaluation": evaluation,
            "evaluation_regressions": regression_files,
            "files": {
                p.name: file_sha256(p)
                for p in (
                    output / "feature_scaler.npz",
                    output / "source_order.npz",
                    output / "fixed_output.json",
                    output / "training_trace.json",
                )
            },
            "evaluation_future_values_read": False,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
