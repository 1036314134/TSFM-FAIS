"""Matched natural-history and corruption-augmented LoRA capacity experiment."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from forecast_lora_core import (
    MODES,
    STEPS_PER_SOURCE,
    ForecastLoRA,
    optimizer_for,
    predict_tensor,
    update_once,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R42_ADAPTATION_CAPACITY_PROTOCOL.md"
SOURCE = ROOT / "artifacts/iclr27-r36/covariance-inputs-v001"
PARENT = ROOT / "artifacts/iclr27-r40"


def canonical(raw, mean, scale, keep):
    return np.array(((raw[:, keep] - mean[keep]) / scale[keep]).T, dtype=np.float32, order="C")


def source_examples():
    for row in read_json(SOURCE / "manifest.json")["source_cases"]:
        d = load_npz(SOURCE / row["path"], row["sha256"])
        labels = load_npz(SOURCE / row["label_path"], row["label_sha256"])
        z = (d["context"] - d["mean"]) / d["scale"]
        clean = np.where(labels["artificial_mask"], labels["hidden_all"], z)
        natural = np.array(clean[:, d["keep"]].T, dtype=np.float32, order="C")
        corrupted = np.array(z[:, d["keep"]].T, dtype=np.float32, order="C")
        if row["origin"] + 24 > 7012 or row["origin"] - 192 < 3506:
            raise ValueError("Beijing training source crossed its registered boundaries")
        yield (
            {
                "case_id": "beijing-" + row["case_id"],
                "dataset": "beijing",
                "station": row["station"],
                "origin": row["origin"],
                "outage_pattern": row["outage_pattern"],
                "outage_age": row["outage_age"],
                "label_end": row["origin"] + 24,
                "fitting_end": 3506,
                "source_path": str(SOURCE / row["path"]),
                "source_sha256": row["sha256"],
                "label_path": str(SOURCE / row["label_path"]),
                "label_sha256": row["label_sha256"],
            },
            {"natural": natural, "corrupted": corrupted, "future": labels["future"].copy()},
        )
    plan_path = ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json"
    plan = read_json(plan_path)
    prefix = load_npz(plan["development_path"], plan["development_sha256"])["values"][:336].copy()
    prepared = read_json(ROOT / "artifacts/iclr27-r39/hdb-inputs-v001/manifest.json")
    for entry in prepared["models"]:
        values = prefix[:, entry["columns"]]
        model_path = ROOT / "artifacts/iclr27-r39/hdb-inputs-v001" / entry["path"]
        model = load_npz(model_path, entry["sha256"])
        for t in (192, 216, 240, 264, 288, 312):
            natural_raw, truth = values[t - 192 : t].copy(), values[t : t + 24, :1]
            if np.isfinite(natural_raw[:, 0]).sum() < 96 or np.isfinite(truth).sum() < 12:
                continue
            keep = np.isfinite(natural_raw).sum(0) >= 2
            keep[0] = True
            natural = canonical(natural_raw, model["mean"], model["scale"], keep)
            for age in (6, 24, 54):
                for pattern, columns in (
                    ("targets_only", [0]),
                    ("target_first3", [0, 1, 2, 3]),
                    ("all_columns", list(range(7))),
                ):
                    raw = natural_raw.copy()
                    raw[-age:, columns] = np.nan
                    key = f"r42|{entry['station']}|{t}|{pattern}|{age}"
                    yield (
                        {
                            "case_id": "hdb-" + hashlib.sha256(key.encode()).hexdigest()[:20],
                            "dataset": "hdb",
                            "station": entry["station"],
                            "origin": t,
                            "outage_pattern": pattern,
                            "outage_age": age,
                            "label_end": t + 24,
                            "fitting_end": 336,
                            "source_columns": entry["columns"],
                            "model_path": str(model_path),
                            "model_sha256": entry["sha256"],
                            "development_sha256": plan["development_sha256"],
                        },
                        {
                            "natural": natural.copy(),
                            "corrupted": canonical(raw, model["mean"], model["scale"], keep),
                            "future": (truth - model["mean"][:1]) / model["scale"][:1],
                        },
                    )


def source_order(rows):
    random = np.random.default_rng(7403)
    groups = {}
    for dataset in ("beijing", "hdb"):
        ids = [r["case_id"] for r in rows if r["dataset"] == dataset]
        if not ids or (dataset == "beijing" and len(ids) != 864):
            raise ValueError("registered adaptation source pool is unavailable")
        permutation = random.permutation(len(ids))
        groups[dataset] = [ids[permutation[i % len(ids)]] for i in range(STEPS_PER_SOURCE)]
    return [name for pair in zip(groups["beijing"], groups["hdb"], strict=True) for name in pair]


def prepare(base, smoke):
    output = base / "adaptation-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed adaptation inputs")
    rows = []
    for metadata, data in source_examples():
        if (
            not np.isfinite(data["future"]).any()
            or data["natural"].shape != data["corrupted"].shape
        ):
            raise ValueError("invalid source input/target shape")
        retained = np.isfinite(data["corrupted"])
        np.testing.assert_array_equal(data["natural"][retained], data["corrupted"][retained])
        path = output / "source" / f"{metadata['case_id']}.npz"
        _save_npz(path, **data)
        rows.append(
            {**metadata, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        )
    order = source_order(rows)
    if smoke:
        order = order[:12]
    inputs = read_json(PARENT / "attention-inputs-v001/manifest.json")
    forecasts = {
        r["case_id"]: r
        for r in read_json(PARENT / "attention-forecasts-v001/manifest.json")["cases"]
    }
    ids = {
        r["case_id"]
        for r in read_json(PARENT / "smoke-v002/attention-inputs-v001/manifest.json")["cases"]
    }
    evaluation = []
    for row in inputs["cases"]:
        if smoke and row["case_id"] not in ids:
            continue
        old = forecasts[row["case_id"]]
        evaluation.append(
            {
                **row,
                "input_path": str(PARENT / "attention-inputs-v001" / row["path"]),
                "input_sha256": row["sha256"],
                "parent_path": str(PARENT / "attention-forecasts-v001" / old["path"]),
                "parent_sha256": old["sha256"],
            }
        )
    counts = {name: sum(r["dataset"] == name for r in rows) for name in ("beijing", "hdb")}
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "source_cases": rows,
            "training_order": order,
            "source_pool_counts": counts,
            "training_updates": len(order),
            "unique_training_cases": len(set(order)),
            "replayed_training_cases": len(order) - len(set(order)),
            "evaluation_cases": evaluation,
            "identity": {
                str(path): file_sha256(path)
                for path in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/forecast_lora_core.py",
                    SOURCE / "manifest.json",
                    ROOT / "artifacts/iclr27-r36/covariance-audit-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json",
                    PARENT / "attention-inputs-v001/manifest.json",
                    PARENT / "attention-forecasts-v001/manifest.json",
                    PARENT / "attention-audit-v001/manifest.json",
                )
            },
            "heldout_value_analysis": False,
        },
    )
    print(
        json.dumps(
            {
                "source_pool_counts": counts,
                "training_updates": len(order),
                "evaluation_cases": len(evaluation),
            }
        ),
        flush=True,
    )


def backbone_and_pipeline():
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    backbone.eval().requires_grad_(False)
    return backbone, adapter._ensure_backend(), digest


def save_adapter(path, bank):
    _save_npz(
        path,
        **{key: value.detach().cpu().numpy().copy() for key, value in bank.state_dict().items()},
    )


def load_adapter(path, sha, bank):
    state = load_npz(path, sha)
    bank.load_state_dict(
        {
            key: torch.tensor(value, device=next(bank.parameters()).device)
            for key, value in state.items()
        },
        strict=True,
    )


def train(base, mode):
    inputs, output = base / "adaptation-inputs-v001", base / "adaptation-training-v001" / mode
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed LoRA fitting")
    prepared = read_json(inputs / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["source_cases"]}
    examples = {
        name: load_npz(inputs / metadata[name]["path"], metadata[name]["sha256"])
        for name in set(prepared["training_order"])
    }
    backbone, pipeline, digest = backbone_and_pipeline()
    bank = ForecastLoRA(backbone).to(device=backbone.device, dtype=backbone.dtype)
    optimizer = optimizer_for(bank)
    output.mkdir(parents=True, exist_ok=True)
    save_adapter(output / "initial.npz", bank)
    losses, started = [], perf_counter()
    with bank.installed():
        for index, name in enumerate(prepared["training_order"]):
            if index == len(prepared["training_order"]) - 1:
                torch.save(
                    {
                        "adapter": {
                            key: value.detach().cpu().clone()
                            for key, value in bank.state_dict().items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "case_id": name,
                        "step": index,
                    },
                    output / "before-final-update.pt",
                )
            loss, norm = update_once(bank, optimizer, backbone, pipeline, examples[name], mode)
            losses.append(
                {
                    "step": index + 1,
                    "case_id": name,
                    "dataset": metadata[name]["dataset"],
                    "loss": loss,
                    "gradient_norm": norm,
                }
            )
            if (index + 1) % 100 == 0:
                _write_json(
                    output / "progress.json",
                    {
                        "mode": mode,
                        "step": index + 1,
                        "total": len(prepared["training_order"]),
                        "elapsed_seconds": perf_counter() - started,
                    },
                )
                print(
                    json.dumps(
                        {"mode": mode, "step": index + 1, "total": len(prepared["training_order"])}
                    ),
                    flush=True,
                )
    if parameter_digest(backbone) != digest:
        raise ValueError("LoRA fitting changed original model parameters")
    save_adapter(output / "final.npz", bank)
    pd.DataFrame(losses).to_csv(output / "training.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "mode": mode,
            "smoke": prepared["smoke"],
            "updates": len(losses),
            "projection_names": bank.projection_names,
            "trainable_parameters": sum(p.numel() for p in bank.parameters()),
            "parameter_sha256": digest,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "initial_sha256": file_sha256(output / "initial.npz"),
            "final_sha256": file_sha256(output / "final.npz"),
            "before_final_sha256": file_sha256(output / "before-final-update.pt"),
            "training_log_sha256": file_sha256(output / "training.csv"),
            "wall_seconds": perf_counter() - started,
        },
    )


def forecast(base):
    inputs, output = base / "adaptation-inputs-v001", base / "adaptation-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed adapter evaluation forecasts")
    prepared = read_json(inputs / "manifest.json")
    backbone, pipeline, digest = backbone_and_pipeline()
    banks, training = {}, {}
    for mode in MODES:
        path = base / "adaptation-training-v001" / mode
        manifest = read_json(path / "manifest.json")
        if (
            manifest["input_sha256"] != file_sha256(inputs / "manifest.json")
            or manifest["parameter_sha256"] != digest
        ):
            raise ValueError("trained adapter source or backbone changed")
        bank = ForecastLoRA(backbone).to(device=backbone.device, dtype=backbone.dtype)
        load_adapter(path / "final.npz", manifest["final_sha256"], bank)
        banks[mode] = bank
        training[mode] = file_sha256(path / "manifest.json")
    zero = (
        ForecastLoRA(backbone).to(device=backbone.device, dtype=backbone.dtype)
        if prepared["smoke"]
        else None
    )
    mid, entries, calls, ordinary, neutral = pipeline.quantiles.index(0.5), [], 0, 0, 0
    for number, row in enumerate(prepared["evaluation_cases"]):
        data = load_npz(row["input_path"], row["input_sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        context = torch.tensor(data["native"], device=backbone.device)
        queries = []
        for mode, bank in banks.items():
            with bank.installed(), torch.inference_mode():
                raw = predict_tensor(backbone, pipeline, context, row["horizon"]).cpu().numpy()
            points[mode] = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
            points["half_var_" + mode] = 0.5 * points[mode] + 0.5 * points["linear_var_direct"]
            path = output / "queries" / f"{row['case_id']}-{mode}.npz"
            _save_npz(path, context_z=data["native"], quantiles=raw)
            queries.append(
                {"name": mode, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
            )
            calls += 1
        with torch.inference_mode():
            raw = predict_tensor(backbone, pipeline, context, row["horizon"]).cpu().numpy()
        np.testing.assert_array_equal(
            raw[: row["target_count"], mid, : row["horizon"]].T.astype(float),
            points[row["native_name"]],
        )
        path = output / "queries" / f"{row['case_id']}-ordinary.npz"
        _save_npz(path, context_z=data["native"], quantiles=raw)
        queries.append(
            {"name": "ordinary", "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        )
        ordinary += 1
        if zero is not None:
            with zero.installed(), torch.inference_mode():
                initial = predict_tensor(backbone, pipeline, context, row["horizon"]).cpu().numpy()
            np.testing.assert_array_equal(initial, raw)
            neutral += 1
        names = sorted(points)
        if (
            len(names) != (166 if row["dataset"] == "beijing" else 51)
            or not np.isfinite(np.stack(list(points.values()))).all()
        ):
            raise ValueError("registered adapter evaluation outputs are incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([points[n] for n in names]),
            queries=np.asarray(json.dumps(queries)),
        )
        entries.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "case_id",
                        "dataset",
                        "panel",
                        "station",
                        "origin",
                        "horizon",
                        "target_count",
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 25 == 0:
            _write_json(output / "progress.json", {"cases": number + 1, "adapter_queries": calls})
            print(json.dumps({"cases": number + 1, "adapter_queries": calls}), flush=True)
    if parameter_digest(backbone) != digest:
        raise ValueError("adapter inference changed original backbone parameters")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "adapter_calls": calls,
            "ordinary_calls": ordinary,
            "neutral_checks": neutral,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "training": training,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "adaptation-inputs-v001",
        base / "adaptation-forecasts-v001",
        base / "adaptation-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed adapter scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all formal adapter predictions before scoring")
    records, hdb = truth_sources()
    metadata = {r["case_id"]: r for r in prepared["evaluation_cases"]}
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(row["input_path"], row["input_sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        truth = case_truth(row, records, hdb)
        valid = np.isfinite(truth)
        if (valid.sum(0) < row["horizon"] // 2).any():
            raise ValueError("registered target support changed")
        expected = (truth - data["mean"][: row["target_count"]]) / data["scale"][
            : row["target_count"]
        ]
        error = np.where(valid[None], saved["points"] - expected[None], 0)
        mae, mse = abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "dataset", "panel", "station", "origin", "horizon")}
        for index, method in enumerate(saved["methods"].tolist()):
            rows.append(
                {
                    **info,
                    "method": method,
                    "mae": float(mae[index].mean()),
                    "mse": float(mse[index].mean()),
                }
            )
            for slot in range(row["target_count"]):
                targets.append(
                    {
                        **info,
                        "method": method,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[index, slot]),
                        "mse": float(mse[index, slot]),
                    }
                )
    if len(rows) != 42579 or len(targets) != 80925:
        raise ValueError("registered adapter score counts changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    stations, summary = aggregate(frame)
    stations.to_csv(output / "stations.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    leave = []
    for (panel, excluded), _ in stations.groupby(["panel", "station"]):
        part = (
            stations.loc[(stations.panel == panel) & (stations.station != excluded)]
            .groupby(["panel", "method"])[["mae", "mse"]]
            .mean()
            .reset_index()
        )
        part["omitted_station"] = excluded
        leave.append(part)
    pd.concat(leave, ignore_index=True).to_csv(output / "leave_one_station_out.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": "corruption_lora",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "independent_confirmation": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "train_natural", "train_corruption", "forecast", "evaluate", "audit"),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "prepare":
        prepare(base, args.smoke)
    elif args.phase in ("train_natural", "train_corruption"):
        train(base, args.phase.removeprefix("train_") + "_lora")
    elif args.phase == "forecast":
        forecast(base)
    elif args.phase == "evaluate":
        evaluate(base)
    else:
        from audit_forecast_lora import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
