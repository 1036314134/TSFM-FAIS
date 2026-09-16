"""Freeze public-runtime repaired forecasts and fixed baselines before reading evaluation targets."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair
from patch_repair_eval_support import (
    TracedRepair,
    fixed_controls,
    pool_catalog,
    read_input,
    source_bank,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def summary_tables(frame):
    common = ["panel", "context_length", "model_id", "method"]
    result, table = {}, frame
    for name, keys in (
        ("histories", ["group_id", "family_id", "dataset_id", "item_id", "origin_id"]),
        ("series", ["group_id", "family_id", "dataset_id", "item_id"]),
        ("datasets", ["group_id", "family_id", "dataset_id"]),
        ("groups", ["group_id"]),
        ("summary", []),
    ):
        table = table.groupby(common + keys)[["mae", "mse"]].mean().reset_index()
        result[name] = table
    return result


def smoke_cases(cases):
    first = next(r for r in cases if r["panel"] == "native_development")
    return [cases[0], *[r for r in cases if r.get("native_case_id") == first["native_case_id"]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--readout-only", action="store_true")
    args = parser.parse_args()
    output, training_root = args.output_root.resolve(), args.training_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair evaluation")
    input_root = ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001"
    prepared, trained = (
        read_json(input_root / "manifest.json"),
        read_json(training_root / "manifest.json"),
    )
    if (
        prepared["status"] != "completed"
        or trained["status"] != "completed"
        or len(trained["models"]) != 4
    ):
        raise ValueError("finish all registered inputs and repair models first")
    if bool(trained["identity"]["smoke_only"]) != bool(args.smoke):
        raise ValueError("smoke and full model populations must remain separate")
    identity = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            input_root / "manifest.json",
            input_root / "matched_controls.json",
            training_root / "manifest.json",
            ROOT / "scripts/learned_patch_repair.py",
            ROOT / "scripts/patch_repair_eval_support.py",
            ROOT / "docs/iclr2027/R24_EXECUTION_PROTOCOL.md",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial repair evaluation definitions changed")
    _write_json(output / "identity.json", identity)
    cases = smoke_cases(prepared["cases"]) if args.smoke else prepared["cases"]
    metadata = {r["case_id"]: r for r in cases}
    matched = read_json(input_root / "matched_controls.json")["models"]
    old = read_json(ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json")[
        "models"
    ]
    native_root = ROOT / "artifacts/iclr27-r21/provenance-forecasts-v001"
    native = {
        (r["model_id"], r["case_id"]): r for r in read_json(native_root / "manifest.json")["cases"]
    }
    long_root = ROOT / "artifacts/iclr27-r23/conditioning-forecasts-v001"
    longer = {
        (r["model_id"], r["case_id"]): r for r in read_json(long_root / "manifest.json")["cases"]
    }
    freeze_path = output / "predictions_frozen.json"
    if args.readout_only and not freeze_path.exists():
        raise ValueError("complete and freeze forecasts before the readout-only stage")
    if not args.readout_only:
        if freeze_path.exists():
            raise ValueError("preserve the completed prediction freeze")
        records, costs = [], []
        torch.set_num_threads(1)
        for model_id in ("chronos2", "timesfm2p5"):
            started = perf_counter()
            runner, adapter, backbone, digest, joint = make_forecaster(
                model_id,
                ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
                ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
            )
            modules, module_records = {}, {}
            for condition in ("pattern", "fraction"):
                record = next(
                    r
                    for r in trained["models"]
                    if r["model_id"] == model_id and r["condition"] == condition
                )
                path = training_root / record["checkpoint_path"]
                if file_sha256(path) != record["checkpoint_sha256"]:
                    raise ValueError("a trained repair checkpoint changed")
                repair = PatchRepair(record["width"], record["patch_size"], record["rank"]).to(
                    "cuda"
                )
                repair.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                modules[condition] = repair.eval().requires_grad_(False)
                module_records[condition] = record
            catalog = pool_catalog(model_id)
            counter = {"forward_batches": 0}

            def before(module, positional, keyword, counter=counter):
                counter["forward_batches"] += 1

            handle = backbone.register_forward_pre_hook(before, with_kwargs=True)
            maximum_base_difference, calls = 0.0, 0
            for row in cases:
                path = input_root / row["path"]
                if file_sha256(path) != row["sha256"]:
                    raise ValueError("a prepared evaluation context changed")
                data = read_input(path)
                raw, observed = data["base"], np.isfinite(data["context"])
                mean, scale = data["mean"][:2], data["scale"][:2]
                spec = ForecastSpec(
                    model_id,
                    "joint_multivariate" if joint else "independent_univariate",
                    96,
                    context_length=row["context_length"],
                    target_indices=[0, 1],
                )
                base_point = (runner.predict_missing(raw[None], spec).point[0] - mean) / scale
                calls += 1
                traces, energy = [], {}
                points = {"base_seasonal": base_point}
                for condition, repair in modules.items():
                    with TracedRepair(
                        backbone, model_id, observed, repair, condition, trace=row["trace"]
                    ) as trace:
                        point = runner.predict_missing(raw[None], spec).point[0]
                    points[condition + "_repair"] = (point - mean) / scale
                    calls += 1
                    energy[condition] = {
                        "relative_energy": float(trace.repair.penalty),
                        "modified_token_instances": trace.repair.changed_tokens,
                    }
                    if row["trace"]:
                        target = output / "traces" / model_id / condition / f"{row['case_id']}.npz"
                        arrays = {
                            f"call_{i}_{name}": record[name]
                            for i, record in enumerate(trace.records)
                            for name in ("before", "after")
                        }
                        _save_npz(target, observed=observed, **arrays)
                        traces.append(
                            {
                                "condition": condition,
                                "path": str(target.relative_to(output)),
                                "sha256": file_sha256(target),
                                "calls": len(trace.records),
                            }
                        )
                if row["panel"] == "source_validation":
                    bank, actions = source_bank(row, model_id, catalog)
                    points.update(fixed_controls(bank, actions, old[model_id], matched[model_id]))
                elif row["context_length"] == 96:
                    previous = native[(model_id, row["native_case_id"])]
                    if file_sha256(native_root / previous["path"]) != previous["sha256"]:
                        raise ValueError("the old native forecast bank changed")
                    with np.load(native_root / previous["path"], allow_pickle=False) as saved:
                        bank, actions = saved["bank"][0], saved["actions"].tolist()
                    points.update(fixed_controls(bank, actions, old[model_id], matched[model_id]))
                else:
                    previous = longer[(model_id, row["native_case_id"])]
                    if file_sha256(long_root / previous["path"]) != previous["sha256"]:
                        raise ValueError("the same-information original forecasts changed")
                    with np.load(long_root / previous["path"], allow_pickle=False) as saved:
                        points.update(
                            native192=saved["native192"], budget_native192=saved["budget_native192"]
                        )
                if row["context_length"] == 96:
                    difference = float(abs(base_point - points["seasonal_lag"]).max())
                    maximum_base_difference = max(maximum_base_difference, difference)
                    if difference > 1e-5:
                        raise ValueError(
                            f"old seasonal forecast replay exceeded its numeric bound: {difference}"
                        )
                if len(points) != (19 if row["context_length"] == 96 else 5):
                    raise ValueError("the fixed method population changed")
                target = output / "predictions" / model_id / f"{row['case_id']}.npz"
                _save_npz(
                    target, methods=np.asarray(list(points)), points=np.stack(list(points.values()))
                )
                records.append(
                    {
                        "model_id": model_id,
                        "case_id": row["case_id"],
                        "path": str(target.relative_to(output)),
                        "sha256": file_sha256(target),
                        "traces": traces,
                        "repair_energy": energy,
                    }
                )
                if len(records) % 100 == 0:
                    print(f"frozen evaluation pairs: {len(records)}", flush=True)
            if parameter_digest(backbone) != digest:
                raise ValueError("the frozen backbone changed during evaluation")
            handle.remove()
            costs.append(
                {
                    "model_id": model_id,
                    "public_forecast_context_calls": calls,
                    "real_target_sequences": calls * 2,
                    "joint_input_groups": calls if joint else 0,
                    "forward_batches": counter["forward_batches"],
                    "maximum_old_base_difference": maximum_base_difference,
                    "wall_seconds": perf_counter() - started,
                    "backbone_sha256": digest,
                }
            )
            del runner, adapter, backbone, modules, repair
            gc.collect()
            torch.cuda.empty_cache()
        _write_json(
            freeze_path,
            {
                "status": "completed",
                "identity": identity,
                "predictions": records,
                "costs": costs,
                "evaluation_future_labels_read": False,
            },
        )
        return
    frozen = read_json(freeze_path)
    if frozen["identity"] != identity:
        raise ValueError("prediction and readout identities differ")
    scores, targets = [], []
    for entry in frozen["predictions"]:
        row = metadata[entry["case_id"]]
        if (
            file_sha256(output / entry["path"]) != entry["sha256"]
            or file_sha256(Path(row["original_path"])) != row["original_sha256"]
        ):
            raise ValueError("a frozen prediction or evaluation target file changed")
        data = read_input(input_root / row["path"])
        with np.load(row["original_path"], allow_pickle=False) as original:
            truth = original["future"][:96, :2]
            observed = np.isfinite(truth)
            if "future_observed" in original.files:
                np.testing.assert_array_equal(observed, original["future_observed"][:96, :2])
        count = observed.sum(0)
        if (count < 48).any():
            raise ValueError("original evaluation future observations are insufficient")
        truth_z = (truth - data["mean"][:2]) / data["scale"][:2]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            error = np.where(observed[None], saved["points"] - truth_z[None], 0.0)
            mae, mse = abs(error).sum(1) / count, (error**2).sum(1) / count
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
                    model_id=entry["model_id"],
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
                            "observed_count": int(count[slot]),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if not args.smoke and (len(frame) != 33936 or len(targets) != 67872):
        raise ValueError("the registered full scoring population changed")
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
            "costs": frozen["costs"],
            "score_rows": len(frame),
            "target_score_rows": len(targets),
            "primary": "pattern_repair",
            "primary_panel": "native_development",
            "primary_context_length": 192,
            "primary_metric": "mae",
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
