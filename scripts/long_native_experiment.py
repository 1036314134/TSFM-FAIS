"""Native Chronos baselines using the frozen model's default available context."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from group_scope_experiment import grouped_forward
from long_native_core import CONTEXT_LIMIT, extend_visible_history, model_inputs, standardized_point
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R45_NATIVE_CONTEXT_PROTOCOL.md"
INPUTS = ROOT / "artifacts/iclr27-r40/attention-inputs-v001"
PARENT = ROOT / "artifacts/iclr27-r44/scope-forecasts-v001"


def model_config_path():
    bundle = read_json(ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001/manifest.json")
    return Path(bundle["identity"]["forecaster_artifacts"]["chronos2"]) / "config.json"


def source_banks():
    records, peers = sources()
    beijing = {name: augmented(name, records, peers)[0] for name in records}
    plan = read_json(ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json")
    values = load_npz(plan["development_path"], plan["development_sha256"])["values"]
    if len(values) != 672:
        raise ValueError("HDB long-context input must use only the development period")
    models = read_json(ROOT / "artifacts/iclr27-r39/hdb-inputs-v001/manifest.json")["models"]
    hdb = {entry["station"]: values[:, entry["columns"]] for entry in models}
    return {"beijing": beijing, "hdb": hdb}


def prepare(base, smoke):
    output = base / "long-native-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long native inputs")
    config_path = model_config_path()
    if read_json(config_path)["chronos_config"]["context_length"] != CONTEXT_LIMIT:
        raise ValueError("the frozen default context limit changed")
    banks = source_banks()
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    rows = read_json(INPUTS / "manifest.json")["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(
                ROOT / "artifacts/iclr27-r44/smoke-v001/scope-inputs-v001/manifest.json"
            )["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    entries = []
    for row in rows:
        source = load_npz(row["source_input_path"], row["source_input_sha256"])
        short = load_npz(INPUTS / row["path"], row["sha256"])
        history, start = extend_visible_history(
            banks[row["dataset"]][row["station"]], row["origin"], source["context"]
        )
        definitions = model_inputs(history, short, row["target_count"])
        if any(np.isinf(context).any() for context in definitions.values()):
            raise ValueError("an infinite source value cannot enter the native model")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            **definitions,
            mean=short["mean"],
            scale=short["scale"],
            selected=short["selected"],
        )
        parent = parents[row["case_id"]]
        entries.append(
            {
                **row,
                "source_attention_path": str(INPUTS / row["path"]),
                "source_attention_sha256": row["sha256"],
                "parent_path": str(PARENT / parent["path"]),
                "parent_sha256": parent["sha256"],
                "history_start": start,
                "actual_context_length": len(history),
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "context_limit": CONTEXT_LIMIT,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/long_native_core.py",
                    config_path,
                    INPUTS / "manifest.json",
                    PARENT / "manifest.json",
                    ROOT / "artifacts/iclr27-r44/scope-audit-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r27/peer-information-preflight-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-inputs-v001/manifest.json",
                )
            },
            "heldout_value_analysis": False,
        },
    )


def query_names():
    return (
        "native_long_raw_peer",
        "native_long_raw_targets",
        "native_long_prefix_peer",
        "native_long_prefix_targets",
        "native_short_restoration",
    )


def forecast(base):
    inputs, output = base / "long-native-inputs-v001", base / "long-native-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed native-context predictions")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    if pipeline.model_context_length != CONTEXT_LIMIT:
        raise ValueError("the loaded backbone does not match the default-context registration")
    mid, entries, calls = pipeline.quantiles.index(0.5), [], 0
    for number, row in enumerate(prepared["cases"]):
        data = load_npz(inputs / row["path"], row["sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        requests = []
        for name in query_names():
            context = data[name]
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            raw, fields = grouped_forward(
                backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), row["horizon"]
            )
            elapsed = perf_counter() - started
            if fields["context_fields"].shape[1] != int(
                np.ceil(context.shape[1] / pipeline.model_output_patch_size)
            ):
                raise ValueError("the native history was implicitly truncated")
            point = standardized_point(
                raw, name, row["target_count"], row["horizon"], mid, data["mean"], data["scale"]
            )
            if name == "native_short_restoration":
                np.testing.assert_array_equal(point, points[row["native_name"]])
            else:
                if name in points or "half_var_" + name in points:
                    raise ValueError("a native-context control would replace a frozen output")
                points[name] = point
                points["half_var_" + name] = 0.5 * point + 0.5 * points["linear_var_direct"]
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            _save_npz(path, context_z=context, quantiles=raw, **fields)
            requests.append(
                {
                    "name": name,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "input_units": "original" if "_raw_" in name else "prefix_standardized",
                    "input_length": context.shape[1],
                    "input_rows": len(context),
                    "wall_seconds": elapsed,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            calls += 1
        names = sorted(points)
        if len(names) != (238 if row["dataset"] == "beijing" else 111):
            raise ValueError("the registered long-native outputs are incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([points[n] for n in names]),
            queries=np.asarray(json.dumps(requests)),
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
                        "history_start",
                        "actual_context_length",
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 10 == 0:
            _write_json(output / "progress.json", {"cases": number + 1, "queries": calls})
            print(json.dumps({"cases": number + 1, "queries": calls}), flush=True)
    if parameter_digest(backbone) != digest or calls != 5 * len(entries):
        raise ValueError("model identity or native-context query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "model_calls": calls,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "context_limit": pipeline.model_context_length,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "long-native-inputs-v001",
        base / "long-native-forecasts-v001",
        base / "long-native-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-native scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all native-context predictions before scoring")
    records, hdb = truth_sources()
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        truth = case_truth(row, records, hdb)
        valid = np.isfinite(truth)
        if (valid.sum(0) < row["horizon"] // 2).any():
            raise ValueError("registered outcome support changed")
        expected = (truth - data["mean"][: row["target_count"]]) / data["scale"][
            : row["target_count"]
        ]
        errors = np.where(valid[None], saved["points"] - expected[None], 0)
        mae, mse = abs(errors).sum(1) / valid.sum(0), np.square(errors).sum(1) / valid.sum(0)
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
    if len(rows) != 64191 or len(targets) != 119169:
        raise ValueError("registered long-native score counts changed")
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
            "diagnostic": True,
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "independent_confirmation": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "forecast", "evaluate", "audit"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "prepare":
        prepare(base, args.smoke)
    elif args.phase == "forecast":
        forecast(base)
    elif args.phase == "evaluate":
        evaluate(base)
    else:
        from audit_long_native import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
