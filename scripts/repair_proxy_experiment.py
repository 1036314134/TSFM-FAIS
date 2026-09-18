"""Preserve native targets while conditioning on frozen repair-reference series."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources
from repair_proxy_core import POINT_NAMES, make_queries, read_points, validate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R43_REPAIR_PROXY_PROTOCOL.md"
INPUTS = ROOT / "artifacts/iclr27-r40/attention-inputs-v001"
PARENT = ROOT / "artifacts/iclr27-r42/adaptation-forecasts-v001"
POOL = ROOT / "artifacts/iclr27-r30/peer-inputs-v001"


def proxy_input(row, pools):
    d = load_npz(INPUTS / row["path"], row["sha256"])
    source = load_npz(row["source_input_path"], row["source_input_sha256"])
    targets = row["target_count"]
    if row["dataset"] == "beijing":
        entry = pools[row["source_case_id"]]
        raw = load_npz(POOL / entry["path"], entry["sha256"])
        values = np.concatenate(
            [
                raw["candidates"][:, :, :targets],
                raw["stat_targets"],
                source["static_values"][None, :, :targets],
            ],
            axis=0,
        )
        names = raw["actions"].tolist() + raw["stat_names"].tolist() + ["gaussian"]
        knn = names.index("knn_multivariate")
        bindings = {
            "pool_path": str(POOL / entry["path"]),
            "pool_sha256": entry["sha256"],
            "upstream_statuses": entry.get("statuses", {}),
        }
    else:
        values = source["fills"][:, :, :targets]
        names = source["fill_names"].tolist()
        knn, bindings = names.index("knn"), {}
    normalized = np.asarray(
        (values - d["mean"][None, None, :targets]) / d["scale"][None, None, :targets],
        dtype=np.float32,
    )
    data = {
        "native": d["native"],
        "pool": normalized,
        "pool_names": np.asarray(names),
        "gaussian_index": np.asarray(names.index("gaussian")),
        "knn_index": np.asarray(knn),
        "mean": d["mean"],
        "scale": d["scale"],
    }
    fallback = validate(data)
    if len(names) != (12 if row["dataset"] == "beijing" else 6):
        raise ValueError("the frozen repair pool membership changed")
    return data, bindings, fallback


def prepare(base, smoke):
    output = base / "proxy-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair-proxy inputs")
    pools = {r["case_id"]: r for r in read_json(POOL / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    smoke_ids = {
        r["case_id"]
        for r in read_json(
            ROOT / "artifacts/iclr27-r40/smoke-v002/attention-inputs-v001/manifest.json"
        )["cases"]
    }
    candidates = []
    for row in read_json(INPUTS / "manifest.json")["cases"]:
        data, bindings, fallback = proxy_input(row, pools)
        parent = parents[row["case_id"]]
        candidates.append(
            (
                {
                    **row,
                    **bindings,
                    "source_attention_path": str(INPUTS / row["path"]),
                    "source_attention_sha256": row["sha256"],
                    "parent_path": str(PARENT / parent["path"]),
                    "parent_sha256": parent["sha256"],
                    "complete_target_fallback": fallback,
                },
                data,
            )
        )
    if smoke:
        complete = [r for r, _ in candidates if r["complete_target_fallback"]]
        if complete:
            smoke_ids.add(complete[0]["case_id"])
        candidates = [(r, d) for r, d in candidates if r["case_id"] in smoke_ids]
    entries = []
    for row, data in candidates:
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(path, **data)
        entries.append(
            {
                **row,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "pool_size": len(data["pool"]),
                "native_rows": len(data["native"]),
                "maximum_query_rows": len(data["native"]) + len(data["pool"]) * row["target_count"],
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "fallback_cases": sum(r["complete_target_fallback"] for r in entries),
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/repair_proxy_core.py",
                    INPUTS / "manifest.json",
                    PARENT / "manifest.json",
                    ROOT / "artifacts/iclr27-r42/adaptation-audit-v001/manifest.json",
                    POOL / "manifest.json",
                    ROOT / "artifacts/iclr27-r35/dynamic-inputs-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-inputs-v001/manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def captured_forward(backbone, pipeline, context, horizon, anchor_rows):
    captured, embeddings, norms = {}, [], []

    def before_embedding(_module, args):
        embeddings.append(args[0][:anchor_rows].detach().cpu().numpy().copy())

    def after_norm(_module, _args, result):
        norms.append(tuple(v[:anchor_rows].detach().cpu().numpy().copy() for v in result[1]))

    handles = [
        backbone.input_patch_embedding.register_forward_pre_hook(before_embedding),
        backbone.instance_norm.register_forward_hook(after_norm),
    ]
    try:
        with torch.inference_mode():
            raw = (
                backbone(
                    context=torch.tensor(context, device="cuda"),
                    group_ids=torch.zeros(len(context), device="cuda", dtype=torch.long),
                    num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
                )
                .quantile_preds.float()
                .cpu()
                .numpy()
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(embeddings) != 2 or len(norms) != 1 or not np.isfinite(raw).all():
        raise ValueError("unexpected proxy forward interface or nonfinite prediction")
    if anchor_rows:
        captured.update(
            anchor_context_fields=embeddings[0],
            anchor_future_fields=embeddings[1],
            anchor_loc=norms[0][0],
            anchor_scale=norms[0][1],
        )
    return raw, captured


def forecast(base):
    inputs, output = base / "proxy-inputs-v001", base / "proxy-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair-proxy forecasts")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, entries, calls, ordinary = pipeline.quantiles.index(0.5), [], 0, 0
    for number, row in enumerate(prepared["cases"]):
        data = load_npz(inputs / row["path"], row["sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        raw_queries, requests, anchor_records = {}, [], []
        for name, context in make_queries(data).items():
            anchor_rows = 0 if name == "proxy_pool_only" else len(data["native"])
            raw, captures = captured_forward(
                backbone, pipeline, context, row["horizon"], anchor_rows
            )
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            _save_npz(path, context_z=context, quantiles=raw, **captures)
            requests.append(
                {
                    "name": name,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "input_rows": len(context),
                }
            )
            raw_queries[name] = raw
            if anchor_rows:
                anchor_records.append(captures)
            calls += 1
        raw, captures = captured_forward(
            backbone, pipeline, data["native"], row["horizon"], len(data["native"])
        )
        np.testing.assert_array_equal(
            raw[: row["target_count"], mid, : row["horizon"]].T.astype(float),
            points[row["native_name"]],
        )
        for entry in anchor_records:
            for field in captures:
                np.testing.assert_array_equal(entry[field], captures[field])
        path = output / "queries" / f"{row['case_id']}-ordinary.npz"
        _save_npz(path, context_z=data["native"], quantiles=raw, **captures)
        requests.append(
            {
                "name": "ordinary",
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "input_rows": len(data["native"]),
            }
        )
        ordinary += 1
        if row["complete_target_fallback"]:
            additions = {name: points[row["native_name"]].copy() for name in POINT_NAMES}
        else:
            additions = read_points(data, raw_queries, row["horizon"], mid)
        for name, point in additions.items():
            if name in points or "half_var_" + name in points:
                raise ValueError("a proxy output would replace an existing control")
            points[name] = point
            points["half_var_" + name] = 0.5 * point + 0.5 * points["linear_var_direct"]
        names = sorted(points)
        if len(names) != (186 if row["dataset"] == "beijing" else 71):
            raise ValueError("the registered proxy output set is incomplete")
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
                        "complete_target_fallback",
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 25 == 0:
            _write_json(output / "progress.json", {"cases": number + 1, "proxy_queries": calls})
            print(json.dumps({"cases": number + 1, "proxy_queries": calls}), flush=True)
    if parameter_digest(backbone) != digest or calls != 7 * (
        len(entries) - prepared["fallback_cases"]
    ):
        raise ValueError("backbone identity or proxy call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "proxy_calls": calls,
            "ordinary_calls": ordinary,
            "fallback_cases": prepared["fallback_cases"],
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "proxy-inputs-v001",
        base / "proxy-forecasts-v001",
        base / "proxy-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed proxy scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all 314 proxy tasks before scoring")
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
    if len(rows) != 48859 or len(targets) != 91825:
        raise ValueError("registered proxy score counts changed")
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
            "primary": "raw_plus_pool",
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
        from audit_repair_proxy import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
