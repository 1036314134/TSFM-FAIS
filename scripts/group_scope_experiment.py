"""Match candidate grouping and forecasting-variable scope without changing repairs."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from group_scope_core import comparison_queries, diagnostic_points
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R44_GROUP_SCOPE_DIAGNOSTIC_PROTOCOL.md"
INPUTS = ROOT / "artifacts/iclr27-r43/proxy-inputs-v001"
PARENT = ROOT / "artifacts/iclr27-r43/proxy-forecasts-v001"


def prepare(base, smoke):
    output = base / "scope-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve the frozen scope diagnostic inputs")
    source = read_json(INPUTS / "manifest.json")
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    smoke_ids = {
        r["case_id"]
        for r in read_json(
            ROOT / "artifacts/iclr27-r43/smoke-v001/proxy-inputs-v001/manifest.json"
        )["cases"]
    }
    rows = []
    for entry in source["cases"]:
        if smoke and entry["case_id"] not in smoke_ids:
            continue
        parent = parents[entry["case_id"]]
        rows.append(
            {
                **entry,
                "input_path": str(INPUTS / entry["path"]),
                "input_sha256": entry["sha256"],
                "parent_path": str(PARENT / parent["path"]),
                "parent_sha256": parent["sha256"],
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": rows,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/group_scope_core.py",
                    ROOT / "scripts/repair_proxy_core.py",
                    INPUTS / "manifest.json",
                    PARENT / "manifest.json",
                    ROOT / "artifacts/iclr27-r43/proxy-audit-v001/manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def grouped_forward(backbone, pipeline, context, groups, horizon):
    fields, norms = [], []

    def before_embedding(_module, args):
        fields.append(args[0].detach().cpu().numpy().copy())

    def after_norm(_module, _args, result):
        norms.append(tuple(v.detach().cpu().numpy().copy() for v in result[1]))

    handles = [
        backbone.input_patch_embedding.register_forward_pre_hook(before_embedding),
        backbone.instance_norm.register_forward_hook(after_norm),
    ]
    try:
        with torch.inference_mode():
            raw = (
                backbone(
                    context=torch.tensor(context, device="cuda"),
                    group_ids=torch.tensor(groups, device="cuda", dtype=torch.long),
                    num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
                )
                .quantile_preds.float()
                .cpu()
                .numpy()
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(fields) != 2 or len(norms) != 1 or not np.isfinite(raw).all():
        raise ValueError("unexpected group-scope interface or nonfinite output")
    return raw, {
        "context_fields": fields[0],
        "future_fields": fields[1],
        "loc": norms[0][0],
        "scale": norms[0][1],
    }


def forecast(base):
    inputs, output = base / "scope-inputs-v001", base / "scope-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed scope predictions")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, entries, calls, serial_calls, restored, maximum_serial = (
        pipeline.quantiles.index(0.5),
        [],
        0,
        0,
        0,
        0.0,
    )
    for number, row in enumerate(prepared["cases"]):
        data = load_npz(row["input_path"], row["input_sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        raw_queries, captures, requests, serial = {}, {}, [], []
        definitions = comparison_queries(data)
        for name, (context, groups) in definitions.items():
            raw, fields = grouped_forward(backbone, pipeline, context, groups, row["horizon"])
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            _save_npz(path, context_z=context, group_ids=groups, quantiles=raw, **fields)
            requests.append(
                {
                    "name": name,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "input_rows": len(context),
                    "logical_groups": len(np.unique(groups)),
                }
            )
            raw_queries[name], captures[name] = raw, fields
            calls += 1
        for field in captures["joint_candidates"]:
            np.testing.assert_array_equal(
                captures["joint_candidates"][field], captures["isolated_candidates"][field]
            )
            np.testing.assert_array_equal(
                captures["target_anchor_candidates"][field][: row["target_count"]],
                captures["native_targets"][field],
            )
        if not row["complete_target_fallback"]:
            old_query = next(
                q for q in json.loads(str(parent["queries"])) if q["name"] == "proxy_pool_only"
            )
            old = load_npz(PARENT / old_query["path"], old_query["sha256"])
            np.testing.assert_array_equal(old["context_z"], definitions["joint_candidates"][0])
            np.testing.assert_array_equal(old["quantiles"], raw_queries["joint_candidates"])
            restored += 1
        if row["dataset"] == "hdb":
            np.testing.assert_array_equal(
                raw_queries["native_targets"][:1, mid, : row["horizon"]].T.astype(float),
                points["native_target192"],
            )
        if prepared["smoke"]:
            target_count = row["target_count"]
            matrix = definitions["isolated_candidates"][0]
            for candidate in range(len(data["pool"])):
                start = candidate * target_count
                context = np.ascontiguousarray(matrix[start : start + target_count])
                raw, _ = grouped_forward(
                    backbone,
                    pipeline,
                    context,
                    np.zeros(target_count, dtype=np.int64),
                    row["horizon"],
                )
                batched = raw_queries["isolated_candidates"][start : start + target_count]
                np.testing.assert_allclose(raw, batched, rtol=2e-5, atol=2e-5)
                maximum_serial = max(maximum_serial, float(abs(raw - batched).max()))
                path = output / "serial" / f"{row['case_id']}-{candidate:02d}.npz"
                _save_npz(path, context_z=context, quantiles=raw)
                serial.append(
                    {
                        "candidate": candidate,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                serial_calls += 1
        additions = diagnostic_points(data, raw_queries, row["horizon"], mid)
        for name, value in additions.items():
            if name in points or "half_var_" + name in points:
                raise ValueError("a scope control would replace a frozen result")
            points[name] = value
            points["half_var_" + name] = 0.5 * value + 0.5 * points["linear_var_direct"]
        names = sorted(points)
        if len(names) != (230 if row["dataset"] == "beijing" else 103):
            raise ValueError("the registered scope control set is incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([points[n] for n in names]),
            queries=np.asarray(json.dumps(requests)),
            serial_queries=np.asarray(json.dumps(serial)),
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
            _write_json(output / "progress.json", {"cases": number + 1, "queries": calls})
            print(json.dumps({"cases": number + 1, "queries": calls}), flush=True)
    if parameter_digest(backbone) != digest or calls != 4 * len(entries):
        raise ValueError("base parameters or grouped-query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "model_calls": calls,
            "serial_verification_calls": serial_calls,
            "maximum_serial_difference": maximum_serial,
            "old_joint_queries_restored": restored,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "scope-inputs-v001",
        base / "scope-forecasts-v001",
        base / "scope-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed scope scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all scope controls before evaluation")
    records, hdb = truth_sources()
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(row["input_path"], row["input_sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        truth = case_truth(row, records, hdb)
        valid = np.isfinite(truth)
        if (valid.sum(0) < row["horizon"] // 2).any():
            raise ValueError("registered future observation support changed")
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
    if len(rows) != 61679 or len(targets) != 114809:
        raise ValueError("registered scope score counts changed")
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
            "contrasts": [
                ["joint_reference_median", "independent_proxy_median"],
                ["target_anchor_pool", "raw_plus_pool"],
            ],
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
        from audit_group_scope import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
