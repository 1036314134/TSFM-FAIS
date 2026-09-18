"""Matched fixed target repairs under the same available long native history."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from future_query_core import array_digest, field_digests
from group_scope_experiment import grouped_forward
from long_repair_core import repaired_long_context, selected_repairs
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R47_LONG_REPAIR_BASELINES_PROTOCOL.md"
INPUTS = ROOT / "artifacts/iclr27-r43/proxy-inputs-v001"
NATIVE = ROOT / "artifacts/iclr27-r45/long-native-forecasts-v001"
PARENT = ROOT / "artifacts/iclr27-r46/query-role-forecasts-v001"


def prepare(base, smoke):
    output = base / "long-repair-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve fixed long-repair inputs")
    native_entries = {r["case_id"]: r for r in read_json(NATIVE / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    rows = read_json(INPUTS / "manifest.json")["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(
                ROOT / "artifacts/iclr27-r45/smoke-v001/long-native-inputs-v001/manifest.json"
            )["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    entries = []
    for row in rows:
        native_row, parent = native_entries[row["case_id"]], parents[row["case_id"]]
        saved = load_npz(NATIVE / native_row["path"], native_row["sha256"])
        queries = json.loads(str(saved["queries"]))
        bindings = {}
        for scope in ("peer", "targets"):
            query = next(q for q in queries if q["name"] == "native_long_prefix_" + scope)
            if query["input_units"] != "prefix_standardized":
                raise ValueError("the baseline long-context units changed")
            bindings[scope] = {"path": str(NATIVE / query["path"]), "sha256": query["sha256"]}
        pool = load_npz(INPUTS / row["path"], row["sha256"])
        long = load_npz(bindings["peer"]["path"], bindings["peer"]["sha256"])["context_z"]
        np.testing.assert_array_equal(long[:, -192:], pool["native"])
        choices = selected_repairs(row["dataset"], pool["pool_names"].tolist())
        entries.append(
            {
                **row,
                "input_path": str(INPUTS / row["path"]),
                "input_sha256": row["sha256"],
                "native_queries": bindings,
                "repairs": [{"name": name, "index": index} for name, index in choices],
                "parent_path": str(PARENT / parent["path"]),
                "parent_sha256": parent["sha256"],
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/long_repair_core.py",
                    INPUTS / "manifest.json",
                    NATIVE / "manifest.json",
                    PARENT / "manifest.json",
                    ROOT / "artifacts/iclr27-r46/query-role-audit-v001/manifest.json",
                )
            },
            "heldout_value_analysis": False,
        },
    )


def case_queries(row, pool, native):
    long = native["peer"]["context_z"]
    np.testing.assert_array_equal(native["targets"]["context_z"], long[: row["target_count"]])
    np.testing.assert_array_equal(long[:, -192:], pool["native"])
    result = {}
    for repair in row["repairs"]:
        completed = repaired_long_context(long, pool["pool"][repair["index"]])
        prefix = "long_repair_" + repair["name"]
        result[prefix + "_peer"] = completed
        result[prefix + "_targets"] = np.ascontiguousarray(completed[: row["target_count"]])
        if row["dataset"] == "hdb":
            result["short_repair_" + repair["name"] + "_peer"] = np.ascontiguousarray(
                completed[:, -192:]
            )
    return result


def forecast(base):
    inputs, output = base / "long-repair-inputs-v001", base / "long-repair-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve fixed long-repair predictions")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    if pipeline.model_context_length != 8192:
        raise ValueError("the registered long context limit changed")
    mid, entries, calls, restores = pipeline.quantiles.index(0.5), [], 0, 0
    for number, row in enumerate(prepared["cases"]):
        pool = load_npz(row["input_path"], row["input_sha256"])
        native = {
            scope: load_npz(binding["path"], binding["sha256"])
            for scope, binding in row["native_queries"].items()
        }
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        requests = []
        for name, context in case_queries(row, pool, native).items():
            started = perf_counter()
            raw, fields = grouped_forward(
                backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), row["horizon"]
            )
            elapsed = perf_counter() - started
            if fields["context_fields"].shape[1] != int(np.ceil(context.shape[1] / 16)):
                raise ValueError("a repaired native history was truncated")
            point = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
            if name in points or "half_var_" + name in points:
                raise ValueError("a baseline supplement would overwrite a frozen output")
            points[name] = point
            points["half_var_" + name] = 0.5 * point + 0.5 * points["linear_var_direct"]
            if name in ("short_repair_knn_peer", "short_repair_gaussian_peer"):
                reference = "target_knn" if name == "short_repair_knn_peer" else "target_gaussian"
                np.testing.assert_array_equal(point, points[reference])
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            _save_npz(
                path,
                quantiles=raw,
                context_sha256=np.asarray(array_digest(context)),
                field_hashes=np.asarray(json.dumps(field_digests(fields))),
            )
            requests.append(
                {
                    "name": name,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "input_length": context.shape[1],
                    "input_rows": len(context),
                    "wall_seconds": elapsed,
                }
            )
            calls += 1
        for scope, reference in native.items():
            context = reference["context_z"]
            raw, fields = grouped_forward(
                backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), row["horizon"]
            )
            np.testing.assert_array_equal(raw, reference["quantiles"])
            path = output / "queries" / f"{row['case_id']}-ordinary_{scope}.npz"
            _save_npz(
                path,
                quantiles=raw,
                context_sha256=np.asarray(array_digest(context)),
                field_hashes=np.asarray(json.dumps(field_digests(fields))),
            )
            requests.append(
                {
                    "name": "ordinary_" + scope,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "input_length": context.shape[1],
                    "input_rows": len(context),
                }
            )
            restores += 1
        names = sorted(points)
        if len(names) != (266 if row["dataset"] == "beijing" else 159):
            raise ValueError("registered long-repair method count changed")
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
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 10 == 0:
            _write_json(
                output / "progress.json",
                {"cases": number + 1, "repair_queries": calls, "ordinary": restores},
            )
            print(json.dumps({"cases": number + 1, "repair_queries": calls}), flush=True)
    if parameter_digest(backbone) != digest:
        raise ValueError("fixed baseline inference changed model parameters")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "repair_calls": calls,
            "ordinary_calls": restores,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "long-repair-inputs-v001",
        base / "long-repair-forecasts-v001",
        base / "long-repair-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve long-repair baseline scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all baseline predictions before scoring")
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
            raise ValueError("registered future support changed")
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
    if len(rows) != 74643 or len(targets) != 136089:
        raise ValueError("registered baseline score counts changed")
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
        from audit_long_repair import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
