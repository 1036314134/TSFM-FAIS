"""Frozen HDB development split and bounded transfer of the existing fixed hybrid."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from hdb_generalization_core import (
    DEVELOPMENT_END,
    PREFIX,
    build_case,
    choose_peers,
    combine,
    fit_dynamics,
    population,
    queries,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

BASE = ROOT / "artifacts/iclr27-r39"
PLAN = BASE / "hdb-plan-v001"
SOURCE = ROOT / "artifacts/iclr27-r38/hdb-hourly-v001"
PROTOCOL = ROOT / "docs/iclr2027/R39_HDB_GENERALIZATION_PROTOCOL.md"


def plan():
    if (PLAN / "manifest.json").exists():
        raise ValueError("preserve the registered development population")
    source = read_json(SOURCE / "manifest.json")
    audited = read_json(SOURCE / "audit_manifest.json")
    if audited["status"] != "completed" or audited["manifest_sha256"] != file_sha256(
        SOURCE / "manifest.json"
    ):
        raise ValueError("official data acquisition has not passed its independent audit")
    panel = load_npz(SOURCE / "hourly_panel.npz", source["panel_sha256"])
    values = panel["fresh_values"][:DEVELOPMENT_END].astype(float)
    identifiers = panel["identifiers"].tolist()
    rows, metadata = population(values, identifiers)
    if not rows or "natural_outage_h24" not in metadata["panels"]:
        raise ValueError("the registered natural-outage task is unsupported")
    peers = []
    for column in metadata["selected_columns"]:
        columns, decisions = choose_peers(
            values[:PREFIX], column, metadata["eligible_columns"], identifiers
        )
        peers.append(
            {
                "station": identifiers[column],
                "columns": columns,
                "identifiers": [identifiers[c] for c in columns],
                "decisions": decisions,
            }
        )
    path = PLAN / "development.npz"
    _save_npz(
        path,
        values=values,
        identifiers=panel["identifiers"],
        timestamps=panel["timestamps"][:DEVELOPMENT_END],
    )
    _write_json(
        PLAN / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(SOURCE / "manifest.json"),
            "source_audit_sha256": file_sha256(SOURCE / "audit_manifest.json"),
            "source_panel_sha256": source["panel_sha256"],
            "development_path": str(path),
            "development_sha256": file_sha256(path),
            "prefix_end": PREFIX,
            "development_end": DEVELOPMENT_END,
            "unscored_confirmation_range": [672, 1008],
            "sealed_range": [1008, 1344],
            "cases": rows,
            "population": metadata,
            "peers": peers,
            "protocol_sha256": file_sha256(PROTOCOL),
            "core_sha256": file_sha256(ROOT / "scripts/hdb_generalization_core.py"),
            "prediction_errors_read": False,
            "heldout_value_analysis": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "planned",
                "panels": metadata["panels"],
                "targets": len(peers),
                "peer_dimensions": sorted({len(p["columns"]) for p in peers}),
            }
        ),
        flush=True,
    )


def prepare(base, smoke):
    output = base / "hdb-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed HDB inputs")
    registered = read_json(PLAN / "manifest.json")
    if registered["protocol_sha256"] != file_sha256(PROTOCOL) or registered[
        "core_sha256"
    ] != file_sha256(ROOT / "scripts/hdb_generalization_core.py"):
        raise ValueError("the population definition changed")
    source = load_npz(registered["development_path"], registered["development_sha256"])
    rows = registered["cases"]
    if smoke:
        selected = set()
        for panel in registered["population"]["panels"]:
            part = [r for r in rows if r["panel"] == panel]
            selected.update((part[0]["case_id"], part[-1]["case_id"]))
        selected.add(max(rows, key=lambda r: r["outage_age"])["case_id"])
        rows = [r for r in rows if r["case_id"] in selected]
    peers = {r["station"]: r for r in registered["peers"]}
    models, entries = {}, []
    for station in sorted({r["station"] for r in rows}):
        columns = peers[station]["columns"]
        fitted = fit_dynamics(source["values"][:PREFIX, columns])
        path = output / "models" / f"{station}.npz"
        _save_npz(path, **fitted)
        models[station] = fitted
        entries.append(
            {
                "station": station,
                "columns": columns,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    cases = []
    for row in rows:
        data = build_case(
            source["values"], peers[row["station"]]["columns"], row, models[row["station"]]
        )
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(path, **data)
        cases.append(
            {
                **row,
                "columns": peers[row["station"]]["columns"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": cases,
            "models": entries,
            "identity": {
                str(path): file_sha256(path)
                for path in (
                    PLAN / "manifest.json",
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/hdb_generalization_core.py",
                    ROOT / "scripts/dynamic_posterior_core.py",
                )
            },
            "current_future_values_used_for_inputs": False,
        },
    )


def forecast(base):
    inputs, output = base / "hdb-inputs-v001", base / "hdb-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed HDB forecasts")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, entries, calls = pipeline.quantiles.index(0.5), [], 0
    for number, row in enumerate(prepared["cases"]):
        data = load_npz(inputs / row["path"], row["sha256"])
        points, requests = {}, []
        for name, context in queries(data).items():
            with torch.inference_mode():
                raw = (
                    backbone(
                        context=torch.tensor(context, device="cuda"),
                        group_ids=torch.zeros(len(context), device="cuda", dtype=torch.long),
                        num_output_patches=int(np.ceil(24 / pipeline.model_output_patch_size)),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            if not np.isfinite(raw).all():
                raise ValueError("nonfinite HDB model query")
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            _save_npz(path, context_z=context, quantiles=raw)
            requests.append(
                {"name": name, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
            )
            points[name] = raw[0, mid, :24].astype(float)
            calls += 1
        combined = combine(points, data)
        names = sorted(combined)
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([combined[n] for n in names]),
            queries=np.asarray(json.dumps(requests)),
        )
        entries.append(
            {
                **{
                    k: row[k]
                    for k in ("case_id", "station", "column", "panel", "origin", "horizon")
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 20 == 0:
            _write_json(output / "progress.json", {"cases": number + 1, "queries": calls})
            print(json.dumps({"cases": number + 1, "queries": calls}), flush=True)
    if parameter_digest(backbone) != digest or calls != 11 * len(entries):
        raise ValueError("model identity or query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "model_calls": calls,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "prediction_errors_read": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "hdb-inputs-v001",
        base / "hdb-forecasts-v001",
        base / "hdb-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed HDB scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    registered = read_json(PLAN / "manifest.json")
    if fm["smoke"] or {r["case_id"] for r in fm["cases"]} != {
        r["case_id"] for r in registered["cases"]
    }:
        raise ValueError("finish the full registered HDB predictions before scoring")
    source = load_npz(registered["development_path"], registered["development_sha256"])
    metadata, rows = {r["case_id"]: r for r in prepared["cases"]}, []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        t = row["origin"]
        truth = source["values"][t : t + 24, row["column"]]
        valid = np.isfinite(truth)
        if valid.sum() < 12 or t + 24 > DEVELOPMENT_END:
            raise ValueError("registered scoring support or time changed")
        expected = (truth[valid] - data["mean"][0]) / data["scale"][0]
        errors = saved["points"][:, valid] - expected
        mae, mse = abs(errors).mean(1), np.square(errors).mean(1)
        for index, method in enumerate(saved["methods"].tolist()):
            rows.append(
                {
                    **{k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")},
                    "method": method,
                    "observed_count": int(valid.sum()),
                    "mae": float(mae[index]),
                    "mse": float(mse[index]),
                }
            )
    if len(rows) != 31 * len(registered["cases"]):
        raise ValueError("registered score count changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
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
            "primary": "half_var_full_gaussian",
            "primary_panel": "natural_outage_h24",
            "score_rows": len(rows),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "independent_source_confirmation": False,
            "development_only": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("plan", "prepare", "forecast", "evaluate", "audit"), required=True
    )
    parser.add_argument("--run-root", type=Path, default=BASE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "plan":
        plan()
    elif args.phase == "prepare":
        prepare(base, args.smoke)
    elif args.phase == "forecast":
        forecast(base)
    elif args.phase == "evaluate":
        evaluate(base)
    else:
        from audit_hdb_generalization import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
