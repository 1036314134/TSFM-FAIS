"""Two-source fixed reliability attention probe with cached strong controls."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from peer_outage_core import sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_core import MODES, attention_intervention, conditional_reliability

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R40_RELIABILITY_ATTENTION_PROTOCOL.md"


def source_entries():
    entries = []
    for dataset, input_root, parent_root, smoke_root in (
        (
            "beijing",
            "artifacts/iclr27-r35/dynamic-inputs-v001",
            "artifacts/iclr27-r38/correlation-forecasts-v001",
            "artifacts/iclr27-r38/smoke-v001/correlation-forecasts-v001",
        ),
        (
            "hdb",
            "artifacts/iclr27-r39/hdb-inputs-v001",
            "artifacts/iclr27-r39/hdb-forecasts-v001",
            "artifacts/iclr27-r39/smoke-v001/hdb-inputs-v001",
        ),
    ):
        inputs, parents = ROOT / input_root, ROOT / parent_root
        im, pm = read_json(inputs / "manifest.json"), read_json(parents / "manifest.json")
        old = {r["case_id"]: r for r in pm["cases"]}
        models = {r["station"]: r for r in im["models"]}
        smoke_ids = {r["case_id"] for r in read_json(ROOT / smoke_root / "manifest.json")["cases"]}
        for row in im["cases"]:
            parent, model = old[row["case_id"]], models[row["station"]]
            entries.append(
                {
                    "case_id": dataset + "-" + row["case_id"],
                    "source_case_id": row["case_id"],
                    "dataset": dataset,
                    "panel": dataset + "/" + row["panel"],
                    "source_panel": row["panel"],
                    **{k: row[k] for k in ("station", "origin", "horizon", "prefix_end")},
                    "target_count": 2 if dataset == "beijing" else 1,
                    "source_column": row.get("column", 0),
                    "smoke_selected": row["case_id"] in smoke_ids,
                    "source_input_path": str(inputs / row["path"]),
                    "source_input_sha256": row["sha256"],
                    "source_model_path": str(inputs / model["path"]),
                    "source_model_sha256": model["sha256"],
                    "parent_path": str(parents / parent["path"]),
                    "parent_sha256": parent["sha256"],
                    "gaussian_name": "full_static_point"
                    if dataset == "beijing"
                    else "full_gaussian",
                    "native_name": "native_peer" if dataset == "beijing" else "native_peer192",
                }
            )
    return entries


def prepare(base, smoke):
    output = base / "attention-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed attention inputs")
    rows = source_entries()
    if smoke:
        rows = [r for r in rows if r["smoke_selected"]]
    entries = []
    for row in rows:
        source = load_npz(row["source_input_path"], row["source_input_sha256"])
        model = load_npz(row["source_model_path"], row["source_model_sha256"])
        context, keep = source["context"], source["keep"]
        if row["dataset"] == "beijing":
            filled = source["static_values"]
        else:
            filled = source["fills"][source["fill_names"].tolist().index("gaussian")]
        selected = np.flatnonzero(keep)
        reliability, variance = conditional_reliability(
            np.isfinite(context), model["initial_covariance"]
        )
        gaussian = np.array(
            ((filled[:, selected] - source["mean"][selected]) / source["scale"][selected]).T,
            dtype=np.float32,
            order="C",
        )
        native = np.array(
            ((context[:, selected] - source["mean"][selected]) / source["scale"][selected]).T,
            dtype=np.float32,
            order="C",
        )
        observed = np.isfinite(context[:, selected]).T
        np.testing.assert_array_equal(gaussian[observed], native[observed])
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            gaussian=gaussian,
            native=native,
            observed=observed,
            reliability=reliability[:, selected].T,
            conditional_variance=variance[:, selected].T,
            mean=source["mean"],
            scale=source["scale"],
            selected=selected,
        )
        entries.append({**row, "path": str(path.relative_to(output)), "sha256": file_sha256(path)})
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "identity": {
                str(path): file_sha256(path)
                for path in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/reliability_attention_core.py",
                    ROOT / "artifacts/iclr27-r38/correlation-forecasts-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r38/correlation-audit-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-forecasts-v001/manifest.json",
                    ROOT / "artifacts/iclr27-r39/hdb-audit-v001/manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def old_points(row):
    old = load_npz(row["parent_path"], row["parent_sha256"])
    points = old["points"][..., None] if row["target_count"] == 1 else old["points"]
    return dict(zip(old["methods"].tolist(), points, strict=True))


def forward(backbone, pipeline, context, horizon):
    with torch.inference_mode():
        return (
            backbone(
                context=torch.tensor(context, device="cuda"),
                group_ids=torch.zeros(len(context), device="cuda", dtype=torch.long),
                num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
            )
            .quantile_preds.float()
            .cpu()
            .numpy()
        )


def forecast(base):
    inputs, output = base / "attention-inputs-v001", base / "attention-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed attention predictions")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, entries, calls, ordinary_calls, neutral_calls = pipeline.quantiles.index(0.5), [], 0, 0, 0
    for number, row in enumerate(prepared["cases"]):
        data = load_npz(inputs / row["path"], row["sha256"])
        points, requests = old_points(row), []
        h, targets = row["horizon"], row["target_count"]
        for mode, (kind, _, _, _) in MODES.items():
            if mode in points or "half_var_" + mode in points:
                raise ValueError("a new method would replace a frozen control")
            context, records = data[kind], {}
            with attention_intervention(backbone, data, mode, records):
                raw = forward(backbone, pipeline, context, h)
            if not np.isfinite(raw).all():
                raise ValueError("nonfinite reliability-attention prediction")
            path = output / "queries" / f"{row['case_id']}-{mode}.npz"
            _save_npz(path, context_z=context, quantiles=raw, **records)
            requests.append(
                {"name": mode, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
            )
            points[mode] = raw[:targets, mid, :h].T.astype(float)
            points["half_var_" + mode] = 0.5 * points[mode] + 0.5 * points["linear_var_direct"]
            calls += 1
        for kind, reference_name in (
            ("gaussian", row["gaussian_name"]),
            ("native", row["native_name"]),
        ):
            raw = forward(backbone, pipeline, data[kind], h)
            np.testing.assert_array_equal(
                raw[:targets, mid, :h].T.astype(float), points[reference_name]
            )
            path = output / "queries" / f"{row['case_id']}-restore_{kind}.npz"
            _save_npz(path, context_z=data[kind], quantiles=raw)
            requests.append(
                {
                    "name": "restore_" + kind,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            ordinary_calls += 1
            if prepared["smoke"] and kind == "gaussian":
                with attention_intervention(
                    backbone, data, "gaussian_conditional_attention", neutral=True
                ):
                    neutral = forward(backbone, pipeline, data[kind], h)
                np.testing.assert_array_equal(neutral, raw)
                neutral_calls += 1
        names = sorted(points)
        expected_count = 162 if row["dataset"] == "beijing" else 47
        if len(names) != expected_count:
            raise ValueError("incomplete registered output set")
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
        if (number + 1) % 25 == 0:
            _write_json(
                output / "progress.json", {"cases": number + 1, "intervention_queries": calls}
            )
            print(json.dumps({"cases": number + 1, "intervention_queries": calls}), flush=True)
    if parameter_digest(backbone) != digest:
        raise ValueError("frozen backbone weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "intervention_calls": calls,
            "ordinary_calls": ordinary_calls,
            "neutral_checks": neutral_calls,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "heldout_value_analysis": False,
        },
    )


def truth_sources():
    records, _ = sources()
    plan = read_json(ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json")
    hdb = load_npz(plan["development_path"], plan["development_sha256"])["values"]
    if len(hdb) != 672:
        raise ValueError("HDB evaluation must remain within the development split")
    return records, hdb


def case_truth(row, records, hdb):
    t, h = row["origin"], row["horizon"]
    if row["dataset"] == "beijing":
        return records[row["station"]]["values"][t : t + h, :2]
    if t + h > 672:
        raise ValueError("HDB forecast crossed the held-out boundary")
    return hdb[t : t + h, row["source_column"] : row["source_column"] + 1]


def evaluate(base):
    inputs, forecasts, output = (
        base / "attention-inputs-v001",
        base / "attention-forecasts-v001",
        base / "attention-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed attention scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze all 314 formal predictions before evaluation")
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
    if len(rows) != 41323 or len(targets) != 78745:
        raise ValueError("registered score counts changed")
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
            "primary": "gaussian_conditional_attention",
            "primary_panels": ["beijing/natural_outage_h24", "hdb/natural_outage_h24"],
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
        from audit_reliability_attention import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
