"""Registered observed-statistics intervention on unchanged R30 query inputs."""

import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import PARENT, ROOT, load_npz, read_json
from normalization_interface import statistical_mask
from peer_outage_core import sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R34_OBSERVED_NORMALIZATION_PROTOCOL.md"


def raw_forward(backbone, context, horizon, patch_size):
    with torch.inference_mode():
        return (
            backbone(
                context=torch.tensor(context, device="cuda"),
                group_ids=torch.zeros(len(context), dtype=torch.long, device="cuda"),
                num_output_patches=int(np.ceil(horizon / patch_size)),
            )
            .quantile_preds.float()
            .cpu()
            .numpy()
        )


def append_pools(methods, actions):
    for prefix in ("peer", "target"):
        bank = np.stack(
            [
                methods["native_peer"].astype(np.float32),
                *[methods["observed_norm_" + prefix + "_" + name] for name in actions],
            ]
        ).astype(np.float32)
        methods["observed_norm_" + prefix + "_mean8"] = bank.mean(0, dtype=np.float64)
        methods["observed_norm_" + prefix + "_median8"] = np.median(bank, axis=0).astype(float)


def forecast(base, smoke):
    output = base / "normalization-forecasts-v001"
    if (output / "identity.json").exists():
        raise ValueError("preserve existing normalization runs, including partial ones")
    inputs = PARENT / "peer-inputs-v001"
    prepared, parent = (
        read_json(inputs / "manifest.json"),
        read_json(PARENT / "peer-forecasts-v001/manifest.json"),
    )
    rows = prepared["cases"]
    old_entries = {r["case_id"]: r for r in parent["cases"]}
    if smoke:
        selected = {
            r["case_id"]
            for panel in ("natural_outage_h24", "synthetic_outage_h24", "legacy_native_h96")
            for r in [v for v in rows if v["panel"] == panel][:2]
        }
        selected.add(
            max(
                [r for r in rows if r["panel"] == "natural_outage_h24"],
                key=lambda r: r["outage_age"],
            )["case_id"]
        )
        selected.add(
            min(
                rows,
                key=lambda r: (
                    np.isfinite(load_npz(inputs / r["path"])["context"][:, :2]).sum(0).min()
                ),
            )["case_id"]
        )
        rows = [r for r in rows if r["case_id"] in selected]
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    identity = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            PROTOCOL,
            ROOT / "scripts/normalization_interface.py",
            inputs / "manifest.json",
            PARENT / "peer-forecasts-v001/manifest.json",
            Path(inspect.getfile(type(backbone))),
            Path(inspect.getfile(type(backbone.instance_norm))),
        )
    }
    _write_json(output / "identity.json", identity)
    mid, entries, calls = pipeline.quantiles.index(0.5), [], 0
    for row in rows:
        d = load_npz(inputs / row["path"], row["sha256"])
        old_entry = old_entries[row["case_id"]]
        old = load_npz(PARENT / "peer-forecasts-v001" / old_entry["path"], old_entry["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        queries = json.loads(str(old["queries"]))
        requests = [("observed", q) for q in queries if not q["name"].startswith("native_")]
        primary = next(q for q in queries if q["name"] == "peer_ridge")
        requests.extend((mode, primary) for mode in ("location", "scale", "shifted"))
        if len(requests) != 21:
            raise ValueError("the registered raw-query set changed")
        new_queries = []
        for mode, query in requests:
            old_path = PARENT / "peer-forecasts-v001/queries" / f"{query['key']}.npz"
            raw = load_npz(old_path, query["sha256"])
            canonical = raw["context_z"]
            columns = query["columns"]
            origin = np.array(
                ((d["context"][:, columns] - d["mean"][columns]) / d["scale"][columns]).T,
                dtype=np.float32,
                order="C",
            )
            observed = np.isfinite(origin)
            np.testing.assert_array_equal(canonical[observed], origin[observed])
            name = mode + "_norm_" + query["name"]
            hooks_before = len(backbone.instance_norm._forward_pre_hooks)
            with statistical_mask(
                backbone.instance_norm, torch.tensor(observed, device="cuda"), mode
            ) as captured:
                q = raw_forward(
                    backbone, canonical, row["horizon"], pipeline.model_output_patch_size
                )
            if (
                captured["calls"] != 1
                or len(backbone.instance_norm._forward_pre_hooks) != hooks_before
            ):
                raise ValueError("normalization intervention escaped its one context call")
            if not np.isfinite(q).all():
                raise ValueError("normalization intervention produced nonfinite predictions")
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            statistics = {
                k: v.detach().cpu().numpy() for k, v in captured.items() if torch.is_tensor(v)
            }
            _save_npz(
                path,
                quantiles=q,
                original_mask=observed,
                eps=np.asarray(backbone.instance_norm.eps),
                use_arcsinh=np.asarray(backbone.instance_norm.use_arcsinh),
                **statistics,
            )
            new_queries.append(
                {
                    "name": name,
                    "mode": mode,
                    "parent_query": query,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            methods[name] = q[:2, mid, : row["horizon"]].T
            calls += 1
        append_pools(methods, d["actions"].tolist())
        original_primary = load_npz(
            PARENT / "peer-forecasts-v001/queries" / f"{primary['key']}.npz", primary["sha256"]
        )
        restored = raw_forward(
            backbone,
            original_primary["context_z"],
            row["horizon"],
            pipeline.model_output_patch_size,
        )
        np.testing.assert_array_equal(restored, original_primary["quantiles"])
        names = sorted(methods)
        points = np.stack([methods[n] for n in names]).astype(float)
        if len(names) != 62 or not np.isfinite(points).all():
            raise ValueError("the 62-method normalization panel is incomplete")
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=points,
            queries=np.asarray(json.dumps(new_queries)),
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "panel": row["panel"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "restored_ordinary_difference": 0,
            }
        )
        _write_json(
            output / "progress.json",
            {"completed": len(entries), "total": len(rows), "new_method_calls": calls},
        )
        if len(entries) % 25 == 0:
            print(json.dumps({"predicted": len(entries), "total": len(rows)}), flush=True)
    if digest != parent["parameter_sha256"] or parameter_digest(backbone) != digest:
        raise ValueError("the frozen forecasting parameters changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "input_root": str(inputs),
            "identity": identity,
            "parameter_sha256": digest,
            "new_method_calls": calls,
            "ordinary_verification_calls": len(entries),
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "evaluation_future_values_read": False,
        },
    )


def evaluate(base):
    forecast_root, output = (
        base / "normalization-forecasts-v001",
        base / "normalization-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed normalization scores")
    fm = read_json(forecast_root / "manifest.json")
    inputs = Path(fm["input_root"])
    metadata = {r["case_id"]: r for r in read_json(inputs / "manifest.json")["cases"]}
    if (
        fm["smoke"]
        or len(fm["cases"]) != 231
        or set(metadata) != {r["case_id"] for r in fm["cases"]}
    ):
        raise ValueError("freeze all formal forecasts before scoring")
    records, _ = sources()
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        pred = load_npz(forecast_root / entry["path"], entry["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        normalized = (truth - d["mean"][:2]) / d["scale"][:2]
        error = np.where(valid[None], pred["points"] - normalized[None], 0)
        mae, mse = abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for i, name in enumerate(pred["methods"].tolist()):
            rows.append(
                {**info, "method": name, "mae": float(mae[i].mean()), "mse": float(mse[i].mean())}
            )
            for slot in (0, 1):
                targets.append(
                    {
                        **info,
                        "method": name,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[i, slot]),
                        "mse": float(mse[i, slot]),
                    }
                )
    if len(rows) != 14322 or len(targets) != 28644:
        raise ValueError("registered normalization score counts changed")
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
            "primary": "observed_norm_peer_ridge",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "evaluate", "audit"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "forecast":
        forecast(base, args.smoke)
    elif args.phase == "evaluate":
        evaluate(base)
    else:
        from audit_normalization import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
