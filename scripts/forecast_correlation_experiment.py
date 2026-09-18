"""Frozen-forecast correlation transport with no new foundation-model queries."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from dynamic_posterior_core import quantile_weights
from forecast_calibration_core import ROOT, load_npz, read_json
from forecast_correlation_core import future_covariances, transport_outputs
from peer_outage_core import sources
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PARENT = ROOT / "artifacts/iclr27-r37"
STATE = ROOT / "artifacts/iclr27-r35"
PROTOCOL = ROOT / "docs/iclr2027/R38_FORECAST_CORRELATION_PROTOCOL.md"


def forecast(base, smoke):
    output = base / "correlation-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed correlation forecasts")
    states = read_json(STATE / "dynamic-inputs-v001/manifest.json")
    parent = read_json(PARENT / "moment-forecasts-v001/manifest.json")
    static_parent = read_json(STATE / "dynamic-forecasts-v001/manifest.json")
    old_entries = {r["case_id"]: r for r in parent["cases"]}
    static_entries = {r["case_id"]: r for r in static_parent["cases"]}
    models = {
        r["station"]: load_npz(STATE / "dynamic-inputs-v001" / r["path"], r["sha256"])
        for r in states["models"]
    }
    rows = states["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(PARENT / "smoke-v002/moment-inputs-v001/manifest.json")["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    weights = quantile_weights(static_parent["quantiles"])
    mid = static_parent["quantiles"].index(0.5)
    entries = []
    for row in rows:
        d = load_npz(STATE / "dynamic-inputs-v001" / row["path"], row["sha256"])
        model = models[row["station"]]
        old_entry = old_entries[row["case_id"]]
        old = load_npz(PARENT / "moment-forecasts-v001" / old_entry["path"], old_entry["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        static_entry = static_entries[row["case_id"]]
        static = load_npz(
            STATE / "dynamic-forecasts-v001" / static_entry["path"], static_entry["sha256"]
        )
        query = next(q for q in json.loads(str(static["queries"])) if q["name"] == "full_static")
        raw = load_npz(STATE / "dynamic-forecasts-v001" / query["path"], query["sha256"])
        h = row["horizon"]
        point = raw["quantiles"][:2, mid, :h].T.astype(float)
        np.testing.assert_array_equal(point, methods["full_static_point"])
        quantiles = np.sort(raw["quantiles"][:2, :, :h].transpose(1, 2, 0).astype(float), axis=0)
        variance = ((quantiles - point[None]) ** 2 * weights[:, None, None]).sum(0) / weights.sum()
        statistical, covariance, initial = future_covariances(
            model["a"],
            model["q"],
            d["filtered_mean"][-1],
            d["filtered_covariance"][-1],
            model["b"],
            h,
        )
        np.testing.assert_array_equal(statistical, methods["linear_var_direct"])
        updated, covariances, fallback = transport_outputs(
            point, methods["static_quantile_mean"], variance, statistical, covariance, initial
        )
        methods.update(updated)
        names = sorted(methods)
        values = np.stack([methods[n] for n in names])
        if len(names) != 146 or not np.isfinite(values).all():
            raise ValueError("the registered correlation output set is incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=values,
            covariance_var=covariance,
            covariance_initial=initial,
            variance_foundation=variance,
            transported_names=np.asarray(list(updated)),
            transported_covariances=np.stack(list(covariances.values())),
            degenerate_fallback=np.asarray(json.dumps(fallback)),
        )
        entries.append(
            {
                **{
                    k: row[k]
                    for k in ("case_id", "panel", "station", "origin", "horizon", "prefix_end")
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "state_path": row["path"],
                "state_sha256": row["sha256"],
                "static_query": query,
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "new_foundation_model_calls": 0,
            "evaluation_future_values_read": False,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/forecast_correlation_core.py",
                    PROTOCOL,
                    PARENT / "moment-forecasts-v001/manifest.json",
                    STATE / "dynamic-inputs-v001/manifest.json",
                    STATE / "dynamic-forecasts-v001/manifest.json",
                )
            },
        },
    )


def evaluate(base):
    forecasts, output = base / "correlation-forecasts-v001", base / "correlation-results-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed correlation scores")
    fm = read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 231:
        raise ValueError("freeze all formal correlation forecasts before scoring")
    records, _ = sources()
    rows, targets = [], []
    for row in fm["cases"]:
        d = load_npz(STATE / "dynamic-inputs-v001" / row["state_path"], row["state_sha256"])
        saved = load_npz(forecasts / row["path"], row["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        expected = (truth - d["mean"][:2]) / d["scale"][:2]
        error = np.where(valid[None], saved["points"] - expected[None], 0)
        mae, mse = abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for i, name in enumerate(saved["methods"].tolist()):
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
    if len(rows) != 33726 or len(targets) != 67452:
        raise ValueError("registered correlation score count changed")
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
            "primary": "correlation_transport",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
        },
    )


def audit(base):
    from audit_forecast_correlation import audit as verify

    verify(base)


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
        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
