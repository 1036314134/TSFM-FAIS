"""Fixed dynamic posterior completion and conditional forecast-distribution mixture."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from dynamic_posterior_core import (
    complete_raw,
    fit_dynamics,
    posterior_samples,
    predictive_mixture,
    smooth_history,
)
from forecast_calibration_core import (
    PARENT,
    ROOT,
    load_npz,
    mixed_context,
    read_json,
    tensor_inputs,
)
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

BASELINES = ROOT / "artifacts/iclr27-r34/normalization-forecasts-v001"
HALF = ROOT / "artifacts/iclr27-r31/calibrated-forecasts-v001"
PROTOCOL = ROOT / "docs/iclr2027/R35_DYNAMIC_POSTERIOR_PROTOCOL.md"


def prepare(base, smoke):
    output = base / "dynamic-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete dynamic inputs")
    records, peers = sources()
    parent = read_json(PARENT / "peer-inputs-v001/manifest.json")
    rows = parent["cases"]
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
                    np.isfinite(load_npz(PARENT / "peer-inputs-v001" / r["path"])["context"][:, :2])
                    .sum(0)
                    .min()
                ),
            )["case_id"]
        )
        rows = [r for r in rows if r["case_id"] in selected]
    models, fits = {}, []
    for station, source in sorted(records.items()):
        prefix = augmented(station, records, peers)[0][: source["prefix_end"]]
        model = fit_dynamics(prefix)
        path = output / "models" / f"{station}.npz"
        _save_npz(path, **model)
        models[station] = model
        fits.append(
            {
                "station": station,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "prefix_end": source["prefix_end"],
                "complete_rows": int(model["complete_rows"]),
                "transition_pairs": int(model["transition_pairs"]),
            }
        )
    entries = []
    for row in rows:
        data = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        model = models[row["station"]]
        np.testing.assert_array_equal(data["mean"], model["mean"])
        np.testing.assert_array_equal(data["scale"], model["scale"])
        x = data["context"]
        z = (x - model["mean"]) / model["scale"]
        state = smooth_history(z, model)
        sampled, seed_hex = posterior_samples(state, np.isfinite(z), row["case_id"])
        fills = {
            "static_values": complete_raw(x, state["static_mean"], model["mean"], model["scale"]),
            "filtered_values": complete_raw(
                x, state["filtered_mean"], model["mean"], model["scale"]
            ),
            "smoothed_values": complete_raw(
                x, state["smoothed_mean"], model["mean"], model["scale"]
            ),
            "sampled_values": complete_raw(x, sampled, model["mean"], model["scale"]),
        }
        for values in (
            fills["static_values"],
            fills["filtered_values"],
            fills["smoothed_values"],
            *fills["sampled_values"],
        ):
            np.testing.assert_array_equal(values[np.isfinite(x)], x[np.isfinite(x)])
            if not np.isfinite(values).all():
                raise ValueError("a dynamic imputation is incomplete")
        paired = (fills["sampled_values"][0::2] + fills["sampled_values"][1::2]) / 2
        np.testing.assert_allclose(
            (paired - fills["smoothed_values"]) / model["scale"], 0, rtol=0, atol=1e-12
        )
        direct, current = [], state["filtered_mean"][-1].copy()
        for _ in range(row["horizon"]):
            current = model["a"] @ current + model["b"]
            direct.append(current[:2].copy())
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            context=x,
            z=z,
            mean=model["mean"],
            scale=model["scale"],
            keep=data["keep"],
            seed_hex=np.asarray(seed_hex),
            direct_var=np.stack(direct),
            **state,
            **fills,
        )
        info = {
            k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon", "prefix_end")
        }
        entries.append(
            {
                **info,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "parent_input_path": row["path"],
                "parent_input_sha256": row["sha256"],
            }
        )
        _write_json(output / "progress.json", {"prepared": len(entries), "total": len(rows)})
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "models": fits,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/dynamic_posterior_core.py",
                    ROOT / "scripts/peer_outage_core.py",
                    PROTOCOL,
                    PARENT / "peer-inputs-v001/manifest.json",
                    BASELINES / "manifest.json",
                    HALF / "manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def scoped_context(data, completed, scope):
    result = data["context"].copy()
    if scope == "full":
        result[:] = completed
    elif scope == "target":
        result[:, :2] = completed[:, :2]
    else:
        raise ValueError("unregistered dynamic imputation scope")
    columns = np.flatnonzero(data["keep"])
    return np.array(
        ((result[:, columns] - data["mean"][columns]) / data["scale"][columns]).T,
        dtype=np.float32,
        order="C",
    )


def forecast(base):
    inputs, output = base / "dynamic-inputs-v001", base / "dynamic-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete dynamic forecasts")
    prepared = read_json(inputs / "manifest.json")
    controls = read_json(BASELINES / "manifest.json")
    parent_entries = {r["case_id"]: r for r in controls["cases"]}
    half_entries = {r["case_id"]: r for r in read_json(HALF / "manifest.json")["cases"]}
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, calls, entries = pipeline.quantiles.index(0.5), 0, []
    for row in prepared["cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        entry = parent_entries[row["case_id"]]
        old = load_npz(BASELINES / entry["path"], entry["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        requests = []

        def query(
            name, canonical, *, horizon=row["horizon"], case_id=row["case_id"], records=requests
        ):
            with torch.inference_mode():
                q = (
                    backbone(
                        context=torch.tensor(canonical, device="cuda"),
                        group_ids=torch.zeros(len(canonical), dtype=torch.long, device="cuda"),
                        num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            if not np.isfinite(q).all():
                raise ValueError("a dynamic conditional forecast is nonfinite")
            path = output / "queries" / f"{case_id}-{name}.npz"
            _save_npz(path, context_z=canonical, quantiles=q)
            records.append(
                {"name": name, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
            )
            return q[:2, :, :horizon].transpose(1, 2, 0)

        for scope in ("full", "target"):
            for label, field in (
                ("static", "static_values"),
                ("filtered", "filtered_values"),
                ("smoothed", "smoothed_values"),
            ):
                q = query(scope + "_" + label, scoped_context(d, d[field], scope))
                methods[scope + "_" + label + "_point"] = q[mid]
                if label == "smoothed":
                    methods[scope + "_smoothed_cdf_median"] = predictive_mixture(
                        q[None], pipeline.quantiles
                    )[1]
            sample_q = np.stack(
                [
                    query(f"{scope}_sample_{i:02d}", scoped_context(d, values, scope))
                    for i, values in enumerate(d["sampled_values"])
                ]
            )
            point = sample_q[:, mid]
            methods[scope + "_sample_point_mean16"] = point.mean(0, dtype=np.float64)
            methods[scope + "_sample_point_median16"] = np.median(point, 0).astype(float)
            mean, median = predictive_mixture(sample_q, pipeline.quantiles)
            methods[scope + "_dynamic_mixture_mean16"] = mean
            methods[scope + "_dynamic_mixture_median16"] = median
        methods["linear_var_direct"] = d["direct_var"]
        methods["half_output_mix"] = (methods["local_ridge"] + methods["peer_ridge"]) * 0.5
        if row["horizon"] == 24:
            half = half_entries[row["case_id"]]
            previous = load_npz(HALF / half["path"], half["sha256"])
            methods["half_input_mix"] = previous["points"][
                previous["methods"].tolist().index("half_input_mix")
            ]
            np.testing.assert_array_equal(
                methods["half_output_mix"],
                previous["points"][previous["methods"].tolist().index("half_output_mix")],
            )
        else:
            original = load_npz(
                PARENT / "peer-inputs-v001" / row["parent_input_path"], row["parent_input_sha256"]
            )
            canonical = (
                mixed_context(tensor_inputs(original), torch.full((2,), 0.5, device="cuda"))
                .T.contiguous()
                .cpu()
                .numpy()
            )
            methods["half_input_mix"] = query("half_input_mix", canonical)[mid]
        names = sorted(methods)
        points = np.stack([methods[n] for n in names]).astype(float)
        if len(names) != 81 or not np.isfinite(points).all():
            raise ValueError("the registered 81-method dynamic panel is incomplete")
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path, methods=np.asarray(names), points=points, queries=np.asarray(json.dumps(requests))
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        calls += len(requests)
        _write_json(
            output / "progress.json",
            {"predicted": len(entries), "total": len(prepared["cases"]), "new_queries": calls},
        )
        if len(entries) % 25 == 0:
            print(
                json.dumps({"predicted": len(entries), "total": len(prepared["cases"])}), flush=True
            )
    if digest != controls["parameter_sha256"] or parameter_digest(backbone) != digest:
        raise ValueError("the frozen forecasting backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "new_queries": calls,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "evaluation_future_values_read": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / n for n in ("dynamic-inputs-v001", "dynamic-forecasts-v001", "dynamic-results-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete dynamic scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if fm["smoke"] or len(metadata) != 231 or set(metadata) != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("freeze all formal forecasts before reading outcomes")
    records, _ = sources()
    rows, targets, auxiliary = [], [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        target = (truth - d["mean"][:2]) / d["scale"][:2]
        error = np.where(valid[None], saved["points"] - target[None], 0)
        mae, mse = abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for index, name in enumerate(saved["methods"].tolist()):
            rows.append(
                {
                    **info,
                    "method": name,
                    "mae": float(mae[index].mean()),
                    "mse": float(mse[index].mean()),
                }
            )
            for slot in (0, 1):
                targets.append(
                    {
                        **info,
                        "method": name,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[index, slot]),
                        "mse": float(mse[index, slot]),
                    }
                )
        if row["panel"] == "synthetic_outage_h24":
            actual = records[row["station"]]["values"][t - 24 : t, :2]
            for label in ("static", "filtered", "smoothed"):
                e = (d[label + "_values"][-24:, :2] - actual) / d["scale"][:2]
                auxiliary.append(
                    {
                        **info,
                        "method": label,
                        "mae": float(abs(e).mean()),
                        "mse": float(np.square(e).mean()),
                    }
                )
    if len(rows) != 18711 or len(targets) != 37422:
        raise ValueError("the registered dynamic score count changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    pd.DataFrame(auxiliary).to_csv(output / "imputation_auxiliary.csv", index=False)
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
            "primary": "full_dynamic_mixture_median16",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
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
        from audit_dynamic_posterior import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
