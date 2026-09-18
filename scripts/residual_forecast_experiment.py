"""A fixed diagnostic separating shared covariate and target residual forecasts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import PARENT, ROOT, load_npz, read_json
from peer_outage_core import PrefixRegression, augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R33_RESIDUAL_FORECAST_PROTOCOL.md"
NEW_METHODS = ("residual_zero", "residual_last", "residual_ar1", "residual_tsfm")


def decompose(stats, data, forecast_z):
    x = data["context"]
    z = (x - stats.mean) / stats.scale
    available = np.flatnonzero(data["keep"] & (np.isfinite(x).sum(0) >= 2))
    available = [int(i) for i in available if i >= 2]
    residuals = np.full((len(x), 2), np.nan)
    component = np.zeros((len(forecast_z), 2))
    models, means, scales = [], [], []
    for target in (0, 1):
        fitted = stats.fit(target, available)
        features = fitted["features"]
        if features:
            beta = np.asarray(fitted["beta"])
            support = np.isfinite(stats.z[:, target]) & np.isfinite(stats.z[:, features]).all(1)
            prefix_residual = stats.z[:, target] - (beta[0] + stats.z[:, features] @ beta[1:])
            seen = np.isfinite(z[:, target]) & np.isfinite(z[:, features]).all(1)
            residuals[seen, target] = z[seen, target] - (beta[0] + z[seen][:, features] @ beta[1:])
            if not np.isfinite(forecast_z[:, features]).all():
                raise ValueError("a selected covariate forecast is unavailable")
            component[:, target] = beta[0] + forecast_z[:, features] @ beta[1:]
        else:
            support = np.isfinite(stats.z[:, target])
            prefix_residual = stats.z[:, target].copy()
            residuals[:, target] = z[:, target]
        prefix_residual = np.where(support, prefix_residual, np.nan)
        mean = float(np.nanmean(prefix_residual))
        scale = float(np.nanstd(prefix_residual, ddof=0))
        scale = 1.0 if scale <= 1e-12 else scale
        centered = prefix_residual - mean
        paired = np.isfinite(centered[:-1]) & np.isfinite(centered[1:])
        left, right = centered[:-1][paired], centered[1:][paired]
        phi = float(np.clip(left @ right / max(left @ left, 1e-12), 0, 0.99))
        models.append(
            {
                **fitted,
                "available_features": available,
                "residual_mean": mean,
                "residual_scale": scale,
                "residual_ar1": phi,
                "residual_prefix_support": int(support.sum()),
                "residual_history_support": int(np.isfinite(residuals[:, target]).sum()),
            }
        )
        means.append(mean)
        scales.append(scale)
    means, scales = np.asarray(means), np.asarray(scales)
    return {
        "context": x,
        "mean": stats.mean,
        "scale": stats.scale,
        "forecast_covariates_z": forecast_z,
        "component": component,
        "residuals": residuals,
        "residual_z": (residuals - means) / scales,
        "residual_means": means,
        "residual_scales": scales,
        "residual_keep": np.isfinite(residuals).sum(0) >= 2,
        "models": np.asarray(json.dumps(models)),
    }


def simple_residuals(data, horizon):
    last = np.zeros((horizon, 2))
    ar = np.zeros_like(last)
    for target, model in enumerate(json.loads(str(data["models"]))):
        indices = np.flatnonzero(np.isfinite(data["residuals"][:, target]))
        if not len(indices):
            continue
        value = data["residuals"][indices[-1], target]
        last[:, target] = value
        distance = len(data["residuals"]) - indices[-1] + np.arange(horizon)
        ar[:, target] = model["residual_mean"] + model["residual_ar1"] ** distance * (
            value - model["residual_mean"]
        )
    return last, ar


def prepare(base, smoke):
    output = base / "residual-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed residual inputs")
    records, peers = sources()
    full = {s: augmented(s, records, peers)[0] for s in records}
    stats = {s: PrefixRegression(full[s][: r["prefix_end"]]) for s, r in records.items()}
    parent = read_json(PARENT / "peer-inputs-v001/manifest.json")
    forecast = read_json(PARENT / "peer-forecasts-v001/manifest.json")
    old_entries = {r["case_id"]: r for r in forecast["cases"]}
    prepared = []
    for row in parent["cases"]:
        d = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        model = stats[row["station"]]
        np.testing.assert_array_equal(d["mean"], model.mean)
        np.testing.assert_array_equal(d["scale"], model.scale)
        entry = old_entries[row["case_id"]]
        old = load_npz(PARENT / "peer-forecasts-v001" / entry["path"], entry["sha256"])
        query = next(q for q in json.loads(str(old["queries"])) if q["name"] == "native_peer")
        path = PARENT / "peer-forecasts-v001/queries" / f"{query['key']}.npz"
        raw = load_npz(path, query["sha256"])
        future = np.full((row["horizon"], 17), np.nan)
        future[:, query["columns"]] = raw["quantiles"][
            :, forecast["quantiles"].index(0.5), : row["horizon"]
        ].T
        data = decompose(model, d, future)
        meta = {
            k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon", "prefix_end")
        }
        meta.update(
            parent_input_path=row["path"],
            parent_input_sha256=row["sha256"],
            parent_forecast=entry,
            native_query={**query, "path": str(path)},
            residual_support=np.isfinite(data["residuals"]).sum(0).tolist(),
            outage_age=row.get("outage_age"),
        )
        prepared.append((meta, data))
    if smoke:
        selected = {
            r["case_id"]
            for panel in ("natural_outage_h24", "synthetic_outage_h24", "legacy_native_h96")
            for r, _ in [v for v in prepared if v[0]["panel"] == panel][:2]
        }
        selected.add(min(prepared, key=lambda v: min(v[0]["residual_support"]))[0]["case_id"])
        natural = [v for v in prepared if v[0]["panel"] == "natural_outage_h24"]
        selected.add(max(natural, key=lambda v: v[0]["outage_age"])[0]["case_id"])
        prepared = [v for v in prepared if v[0]["case_id"] in selected]
    entries = []
    for row, data in prepared:
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(path, **data)
        entries.append({**row, "path": str(path.relative_to(output)), "sha256": file_sha256(path)})
    fits = []
    for station, model in stats.items():
        path = output / "regressions" / f"{station}.json"
        model.save(path)
        fits.append(
            {"station": station, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "regression_fits": fits,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    PROTOCOL,
                    ROOT / "scripts/peer_outage_core.py",
                    PARENT / "peer-inputs-v001/manifest.json",
                    PARENT / "peer-forecasts-v001/manifest.json",
                )
            },
            "parameter_sha256": forecast["parameter_sha256"],
            "evaluation_future_values_read": False,
        },
    )


def get_backbone():
    torch.set_num_threads(1)
    _, adapter, model, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    return adapter._ensure_backend(), model, digest


def forecast(base):
    inputs, output = base / "residual-inputs-v001", base / "residual-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed residual forecasts")
    prepared = read_json(inputs / "manifest.json")
    pipeline, backbone, digest = get_backbone()
    mid = pipeline.quantiles.index(0.5)
    entries, calls = [], 0
    for row in prepared["cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        parent = row["parent_forecast"]
        old = load_npz(PARENT / "peer-forecasts-v001" / parent["path"], parent["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        horizon = row["horizon"]
        selected = np.flatnonzero(d["residual_keep"])
        canonical = np.array(d["residual_z"][:, selected].T, dtype=np.float32, order="C")
        quantiles = np.empty(
            (
                0,
                len(pipeline.quantiles),
                int(np.ceil(horizon / pipeline.model_output_patch_size))
                * pipeline.model_output_patch_size,
            ),
            np.float32,
        )
        predicted = np.zeros((horizon, 2))
        if len(selected):
            with torch.inference_mode():
                quantiles = (
                    backbone(
                        context=torch.tensor(canonical, device="cuda"),
                        group_ids=torch.zeros(len(selected), dtype=torch.long, device="cuda"),
                        num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            if not np.isfinite(quantiles).all():
                raise ValueError("residual forecasting produced nonfinite values")
            predicted[:, selected] = (
                quantiles[:, mid, :horizon].T.astype(float) * d["residual_scales"][selected]
                + d["residual_means"][selected]
            )
            calls += 1
        last, ar = simple_residuals(d, horizon)
        for name, change in (
            ("residual_zero", np.zeros_like(predicted)),
            ("residual_last", last),
            ("residual_ar1", ar),
            ("residual_tsfm", predicted),
        ):
            methods[name] = d["component"] + change
        names = sorted(methods)
        points = np.stack([methods[n] for n in names])
        if len(names) != 41 or not np.isfinite(points).all():
            raise ValueError("the registered residual forecast panel is incomplete")
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=points,
            context_z=canonical,
            residual_targets=selected,
            quantiles=quantiles,
            predicted_residual=predicted,
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json", {"completed": len(entries), "total": len(prepared["cases"])}
        )
    if digest != prepared["parameter_sha256"] or parameter_digest(backbone) != digest:
        raise ValueError("the forecasting backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "new_model_calls": calls,
            "reused_native_calls": len(entries),
            "evaluation_future_values_read": False,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
        },
    )


def evaluate(base):
    inputs, forecast_root, output = (
        base / n
        for n in ("residual-inputs-v001", "residual-forecasts-v001", "residual-results-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed residual results")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecast_root / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if fm["smoke"] or len(metadata) != 231 or set(metadata) != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("freeze all formal predictions before scoring")
    records, _ = sources()
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(inputs / row["path"], row["sha256"])
        pred = load_npz(forecast_root / entry["path"], entry["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        truth = (truth - data["mean"][:2]) / data["scale"][:2]
        error = np.where(valid[None], pred["points"] - truth[None], 0)
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
    if len(rows) != 9471 or len(targets) != 18942:
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
            "primary": "residual_tsfm",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
        },
    )


def audit(base):
    # Imported here so the numerical verifier remains a separately readable implementation.
    from audit_residual_forecast import audit as verify

    verify(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "forecast", "evaluate", "audit"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    base = args.run_root.resolve()
    if args.phase == "prepare":
        prepare(base, args.smoke)
    else:
        {"forecast": forecast, "evaluate": evaluate, "audit": audit}[args.phase](base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
