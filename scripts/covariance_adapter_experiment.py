"""Source-supervised bounded conditional covariance adaptation and matched controls."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from covariance_adapter_core import (
    CovarianceAdapter,
    forecast_mix,
    geometry,
    masked_smooth_mae,
    repaired_context,
    static_values,
)
from dynamic_posterior_core import condition_state, fit_dynamics
from forecast_calibration_core import ROOT, calibration_sources, fixed_mae_fit, load_npz, read_json
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

SOURCE = ROOT / "artifacts/iclr27-r32/calibration-inputs-v001"
SOURCE_FORECASTS = ROOT / "artifacts/iclr27-r32/calibration-forecasts-v001"
BASELINE = ROOT / "artifacts/iclr27-r35"
PROTOCOL = ROOT / "docs/iclr2027/R36_COVARIANCE_ADAPTER_PROTOCOL.md"
LEARNED = ("forecast_covariance", "imputation_covariance_targets", "imputation_covariance_all")


def static_model(dynamic):
    return {
        "mean": dynamic["mean"],
        "scale": dynamic["scale"],
        "center": dynamic["initial_mean"],
        "covariance": dynamic["initial_covariance"],
        "support": dynamic["complete_rows"],
    }


def direct_var(z, model, horizon):
    mean, covariance = model["initial_mean"].copy(), model["initial_covariance"].copy()
    for t, row in enumerate(z):
        if t:
            mean = model["a"] @ mean + model["b"]
            covariance = model["a"] @ covariance @ model["a"].T + model["q"]
        mean, covariance = condition_state(mean, covariance, row)
    result = []
    for _ in range(horizon):
        mean = model["a"] @ mean + model["b"]
        result.append(mean[:2].copy())
    return np.stack(result)


def prepare(base, smoke):
    output = base / "covariance-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed covariance preparation")
    source_manifest = read_json(SOURCE / "manifest.json")
    rows = source_manifest["cases"]
    if smoke:
        rows = [
            r
            for station in sorted({r["station"] for r in rows})
            for i, r in enumerate([v for v in rows if v["station"] == station])
            if i in (0, 4, 8)
        ]
    records, peers, _ = calibration_sources()
    full = {s: augmented(s, records, peers)[0] for s in records}
    source_models, source_fits = {}, []
    for station, record in sorted(records.items()):
        dynamic = fit_dynamics(full[station][: record["prefix_end"]])
        model = static_model(dynamic)
        source_models[station] = model, dynamic
        path = output / "source-models" / f"{station}.npz"
        dynamic_path = output / "source-models" / f"{station}-var.npz"
        _save_npz(path, **model)
        _save_npz(dynamic_path, **dynamic)
        source_fits.append(
            {
                "station": station,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "var_path": str(dynamic_path.relative_to(output)),
                "var_sha256": file_sha256(dynamic_path),
                "prefix_end": record["prefix_end"],
            }
        )
    source_cases = []
    for row in rows:
        old = load_npz(SOURCE / row["path"], row["sha256"])
        model, dynamic = source_models[row["station"]]
        np.testing.assert_array_equal(model["mean"], old["mean"])
        np.testing.assert_array_equal(model["scale"], old["scale"])
        context = old["context"]
        values = static_values(context, model)
        path = output / "source" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            context=context,
            mean=model["mean"],
            scale=model["scale"],
            keep=old["keep"],
            base_values=values,
            direct_var=direct_var((context - model["mean"]) / model["scale"], dynamic, 24),
        )
        t = row["origin"]
        if t - 192 < row["prefix_end"] or t + 24 > row["calibration_end"]:
            raise ValueError("a calibration label crosses the registered temporal boundary")
        truth = full[row["station"]][t - 192 : t]
        artificial = np.isfinite(truth) & ~np.isfinite(context)
        hidden = np.where(artificial, (truth - model["mean"]) / model["scale"], np.nan)
        old_labels = load_npz(SOURCE / row["label_path"], row["label_sha256"])
        future = (old_labels["future"] - model["mean"][:2]) / model["scale"][:2]
        label_path = output / "labels" / f"{row['case_id']}.npz"
        _save_npz(label_path, hidden_all=hidden, future=future, artificial_mask=artificial)
        source_cases.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "case_id",
                        "station",
                        "origin",
                        "horizon",
                        "prefix_end",
                        "calibration_end",
                        "outage_pattern",
                        "outage_age",
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "label_path": str(label_path.relative_to(output)),
                "label_sha256": file_sha256(label_path),
                "parent_input_path": row["path"],
                "parent_input_sha256": row["sha256"],
            }
        )
    prior_inputs = read_json(BASELINE / "dynamic-inputs-v001/manifest.json")
    eval_rows = prior_inputs["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(BASELINE / "smoke-v001/dynamic-inputs-v001/manifest.json")["cases"]
        }
        eval_rows = [r for r in eval_rows if r["case_id"] in ids]
    evaluation_fits = []
    for row in prior_inputs["models"]:
        dynamic = load_npz(BASELINE / "dynamic-inputs-v001" / row["path"], row["sha256"])
        path = output / "evaluation-models" / f"{row['station']}.npz"
        _save_npz(path, **static_model(dynamic))
        evaluation_fits.append(
            {
                "station": row["station"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "parent_path": row["path"],
                "parent_sha256": row["sha256"],
            }
        )
    evaluation = []
    for row in eval_rows:
        old = load_npz(BASELINE / "dynamic-inputs-v001" / row["path"], row["sha256"])
        path = output / "evaluation" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            context=old["context"],
            mean=old["mean"],
            scale=old["scale"],
            keep=old["keep"],
            base_values=old["static_values"],
        )
        evaluation.append(
            {
                **{
                    k: row[k]
                    for k in ("case_id", "panel", "station", "origin", "horizon", "prefix_end")
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "parent_path": row["path"],
                "parent_sha256": row["sha256"],
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "source_cases": source_cases,
            "source_models": source_fits,
            "cases": evaluation,
            "evaluation_models": evaluation_fits,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    Path(__file__),
                    PROTOCOL,
                    ROOT / "scripts/covariance_adapter_core.py",
                    ROOT / "scripts/dynamic_posterior_core.py",
                    SOURCE / "manifest.json",
                    SOURCE_FORECASTS / "manifest.json",
                    BASELINE / "dynamic-inputs-v001/manifest.json",
                    BASELINE / "dynamic-forecasts-v001/manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def backbone_and_pipeline():
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    backbone.eval().requires_grad_(False)
    return adapter._ensure_backend(), backbone, digest


def source_forecast(base):
    inputs, output = base / "covariance-inputs-v001", base / "covariance-source-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete covariance source forecasts")
    prepared = read_json(inputs / "manifest.json")
    old_entries = {r["case_id"]: r for r in read_json(SOURCE_FORECASTS / "manifest.json")["cases"]}
    pipeline, backbone, digest = backbone_and_pipeline()
    entries = []
    for row in prepared["source_cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        old_entry = old_entries[row["case_id"]]
        old = load_npz(SOURCE_FORECASTS / old_entry["path"], old_entry["sha256"])
        methods = {
            n: old["points"][i]
            for i, n in enumerate(old["methods"].tolist())
            if "_source_" not in n
        }
        contexts = []
        for scope, name in (("full", "full_static_point"), ("target", "target_static_point")):
            raw = d["context"].copy()
            if scope == "full":
                raw[:] = d["base_values"]
            else:
                raw[:, :2] = d["base_values"][:, :2]
            canonical = np.array(
                ((raw[:, d["keep"]] - d["mean"][d["keep"]]) / d["scale"][d["keep"]]).T,
                dtype=np.float32,
                order="C",
            )
            with torch.inference_mode():
                point = (
                    chronos_median(pipeline, torch.tensor(canonical.T, device="cuda"), 24, [0, 1])
                    .cpu()
                    .numpy()
                )
            methods[name] = point
            contexts.append(canonical)
        methods["linear_var_direct"] = d["direct_var"]
        names = sorted(methods)
        if len(names) != 28:
            raise ValueError("the 28-output source portfolio changed")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([methods[n] for n in names]).astype(float),
            contexts_z=np.stack(contexts),
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json",
            {"completed": len(entries), "total": len(prepared["source_cases"])},
        )
    if parameter_digest(backbone) != digest:
        raise ValueError("source forecasting changed backbone parameters")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "parameter_sha256": digest,
            "new_model_calls": 2 * len(entries),
            "evaluation_future_values_read": False,
        },
    )


def train(base):
    inputs, forecast_root, output = (
        base / n
        for n in (
            "covariance-inputs-v001",
            "covariance-source-forecasts-v001",
            "covariance-training-v001",
        )
    )
    if (output / "initial_state.json").exists():
        raise ValueError("preserve partial and completed covariance training")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecast_root / "manifest.json")
    rows = prepared["source_cases"]
    if not prepared["smoke"] and len(rows) != 864:
        raise ValueError("the registered source task count changed")
    models = {
        r["station"]: load_npz(inputs / r["path"], r["sha256"]) for r in prepared["source_models"]
    }
    source_entries = {r["case_id"]: r for r in fm["cases"]}
    if set(source_entries) != {r["case_id"] for r in rows}:
        raise ValueError("complete all matched source controls before training")
    _write_json(
        output / "initial_state.json",
        {
            "status": "started",
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "source_forecast_sha256": file_sha256(forecast_root / "manifest.json"),
        },
    )
    pipeline, backbone, digest = backbone_and_pipeline()
    samples, bank, truth, names = [], [], [], None
    for row in rows:
        d = load_npz(inputs / row["path"], row["sha256"])
        labels = load_npz(inputs / row["label_path"], row["label_sha256"])
        samples.append(
            {
                "geometry": geometry(d, models[row["station"]]),
                "future": torch.tensor(labels["future"], dtype=torch.float32, device="cuda"),
                "hidden": torch.tensor(labels["hidden_all"], dtype=torch.float32, device="cuda"),
            }
        )
        f = source_entries[row["case_id"]]
        forecasts = load_npz(forecast_root / f["path"], f["sha256"])
        current = forecasts["methods"].tolist()
        if names is not None and current != names:
            raise ValueError("source portfolio ordering changed")
        names = current
        bank.append(forecasts["points"])
        truth.append(labels["future"])
    bank, truth = np.stack(bank), np.stack(truth)
    fixed = {
        "methods": names,
        "global": [fixed_mae_fit(bank[:, :, :, t], truth[:, :, t]) for t in (0, 1)],
        "stations": {},
    }
    for station in sorted(models):
        selected = [i for i, r in enumerate(rows) if r["station"] == station]
        fixed["stations"][station] = [
            fixed_mae_fit(bank[selected, :, :, t], truth[selected, :, t]) for t in (0, 1)
        ]
    _write_json(output / "fixed_portfolios.json", fixed)
    order = np.random.default_rng(5101).permutation(len(rows))
    _save_npz(
        output / "source_order.npz",
        indices=order,
        case_ids=np.asarray([r["case_id"] for r in rows]),
    )
    learned, trace = [], []
    for name in LEARNED:
        net = CovarianceAdapter().cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.001)
        then = perf_counter()
        for step, index in enumerate(order):
            sample = samples[index]
            optimizer.zero_grad(set_to_none=True)
            if step == len(order) - 1:
                torch.save(
                    {
                        "model": copy.deepcopy(net.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "case_id": rows[index]["case_id"],
                        "step": step,
                    },
                    output / f"{name}-before-last.pt",
                )
            repaired = repaired_context(net, sample["geometry"])
            if name == "forecast_covariance":
                prediction = chronos_median(
                    pipeline, repaired[:, sample["geometry"]["keep"]], 24, [0, 1]
                )
                loss = masked_smooth_mae(prediction, sample["future"])
            elif name == "imputation_covariance_targets":
                loss = masked_smooth_mae(repaired[:, :2], sample["hidden"][:, :2])
            else:
                loss = masked_smooth_mae(repaired, sample["hidden"])
            data_loss = float(loss.detach())
            loss = loss + 0.001 * net.penalty()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            if net.correction.grad is None or not bool(torch.isfinite(loss)):
                raise ValueError("covariance adaptation produced an invalid gradient or loss")
            optimizer.step()
            trace.append(
                {
                    "method": name,
                    "step": step,
                    "case_id": rows[index]["case_id"],
                    "loss": float(loss.detach()),
                    "data_loss": data_loss,
                    "gradient_norm": float(norm),
                }
            )
            if (step + 1) % 48 == 0 or step + 1 == len(order):
                _write_json(
                    output / "progress.json",
                    {"method": name, "step": step + 1, "total": len(order)},
                )
                print(
                    json.dumps({"method": name, "step": step + 1, "total": len(order)}), flush=True
                )
        path = output / f"{name}.pt"
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict()}, path)
        learned.append(
            {
                "method": name,
                "path": path.name,
                "sha256": file_sha256(path),
                "before_last_sha256": file_sha256(output / f"{name}-before-last.pt"),
                "steps": len(order),
                "parameter_count": sum(p.numel() for p in net.parameters()),
                "seconds": perf_counter() - then,
            }
        )
    _write_json(output / "training_trace.json", trace)
    if parameter_digest(backbone) != digest or any(
        p.grad is not None for p in backbone.parameters()
    ):
        raise ValueError("the forecasting backbone was not frozen during training")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "learned": learned,
            "parameter_sha256": digest,
            "source_cases": len(rows),
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "source_forecast_sha256": file_sha256(forecast_root / "manifest.json"),
            "files": {
                p.name: file_sha256(p)
                for p in (
                    output / "fixed_portfolios.json",
                    output / "source_order.npz",
                    output / "training_trace.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def forecast(base):
    inputs, training, output = (
        base / n
        for n in ("covariance-inputs-v001", "covariance-training-v001", "covariance-forecasts-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed covariance forecasts")
    prepared, tm = read_json(inputs / "manifest.json"), read_json(training / "manifest.json")
    models = {
        r["station"]: load_npz(inputs / r["path"], r["sha256"])
        for r in prepared["evaluation_models"]
    }
    baseline = read_json(BASELINE / "dynamic-forecasts-v001/manifest.json")
    old_entries = {r["case_id"]: r for r in baseline["cases"]}
    fixed = read_json(training / "fixed_portfolios.json")
    pipeline, backbone, digest = backbone_and_pipeline()
    nets = {}
    for record in tm["learned"]:
        path = training / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a trained covariance adapter changed")
        net = CovarianceAdapter().cuda()
        net.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model"])
        nets[record["method"]] = net.eval()
    zero = CovarianceAdapter().cuda()
    entries = []
    for row in prepared["cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        g = geometry(d, models[row["station"]])
        old_entry = old_entries[row["case_id"]]
        old = load_npz(BASELINE / "dynamic-forecasts-v001" / old_entry["path"], old_entry["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        contexts, corrected = [], []
        with torch.inference_mode():
            baseline_input = repaired_context(zero, g)[:, g["keep"]].T.contiguous().cpu().numpy()
            original_query = next(
                q for q in json.loads(str(old["queries"])) if q["name"] == "full_static"
            )
            raw = load_npz(
                BASELINE / "dynamic-forecasts-v001" / original_query["path"],
                original_query["sha256"],
            )
            np.testing.assert_array_equal(baseline_input, raw["context_z"])
            for name, net in nets.items():
                context = repaired_context(net, g)[:, g["keep"]]
                methods[name] = (
                    chronos_median(pipeline, context, row["horizon"], [0, 1]).cpu().numpy()
                )
                contexts.append(context.T.contiguous().cpu().numpy())
                corrected.append(net(g["covariance"]).cpu().numpy())
        methods["half_static_var"] = (
            0.5 * methods["full_static_point"] + 0.5 * methods["linear_var_direct"]
        )
        bank = np.stack([methods[n] for n in fixed["methods"]])
        for name, weights in (
            ("covariance_portfolio_global", fixed["global"]),
            ("covariance_portfolio_station", fixed["stations"][row["station"]]),
        ):
            methods[name] = forecast_mix(bank, np.stack([w["weights"] for w in weights], axis=1))
        names = sorted(methods)
        points = np.stack([methods[n] for n in names]).astype(float)
        if len(names) != 87 or not np.isfinite(points).all():
            raise ValueError("the registered 87-method covariance panel is incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=points,
            adapted_methods=np.asarray(list(nets)),
            contexts_z=np.stack(contexts),
            covariances=np.stack(corrected),
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "zero_input_difference": 0,
            }
        )
        _write_json(
            output / "progress.json", {"predicted": len(entries), "total": len(prepared["cases"])}
        )
    if (
        digest != tm["parameter_sha256"]
        or digest != baseline["parameter_sha256"]
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("the forecasting backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "parameter_sha256": digest,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "training_sha256": file_sha256(training / "manifest.json"),
            "new_model_calls": 3 * len(entries),
            "evaluation_future_values_read": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / n
        for n in ("covariance-inputs-v001", "covariance-forecasts-v001", "covariance-results-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete covariance results")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if fm["smoke"] or len(metadata) != 231 or set(metadata) != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("complete formal covariance predictions before scoring")
    records, _ = sources()
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        pred = load_npz(forecasts / entry["path"], entry["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        expected = (truth - d["mean"][:2]) / d["scale"][:2]
        error = np.where(valid[None], pred["points"] - expected[None], 0)
        mae, mse = abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for index, name in enumerate(pred["methods"].tolist()):
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
    if len(rows) != 20097 or len(targets) != 40194:
        raise ValueError("registered covariance score count changed")
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
            "primary": "forecast_covariance",
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
        "--phase",
        choices=("prepare", "source_forecast", "train", "forecast", "evaluate", "audit"),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "prepare":
        prepare(base, args.smoke)
    elif args.phase == "audit":
        from audit_covariance_adapter import audit

        audit(base)
    else:
        {
            "source_forecast": source_forecast,
            "train": train,
            "forecast": forecast,
            "evaluate": evaluate,
        }[args.phase](base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
