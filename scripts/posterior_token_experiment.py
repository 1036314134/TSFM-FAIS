"""Fixed posterior-moment input encoding with matched hybrid and Monte Carlo controls."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from dynamic_posterior_core import predictive_mixture
from forecast_calibration_core import ROOT, load_npz, read_json
from normalization_interface import constant_statistics
from peer_outage_core import sources
from posterior_token_core import (
    conditional_uncertainty,
    mc_embedding,
    moment_embedding,
    posterior_normalization,
    predicted_quantiles,
    static_samples,
    transformed_moments,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PARENT = ROOT / "artifacts/iclr27-r36"
STATIC = ROOT / "artifacts/iclr27-r35"
PROTOCOL = ROOT / "docs/iclr2027/R37_POSTERIOR_MOMENT_PROTOCOL.md"
MODES = (
    "posterior_norm",
    "posterior_mean",
    "posterior_variance",
    "posterior_moment",
    "unconditional_moment",
    "mc_embedding",
)


def prepare(base, smoke):
    output = base / "moment-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete moment inputs")
    parent = read_json(PARENT / "covariance-inputs-v001/manifest.json")
    models = {
        r["station"]: load_npz(PARENT / "covariance-inputs-v001" / r["path"], r["sha256"])
        for r in parent["evaluation_models"]
    }
    rows = parent["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(PARENT / "smoke-v001/covariance-inputs-v001/manifest.json")["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    entries = []
    for row in rows:
        d = load_npz(PARENT / "covariance-inputs-v001" / row["path"], row["sha256"])
        start = perf_counter()
        covariance, roots, unconditional = conditional_uncertainty(
            d["context"], models[row["station"]]["covariance"], d["keep"]
        )
        conditioning_seconds = perf_counter() - start
        selected = np.flatnonzero(d["keep"])
        mean_z = np.array(
            ((d["base_values"][:, selected] - d["mean"][selected]) / d["scale"][selected]).T,
            dtype=np.float32,
            order="C",
        )
        observed = np.isfinite(d["context"][:, selected]).T
        start = perf_counter()
        samples, seed = static_samples(mean_z, roots, observed, row["case_id"])
        sampling_seconds = perf_counter() - start
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            mean_z=mean_z,
            observed=observed,
            covariance=covariance,
            roots=roots,
            variance=np.diagonal(covariance, axis1=1, axis2=2).T.copy(),
            unconditional_variance=unconditional,
            samples_z=samples,
            seed=np.asarray(seed),
            mean=d["mean"],
            scale=d["scale"],
            keep=d["keep"],
        )
        entries.append(
            {
                **{
                    k: row[k]
                    for k in ("case_id", "panel", "station", "origin", "horizon", "prefix_end")
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "parent_path": row["path"],
                "parent_sha256": row["sha256"],
                "conditioning_and_reference_factor_seconds": conditioning_seconds,
                "reference_sampling_seconds": sampling_seconds,
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
                    Path(__file__),
                    PROTOCOL,
                    ROOT / "scripts/posterior_token_core.py",
                    ROOT / "scripts/normalization_interface.py",
                    ROOT / "scripts/dynamic_posterior_core.py",
                    PARENT / "covariance-inputs-v001/manifest.json",
                    PARENT / "covariance-forecasts-v001/manifest.json",
                    PARENT / "covariance-training-v001/fixed_portfolios.json",
                    STATIC / "dynamic-forecasts-v001/manifest.json",
                )
            },
            "evaluation_future_values_read": False,
        },
    )


def fixed_controls(old, static_q, levels, base_names):
    methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
    var = methods["linear_var_direct"]
    for name in [
        *[n for n in base_names if n not in ("full_static_point", "linear_var_direct")],
        "forecast_covariance",
        "imputation_covariance_targets",
        "imputation_covariance_all",
    ]:
        methods["half_var_" + name] = 0.5 * methods[name] + 0.5 * var
    mean, median = predictive_mixture(static_q[None], levels)
    for label, value in (("mean", mean), ("median", median)):
        methods["static_quantile_" + label] = value
        methods["hybrid_static_quantile_" + label] = 0.5 * value + 0.5 * var
    return methods


def forecast(base):
    inputs, output = base / "moment-inputs-v001", base / "moment-forecasts-v001"
    if (output / "identity.json").exists():
        raise ValueError("preserve completed and partial moment forecast runs")
    prepared = read_json(inputs / "manifest.json")
    parent = read_json(PARENT / "covariance-forecasts-v001/manifest.json")
    old_entries = {r["case_id"]: r for r in parent["cases"]}
    static_parent = read_json(STATIC / "dynamic-forecasts-v001/manifest.json")
    static_entries = {r["case_id"]: r for r in static_parent["cases"]}
    base_names = read_json(PARENT / "covariance-training-v001/fixed_portfolios.json")["methods"]
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    block, norm = backbone.input_patch_embedding, backbone.instance_norm
    if backbone.config.dense_act_fn != "relu" or block.use_layer_norm or not norm.use_arcsinh:
        raise ValueError("the registered ReLU/arcsinh architecture changed")
    identity = {
        "input_sha256": file_sha256(inputs / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "core_sha256": file_sha256(ROOT / "scripts/posterior_token_core.py"),
        "parameter_sha256": digest,
    }
    _write_json(output / "identity.json", identity)
    entries, method_calls, ordinary_calls = [], 0, 0
    mid = pipeline.quantiles.index(0.5)
    for row in prepared["cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        parent_entry = old_entries[row["case_id"]]
        old = load_npz(
            PARENT / "covariance-forecasts-v001" / parent_entry["path"], parent_entry["sha256"]
        )
        static_entry = static_entries[row["case_id"]]
        old_static = load_npz(
            STATIC / "dynamic-forecasts-v001" / static_entry["path"], static_entry["sha256"]
        )
        static_query = next(
            q for q in json.loads(str(old_static["queries"])) if q["name"] == "full_static"
        )
        native = load_npz(
            STATIC / "dynamic-forecasts-v001" / static_query["path"], static_query["sha256"]
        )
        np.testing.assert_array_equal(d["mean_z"], native["context_z"])
        h = row["horizon"]
        methods = fixed_controls(
            old, native["quantiles"][:2, :, :h].transpose(1, 2, 0), pipeline.quantiles, base_names
        )
        context = torch.tensor(d["mean_z"], device="cuda")
        variance = torch.tensor(d["variance"], dtype=torch.float64, device="cuda")
        samples = torch.tensor(d["samples_z"], device="cuda")
        requests = []
        with torch.inference_mode():
            torch.cuda.synchronize()
            begin = perf_counter()
            loc, scale, _, _ = posterior_normalization(norm, context, variance)
            value_mean, value_var = transformed_moments(norm, context, variance, loc, scale)
            with constant_statistics(norm, loc, scale):
                patches, _, _ = backbone._prepare_patched_context(context)
            torch.cuda.synchronize()
            common_seconds = perf_counter() - begin
            for mode in MODES:
                torch.cuda.synchronize()
                begin = perf_counter()
                selected_loc, selected_scale = loc, scale
                replacement = None
                if mode == "unconditional_moment":
                    prior_var = torch.tensor(
                        d["unconditional_variance"], dtype=torch.float64, device="cuda"
                    )
                    selected_loc, selected_scale, _, _ = posterior_normalization(
                        norm, context, prior_var
                    )
                    mean_value, var_value = transformed_moments(
                        norm, context, prior_var, selected_loc, selected_scale
                    )
                    with constant_statistics(norm, selected_loc, selected_scale):
                        p, _, _ = backbone._prepare_patched_context(context)
                    replacement = moment_embedding(block, p, mean_value, var_value, "moment")
                elif mode == "mc_embedding":
                    normalized, _ = norm.forward(samples, (loc, scale))
                    replacement = mc_embedding(block, patches, normalized, value_var)
                elif mode != "posterior_norm":
                    kind = {
                        "posterior_mean": "mean_only",
                        "posterior_variance": "variance_only",
                        "posterior_moment": "moment",
                    }[mode]
                    replacement = moment_embedding(block, patches, value_mean, value_var, kind)
                torch.cuda.synchronize()
                prepare_seconds = perf_counter() - begin
                begin = perf_counter()
                q = predicted_quantiles(
                    backbone,
                    context,
                    h,
                    pipeline.model_output_patch_size,
                    selected_loc,
                    selected_scale,
                    replacement,
                )
                forward_seconds = perf_counter() - begin
                path = output / "queries" / f"{row['case_id']}-{mode}.npz"
                _save_npz(
                    path,
                    quantiles=q,
                    loc=selected_loc.cpu().numpy(),
                    scale=selected_scale.cpu().numpy(),
                    replacement=np.empty(0, np.float32)
                    if replacement is None
                    else replacement.cpu().numpy(),
                )
                requests.append(
                    {
                        "name": mode,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "common_setup_seconds": common_seconds,
                        "method_setup_seconds": prepare_seconds,
                        "forward_seconds": forward_seconds,
                    }
                )
                point = q[:2, mid, :h].T.astype(float)
                methods[mode + "_component"] = point
                methods["hybrid_" + mode] = 0.5 * point + 0.5 * methods["linear_var_direct"]
                method_calls += 1
            sample_quantiles = []
            for index, sample in enumerate(samples):
                begin = perf_counter()
                q = predicted_quantiles(
                    backbone, sample, h, pipeline.model_output_patch_size, loc, scale
                )
                path = output / "queries" / f"{row['case_id']}-mc_sample_{index:02d}.npz"
                _save_npz(
                    path,
                    quantiles=q,
                    loc=loc.cpu().numpy(),
                    scale=scale.cpu().numpy(),
                    replacement=np.empty(0, np.float32),
                )
                requests.append(
                    {
                        "name": f"mc_sample_{index:02d}",
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "forward_seconds": perf_counter() - begin,
                    }
                )
                sample_quantiles.append(q[:2, :, :h].transpose(1, 2, 0))
                method_calls += 1
            samples_q = np.stack(sample_quantiles)
            mean, median = predictive_mixture(samples_q, pipeline.quantiles)
            combined = {
                "mc_point_mean16": samples_q[:, mid].mean(0, dtype=np.float64),
                "mc_point_median16": np.median(samples_q[:, mid], 0).astype(float),
                "mc_quantile_mean16": mean,
                "mc_quantile_median16": median,
            }
            for name, point in combined.items():
                methods[name + "_component"] = point
                methods["hybrid_" + name] = 0.5 * point + 0.5 * methods["linear_var_direct"]
            restored = predicted_quantiles(backbone, context, h, pipeline.model_output_patch_size)
            np.testing.assert_array_equal(restored, native["quantiles"])
            ordinary_calls += 1
        names = sorted(methods)
        points = np.stack([methods[n] for n in names]).astype(float)
        if len(names) != 140 or not np.isfinite(points).all():
            raise ValueError("the registered 140-method posterior-moment panel is incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path, methods=np.asarray(names), points=points, queries=np.asarray(json.dumps(requests))
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "ordinary_replay_difference": 0,
            }
        )
        _write_json(
            output / "progress.json",
            {
                "predicted": len(entries),
                "total": len(prepared["cases"]),
                "new_method_calls": method_calls,
            },
        )
        if len(entries) % 25 == 0:
            print(
                json.dumps({"predicted": len(entries), "total": len(prepared["cases"])}), flush=True
            )
    if digest != parent["parameter_sha256"] or parameter_digest(backbone) != digest:
        raise ValueError("the forecasting backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "identity": identity,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "new_method_calls": method_calls,
            "ordinary_verification_calls": ordinary_calls,
            "evaluation_future_values_read": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / n for n in ("moment-inputs-v001", "moment-forecasts-v001", "moment-results-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete moment scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if fm["smoke"] or len(metadata) != 231 or set(metadata) != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("complete all formal moment forecasts before scoring")
    records, _ = sources()
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        t, h = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + h, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < h // 2).any():
            raise ValueError("registered outcome support changed")
        target = (truth - data["mean"][:2]) / data["scale"][:2]
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
    if len(rows) != 32340 or len(targets) != 64680:
        raise ValueError("registered posterior-moment score count changed")
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
            "primary": "hybrid_posterior_moment",
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
        from audit_posterior_token import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
