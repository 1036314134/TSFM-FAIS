"""Verify dynamic conditional inputs, posterior factors, predictive mixtures and scores."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from dynamic_posterior_core import dense_conditional_tail
from forecast_calibration_core import PARENT, ROOT, load_npz, read_json
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_mixture(quantiles, levels):
    grid = np.rint(np.asarray(levels) * 1_000_000).astype(np.int64)
    np.testing.assert_allclose(grid / 1_000_000, levels, rtol=0, atol=1e-12)
    edges = np.r_[0, (grid[:-1] + grid[1:]) // 2, 1_000_000]
    weights = np.diff(edges)
    weights //= np.gcd.reduce(weights)
    values = np.sort(quantiles.astype(float), axis=1)
    nodes = values.reshape(-1, *values.shape[2:])
    mass = np.tile(weights, len(values))
    total = int(mass.sum())
    mean = (nodes * mass[:, None, None]).sum(0) / total
    median = np.empty(values.shape[2:])
    for step in range(median.shape[0]):
        for target in range(median.shape[1]):
            items = sorted(zip(nodes[:, step, target].tolist(), mass.tolist(), strict=True))
            accumulated = 0
            for index, (value, weight) in enumerate(items):
                accumulated += weight
                if 2 * accumulated >= total:
                    median[step, target] = (
                        (value + items[index + 1][0]) / 2 if 2 * accumulated == total else value
                    )
                    break
    return mean, median


def audit(base):
    inputs, forecasts, output = (
        base / n for n in ("dynamic-inputs-v001", "dynamic-forecasts-v001", "dynamic-audit-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed dynamic audits")
    started = perf_counter()
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("incomplete or changed dynamic prediction inputs")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a registered dynamic definition changed")
    if fm["script_sha256"] != file_sha256(ROOT / "scripts/dynamic_posterior_experiment.py"):
        raise ValueError("dynamic forecasting implementation changed")
    records, peers = sources()
    full = {s: augmented(s, records, peers)[0] for s in records}
    models = {}
    for entry in prepared["models"]:
        model = load_npz(inputs / entry["path"], entry["sha256"])
        raw = full[entry["station"]][: entry["prefix_end"]]
        mean, scale = np.nanmean(raw, 0), np.nanstd(raw, 0, ddof=0)
        scale = np.where(scale <= 1e-12, 1, scale)
        np.testing.assert_array_equal(mean, model["mean"])
        np.testing.assert_array_equal(scale, model["scale"])
        z = (raw - mean) / scale
        complete = np.isfinite(z).all(1)
        pair = complete[:-1] & complete[1:]
        x, y = z[:-1][pair], z[1:][pair]
        if (
            int(model["complete_rows"]) != complete.sum()
            or int(model["transition_pairs"]) != pair.sum()
        ):
            raise ValueError("original complete-case fitting support changed")
        if min(complete.sum(), pair.sum()) < 128:
            raise ValueError("the transition model lacks registered support")
        xm, ym = x.mean(0), y.mean(0)
        design = np.vstack([(x - xm) / np.sqrt(len(x)), np.sqrt(0.001) * np.eye(17)])
        target = np.vstack([(y - ym) / np.sqrt(len(y)), np.zeros((17, 17))])
        beta = np.linalg.lstsq(design, target, rcond=None)[0].T
        np.testing.assert_allclose(model["a_unconstrained"], beta, rtol=1e-8, atol=1e-10)
        radius = np.max(abs(np.linalg.eigvals(model["a_unconstrained"])))
        np.testing.assert_allclose(
            model["a"], model["a_unconstrained"] * min(1, 0.99 / max(radius, 1e-12)), rtol=0, atol=0
        )
        np.testing.assert_array_equal(model["b"], ym - model["a"] @ xm)
        residual = y - (x @ model["a"].T + model["b"])
        for key, values in (("q", residual), ("initial_covariance", z[complete])):
            covariance = np.cov(values, rowvar=False, bias=True)
            expected = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance)) + 1e-6 * np.eye(17)
            np.testing.assert_allclose(model[key], expected, rtol=1e-10, atol=1e-10)
            if np.linalg.eigvalsh(model[key]).min() <= 0:
                raise ValueError("a registered prior covariance is not positive definite")
        np.testing.assert_array_equal(model["initial_mean"], z[complete].mean(0))
        models[entry["station"]] = model
    baseline_root = ROOT / "artifacts/iclr27-r34/normalization-forecasts-v001"
    baseline = read_json(baseline_root / "manifest.json")
    old_entries = {r["case_id"]: r for r in baseline["cases"]}
    half_root = ROOT / "artifacts/iclr27-r31/calibrated-forecasts-v001"
    half_entries = {r["case_id"]: r for r in read_json(half_root / "manifest.json")["cases"]}
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if set(metadata) != {r["case_id"] for r in fm["cases"]} or (
        not fm["smoke"] and len(metadata) != 231
    ):
        raise ValueError("dynamic forecast population changed")
    scores = None
    if not fm["smoke"]:
        result = base / "dynamic-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("dynamic scoring provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, calls, max_dense_difference, rebuilt_scores = pipeline.quantiles.index(0.5), 0, 0.0, []
    for number, entry in enumerate(fm["cases"]):
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        original = load_npz(
            PARENT / "peer-inputs-v001" / row["parent_input_path"], row["parent_input_sha256"]
        )
        np.testing.assert_array_equal(d["context"], original["context"])
        expected_raw = full[row["station"]][row["origin"] - 192 : row["origin"]].copy()
        if row["panel"] == "synthetic_outage_h24":
            expected_raw[-24:, :2] = np.nan
        np.testing.assert_array_equal(d["context"], expected_raw)
        model = models[row["station"]]
        np.testing.assert_array_equal(d["mean"], model["mean"])
        np.testing.assert_array_equal(d["scale"], model["scale"])
        np.testing.assert_array_equal(d["z"], (expected_raw - model["mean"]) / model["scale"])
        np.testing.assert_array_equal(d["keep"], original["keep"])
        observed = np.isfinite(d["z"])
        for t in range(192):
            pm = (
                model["initial_mean"]
                if t == 0
                else model["a"] @ d["filtered_mean"][t - 1] + model["b"]
            )
            pc = (
                model["initial_covariance"]
                if t == 0
                else model["a"] @ d["filtered_covariance"][t - 1] @ model["a"].T + model["q"]
            )
            np.testing.assert_array_equal(pm, d["predicted_mean"][t])
            np.testing.assert_array_equal(pc, d["predicted_covariance"][t])
            obs, missing = np.flatnonzero(observed[t]), np.flatnonzero(~observed[t])
            for label, prior_mean, prior_cov in (
                ("filtered", pm, pc),
                ("static", model["initial_mean"], model["initial_covariance"]),
            ):
                expected_mean, expected_cov = prior_mean.copy(), prior_cov.copy()
                if len(obs):
                    expected_mean[obs] = d["z"][t, obs]
                    expected_cov[:] = 0
                    if len(missing):
                        cross = prior_cov[np.ix_(missing, obs)]
                        obs_cov = prior_cov[np.ix_(obs, obs)]
                        expected_mean[missing] += cross @ np.linalg.solve(
                            obs_cov, d["z"][t, obs] - prior_mean[obs]
                        )
                        expected_cov[np.ix_(missing, missing)] = prior_cov[
                            np.ix_(missing, missing)
                        ] - cross @ np.linalg.solve(obs_cov, cross.T)
                np.testing.assert_allclose(
                    d[label + "_mean"][t], expected_mean, rtol=1e-9, atol=1e-10
                )
                if label == "filtered":
                    np.testing.assert_allclose(
                        d["filtered_covariance"][t], expected_cov, rtol=1e-9, atol=1e-10
                    )
            root = d["conditional_roots"][t]
            if t == 191:
                conditional_cov = d["filtered_covariance"][t]
                np.testing.assert_array_equal(d["smoothed_mean"][t], d["filtered_mean"][t])
            else:
                gain = np.linalg.solve(
                    d["predicted_covariance"][t + 1], model["a"] @ d["filtered_covariance"][t]
                ).T
                # Production assigns the solve result into a contiguous time-indexed array.
                gain = np.ascontiguousarray(gain)
                np.testing.assert_array_equal(d["backward_gain"][t], gain)
                smooth = d["filtered_mean"][t] + gain @ (
                    d["smoothed_mean"][t + 1] - d["predicted_mean"][t + 1]
                )
                smooth[observed[t]] = d["z"][t, observed[t]]
                np.testing.assert_array_equal(d["smoothed_mean"][t], smooth)
                conditional_cov = (
                    d["filtered_covariance"][t] - gain @ d["predicted_covariance"][t + 1] @ gain.T
                )
            np.testing.assert_allclose(root @ root.T, conditional_cov, rtol=1e-9, atol=1e-10)
            np.testing.assert_array_equal(root[observed[t]], np.zeros((observed[t].sum(), 17)))
        dense_mean, dense_cov = dense_conditional_tail(
            d["predicted_mean"][-12],
            d["predicted_covariance"][-12],
            model["a"],
            model["b"],
            model["q"],
            d["z"][-12:],
        )
        np.testing.assert_allclose(dense_mean, d["smoothed_mean"][-12:], rtol=1e-9, atol=1e-10)
        block_cov = np.zeros((12, 17, 12, 17))
        block_cov[-1, :, -1, :] = d["conditional_roots"][-1] @ d["conditional_roots"][-1].T
        for t in range(10, -1, -1):
            gain, root = d["backward_gain"][180 + t], d["conditional_roots"][180 + t]
            block_cov[t, :, t, :] = root @ root.T + gain @ block_cov[t + 1, :, t + 1, :] @ gain.T
            for later in range(t + 1, 12):
                block_cov[t, :, later, :] = gain @ block_cov[t + 1, :, later, :]
                block_cov[later, :, t, :] = block_cov[t, :, later, :].T
        np.testing.assert_allclose(block_cov.reshape(204, 204), dense_cov, rtol=1e-9, atol=1e-10)
        max_dense_difference = max(
            max_dense_difference,
            float(abs(dense_mean - d["smoothed_mean"][-12:]).max()),
            float(abs(block_cov.reshape(204, 204) - dense_cov).max()),
        )
        seed_hex = hashlib.sha256(("r35|6103|" + row["case_id"]).encode()).hexdigest()[:16]
        if str(d["seed_hex"]) != seed_hex:
            raise ValueError("the conditional simulation seed changed")
        noise = np.random.default_rng(int(seed_hex, 16)).standard_normal((8, 192, 17))
        deviation = np.zeros_like(noise)
        deviation[:, -1] = noise[:, -1] @ d["conditional_roots"][-1].T
        for t in range(190, -1, -1):
            deviation[:, t] = (
                deviation[:, t + 1] @ d["backward_gain"][t].T
                + noise[:, t] @ d["conditional_roots"][t].T
            )
        deviation[:, observed] = 0
        paired = np.stack([deviation, -deviation], axis=1).reshape(16, 192, 17)
        draws = (d["smoothed_mean"][None] + paired) * d["scale"] + d["mean"]
        draws[:, observed] = d["context"][observed]
        np.testing.assert_array_equal(draws, d["sampled_values"])
        for label in ("static", "filtered", "smoothed"):
            raw = d[label + "_mean"] * d["scale"] + d["mean"]
            raw[observed] = d["context"][observed]
            np.testing.assert_array_equal(raw, d[label + "_values"])
        np.testing.assert_allclose(
            ((draws[0::2] + draws[1::2]) / 2 - d["smoothed_values"]) / d["scale"],
            0,
            rtol=0,
            atol=1e-12,
        )
        old_entry = old_entries[row["case_id"]]
        old = load_npz(baseline_root / old_entry["path"], old_entry["sha256"])
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        pred = load_npz(forecasts / entry["path"], entry["sha256"])
        query_outputs = {}
        for query in json.loads(str(pred["queries"])):
            saved = load_npz(forecasts / query["path"], query["sha256"])
            name = query["name"]
            if name == "half_input_mix":
                canonical = ((original["context"] - original["mean"]) / original["scale"]).astype(
                    np.float32
                )
                names = original["stat_names"].tolist()
                local = (
                    (original["stat_targets"][names.index("local_ridge")] - original["mean"][:2])
                    / original["scale"][:2]
                ).astype(np.float32)
                peer = (
                    (original["stat_targets"][names.index("peer_ridge")] - original["mean"][:2])
                    / original["scale"][:2]
                ).astype(np.float32)
                canonical[:, :2] = np.where(
                    np.isfinite(canonical[:, :2]), canonical[:, :2], 0.5 * local + 0.5 * peer
                )
                canonical = np.ascontiguousarray(canonical[:, original["keep"]].T)
            else:
                scope, label = name.split("_", 1)
                values = (
                    d["sampled_values"][int(label.split("_")[1])]
                    if label.startswith("sample_")
                    else d[label + "_values"]
                )
                raw = d["context"].copy()
                if scope == "full":
                    raw[:] = values
                else:
                    if scope != "target":
                        raise ValueError("unregistered imputation scope")
                    raw[:, :2] = values[:, :2]
                columns = np.flatnonzero(d["keep"])
                canonical = np.array(
                    ((raw[:, columns] - d["mean"][columns]) / d["scale"][columns]).T,
                    dtype=np.float32,
                    order="C",
                )
            np.testing.assert_array_equal(canonical, saved["context_z"])
            with torch.inference_mode():
                q = (
                    backbone(
                        context=torch.tensor(canonical, device="cuda"),
                        group_ids=torch.zeros(len(canonical), device="cuda", dtype=torch.long),
                        num_output_patches=int(
                            np.ceil(row["horizon"] / pipeline.model_output_patch_size)
                        ),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            np.testing.assert_array_equal(q, saved["quantiles"])
            query_outputs[name] = q[:2, :, : row["horizon"]].transpose(1, 2, 0)
            calls += 1
        for scope in ("full", "target"):
            for label in ("static", "filtered", "smoothed"):
                q = query_outputs[scope + "_" + label]
                methods[scope + "_" + label + "_point"] = q[mid]
                if label == "smoothed":
                    methods[scope + "_smoothed_cdf_median"] = reference_mixture(
                        q[None], pipeline.quantiles
                    )[1]
            q = np.stack([query_outputs[f"{scope}_sample_{i:02d}"] for i in range(16)])
            methods[scope + "_sample_point_mean16"] = q[:, mid].mean(0, dtype=np.float64)
            methods[scope + "_sample_point_median16"] = np.median(q[:, mid], 0).astype(float)
            mean, median = reference_mixture(q, pipeline.quantiles)
            methods[scope + "_dynamic_mixture_mean16"] = mean
            methods[scope + "_dynamic_mixture_median16"] = median
        direct, current = [], d["filtered_mean"][-1].copy()
        for _ in range(row["horizon"]):
            current = model["a"] @ current + model["b"]
            direct.append(current[:2].copy())
        np.testing.assert_array_equal(np.stack(direct), d["direct_var"])
        methods["linear_var_direct"] = np.stack(direct)
        methods["half_output_mix"] = (methods["local_ridge"] + methods["peer_ridge"]) * 0.5
        if row["horizon"] == 24:
            old_half = half_entries[row["case_id"]]
            half = load_npz(half_root / old_half["path"], old_half["sha256"])
            methods["half_input_mix"] = half["points"][
                half["methods"].tolist().index("half_input_mix")
            ]
            np.testing.assert_array_equal(
                methods["half_output_mix"],
                half["points"][half["methods"].tolist().index("half_output_mix")],
            )
        else:
            methods["half_input_mix"] = query_outputs["half_input_mix"][mid]
        names = pred["methods"].tolist()
        if len(names) != 81 or set(names) != set(methods):
            raise ValueError("a dynamic output is missing")
        np.testing.assert_array_equal(pred["points"], np.stack([methods[n] for n in names]))
        if scores is not None:
            t, h = row["origin"], row["horizon"]
            truth = records[row["station"]]["values"][t : t + h, :2]
            for name in names:
                metrics = []
                for slot in (0, 1):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        methods[name][valid, slot]
                        - (truth[valid, slot] - d["mean"][slot]) / d["scale"][slot]
                    )
                    values = [float(abs(error).mean()), float(np.square(error).mean())]
                    actual = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        values, actual[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if actual["observed_count"] != valid.sum():
                        raise ValueError("outcome support changed")
                    metrics.append(values)
                values = np.mean(metrics, 0)
                rebuilt_scores.append(
                    {
                        "case_id": row["case_id"],
                        "panel": row["panel"],
                        "station": row["station"],
                        "method": name,
                        "mae": values[0],
                        "mse": values[1],
                    }
                )
        if (number + 1) % 25 == 0:
            _write_json(output / "progress.json", {"audited": number + 1, "queries": calls})
            print(json.dumps({"audited": number + 1, "queries": calls}), flush=True)
    if rebuilt_scores:
        frame = pd.DataFrame(rebuilt_scores)
        saved_case = pd.read_parquet(result / "case_scores.parquet")
        np.testing.assert_allclose(
            frame.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            saved_case.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            rtol=1e-12,
            atol=1e-12,
        )
        station = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean().sort_index()
        np.testing.assert_allclose(
            station,
            pd.read_csv(result / "stations.csv")
            .set_index(["panel", "method", "station"])
            .sort_index()[["mae", "mse"]],
            rtol=1e-12,
            atol=1e-12,
        )
        summary = station.groupby(["panel", "method"]).mean().sort_index()
        np.testing.assert_allclose(
            summary,
            pd.read_csv(result / "summary.csv")
            .set_index(["panel", "method"])
            .sort_index()[["mae", "mse"]],
            rtol=1e-12,
            atol=1e-12,
        )
        leave = pd.read_csv(result / "leave_one_station_out.csv")
        for panel, part in station.groupby(level="panel"):
            for excluded in part.index.get_level_values("station").unique():
                expected = (
                    part.loc[part.index.get_level_values("station") != excluded]
                    .groupby(["panel", "method"])
                    .mean()
                    .sort_index()
                )
                actual = (
                    leave.loc[(leave.panel == panel) & (leave.omitted_station == excluded)]
                    .set_index(["panel", "method"])
                    .sort_index()[["mae", "mse"]]
                )
                np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    if (
        digest != fm["parameter_sha256"]
        or digest != baseline["parameter_sha256"]
        or parameter_digest(backbone) != digest
        or calls != fm["new_queries"]
    ):
        raise ValueError("model parameters or call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(fm["cases"]),
            "prefix_models": len(models),
            "queries_replayed": calls,
            "prediction_difference": 0,
            "maximum_dense_posterior_difference": max_dense_difference,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
