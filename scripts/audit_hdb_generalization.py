"""Audit HDB time boundaries, statistical fits, imputation inputs and frozen-model replay."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import ROOT, load_npz, read_json
from hdb_generalization_core import build_case, choose_peers, population, queries
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def check_model(prefix, model):
    mean, scale = np.nanmean(prefix, 0), np.nanstd(prefix, 0)
    scale = np.where(scale <= 1e-12, 1, scale)
    np.testing.assert_array_equal(mean, model["mean"])
    np.testing.assert_array_equal(scale, model["scale"])
    z = (prefix - mean) / scale
    complete = np.isfinite(z).all(1)
    pair = complete[:-1] & complete[1:]
    if min(complete.sum(), pair.sum()) < 128:
        raise ValueError("prefix support is below its fixed minimum")
    x, y, dimension = z[:-1][pair], z[1:][pair], z.shape[1]
    xm, ym = x.mean(0), y.mean(0)
    design = np.vstack([(x - xm) / np.sqrt(len(x)), np.sqrt(0.001) * np.eye(dimension)])
    target = np.vstack([(y - ym) / np.sqrt(len(y)), np.zeros((dimension, dimension))])
    beta = np.linalg.lstsq(design, target, rcond=None)[0].T
    np.testing.assert_allclose(model["a_unconstrained"], beta, rtol=1e-8, atol=1e-10)
    radius = np.max(abs(np.linalg.eigvals(model["a_unconstrained"])))
    np.testing.assert_array_equal(
        model["a"], model["a_unconstrained"] * min(1, 0.99 / max(radius, 1e-12))
    )
    np.testing.assert_array_equal(model["b"], ym - model["a"] @ xm)
    residual = y - (x @ model["a"].T + model["b"])
    for name, values in (("q", residual), ("initial_covariance", z[complete])):
        covariance = np.atleast_2d(np.cov(values, rowvar=False, bias=True))
        expected = (
            0.95 * covariance + 0.05 * np.diag(np.diag(covariance)) + 1e-6 * np.eye(dimension)
        )
        np.testing.assert_allclose(model[name], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(model["initial_mean"], z[complete].mean(0))
    if (
        int(model["complete_rows"]) != complete.sum()
        or int(model["transition_pairs"]) != pair.sum()
    ):
        raise ValueError("recorded fitting support changed")


def reference_condition(mu, covariance, row):
    observed, missing = np.flatnonzero(np.isfinite(row)), np.flatnonzero(~np.isfinite(row))
    result, cov = mu.copy(), np.zeros_like(covariance)
    if not len(observed):
        return mu.copy(), covariance.copy()
    result[observed] = row[observed]
    if len(missing):
        cross = covariance[np.ix_(missing, observed)]
        observed_cov = covariance[np.ix_(observed, observed)]
        result[missing] += cross @ np.linalg.solve(observed_cov, row[observed] - mu[observed])
        cov[np.ix_(missing, missing)] = covariance[
            np.ix_(missing, missing)
        ] - cross @ np.linalg.solve(observed_cov, cross.T)
    return result, (cov + cov.T) / 2


def check_conditional_inputs(data, model):
    z = (data["context"] - model["mean"]) / model["scale"]
    gaussian = data["fills"][data["fill_names"].tolist().index("gaussian")]
    current, covariance = model["initial_mean"].copy(), model["initial_covariance"].copy()
    maximum = 0.0
    for index, row in enumerate(z):
        static, _ = reference_condition(model["initial_mean"], model["initial_covariance"], row)
        expected_raw = static * model["scale"] + model["mean"]
        seen = np.isfinite(data["context"][index])
        expected_raw[seen] = data["context"][index, seen]
        np.testing.assert_allclose(gaussian[index], expected_raw, rtol=1e-9, atol=1e-10)
        if index:
            current = model["a"] @ current + model["b"]
            covariance = model["a"] @ covariance @ model["a"].T + model["q"]
        current, covariance = reference_condition(current, covariance, row)
        np.testing.assert_allclose(current, data["filtered_mean"][index], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            covariance, data["filtered_covariance"][index], rtol=1e-9, atol=1e-10
        )
        maximum = max(maximum, float(abs(current - data["filtered_mean"][index]).max()))
    direct = []
    for _ in range(24):
        current = model["a"] @ current + model["b"]
        direct.append(current[0])
    saved = data["controls"][data["control_names"].tolist().index("linear_var_direct")]
    np.testing.assert_allclose(direct, saved, rtol=1e-9, atol=1e-10)
    return maximum


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "hdb-inputs-v001",
        base / "hdb-forecasts-v001",
        base / "hdb-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed HDB experiment audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("incomplete or changed HDB forecast inputs")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("an HDB experiment definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered HDB runtime changed")
    plan_path = ROOT / "artifacts/iclr27-r39/hdb-plan-v001/manifest.json"
    plan = read_json(plan_path)
    source = load_npz(plan["development_path"], plan["development_sha256"])
    if source["values"].shape[0] != 672 or source["timestamps"][-1] != "2025-06-28T23:00:00+08:00":
        raise ValueError("the experiment source crossed into held-out data")
    rows, metadata = population(source["values"], source["identifiers"].tolist())
    if rows != plan["cases"] or metadata != plan["population"]:
        raise ValueError("the development population changed")
    peers = {p["station"]: p for p in plan["peers"]}
    for peer in peers.values():
        columns, decisions = choose_peers(
            source["values"][:336],
            peer["columns"][0],
            metadata["eligible_columns"],
            source["identifiers"].tolist(),
        )
        if columns != peer["columns"] or decisions != peer["decisions"]:
            raise ValueError("prefix-only peer ranking or support changed")
        target = source["values"][:336, columns[0]]
        for decision in decisions:
            other = source["values"][:336, decision["column"]]
            observed = np.isfinite(target) & np.isfinite(other)
            x, y = target[observed], other[observed]
            x, y = x - x.mean(), y - y.mean()
            rho = (x @ y) / np.sqrt((x @ x) * (y @ y))
            np.testing.assert_allclose(rho, decision["correlation"], rtol=1e-12, atol=1e-12)
    models = {}
    for entry in prepared["models"]:
        model = load_npz(inputs / entry["path"], entry["sha256"])
        if entry["columns"] != peers[entry["station"]]["columns"]:
            raise ValueError("a fitted model uses different peer information")
        check_model(source["values"][:336, entry["columns"]], model)
        models[entry["station"]] = model
    if {r["case_id"] for r in prepared["cases"]} != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("a prepared case is missing its predictions")
    if not fm["smoke"] and {r["case_id"] for r in rows} != {r["case_id"] for r in fm["cases"]}:
        raise ValueError("the complete registered population was not evaluated")
    scores = None
    if not fm["smoke"]:
        result = base / "hdb-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("HDB score provenance changed")
        scores = pd.read_parquet(result / "case_scores.parquet").set_index(["case_id", "method"])
        if not scores.index.is_unique or len(scores) != 31 * len(rows):
            raise ValueError("HDB score population changed")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, calls, maximum, reconstructed = pipeline.quantiles.index(0.5), 0, 0.0, []
    original = {r["case_id"]: r for r in prepared["cases"]}
    for number, entry in enumerate(fm["cases"]):
        row = original[entry["case_id"]]
        model = models[row["station"]]
        saved = load_npz(inputs / row["path"], row["sha256"])
        rebuilt = build_case(source["values"], row["columns"], row, model)
        for name in saved:
            np.testing.assert_array_equal(saved[name], rebuilt[name])
        maximum = max(maximum, check_conditional_inputs(saved, model))
        data = load_npz(forecasts / entry["path"], entry["sha256"])
        requests, expected_queries = json.loads(str(data["queries"])), queries(saved)
        if {r["name"] for r in requests} != set(expected_queries) or len(requests) != 11:
            raise ValueError("a registered model input is missing")
        points = {}
        for request in requests:
            query = load_npz(forecasts / request["path"], request["sha256"])
            np.testing.assert_array_equal(query["context_z"], expected_queries[request["name"]])
            with torch.inference_mode():
                raw = (
                    backbone(
                        context=torch.tensor(query["context_z"], device="cuda"),
                        group_ids=torch.zeros(
                            len(query["context_z"]), device="cuda", dtype=torch.long
                        ),
                        num_output_patches=int(np.ceil(24 / pipeline.model_output_patch_size)),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            np.testing.assert_array_equal(raw, query["quantiles"])
            points[request["name"]] = raw[0, mid, :24].astype(float)
            calls += 1
        pool_names = [
            "native_peer192",
            *["full_" + n for n in ("median", "ffill", "linear", "seasonal24", "knn", "gaussian")],
            "target_knn",
            "target_gaussian",
        ]
        pool = np.stack([points[n] for n in pool_names])
        points["mean9"], points["median9"] = pool.mean(0), np.median(pool, axis=0)
        controls = dict(zip(saved["control_names"].tolist(), saved["controls"], strict=True))
        for name, values in list(points.items()):
            points["half_var_" + name] = 0.5 * values + 0.5 * controls["linear_var_direct"]
        points.update(controls)
        points["half_gaussian_seasonal168"] = (
            0.5 * points["full_gaussian"] + 0.5 * points["seasonal168_direct"]
        )
        names = sorted(points)
        np.testing.assert_array_equal(data["methods"], np.asarray(names))
        np.testing.assert_array_equal(data["points"], np.stack([points[n] for n in names]))
        if scores is not None:
            t = row["origin"]
            truth = source["values"][t : t + 24, row["column"]]
            observed = np.flatnonzero(np.isfinite(truth))
            for name, values in points.items():
                errors = [
                    values[i] - (truth[i] - model["mean"][0]) / model["scale"][0] for i in observed
                ]
                mae, mse = float(np.mean(np.abs(errors))), float(np.mean(np.square(errors)))
                record = scores.loc[(row["case_id"], name)]
                np.testing.assert_allclose(
                    [mae, mse], record[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                )
                if record["observed_count"] != len(observed):
                    raise ValueError("observed outcome count changed")
                reconstructed.append(
                    {
                        **{k: row[k] for k in ("case_id", "panel", "station")},
                        "method": name,
                        "mae": mae,
                        "mse": mse,
                    }
                )
        if (number + 1) % 20 == 0:
            _write_json(
                output / "progress.json", {"cases_audited": number + 1, "queries_replayed": calls}
            )
            print(json.dumps({"cases_audited": number + 1, "queries_replayed": calls}), flush=True)
    if reconstructed:
        check_metrics(pd.DataFrame(reconstructed), result)
    if (
        digest != fm["parameter_sha256"]
        or parameter_digest(backbone) != digest
        or calls != fm["model_calls"]
    ):
        raise ValueError("frozen model or call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(fm["cases"]),
            "prefix_models_checked": len(models),
            "queries_replayed": calls,
            "prediction_difference": 0,
            "maximum_state_reference_difference": maximum,
            "score_rows": len(reconstructed),
            "heldout_value_analysis": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
