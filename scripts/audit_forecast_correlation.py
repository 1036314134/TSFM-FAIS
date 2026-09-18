"""Independent innovation-factor covariance and whitened forecast-pool audit."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from forecast_calibration_core import ROOT, load_npz, read_json
from peer_outage_core import sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_covariances(model, state, horizon):
    a, q, p = model["a"], model["q"], state["filtered_covariance"][-1]
    eigenvalues, eigenvectors = np.linalg.eigh((p + p.T) / 2)
    if eigenvalues.min() < -1e-10 * max(1, np.linalg.norm(p, 2)):
        raise ValueError("the frozen current-state covariance is not positive semidefinite")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    innovation = np.linalg.cholesky(q)
    dimension = len(a)
    response = np.zeros((2 * horizon, dimension * (horizon + 1)))
    for t in range(horizon):
        response[2 * t : 2 * t + 2, :dimension] = (np.linalg.matrix_power(a, t + 1) @ root)[:2]
        for k in range(t + 1):
            response[2 * t : 2 * t + 2, dimension * (k + 1) : dimension * (k + 2)] = (
                np.linalg.matrix_power(a, t - k) @ innovation
            )[:2]
    return response @ response.T, response[:, :dimension] @ response[:, :dimension].T


def reference_pool(foundation, statistical, covariance_v, covariance_f, match):
    cf = covariance_f.copy()
    fallback = bool(np.trace(cf) <= 1e-12)
    if fallback:
        cf = covariance_v.copy()
    elif match:
        cf *= np.trace(covariance_v) / np.trace(cf)
    cf = (cf + cf.T) / 2
    factor = np.linalg.cholesky(covariance_v)
    whitened = np.linalg.solve(factor, np.linalg.solve(factor, cf).T).T
    correction = np.linalg.solve(
        np.eye(len(factor)) + whitened,
        np.linalg.solve(factor, (foundation - statistical).ravel()),
    )
    return statistical + (factor @ correction).reshape(statistical.shape), cf, fallback


def check_metrics(frame, result):
    saved = pd.read_parquet(result / "case_scores.parquet")
    keys = ["case_id", "method"]
    pd.testing.assert_frame_equal(
        frame.set_index(keys).sort_index()[["mae", "mse"]],
        saved.set_index(keys).sort_index()[["mae", "mse"]],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    stations = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean()
    for name, reference, keys in (
        ("stations", stations, ["panel", "method", "station"]),
        ("summary", stations.groupby(["panel", "method"]).mean(), ["panel", "method"]),
    ):
        actual = pd.read_csv(result / f"{name}.csv").set_index(keys)[["mae", "mse"]]
        pd.testing.assert_frame_equal(
            reference.sort_index(), actual.sort_index(), check_exact=False, rtol=1e-12, atol=1e-12
        )
    leave = pd.read_csv(result / "leave_one_station_out.csv")
    for panel, part in stations.groupby(level="panel"):
        for excluded in part.index.get_level_values("station").unique():
            reference = (
                part.loc[part.index.get_level_values("station") != excluded]
                .groupby(["panel", "method"])
                .mean()
            )
            actual = leave.loc[(leave.panel == panel) & (leave.omitted_station == excluded)]
            actual = actual.set_index(["panel", "method"])[["mae", "mse"]]
            pd.testing.assert_frame_equal(
                reference.sort_index(),
                actual.sort_index(),
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )


def audit(base):
    forecasts, output = base / "correlation-forecasts-v001", base / "correlation-audit-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed correlation audits")
    started = perf_counter()
    fm = read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["new_foundation_model_calls"] != 0:
        raise ValueError("incomplete predictions or unexpected model calls")
    for path, sha in fm["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a frozen prediction definition changed")
    if not fm["smoke"]:
        registration = read_json(base / "method_manifest.json")
        for row in registration["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered runtime file changed")
    parent_root, state_root = ROOT / "artifacts/iclr27-r37", ROOT / "artifacts/iclr27-r35"
    parent = read_json(parent_root / "moment-forecasts-v001/manifest.json")
    parent_audit = read_json(parent_root / "moment-audit-v001/manifest.json")
    if parent_audit["status"] != "completed" or parent_audit["forecast_sha256"] != file_sha256(
        parent_root / "moment-forecasts-v001/manifest.json"
    ):
        raise ValueError("the prior forecasts no longer match their completed audit")
    old_entries = {r["case_id"]: r for r in parent["cases"]}
    inputs = read_json(state_root / "dynamic-inputs-v001/manifest.json")
    states = {r["case_id"]: r for r in inputs["cases"]}
    models = {
        r["station"]: load_npz(state_root / "dynamic-inputs-v001" / r["path"], r["sha256"])
        for r in inputs["models"]
    }
    static = read_json(state_root / "dynamic-forecasts-v001/manifest.json")
    levels = np.asarray(static["quantiles"])
    edges = np.r_[0, (levels[:-1] + levels[1:]) / 2, 1]
    weights = np.rint(np.diff(edges) * 200).astype(np.int64)
    if weights.sum() != 200:
        raise ValueError("the registered quantile mass changed")
    mid = static["quantiles"].index(0.5)
    ids = [r["case_id"] for r in fm["cases"]]
    if len(ids) != len(set(ids)) or (not fm["smoke"] and set(ids) != set(states)):
        raise ValueError("the registered case population changed")
    scores, records = None, None
    if not fm["smoke"]:
        result = base / "correlation-results-v001"
        rm = read_json(result / "manifest.json")
        if rm["forecast_sha256"] != file_sha256(forecasts / "manifest.json"):
            raise ValueError("the score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 67452:
            raise ValueError("the target score population changed")
        records, _ = sources()
    maximum_covariance, maximum_prediction, reconstructed, new_outputs = 0.0, 0.0, [], 0
    for number, row in enumerate(fm["cases"]):
        original = states[row["case_id"]]
        for field in ("panel", "station", "origin", "horizon", "prefix_end"):
            if original[field] != row[field]:
                raise ValueError("case metadata changed")
        if row["state_path"] != original["path"] or row["state_sha256"] != original["sha256"]:
            raise ValueError("state provenance changed")
        state = load_npz(state_root / "dynamic-inputs-v001" / original["path"], original["sha256"])
        old_row = old_entries[row["case_id"]]
        old = load_npz(parent_root / "moment-forecasts-v001" / old_row["path"], old_row["sha256"])
        saved = load_npz(forecasts / row["path"], row["sha256"])
        prior = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        actual = dict(zip(saved["methods"].tolist(), saved["points"], strict=True))
        for name, point in prior.items():
            np.testing.assert_array_equal(actual[name], point)
        h, model = row["horizon"], models[row["station"]]
        cv, ci = reference_covariances(model, state, h)
        for key, reference in (("covariance_var", cv), ("covariance_initial", ci)):
            np.testing.assert_allclose(saved[key], reference, rtol=1e-9, atol=1e-10)
            maximum_covariance = max(maximum_covariance, float(abs(saved[key] - reference).max()))
        query = row["static_query"]
        raw = load_npz(state_root / "dynamic-forecasts-v001" / query["path"], query["sha256"])
        f = raw["quantiles"][:2, mid, :h].T.astype(float)
        q = np.sort(raw["quantiles"][:2, :, :h].transpose(1, 2, 0).astype(float), axis=0)
        variance = sum(w * (value - f) ** 2 for w, value in zip(weights, q, strict=True)) / 200
        np.testing.assert_array_equal(f, prior["full_static_point"])
        np.testing.assert_allclose(variance, saved["variance_foundation"], rtol=1e-9, atol=1e-10)
        mean, expected_mean = state["filtered_mean"][-1].copy(), []
        for _ in range(h):
            mean = model["a"] @ mean + model["b"]
            expected_mean.append(mean[:2].copy())
        v = np.stack(expected_mean)
        np.testing.assert_array_equal(v, prior["linear_var_direct"])
        ratio = np.sqrt(variance.ravel() / np.diag(cv))
        transported = np.diag(ratio) @ cv @ np.diag(ratio)
        complete = transported + ci
        definitions = {
            "correlation_transport": (f, cv, complete, True),
            "transport_no_initial": (f, cv, transported, True),
            "transport_independent_f": (f, cv, np.diag(variance.ravel()) + ci, True),
            "transport_unscaled": (f, cv, complete, False),
            "transport_diagonal": (f, np.diag(np.diag(cv)), np.diag(np.diag(complete)), True),
            "transport_mean_center": (prior["static_quantile_mean"], cv, complete, True),
        }
        if set(actual) != set(prior) | set(definitions) or len(actual) != 146:
            raise ValueError("registered method identities changed")
        cached_covariances = dict(
            zip(saved["transported_names"].tolist(), saved["transported_covariances"], strict=True)
        )
        fallbacks = json.loads(str(saved["degenerate_fallback"]))
        for name, (center, c1, c2, match) in definitions.items():
            point, used, fallback = reference_pool(center, v, c1, c2, match)
            np.testing.assert_allclose(used, cached_covariances[name], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(point, actual[name], rtol=1e-9, atol=1e-10)
            if fallback != fallbacks[name]:
                raise ValueError("degenerate covariance fallback changed")
            maximum_prediction = max(maximum_prediction, float(abs(point - actual[name]).max()))
            new_outputs += 1
        if scores is not None:
            t = row["origin"]
            truth = records[row["station"]]["values"][t : t + h, :2]
            for name, point in actual.items():
                values = []
                for slot in (0, 1):
                    observed = np.isfinite(truth[:, slot])
                    error = (
                        point[observed, slot]
                        - (truth[observed, slot] - state["mean"][slot]) / state["scale"][slot]
                    )
                    metrics = [float(np.abs(error).mean()), float(np.square(error).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metrics, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != observed.sum():
                        raise ValueError("outcome support changed")
                    values.append(metrics)
                mae, mse = np.mean(values, axis=0)
                reconstructed.append(
                    {
                        **{k: row[k] for k in ("case_id", "panel", "station")},
                        "method": name,
                        "mae": mae,
                        "mse": mse,
                    }
                )
        if (number + 1) % 25 == 0:
            _write_json(output / "progress.json", {"audited": number + 1})
            print(json.dumps({"audited": number + 1}), flush=True)
    if reconstructed:
        check_metrics(pd.DataFrame(reconstructed), result)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "old_outputs_preserved": len(ids) * 140,
            "new_outputs_replayed": new_outputs,
            "maximum_covariance_reference_difference": maximum_covariance,
            "maximum_prediction_reference_difference": maximum_prediction,
            "score_rows": len(reconstructed),
            "new_foundation_model_calls": 0,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
