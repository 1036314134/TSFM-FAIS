"""Independent numerical and information-boundary checks for R33."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import PARENT, ROOT, load_npz, read_json
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def audit(base):
    inputs, forecasts, output = (
        base / n for n in ("residual-inputs-v001", "residual-forecasts-v001", "residual-audit-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed residual audits")
    started = perf_counter()
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("the forecast input collection is incomplete or changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a registered definition changed")
    if file_sha256(ROOT / "scripts/residual_forecast_experiment.py") != fm["script_sha256"]:
        raise ValueError("the forecast implementation changed")
    records, peers = sources()
    full = {s: augmented(s, records, peers)[0] for s in records}
    prefixes = {}
    for s, r in records.items():
        values = full[s][: r["prefix_end"]]
        mean, scale = np.nanmean(values, 0), np.nanstd(values, 0, ddof=0)
        scale = np.where(scale <= 1e-12, 1, scale)
        prefixes[s] = (mean, scale, (values - mean) / scale)
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if set(metadata) != {r["case_id"] for r in fm["cases"]} or (
        not fm["smoke"] and len(metadata) != 231
    ):
        raise ValueError("the residual evaluation population changed")
    for row in prepared["regression_fits"]:
        if file_sha256(inputs / row["path"]) != row["sha256"]:
            raise ValueError("a saved prefix regression changed")
    scores = None
    if not fm["smoke"]:
        result = base / "residual-results-v001"
        study = read_json(result / "manifest.json")
        if study["forecast_sha256"] != file_sha256(forecasts / "manifest.json"):
            raise ValueError("score provenance changed")
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
    mid = pipeline.quantiles.index(0.5)
    parent_manifest = read_json(PARENT / "peer-forecasts-v001/manifest.json")
    checked_fits, raw_calls, rebuilt_scores = set(), 0, []
    support_rows = []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        parent_data = load_npz(
            PARENT / "peer-inputs-v001" / row["parent_input_path"], row["parent_input_sha256"]
        )
        original = full[row["station"]][row["origin"] - 192 : row["origin"]].copy()
        if row["panel"] == "synthetic_outage_h24":
            original[-24:, :2] = np.nan
        if row["origin"] - 192 < row["prefix_end"]:
            raise ValueError("evaluation history overlaps its fitting prefix")
        np.testing.assert_array_equal(original, d["context"])
        np.testing.assert_array_equal(original, parent_data["context"])
        mean, scale, prefix_z = prefixes[row["station"]]
        np.testing.assert_array_equal(mean, d["mean"])
        np.testing.assert_array_equal(scale, d["scale"])
        z = (original - mean) / scale
        parent_entry = row["parent_forecast"]
        parent_prediction = load_npz(
            PARENT / "peer-forecasts-v001" / parent_entry["path"], parent_entry["sha256"]
        )
        native = next(
            q for q in json.loads(str(parent_prediction["queries"])) if q["name"] == "native_peer"
        )
        query = row["native_query"]
        if any(query[k] != native[k] for k in native):
            raise ValueError("the contemporaneous component uses a different base forecast")
        raw_native = load_npz(query["path"], query["sha256"])
        h = row["horizon"]
        future = np.full((h, 17), np.nan)
        future[:, query["columns"]] = raw_native["quantiles"][
            :, parent_manifest["quantiles"].index(0.5), :h
        ].T
        np.testing.assert_array_equal(future, d["forecast_covariates_z"])
        available = [
            i
            for i in range(2, 17)
            if parent_data["keep"][i] and np.isfinite(original[:, i]).sum() >= 2
        ]
        models = json.loads(str(d["models"]))
        residuals = np.full((192, 2), np.nan)
        component = np.zeros((h, 2))
        residual_means, residual_scales, phi_values = [], [], []
        for slot, model in enumerate(models):
            if model["available_features"] != available or not set(model["features"]).issubset(
                available
            ):
                raise ValueError("a decomposition uses an unavailable predictor or target channel")
            features = available.copy()
            correlations = {}
            for feature in features:
                valid = np.isfinite(prefix_z[:, slot]) & np.isfinite(prefix_z[:, feature])
                c = (
                    np.corrcoef(prefix_z[valid, slot], prefix_z[valid, feature])[0, 1]
                    if valid.sum() >= 128 and np.std(prefix_z[valid, feature]) > 1e-12
                    else 0.0
                )
                correlations[feature] = abs(c) if np.isfinite(c) else 0.0
            while features:
                support = np.isfinite(prefix_z[:, slot]) & np.isfinite(prefix_z[:, features]).all(1)
                if support.sum() >= 128:
                    break
                features.remove(min(features, key=lambda i: (correlations[i], -i)))
            if features != model["features"]:
                raise ValueError("prefix support-based feature reduction changed")
            if features:
                beta = np.asarray(model["beta"])
                key = row["station"], slot, tuple(features)
                if key not in checked_fits:
                    design = np.column_stack(
                        [np.ones(support.sum()), prefix_z[support][:, features]]
                    )
                    penalty = np.diag([0.0] + [np.sqrt(0.001)] * len(features))
                    fitted = np.linalg.lstsq(
                        np.vstack([design / np.sqrt(support.sum()), penalty]),
                        np.r_[
                            prefix_z[support, slot] / np.sqrt(support.sum()),
                            np.zeros(len(features) + 1),
                        ],
                        rcond=None,
                    )[0]
                    np.testing.assert_allclose(fitted, beta, rtol=1e-8, atol=1e-10)
                    checked_fits.add(key)
                if model["support"] != support.sum():
                    raise ValueError("prefix ridge label support changed")
                residual_prefix = prefix_z[:, slot] - (beta[0] + prefix_z[:, features] @ beta[1:])
                usable = np.isfinite(z[:, slot]) & np.isfinite(z[:, features]).all(1)
                residuals[usable, slot] = z[usable, slot] - (
                    beta[0] + z[usable][:, features] @ beta[1:]
                )
                component[:, slot] = beta[0] + future[:, features] @ beta[1:]
            else:
                support = np.isfinite(prefix_z[:, slot])
                residual_prefix = prefix_z[:, slot].copy()
                residuals[:, slot] = z[:, slot]
            residual_prefix[~support] = np.nan
            center, unit = (
                float(np.nanmean(residual_prefix)),
                float(np.nanstd(residual_prefix, ddof=0)),
            )
            unit = 1.0 if unit <= 1e-12 else unit
            centered = residual_prefix - center
            pairs = np.isfinite(centered[1:]) & np.isfinite(centered[:-1])
            left, right = centered[:-1][pairs], centered[1:][pairs]
            phi = float(np.clip(np.dot(left, right) / max(np.dot(left, left), 1e-12), 0, 0.99))
            np.testing.assert_allclose(
                [center, unit, phi],
                [model["residual_mean"], model["residual_scale"], model["residual_ar1"]],
                rtol=1e-12,
                atol=1e-12,
            )
            if (
                model["residual_prefix_support"] != support.sum()
                or model["residual_history_support"] != np.isfinite(residuals[:, slot]).sum()
            ):
                raise ValueError("residual observation support changed")
            residual_means.append(center)
            residual_scales.append(unit)
            phi_values.append(phi)
        np.testing.assert_array_equal(component, d["component"])
        np.testing.assert_array_equal(residuals, d["residuals"])
        np.testing.assert_array_equal(residual_means, d["residual_means"])
        np.testing.assert_array_equal(residual_scales, d["residual_scales"])
        expected_z = (residuals - np.asarray(residual_means)) / residual_scales
        np.testing.assert_array_equal(expected_z, d["residual_z"])
        selected = np.where(np.isfinite(residuals).sum(0) >= 2)[0]
        np.testing.assert_array_equal(selected, saved["residual_targets"])
        np.testing.assert_array_equal(np.flatnonzero(d["residual_keep"]), selected)
        canonical = np.array(expected_z[:, selected].T, dtype=np.float32, order="C")
        np.testing.assert_array_equal(canonical, saved["context_z"])
        predicted = np.zeros((h, 2))
        if len(selected):
            with torch.inference_mode():
                raw = (
                    backbone(
                        context=torch.tensor(canonical, device="cuda"),
                        group_ids=torch.zeros(len(selected), dtype=torch.long, device="cuda"),
                        num_output_patches=int(np.ceil(h / pipeline.model_output_patch_size)),
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
            np.testing.assert_array_equal(raw, saved["quantiles"])
            predicted[:, selected] = (
                raw[:, mid, :h].T.astype(float) * np.asarray(residual_scales)[selected]
                + np.asarray(residual_means)[selected]
            )
            raw_calls += 1
        elif saved["quantiles"].shape[0] != 0:
            raise ValueError("a residual without enough observed history was forecast")
        np.testing.assert_array_equal(predicted, saved["predicted_residual"])
        last, ar = np.zeros((h, 2)), np.zeros((h, 2))
        for slot in (0, 1):
            seen = np.where(np.isfinite(residuals[:, slot]))[0]
            if len(seen):
                value = residuals[seen[-1], slot]
                last[:, slot] = value
                steps = 192 - seen[-1] + np.arange(h)
                ar[:, slot] = residual_means[slot] + np.power(phi_values[slot], steps) * (
                    value - residual_means[slot]
                )
        methods = dict(
            zip(parent_prediction["methods"].tolist(), parent_prediction["points"], strict=True)
        )
        methods.update(
            residual_zero=component,
            residual_last=component + last,
            residual_ar1=component + ar,
            residual_tsfm=component + predicted,
        )
        names = saved["methods"].tolist()
        if len(names) != 41 or set(names) != set(methods):
            raise ValueError("a registered method was dropped")
        np.testing.assert_array_equal(saved["points"], np.stack([methods[n] for n in names]))
        support_rows.append(
            {
                "case_id": row["case_id"],
                "panel": row["panel"],
                "support": np.isfinite(residuals).sum(0).tolist(),
                "forecasted_targets": selected.tolist(),
            }
        )
        if scores is not None:
            truth = records[row["station"]]["values"][row["origin"] : row["origin"] + h, :2]
            for method in names:
                metrics = []
                for slot in (0, 1):
                    observed = np.isfinite(truth[:, slot])
                    error = (
                        methods[method][observed, slot]
                        - (truth[observed, slot] - mean[slot]) / scale[slot]
                    )
                    values = [float(np.abs(error).mean()), float(np.square(error).mean())]
                    actual = scores.loc[(row["case_id"], method, slot)]
                    np.testing.assert_allclose(
                        values, actual[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if actual["observed_count"] != observed.sum():
                        raise ValueError("scoring support changed")
                    metrics.append(values)
                values = np.mean(metrics, 0)
                rebuilt_scores.append(
                    {
                        "case_id": row["case_id"],
                        "panel": row["panel"],
                        "station": row["station"],
                        "method": method,
                        "mae": values[0],
                        "mse": values[1],
                    }
                )
        if len(support_rows) % 50 == 0:
            _write_json(
                output / "progress.json", {"audited": len(support_rows), "total": len(fm["cases"])}
            )
    if rebuilt_scores:
        frame = pd.DataFrame(rebuilt_scores)
        case = pd.read_parquet(result / "case_scores.parquet")
        np.testing.assert_allclose(
            frame.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            case.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            rtol=1e-12,
            atol=1e-12,
        )
        station = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean().sort_index()
        actual_station = (
            pd.read_csv(result / "stations.csv")
            .set_index(["panel", "method", "station"])
            .sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(station, actual_station, rtol=1e-12, atol=1e-12)
        summary = station.groupby(["panel", "method"]).mean().sort_index()
        actual = (
            pd.read_csv(result / "summary.csv")
            .set_index(["panel", "method"])
            .sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(summary, actual, rtol=1e-12, atol=1e-12)
        leave = pd.read_csv(result / "leave_one_station_out.csv")
        for panel, part in station.groupby(level="panel"):
            for omitted in part.index.get_level_values("station").unique():
                expected = (
                    part.loc[part.index.get_level_values("station") != omitted]
                    .groupby(["panel", "method"])
                    .mean()
                    .sort_index()
                )
                actual = (
                    leave.loc[(leave.panel == panel) & (leave.omitted_station == omitted)]
                    .set_index(["panel", "method"])
                    .sort_index()[["mae", "mse"]]
                )
                np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    if (
        digest != fm["parameter_sha256"]
        or digest != parent_manifest["parameter_sha256"]
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("the forecasting weights changed")
    if raw_calls != fm["new_model_calls"]:
        raise ValueError("residual forecast call counts changed")
    _write_json(output / "support.json", support_rows)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(fm["cases"]),
            "prefix_models_verified": len(checked_fits),
            "residual_queries_replayed": raw_calls,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
