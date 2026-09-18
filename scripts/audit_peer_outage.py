"""Verify peer timing, prefix fits, residual bridges, every query and forecast score."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from conditioning_core import source_controls
from peer_outage_core import BASE, ROOT, STAT_METHODS, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def conditional_residual(phi, t, indices, values, linear):
    before, after = indices[indices < t], indices[indices > t]
    nearest = ([] if len(before) == 0 else [int(before[-1])]) + (
        [] if len(after) == 0 else [int(after[0])]
    )
    if not nearest:
        return 0.0
    residual = np.array([values[np.flatnonzero(indices == i)[0]] for i in nearest])
    if linear:
        return float(np.interp(t, nearest, residual))
    locations = np.asarray(nearest)
    covariance = phi ** abs(locations[:, None] - locations[None, :])
    return float(phi ** abs(t - locations) @ np.linalg.solve(covariance, residual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    forecast, output = args.forecast_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed peer audits")
    fm = json.loads((forecast / "manifest.json").read_text(encoding="utf-8"))
    inputs = Path(fm["input_root"])
    prepared = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    if fm["smoke"] != args.smoke or fm["status"] != "completed":
        raise ValueError("the corresponding forecasts are incomplete")
    for mapping in (prepared["identity"]["files"], fm["identity"]):
        for path, sha in mapping.items():
            if file_sha256(Path(path)) != sha:
                raise ValueError("a frozen preparation or forecasting definition changed")
    records, peer_map = sources()
    full, fits = {}, {}
    for station, source in records.items():
        columns = [source["values"]]
        for slot in (0, 1):
            for peer in peer_map[station, slot]:
                columns.append(records[peer]["values"][:, slot : slot + 1])
        full[station] = np.concatenate(columns, axis=1)
    for item in prepared["regression_fits"]:
        path = inputs / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a frozen regression fit changed")
        fit = json.loads(path.read_text(encoding="utf-8"))
        source = records[item["station"]]
        prefix = full[item["station"]][: source["prefix_end"]]
        center, units = np.nanmean(prefix, axis=0), np.nanstd(prefix, axis=0, ddof=0)
        units = np.where(units <= 1e-12, 1, units)
        np.testing.assert_array_equal(fit["mean"], center)
        np.testing.assert_array_equal(fit["scale"], units)
        z = (prefix - center) / units
        for model in fit["models"].values():
            features, target = model["features"], model["target"]
            if not features:
                continue
            valid = np.isfinite(z[:, target]) & np.isfinite(z[:, features]).all(1)
            if int(valid.sum()) != model["support"] or valid.sum() < 128 or target in features:
                raise ValueError("prefix regression support or target exclusion changed")
            design = np.column_stack([np.ones(valid.sum()), z[valid][:, features]])
            penalty = np.eye(len(features) + 1) * np.sqrt(0.001)
            penalty[0, 0] = 0
            beta = np.linalg.lstsq(
                np.vstack([design / np.sqrt(valid.sum()), penalty]),
                np.r_[z[valid, target] / np.sqrt(valid.sum()), np.zeros(len(features) + 1)],
                rcond=None,
            )[0]
            np.testing.assert_allclose(model["beta"], beta, rtol=1e-8, atol=1e-10)
            residual = np.full(len(z), np.nan)
            residual[valid] = z[valid, target] - design @ np.asarray(model["beta"])
            paired = np.isfinite(residual[:-1]) & np.isfinite(residual[1:])
            a, b = residual[:-1][paired], residual[1:][paired]
            phi = float(np.clip(np.dot(a, b) / max(np.dot(a, a), 1e-12), 0, 0.99))
            np.testing.assert_allclose(model["phi"], phi, rtol=1e-12, atol=1e-12)
        fits[item["station"]] = fit
    training = inputs / "imputers/training_batch.npz"
    if file_sha256(training) != prepared["training_batch_sha256"]:
        raise ValueError("the neural training batch changed")
    with np.load(training, allow_pickle=False) as saved:
        if (
            saved["values"].shape[1:] != (192, 17)
            or len(saved["values"]) > prepared["identity"]["training_window_cap"]
        ):
            raise ValueError("invalid augmented neural training shape or count")
        for i, identifier in enumerate(saved["window_ids"].tolist()):
            station, tail = identifier.rsplit("@", 1)
            first = int(tail.split("|", 1)[0])
            if first + 192 > records[station]["prefix_end"]:
                raise ValueError("neural fitting used post-prefix data")
            observed = saved["observed"][i]
            raw = full[station][first : first + 192]
            if (observed & ~np.isfinite(raw)).any():
                raise ValueError("training exposed an originally missing value")
            np.testing.assert_array_equal(saved["values"][i][observed], raw[observed])
    for fitted in prepared["deep_fits"]:
        marker = Path(fitted["marker"])
        if file_sha256(marker) != fitted["sha256"]:
            raise ValueError("a neural fitting record changed")
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info["status"] != "fitted":
            raise ValueError("strong baseline fitting did not complete")
        for entry in info["files"]:
            if (
                file_sha256(marker.parent / fitted["candidate_id"] / entry["path"])
                != entry["sha256"]
            ):
                raise ValueError("a neural baseline artifact changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if set(metadata) != {r["case_id"] for r in fm["cases"]} or (
        not args.smoke and len(metadata) != 231
    ):
        raise ValueError("forecast population is incomplete")
    scores = None
    if not args.smoke:
        results = BASE / "peer-results-v001"
        study = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
        if study["forecast_sha256"] != file_sha256(forecast / "manifest.json"):
            raise ValueError("score provenance differs")
        scores = pd.read_parquet(results / "case_scores.parquet").set_index(["case_id", "method"])
        targets = pd.read_parquet(results / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
    torch.set_num_threads(1)
    started = perf_counter()
    _, _, model, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    mid = fm["quantiles"].index(0.5)
    checked, rebuilt_scores, metric_max, input_max = set(), [], 0.0, 0.0
    controls = source_controls()
    for number, entry in enumerate(fm["cases"]):
        row = metadata[entry["case_id"]]
        station, t, h = row["station"], row["origin"], row["horizon"]
        for p, sha in (
            (inputs / row["path"], row["sha256"]),
            (forecast / entry["path"], entry["sha256"]),
        ):
            if file_sha256(p) != sha:
                raise ValueError("a saved input or forecast changed")
        with np.load(inputs / row["path"], allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        with np.load(forecast / entry["path"], allow_pickle=False) as saved:
            names, points = saved["methods"].tolist(), saved["points"]
            queries = json.loads(str(saved["queries"]))
        original = full[station][t - 192 : t].copy()
        if row["panel"] == "synthetic_outage_h24":
            if not np.isfinite(full[station][t - 192 : t + 24, :2]).all():
                raise ValueError("the synthetic anchor was not complete")
            original[-24:, :2] = np.nan
        if row["panel"] == "natural_outage_h24":
            if (
                not np.isnan(original[-6:, :2]).all()
                or row["outage_age"] < 6
                or (row["outage_age"] - 6) % 24
            ):
                raise ValueError("a native outage violates its input criterion")
            if not np.isnan(full[station][t - row["outage_age"] : t, :2]).all():
                raise ValueError("recorded outage age is incorrect")
        if t - 192 < records[station]["prefix_end"]:
            raise ValueError("an evaluation history overlaps prefix fitting")
        np.testing.assert_array_equal(original, data["context"])
        expected_peers = [{"station": p, "slot": s} for s in (0, 1) for p in peer_map[station, s]]
        if row["peer_columns"] != expected_peers:
            raise ValueError("peer order or source differs")
        np.testing.assert_array_equal(data["mean"], fits[station]["mean"])
        np.testing.assert_array_equal(data["scale"], fits[station]["scale"])
        observed = np.isfinite(original)
        keep = np.r_[np.ones(11, bool), observed[:, 11:].any(0)]
        np.testing.assert_array_equal(keep, data["keep"])
        for value in data["candidates"]:
            np.testing.assert_array_equal(value[observed], original[observed])
            if not np.isfinite(value).all():
                raise ValueError("a candidate fallback is incomplete")
        mean, scale = data["mean"], data["scale"]
        z = (original - mean) / scale
        model_map = fits[station]["models"]
        regression = {name: original[:, :2].copy() for name in STAT_METHODS}
        for target in (0, 1):
            for position in np.flatnonzero(~observed[:, target]):
                for local in (True, False):
                    available = [
                        int(i)
                        for i in np.flatnonzero(
                            observed[position, :11] if local else observed[position]
                        )
                        if i != target
                    ]
                    key = f"{target}:" + ",".join(map(str, available))
                    fitted = model_map[key]
                    features = fitted["features"]
                    outputs = ("local_ridge",) if local else STAT_METHODS[1:]
                    if not features:
                        fallback = data["candidates"][
                            data["actions"].tolist().index("linear_interp"), position, target
                        ]
                        for name in outputs:
                            regression[name][position, target] = fallback
                        continue
                    beta = np.asarray(fitted["beta"])
                    prediction = beta[0] + np.dot(z[position, features], beta[1:])
                    regression[outputs[0]][position, target] = (
                        prediction * scale[target] + mean[target]
                    )
                    if not local:
                        anchors = np.flatnonzero(observed[:, target] & observed[:, features].all(1))
                        residuals = (
                            z[anchors, target] - beta[0] - z[anchors][:, features] @ beta[1:]
                        )
                        for name, linear in (
                            ("peer_residual_linear", True),
                            ("peer_ar_bridge", False),
                        ):
                            delta = conditional_residual(
                                fitted["phi"], position, anchors, residuals, linear
                            )
                            regression[name][position, target] = (prediction + delta) * scale[
                                target
                            ] + mean[target]
        for name, values in zip(data["stat_names"].tolist(), data["stat_targets"], strict=True):
            np.testing.assert_allclose(
                regression[name] / scale[:2], values / scale[:2], rtol=1e-9, atol=1e-9
            )
            input_max = max(input_max, float(np.max(abs(regression[name] - values) / scale[:2])))
            np.testing.assert_array_equal(values[observed[:, :2]], original[:, :2][observed[:, :2]])
        raw_queries = {}
        for item in queries:
            path = forecast / "queries" / f"{item['key']}.npz"
            if file_sha256(path) != item["sha256"]:
                raise ValueError("a raw forecasting query changed")
            with np.load(path, allow_pickle=False) as saved:
                qx, quantiles = saved["context_z"], saved["quantiles"]
            raw_queries[item["name"]] = (qx, quantiles, item["columns"])
            if item["key"] not in checked:
                with torch.inference_mode():
                    again = (
                        model(
                            context=torch.tensor(qx, device="cuda"),
                            group_ids=torch.zeros(len(qx), device="cuda", dtype=torch.long),
                            num_output_patches=int(np.ceil(h / fm["patch_size"])),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_array_equal(again, quantiles)
                checked.add(item["key"])
        rebuilt = {}
        for name, (qx, quantiles, columns) in raw_queries.items():
            if name == "native_local":
                raw, expected_columns = original, list(range(11))
            else:
                expected_columns = np.flatnonzero(keep).tolist()
                if name == "native_peer":
                    raw = original
                elif name in STAT_METHODS:
                    raw = original.copy()
                    raw[:, :2] = data["stat_targets"][data["stat_names"].tolist().index(name)]
                else:
                    prefix, action = name.split("_", 1)
                    values = data["candidates"][data["actions"].tolist().index(action)]
                    raw = values.copy() if prefix == "peer" else original.copy()
                    if prefix == "target":
                        raw[:, :2] = values[:, :2]
            if columns != expected_columns:
                raise ValueError("a forecast uses a different information scope")
            expected = np.array(
                ((raw[:, columns] - mean[columns]) / scale[columns]).T, dtype=np.float32, order="C"
            )
            expected[~np.isfinite(expected)] = np.nan
            np.testing.assert_array_equal(qx, expected)
            rebuilt[name] = quantiles[:2, mid, :h].T
        actions = data["actions"].tolist()
        for prefix in ("peer", "target"):
            bank = np.stack([rebuilt["native_peer"], *[rebuilt[prefix + "_" + a] for a in actions]])
            rebuilt[prefix + "_mean8"] = bank.mean(0, dtype=np.float64)
            rebuilt[prefix + "_median8"] = np.median(bank, axis=0).astype(float)
            for source_name, control in zip(("source", "matched_source"), controls, strict=True):
                ordered = np.stack(
                    [
                        rebuilt["native_peer"]
                        if a == "guarded_direct"
                        else rebuilt[prefix + "_" + a]
                        for a in control["actions"]
                    ]
                ).astype(float)
                rebuilt[prefix + "_" + source_name + "_single_mae"] = ordered[
                    control["single_index"]
                ]
                for label, weights in (
                    ("mae", control["fixed_mae"]["weights"]),
                    (
                        "joint",
                        control.get(
                            "fixed_joint_weights", control.get("fixed_joint", {}).get("weights")
                        ),
                    ),
                ):
                    rebuilt[prefix + "_" + source_name + "_fixed_" + label] = (
                        ordered * np.asarray(weights)[:, None, None]
                    ).sum(0)
        future_z = np.full((h, 17), np.nan)
        future_z[:, keep] = raw_queries["native_peer"][1][:, mid, :h].T
        reconciled = rebuilt["native_peer"].astype(float).copy()
        for target, fitted in enumerate(json.loads(str(data["future_models"]))):
            features = fitted["features"]
            if features:
                if not keep[features].all():
                    raise ValueError("future reconciliation uses an unavailable channel")
                beta = np.asarray(fitted["beta"])
                reconciled[:, target] = beta[0] + future_z[:, features] @ beta[1:]
        rebuilt["peer_future_ridge"] = reconciled
        if set(rebuilt) != set(names) or len(names) != 37:
            raise ValueError("a registered method was omitted")
        for name, value in rebuilt.items():
            np.testing.assert_array_equal(value, points[names.index(name)])
        if scores is not None:
            truth = records[station]["values"][t : t + h, :2]
            valid = np.isfinite(truth)
            for i, name in enumerate(names):
                metrics = []
                for slot in (0, 1):
                    error = (
                        points[i, valid[:, slot], slot]
                        - (truth[valid[:, slot], slot] - mean[slot]) / scale[slot]
                    )
                    actual = np.array([np.abs(error).mean(), np.square(error).mean()])
                    saved = targets.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        actual, saved[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if int(saved["observed_count"]) != int(valid[:, slot].sum()):
                        raise ValueError("outcome support changed")
                    metrics.append(actual)
                expected = np.mean(metrics, axis=0)
                actual = scores.loc[(row["case_id"], name), ["mae", "mse"]].to_numpy(float)
                np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
                metric_max = max(metric_max, float(abs(expected - actual).max()))
                rebuilt_scores.append(
                    {
                        "panel": row["panel"],
                        "station": station,
                        "method": name,
                        "mae": expected[0],
                        "mse": expected[1],
                    }
                )
        if (number + 1) % 20 == 0:
            print(json.dumps({"audited": number + 1, "queries": len(checked)}), flush=True)
    if rebuilt_scores:
        frame = pd.DataFrame(rebuilt_scores)
        stations = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean()
        expected = stations.groupby(["panel", "method"]).mean().sort_index()
        actual = (
            pd.read_csv(results / "summary.csv")
            .set_index(["panel", "method"])
            .sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        leave = pd.read_csv(results / "leave_one_station_out.csv")
        for panel, part in stations.groupby(level="panel"):
            for station in part.index.get_level_values("station").unique():
                expected = (
                    part.loc[part.index.get_level_values("station") != station]
                    .groupby(["panel", "method"])
                    .mean()
                    .sort_index()
                )
                actual = (
                    leave.loc[(leave["panel"] == panel) & (leave["omitted_station"] == station)]
                    .set_index(["panel", "method"])
                    .sort_index()[["mae", "mse"]]
                )
                np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    if digest != fm["parameter_sha256"] or parameter_digest(model) != digest:
        raise ValueError("the fixed forecasting weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "cases": len(fm["cases"]),
            "raw_queries_replayed": len(checked),
            "raw_prediction_difference": 0,
            "method_difference": 0,
            "stat_input_max_scaled_difference": input_max,
            "metric_max_difference": metric_max,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecast / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
