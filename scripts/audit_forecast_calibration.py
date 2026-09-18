"""Audit temporal splits, source fits, optimizer replay, frozen forecasts and scores."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_peer_outage import conditional_residual
from conditioning_core import source_controls
from forecast_calibration_core import (
    BASE,
    LEARNED,
    PARENT,
    ROOT,
    SEED,
    calibration_population,
    calibration_sources,
    fixed_mae_fit,
    load_npz,
    mixed_context,
    new_gate,
    normalized_features,
    read_json,
    smooth_mae,
    tensor_inputs,
)
from peer_outage_core import STAT_METHODS, augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.utility_experiment import _write_json, file_sha256


def check_hash(path, expected):
    if file_sha256(Path(path)) != expected:
        raise ValueError(f"a registered file changed: {path}")


def audit_fits(manifest, inputs, full, records):
    fits = {}
    for row in manifest["regression_fits"]:
        check_hash(inputs / row["path"], row["sha256"])
        fit = read_json(inputs / row["path"])
        prefix = full[row["station"]][: records[row["station"]]["prefix_end"]]
        mean, scale = np.nanmean(prefix, 0), np.nanstd(prefix, 0, ddof=0)
        scale = np.where(scale <= 1e-12, 1, scale)
        np.testing.assert_array_equal(fit["mean"], mean)
        np.testing.assert_array_equal(fit["scale"], scale)
        z = (prefix - mean) / scale
        for key, model in fit["models"].items():
            features, target = model["features"], model["target"]
            requested = [int(v) for v in key.split(":")[1].split(",") if v]
            if target in features or not set(features).issubset(requested):
                raise ValueError("regression target exclusion failed")
            if not features:
                continue
            valid = np.isfinite(z[:, target]) & np.isfinite(z[:, features]).all(1)
            if valid.sum() != model["support"] or valid.sum() < 128:
                raise ValueError("regression fitting support changed")
            design = np.column_stack([np.ones(valid.sum()), z[valid][:, features]])
            penalty = np.diag([0.0] + [np.sqrt(0.001)] * len(features))
            beta = np.linalg.lstsq(
                np.vstack([design / np.sqrt(valid.sum()), penalty]),
                np.r_[z[valid, target] / np.sqrt(valid.sum()), np.zeros(len(features) + 1)],
                rcond=None,
            )[0]
            np.testing.assert_allclose(beta, model["beta"], rtol=1e-8, atol=1e-10)
        fits[row["station"]] = fit
    return fits


def independently_rebuild_features(data, fit):
    x = data["context"]
    observed = np.isfinite(x)
    z = (x - data["mean"]) / data["scale"]
    repairs = {
        n: (v - data["mean"][:2]) / data["scale"][:2]
        for n, v in zip(data["stat_names"].tolist(), data["stat_targets"], strict=True)
    }
    result = np.empty((2, 10))
    for slot in (0, 1):
        indices = np.where(observed[:, slot])[0]
        age = 192 - indices[-1] - 1 if indices.size else 192
        result[slot, :3] = [
            age / 192,
            len(indices) / 192,
            observed[-age:, 11 + 3 * slot : 14 + 3 * slot].mean(),
        ]
        for col, limit in ((3, 11), (4, 17)):
            errors = []
            for time in indices:
                features = [i for i in range(2, limit) if observed[time, i]]
                model = fit["models"][f"{slot}:" + ",".join(map(str, features))]
                if 0 in model["features"] or 1 in model["features"]:
                    raise ValueError("a gate quality feature exposes PM target values")
                prediction = (
                    model["beta"][0] + np.dot(z[time, model["features"]], model["beta"][1:])
                    if model["features"]
                    else 0
                )
                errors.append(abs(z[time, slot] - prediction))
            result[slot, col] = np.mean(errors) if errors else 1
        local, peer = repairs["local_ridge"][:, slot], repairs["peer_ridge"][:, slot]
        result[slot, 5:] = [
            np.abs(local[-age:] - peer[-age:]).mean(),
            z[indices[-1], slot] if indices.size else 0,
            local[-1],
            peer[-1],
            observed[-age:, 2:11].mean(),
        ]
    return result


def audit_repairs(data, fit):
    x, center, units = data["context"], data["mean"], data["scale"]
    z, observed = (x - center) / units, np.isfinite(x)
    rebuilt = {n: x[:, :2].copy() for n in STAT_METHODS}
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
                fitted = fit["models"][f"{target}:" + ",".join(map(str, available))]
                features = fitted["features"]
                names = ("local_ridge",) if local else STAT_METHODS[1:]
                if not features:
                    fallback = data["candidates"][
                        data["actions"].tolist().index("linear_interp"), position, target
                    ]
                    for name in names:
                        rebuilt[name][position, target] = fallback
                    continue
                beta = np.asarray(fitted["beta"])
                value = beta[0] + z[position, features] @ beta[1:]
                rebuilt[names[0]][position, target] = value * units[target] + center[target]
                if not local:
                    indices = np.where(observed[:, target] & observed[:, features].all(1))[0]
                    residual = z[indices, target] - (beta[0] + z[indices][:, features] @ beta[1:])
                    for name, linear in (("peer_residual_linear", True), ("peer_ar_bridge", False)):
                        delta = conditional_residual(
                            fitted["phi"], position, indices, residual, linear
                        )
                        rebuilt[name][position, target] = (value + delta) * units[target] + center[
                            target
                        ]
    for name, saved in zip(data["stat_names"].tolist(), data["stat_targets"], strict=True):
        np.testing.assert_allclose((rebuilt[name] - saved) / units[:2], 0, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(saved[observed[:, :2]], x[:, :2][observed[:, :2]])
    for candidate in data["candidates"]:
        np.testing.assert_array_equal(candidate[observed], x[observed])
        if not np.isfinite(candidate).all():
            raise ValueError("a source candidate is not complete")


def audit_source_forecasts(root, fm, prepared, inputs, backbone, pipeline):
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    checked, points = set(), {}
    mid = pipeline.quantiles.index(0.5)
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(root / entry["path"], entry["sha256"])
        x, keep, mean, scale = (data[k] for k in ("context", "keep", "mean", "scale"))
        raw, rebuilt = {}, {}
        for query in json.loads(str(saved["queries"])):
            path = root / "queries" / f"{query['key']}.npz"
            q = load_npz(path, query["sha256"])
            name, columns = query["name"], query["columns"]
            expect_columns = (
                list(range(11)) if name == "native_local" else np.flatnonzero(keep).tolist()
            )
            if columns != expect_columns:
                raise ValueError("source forecast information set differs")
            context = x.copy()
            if name in data["stat_names"].tolist():
                context[:, :2] = data["stat_targets"][data["stat_names"].tolist().index(name)]
            elif not name.startswith("native_"):
                prefix, action = name.split("_", 1)
                candidate = data["candidates"][data["actions"].tolist().index(action)]
                if prefix == "peer":
                    context = candidate
                else:
                    context[:, :2] = candidate[:, :2]
            expected = np.array(
                ((context[:, columns] - mean[columns]) / scale[columns]).T,
                dtype=np.float32,
                order="C",
            )
            expected[~np.isfinite(expected)] = np.nan
            np.testing.assert_array_equal(q["context_z"], expected)
            if (
                hashlib.sha256(str(q["binding"]).encode() + expected.tobytes()).hexdigest()
                != query["key"]
            ):
                raise ValueError("a source cache binding changed")
            if query["key"] not in checked:
                with torch.inference_mode():
                    actual = (
                        backbone(
                            context=torch.tensor(expected, device="cuda"),
                            group_ids=torch.zeros(len(columns), dtype=torch.long, device="cuda"),
                            num_output_patches=2,
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_array_equal(actual, q["quantiles"])
                checked.add(query["key"])
            raw[name] = q["quantiles"]
            rebuilt[name] = q["quantiles"][:2, mid, :24].T
        for prefix in ("peer", "target"):
            bank = np.stack(
                [
                    rebuilt["native_peer"],
                    *[rebuilt[prefix + "_" + a] for a in data["actions"].tolist()],
                ]
            )
            rebuilt[prefix + "_mean8"] = bank.mean(0, dtype=np.float64)
            rebuilt[prefix + "_median8"] = np.median(bank, 0).astype(float)
            for key, control in zip(("source", "matched_source"), source_controls(), strict=True):
                ordered = np.stack(
                    [
                        rebuilt["native_peer"]
                        if a == "guarded_direct"
                        else rebuilt[prefix + "_" + a]
                        for a in control["actions"]
                    ]
                ).astype(float)
                rebuilt[prefix + "_" + key + "_single_mae"] = ordered[control["single_index"]]
                for loss, weights in (
                    ("mae", control["fixed_mae"]["weights"]),
                    (
                        "joint",
                        control.get(
                            "fixed_joint_weights", control.get("fixed_joint", {}).get("weights")
                        ),
                    ),
                ):
                    rebuilt[prefix + "_" + key + "_fixed_" + loss] = (
                        ordered * np.asarray(weights)[:, None, None]
                    ).sum(0)
        future = np.full((24, 17), np.nan)
        future[:, keep] = raw["native_peer"][:, mid, :24].T
        reconciled = rebuilt["native_peer"].astype(float).copy()
        for target, model in enumerate(json.loads(str(data["future_models"]))):
            if model["features"]:
                if target in model["features"] or not keep[model["features"]].all():
                    raise ValueError("source future reconciliation uses unavailable features")
                reconciled[:, target] = (
                    model["beta"][0] + future[:, model["features"]] @ model["beta"][1:]
                )
        rebuilt["peer_future_ridge"] = reconciled
        names = saved["methods"].tolist()
        if len(names) != 37 or set(names) != set(rebuilt):
            raise ValueError("a source method is missing")
        np.testing.assert_array_equal(saved["points"], np.stack([rebuilt[n] for n in names]))
        points[row["case_id"]] = saved
    return len(checked), points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    forecast, output = args.forecast_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed calibration audits")
    started = perf_counter()
    fm = read_json(forecast / "manifest.json")
    training = Path(fm["training_root"])
    tm = read_json(training / "manifest.json")
    inputs, source_forecast = Path(tm["input_root"]), Path(tm["source_forecast_root"])
    prepared, source_fm = (
        read_json(inputs / "manifest.json"),
        read_json(source_forecast / "manifest.json"),
    )
    if any(
        m["status"] != "completed" or m["smoke"] != args.smoke
        for m in (fm, tm, prepared, source_fm)
    ):
        raise ValueError("the corresponding study stages are incomplete")
    for mapping in (
        fm["identity"],
        tm["identity"],
        prepared["identity"]["files"],
        source_fm["identity"],
    ):
        for path, sha in mapping.items():
            check_hash(path, sha)
    for path, sha in tm["files"].items():
        check_hash(training / path, sha)
    records, peers, selection = calibration_sources()
    if selection != prepared["identity"]["peer_selection"]:
        raise ValueError("source peers were not selected on the first half prefix")
    planned, eligibility = calibration_population(records)
    plan = read_json(inputs / "case_plan.json")
    if eligibility != plan["eligibility"] or any(r not in planned for r in plan["cases"]):
        raise ValueError("source input eligibility changed")
    if not args.smoke and len(prepared["cases"]) != 288:
        raise ValueError("source calibration cases are missing")
    full = {s: augmented(s, records, peers)[0] for s in records}
    fits = audit_fits(prepared, inputs, full, records)
    training_path = inputs / "imputers/training_batch.npz"
    batch = load_npz(training_path, prepared["training_batch_sha256"])
    for index, identifier in enumerate(batch["window_ids"].tolist()):
        station, tail = identifier.rsplit("@", 1)
        start = int(tail.split("|", 1)[0])
        if start + 192 > records[station]["prefix_end"]:
            raise ValueError("source imputer training sees calibration outcomes")
        original = full[station][start : start + 192]
        observed = batch["observed"][index]
        if (observed & ~np.isfinite(original)).any():
            raise ValueError("a training mask exposes an originally absent value")
        np.testing.assert_array_equal(batch["values"][index][observed], original[observed])
    for record in prepared["deep_fits"]:
        check_hash(record["marker"], record["sha256"])
        fit = read_json(record["marker"])
        if fit["status"] != "fitted" or fit["training_windows"] != len(batch["values"]):
            raise ValueError("a strong source imputer did not finish")
        for file in fit["files"]:
            check_hash(
                Path(record["marker"]).parent / record["candidate_id"] / file["path"],
                file["sha256"],
            )
    source_data, labels, feature_rows = {}, {}, []
    for row in prepared["cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        label = load_npz(inputs / row["label_path"], row["label_sha256"])
        t, age, station = row["origin"], row["outage_age"], row["station"]
        if t - 192 < row["prefix_end"] or t + 24 > row["calibration_end"]:
            raise ValueError("calibration crosses a time boundary")
        original = full[station][t - 192 : t].copy()
        np.testing.assert_array_equal(label["original_history"], original[:, :2])
        np.testing.assert_array_equal(label["future"], full[station][t : t + 24, :2])
        hidden = np.zeros((192, 2), bool)
        hidden[-age:] = True
        np.testing.assert_array_equal(label["hidden"], hidden)
        original[-age:, :2] = np.nan
        np.testing.assert_array_equal(original, d["context"])
        np.testing.assert_array_equal(d["mean"], fits[station]["mean"])
        np.testing.assert_array_equal(d["scale"], fits[station]["scale"])
        np.testing.assert_array_equal(
            d["keep"], np.r_[np.ones(11, bool), np.isfinite(original[:, 11:]).any(0)]
        )
        audit_repairs(d, fits[station])
        features = independently_rebuild_features(d, fits[station])
        np.testing.assert_allclose(d["features"], features, rtol=1e-12, atol=1e-12)
        source_data[row["case_id"]], labels[row["case_id"]] = d, label
        feature_rows.append(features)
    scaler = load_npz(training / "feature_scaler.npz")
    values = np.stack(feature_rows)
    np.testing.assert_allclose(scaler["values"], values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler["mean"], values.mean((0, 1)), rtol=1e-12, atol=1e-12)
    scale = values.std((0, 1), ddof=0)
    np.testing.assert_allclose(
        scaler["scale"], np.where(scale < 1e-6, 1, scale), rtol=1e-12, atol=1e-12
    )
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    backbone.eval().requires_grad_(False)
    raw_queries, bank = audit_source_forecasts(
        source_forecast, source_fm, prepared, inputs, backbone, pipeline
    )
    fixed = read_json(training / "fixed_output.json")
    ids = [r["case_id"] for r in prepared["cases"]]
    predictions = np.stack(
        [
            bank[i]["points"][[bank[i]["methods"].tolist().index(n) for n in fixed["methods"]]]
            for i in ids
        ]
    )
    truth = np.stack(
        [
            (labels[i]["future"] - source_data[i]["mean"][:2]) / source_data[i]["scale"][:2]
            for i in ids
        ]
    )
    for key, selected in [
        ("global", list(range(len(ids)))),
        *[
            (s, [j for j, r in enumerate(prepared["cases"]) if r["station"] == s])
            for s in fixed["stations"]
        ],
    ]:
        fitted = fixed["global"] if key == "global" else fixed["stations"][key]
        for target in (0, 1):
            certificate = fixed_mae_fit(
                predictions[selected, :, :, target], truth[selected, :, target]
            )
            np.testing.assert_allclose(
                certificate["objective"], fitted[target]["objective"], rtol=1e-10, atol=1e-10
            )
            weights = np.asarray(fitted[target]["weights"])
            if abs(weights.sum() - 1) > 1e-7 or weights.min() < -1e-7:
                raise ValueError("a saved fixed control is outside the simplex")
            errors = abs(
                np.einsum("k,nkh->nh", weights, predictions[selected, :, :, target])
                - truth[selected, :, target]
            )
            np.testing.assert_allclose(
                np.nanmean(errors, 1).mean(), certificate["objective"], rtol=1e-7, atol=1e-7
            )
    order = load_npz(training / "source_order.npz")
    random = np.random.default_rng(SEED)
    expected_order = np.stack([random.permutation(len(ids)) for _ in range(tm["epochs"])])
    np.testing.assert_array_equal(order["indices"], expected_order)
    np.testing.assert_array_equal(order["case_ids"], ids)
    gates, replay = {}, []
    trace = read_json(training / "training_trace.json")
    for record in tm["learned"]:
        name = record["method"]
        check_hash(training / record["path"], record["sha256"])
        check_hash(training / f"{name}-before-last.pt", record["before_last_sha256"])
        before = torch.load(
            training / f"{name}-before-last.pt", map_location="cuda", weights_only=False
        )
        after = torch.load(training / record["path"], map_location="cuda", weights_only=False)
        if (
            before["case_id"] != ids[expected_order.ravel()[-1]]
            or record["steps"] != expected_order.size
        ):
            raise ValueError("the optimizer step count or last case changed")
        gate = new_gate(name)
        gate.load_state_dict(before["model"])
        optimizer = torch.optim.AdamW(gate.parameters(), lr=0.01, weight_decay=0.001)
        optimizer.load_state_dict(before["optimizer"])
        d, label = source_data[before["case_id"]], labels[before["case_id"]]
        alpha = gate(normalized_features(d["features"], scaler["mean"], scaler["scale"]))
        context = mixed_context(tensor_inputs(d), alpha)
        if name == "imputation_gate":
            target = np.where(
                label["hidden"],
                (label["original_history"] - d["mean"][:2]) / d["scale"][:2],
                np.nan,
            )
            predicted = context[:, :2]
        else:
            target = (label["future"] - d["mean"][:2]) / d["scale"][:2]
            predicted = chronos_median(pipeline, context, 24, [0, 1])
        loss = smooth_mae(predicted, torch.tensor(target, dtype=torch.float32, device="cuda"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for key, value in after["model"].items():
            torch.testing.assert_close(gate.state_dict()[key], value, rtol=0, atol=0)
        for index, state in after["optimizer"]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    optimizer.state_dict()["state"][index][key], value, rtol=0, atol=0
                )
        log = [r for r in trace if r["method"] == name]
        if len(log) != expected_order.size or [r["case_id"] for r in log] != [
            ids[i] for i in expected_order.ravel()
        ]:
            raise ValueError("training trace order changed")
        np.testing.assert_array_equal(
            [float(loss.detach()), float(norm)], [log[-1]["loss"], log[-1]["gradient_norm"]]
        )
        gates[name] = gate.eval()
        replay.append({"method": name, "parameter_difference": 0, "optimizer_difference": 0})
    eval_fits = {}
    for entry in tm["evaluation_regressions"]:
        check_hash(training / entry["path"], entry["sha256"])
        eval_fits[entry["station"]] = read_json(training / entry["path"])
    original_records, original_peers = sources()
    original_full = {s: augmented(s, original_records, original_peers)[0] for s in original_records}
    audit_fits(
        {"regression_fits": tm["evaluation_regressions"]}, training, original_full, original_records
    )
    metadata = {r["case_id"]: r for r in tm["evaluation"]}
    original_fm = read_json(PARENT / "peer-forecasts-v001/manifest.json")
    old_entries = {r["case_id"]: r for r in original_fm["cases"]}
    rebuilt_scores = []
    score_frame = (
        pd.read_parquet(BASE / "calibrated-results-v001/target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not args.smoke
        else None
    )
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        saved = load_npz(forecast / entry["path"], entry["sha256"])
        feature_file = load_npz(training / row["feature_path"], row["feature_sha256"])
        features = independently_rebuild_features(d, eval_fits[row["station"]])
        np.testing.assert_allclose(features, feature_file["features"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(saved["features"], feature_file["features"])
        tensor = normalized_features(feature_file["features"], scaler["mean"], scaler["scale"])
        old_entry = old_entries[entry["case_id"]]
        old = load_npz(PARENT / "peer-forecasts-v001" / old_entry["path"], old_entry["sha256"])
        rebuilt = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        z = ((d["context"] - d["mean"]) / d["scale"]).astype(np.float32)
        experts = {
            n: ((v - d["mean"][:2]) / d["scale"][:2]).astype(np.float32)
            for n, v in zip(d["stat_names"].tolist(), d["stat_targets"], strict=True)
        }
        with torch.inference_mode():
            for i, name in enumerate(saved["input_methods"].tolist()):
                alpha = (
                    gates[name](tensor).cpu().numpy()
                    if name in LEARNED
                    else np.full(2, 0.5, np.float32)
                )
                np.testing.assert_array_equal(alpha, saved["alphas"][i])
                context = z.copy()
                mix = (1 - alpha) * experts["local_ridge"] + alpha * experts["peer_ridge"]
                context[:, :2] = np.where(np.isfinite(z[:, :2]), z[:, :2], mix)
                context = context[:, d["keep"]]
                np.testing.assert_array_equal(context, saved["contexts_z"][i])
                canonical = np.ascontiguousarray(context.T)
                q = (
                    backbone(
                        context=torch.tensor(canonical, device="cuda"),
                        group_ids=torch.zeros(len(canonical), dtype=torch.long, device="cuda"),
                        num_output_patches=2,
                    )
                    .quantile_preds.float()
                    .cpu()
                    .numpy()
                )
                rebuilt[name] = q[:2, pipeline.quantiles.index(0.5), :24].T.astype(float)
        rebuilt["half_output_mix"] = (rebuilt["local_ridge"] + rebuilt["peer_ridge"]) * 0.5
        # Preserve the strided [method, horizon, target] views used in forecasting.
        # Packed target copies select a different float64 reduction layout.
        fixed_bank = np.stack([rebuilt[n] for n in fixed["methods"]])
        for name, weights in (
            ("calibrated_output_global", fixed["global"]),
            ("calibrated_output_station", fixed["stations"][row["station"]]),
        ):
            reconstructed = np.empty((24, 2))
            for target in (0, 1):
                matrix = fixed_bank[:, :, target]
                reconstructed[:, target] = np.asarray(weights[target]["weights"]) @ matrix
            rebuilt[name] = reconstructed
        names = saved["methods"].tolist()
        if len(names) != 44 or set(names) != set(rebuilt):
            raise ValueError("an evaluation method disappeared")
        np.testing.assert_array_equal(saved["points"], np.stack([rebuilt[n] for n in names]))
        if score_frame is not None:
            t = row["origin"]
            truth = original_records[row["station"]]["values"][t : t + 24, :2]
            for name in names:
                errors = []
                for slot in (0, 1):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        rebuilt[name][valid, slot]
                        - (truth[valid, slot] - d["mean"][slot]) / d["scale"][slot]
                    )
                    metrics = [float(abs(error).mean()), float(np.square(error).mean())]
                    actual = score_frame.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metrics, actual[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if actual["observed_count"] != valid.sum():
                        raise ValueError("scoring support differs")
                    errors.append(metrics)
                m = np.mean(errors, 0)
                rebuilt_scores.append(
                    {
                        "case_id": row["case_id"],
                        "panel": row["panel"],
                        "station": row["station"],
                        "method": name,
                        "mae": m[0],
                        "mse": m[1],
                    }
                )
    if rebuilt_scores:
        result = BASE / "calibrated-results-v001"
        frame = pd.DataFrame(rebuilt_scores)
        case_scores = pd.read_parquet(result / "case_scores.parquet")
        for key, saved in ((["case_id", "method"], case_scores),):
            expected = frame.set_index(key).sort_index()[["mae", "mse"]]
            actual = saved.set_index(key).sort_index()[["mae", "mse"]]
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
        stations = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean()
        summary = stations.groupby(["panel", "method"]).mean().sort_index()
        actual = (
            pd.read_csv(result / "summary.csv")
            .set_index(["panel", "method"])
            .sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(summary, actual, rtol=1e-12, atol=1e-12)
        leave = pd.read_csv(result / "leave_one_station_out.csv")
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
    if (
        any(m["parameter_sha256"] != digest for m in (fm, tm, source_fm, original_fm))
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("the frozen forecasting weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "source_cases": len(prepared["cases"]),
            "evaluation_cases": len(fm["cases"]),
            "source_raw_queries_replayed": raw_queries,
            "new_evaluation_queries_replayed": 4 * len(fm["cases"]),
            "optimizer_replays": replay,
            "forecast_difference": 0,
            "audit_version": "v002_preserve_fixed_output_reduction_layout",
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecast / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
