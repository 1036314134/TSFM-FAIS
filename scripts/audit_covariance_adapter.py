"""Verify the calibrated covariance, source objectives, fixed controls and outputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from covariance_adapter_core import CovarianceAdapter, geometry, masked_smooth_mae, repaired_context
from forecast_calibration_core import ROOT, calibration_sources, fixed_mae_fit, load_npz, read_json
from peer_outage_core import augmented, sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.utility_experiment import _write_json, file_sha256


def check(path, sha):
    if file_sha256(Path(path)) != sha:
        raise ValueError("a frozen covariance study artifact changed")


def independent_static(z, center, covariance):
    result = np.tile(center, (len(z), 1))
    for t, value in enumerate(z):
        observed, missing = np.where(np.isfinite(value))[0], np.where(~np.isfinite(value))[0]
        result[t, observed] = value[observed]
        if len(observed) and len(missing):
            result[t, missing] += covariance[np.ix_(missing, observed)] @ np.linalg.solve(
                covariance[np.ix_(observed, observed)], value[observed] - center[observed]
            )
    return result


def reference_var(z, model, horizon):
    mean, covariance = model["initial_mean"].copy(), model["initial_covariance"].copy()
    for t, values in enumerate(z):
        if t:
            mean = model["a"] @ mean + model["b"]
            covariance = model["a"] @ covariance @ model["a"].T + model["q"]
        observed, missing = np.where(np.isfinite(values))[0], np.where(~np.isfinite(values))[0]
        if len(observed):
            prior = mean.copy()
            posterior = np.zeros_like(covariance)
            mean[observed] = values[observed]
            if len(missing):
                cross = covariance[np.ix_(missing, observed)]
                cov = covariance[np.ix_(observed, observed)]
                mean[missing] = prior[missing] + cross @ np.linalg.solve(
                    cov, values[observed] - prior[observed]
                )
                posterior[np.ix_(missing, missing)] = covariance[
                    np.ix_(missing, missing)
                ] - cross @ np.linalg.solve(cov, cross.T)
            covariance = (posterior + posterior.T) / 2
    result = []
    for _ in range(horizon):
        mean = model["a"] @ mean + model["b"]
        result.append(mean[:2].copy())
    return np.stack(result)


def audit(base):
    inputs, source_forecasts, training, forecasts, output = (
        base / n
        for n in (
            "covariance-inputs-v001",
            "covariance-source-forecasts-v001",
            "covariance-training-v001",
            "covariance-forecasts-v001",
            "covariance-audit-v001",
        )
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed covariance audits")
    started = perf_counter()
    prepared, sfm, tm, fm = (
        read_json(p / "manifest.json") for p in (inputs, source_forecasts, training, forecasts)
    )
    if any(m["status"] != "completed" or m["smoke"] != prepared["smoke"] for m in (sfm, tm, fm)):
        raise ValueError("the corresponding covariance stages are incomplete")
    for path, sha in prepared["identity"].items():
        check(path, sha)
    for m in (sfm, tm, fm):
        check(inputs / "manifest.json", m["input_sha256"])
    check(source_forecasts / "manifest.json", tm["source_forecast_sha256"])
    check(training / "manifest.json", fm["training_sha256"])
    for path, sha in tm["files"].items():
        check(training / path, sha)
    records, peers, _ = calibration_sources()
    full = {s: augmented(s, records, peers)[0] for s in records}
    source_models, var_models = {}, {}
    for entry in prepared["source_models"]:
        model = load_npz(inputs / entry["path"], entry["sha256"])
        var = load_npz(inputs / entry["var_path"], entry["var_sha256"])
        prefix = full[entry["station"]][: entry["prefix_end"]]
        mean, scale = np.nanmean(prefix, 0), np.nanstd(prefix, 0, ddof=0)
        scale = np.where(scale <= 1e-12, 1, scale)
        np.testing.assert_array_equal(mean, model["mean"])
        np.testing.assert_array_equal(scale, model["scale"])
        z = (prefix - mean) / scale
        complete = np.isfinite(z).all(1)
        covariance = np.cov(z[complete], rowvar=False, bias=True)
        covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance)) + 1e-6 * np.eye(17)
        np.testing.assert_allclose(model["covariance"], covariance, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(model["center"], z[complete].mean(0))
        if int(model["support"]) != complete.sum() or complete.sum() < 128:
            raise ValueError("static fitting support changed")
        for key, other in (
            ("mean", "mean"),
            ("scale", "scale"),
            ("center", "initial_mean"),
            ("covariance", "initial_covariance"),
        ):
            np.testing.assert_array_equal(model[key], var[other])
        pairs = complete[:-1] & complete[1:]
        x, y = z[:-1][pairs], z[1:][pairs]
        design = np.vstack([(x - x.mean(0)) / np.sqrt(len(x)), np.sqrt(0.001) * np.eye(17)])
        target = np.vstack([(y - y.mean(0)) / np.sqrt(len(y)), np.zeros((17, 17))])
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0].T
        np.testing.assert_allclose(var["a_unconstrained"], coefficient, rtol=1e-8, atol=1e-10)
        radius = np.max(abs(np.linalg.eigvals(var["a_unconstrained"])))
        np.testing.assert_array_equal(
            var["a"], var["a_unconstrained"] * min(1, 0.99 / max(radius, 1e-12))
        )
        np.testing.assert_array_equal(var["b"], y.mean(0) - var["a"] @ x.mean(0))
        error = y - (x @ var["a"].T + var["b"])
        cov = np.cov(error, rowvar=False, bias=True)
        np.testing.assert_allclose(
            var["q"],
            0.95 * cov + 0.05 * np.diag(np.diag(cov)) + 1e-6 * np.eye(17),
            rtol=1e-12,
            atol=1e-12,
        )
        source_models[entry["station"]], var_models[entry["station"]] = model, var
    source_parent = ROOT / "artifacts/iclr27-r32/calibration-inputs-v001"
    parent_bank = ROOT / "artifacts/iclr27-r32/calibration-forecasts-v001"
    parent_entries = {r["case_id"]: r for r in read_json(parent_bank / "manifest.json")["cases"]}
    source_entries = {r["case_id"]: r for r in sfm["cases"]}
    if set(source_entries) != {r["case_id"] for r in prepared["source_cases"]}:
        raise ValueError("source forecasts are incomplete")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    backbone.eval().requires_grad_(False)
    source_data, labels, bank, truths, names = {}, {}, [], [], None
    source_calls = 0
    for row in prepared["source_cases"]:
        d = load_npz(inputs / row["path"], row["sha256"])
        label = load_npz(inputs / row["label_path"], row["label_sha256"])
        original = load_npz(source_parent / row["parent_input_path"], row["parent_input_sha256"])
        np.testing.assert_array_equal(d["context"], original["context"])
        model = source_models[row["station"]]
        np.testing.assert_array_equal(d["mean"], model["mean"])
        np.testing.assert_array_equal(d["scale"], model["scale"])
        t, age = row["origin"], row["outage_age"]
        if t - 192 < row["prefix_end"] or t + 24 > row["calibration_end"]:
            raise ValueError("source calibration crosses a time boundary")
        unmasked = full[row["station"]][t - 192 : t].copy()
        context = unmasked.copy()
        context[-age:, :2] = np.nan
        if row["outage_pattern"] in ("local_pollutants", "regional_pollutants"):
            context[-age:, :6] = np.nan
        if row["outage_pattern"] == "regional_pollutants":
            context[-age:, 11:17] = np.nan
        np.testing.assert_array_equal(d["context"], context)
        hidden = np.isfinite(unmasked) & ~np.isfinite(context)
        np.testing.assert_array_equal(hidden, label["artificial_mask"])
        expected_hidden = np.where(hidden, (unmasked - model["mean"]) / model["scale"], np.nan)
        np.testing.assert_array_equal(expected_hidden, label["hidden_all"])
        expected_future = (full[row["station"]][t : t + 24, :2] - model["mean"][:2]) / model[
            "scale"
        ][:2]
        np.testing.assert_array_equal(expected_future, label["future"])
        z = (context - model["mean"]) / model["scale"]
        expected = independent_static(z, model["center"], model["covariance"])
        np.testing.assert_allclose(
            (d["base_values"] - model["mean"]) / model["scale"], expected, rtol=1e-9, atol=1e-10
        )
        observed = np.isfinite(context)
        np.testing.assert_array_equal(d["base_values"][observed], context[observed])
        np.testing.assert_allclose(
            d["direct_var"], reference_var(z, var_models[row["station"]], 24), rtol=1e-9, atol=1e-10
        )
        e = source_entries[row["case_id"]]
        f = load_npz(source_forecasts / e["path"], e["sha256"])
        old_entry = parent_entries[row["case_id"]]
        old = load_npz(parent_bank / old_entry["path"], old_entry["sha256"])
        reconstructed = {
            n: old["points"][i]
            for i, n in enumerate(old["methods"].tolist())
            if "_source_" not in n
        }
        for index, name in enumerate(("full_static_point", "target_static_point")):
            raw = d["base_values"].copy() if index == 0 else context.copy()
            if index:
                raw[:, :2] = d["base_values"][:, :2]
            canonical = np.array(
                ((raw[:, d["keep"]] - d["mean"][d["keep"]]) / d["scale"][d["keep"]]).T,
                dtype=np.float32,
                order="C",
            )
            np.testing.assert_array_equal(canonical, f["contexts_z"][index])
            with torch.inference_mode():
                reconstructed[name] = (
                    chronos_median(pipeline, torch.tensor(canonical.T, device="cuda"), 24, [0, 1])
                    .cpu()
                    .numpy()
                )
            source_calls += 1
        reconstructed["linear_var_direct"] = d["direct_var"]
        current_names = f["methods"].tolist()
        if (
            len(current_names) != 28
            or set(current_names) != set(reconstructed)
            or (names is not None and current_names != names)
        ):
            raise ValueError("a matched source control is missing")
        np.testing.assert_array_equal(
            f["points"], np.stack([reconstructed[n] for n in current_names])
        )
        names = current_names
        source_data[row["case_id"]], labels[row["case_id"]] = d, label
        bank.append(f["points"])
        truths.append(label["future"])
    bank, truths = np.stack(bank), np.stack(truths)
    fixed = read_json(training / "fixed_portfolios.json")
    if names != fixed["methods"]:
        raise ValueError("the source portfolio order changed")
    for key, selected in [
        ("global", list(range(len(bank)))),
        *[
            (s, [i for i, r in enumerate(prepared["source_cases"]) if r["station"] == s])
            for s in fixed["stations"]
        ],
    ]:
        records_fit = fixed["global"] if key == "global" else fixed["stations"][key]
        for target in (0, 1):
            certificate = fixed_mae_fit(bank[selected, :, :, target], truths[selected, :, target])
            fitted = records_fit[target]
            np.testing.assert_allclose(
                certificate["objective"], fitted["objective"], rtol=1e-10, atol=1e-10
            )
            weight = np.asarray(fitted["weights"])
            if abs(weight.sum() - 1) > 1e-7 or weight.min() < -1e-7:
                raise ValueError("a portfolio left its simplex")
            predicted = np.zeros((len(selected), 24))
            for index, value in enumerate(weight):
                predicted += value * bank[selected, index, :, target]
            actual = np.nanmean(abs(predicted - truths[selected, :, target]), axis=1).mean()
            np.testing.assert_allclose(actual, certificate["objective"], rtol=1e-7, atol=1e-7)
    order = load_npz(training / "source_order.npz")
    ids = [r["case_id"] for r in prepared["source_cases"]]
    np.testing.assert_array_equal(order["case_ids"], ids)
    np.testing.assert_array_equal(
        order["indices"], np.random.default_rng(5101).permutation(len(ids))
    )
    trace = read_json(training / "training_trace.json")
    nets, replays = {}, []
    source_by_id = {r["case_id"]: r for r in prepared["source_cases"]}
    for entry in tm["learned"]:
        name = entry["method"]
        check(training / entry["path"], entry["sha256"])
        check(training / f"{name}-before-last.pt", entry["before_last_sha256"])
        before = torch.load(
            training / f"{name}-before-last.pt", map_location="cuda", weights_only=False
        )
        after = torch.load(training / entry["path"], map_location="cuda", weights_only=False)
        if (
            before["case_id"] != ids[order["indices"][-1]]
            or entry["steps"] != len(ids)
            or entry["parameter_count"] != 34
        ):
            raise ValueError("the last update or model capacity changed")
        net = CovarianceAdapter().cuda()
        net.load_state_dict(before["model"])
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.001)
        optimizer.load_state_dict(before["optimizer"])
        row = source_by_id[before["case_id"]]
        g = geometry(source_data[row["case_id"]], source_models[row["station"]])
        repaired = repaired_context(net, g)
        lab = labels[row["case_id"]]
        if name == "forecast_covariance":
            pred = chronos_median(pipeline, repaired[:, g["keep"]], 24, [0, 1])
            loss = masked_smooth_mae(
                pred, torch.tensor(lab["future"], dtype=torch.float32, device="cuda")
            )
        elif name == "imputation_covariance_targets":
            loss = masked_smooth_mae(
                repaired[:, :2],
                torch.tensor(lab["hidden_all"][:, :2], dtype=torch.float32, device="cuda"),
            )
        else:
            loss = masked_smooth_mae(
                repaired, torch.tensor(lab["hidden_all"], dtype=torch.float32, device="cuda")
            )
        loss = loss + 0.001 * net.penalty()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for key, value in after["model"].items():
            torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)
        for index, state in after["optimizer"]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    optimizer.state_dict()["state"][index][key], value, rtol=0, atol=0
                )
        log = [r for r in trace if r["method"] == name]
        if [r["case_id"] for r in log] != [ids[i] for i in order["indices"]]:
            raise ValueError("source update ordering changed")
        np.testing.assert_array_equal(
            [float(loss.detach()), float(norm)], [log[-1]["loss"], log[-1]["gradient_norm"]]
        )
        transform = net.transform().detach().cpu().numpy()
        if (
            np.linalg.norm(transform - np.eye(17), 2) > 0.5 + 1e-12
            or np.linalg.svd(transform, compute_uv=False).min() < 0.5 - 1e-12
        ):
            raise ValueError("the bounded covariance transform violated its spectral constraint")
        nets[name] = net.eval()
        replays.append({"method": name, "parameter_difference": 0, "optimizer_difference": 0})
    baseline_root = ROOT / "artifacts/iclr27-r35"
    old_entries = {
        r["case_id"]: r
        for r in read_json(baseline_root / "dynamic-forecasts-v001/manifest.json")["cases"]
    }
    evaluation_models = {}
    for entry in prepared["evaluation_models"]:
        model = load_npz(inputs / entry["path"], entry["sha256"])
        old = load_npz(
            baseline_root / "dynamic-inputs-v001" / entry["parent_path"], entry["parent_sha256"]
        )
        for key, prior in (
            ("mean", "mean"),
            ("scale", "scale"),
            ("center", "initial_mean"),
            ("covariance", "initial_covariance"),
        ):
            np.testing.assert_array_equal(model[key], old[prior])
        evaluation_models[entry["station"]] = model
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if set(metadata) != {r["case_id"] for r in fm["cases"]} or (
        not fm["smoke"] and len(metadata) != 231
    ):
        raise ValueError("the evaluation population is incomplete")
    scores = None
    if not fm["smoke"]:
        result = base / "covariance-results-v001"
        check(forecasts / "manifest.json", read_json(result / "manifest.json")["forecast_sha256"])
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
    original_records, _ = sources()
    zero = CovarianceAdapter().cuda()
    evaluation_calls, baseline_calls, rebuilt_scores, max_conditional_difference = 0, 0, [], 0.0
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        prior = load_npz(
            baseline_root / "dynamic-inputs-v001" / row["parent_path"], row["parent_sha256"]
        )
        for key in ("context", "mean", "scale", "keep"):
            np.testing.assert_array_equal(d[key], prior[key])
        np.testing.assert_array_equal(d["base_values"], prior["static_values"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        old_entry = old_entries[row["case_id"]]
        old = load_npz(
            baseline_root / "dynamic-forecasts-v001" / old_entry["path"], old_entry["sha256"]
        )
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        model = evaluation_models[row["station"]]
        g = geometry(d, model)
        with torch.inference_mode():
            baseline_context = repaired_context(zero, g)[:, g["keep"]]
            old_query = next(
                q for q in json.loads(str(old["queries"])) if q["name"] == "full_static"
            )
            raw = load_npz(
                baseline_root / "dynamic-forecasts-v001" / old_query["path"], old_query["sha256"]
            )
            np.testing.assert_array_equal(
                baseline_context.T.contiguous().cpu().numpy(), raw["context_z"]
            )
            base_prediction = (
                chronos_median(pipeline, baseline_context, row["horizon"], [0, 1]).cpu().numpy()
            )
            np.testing.assert_array_equal(base_prediction, methods["full_static_point"])
            baseline_calls += 1
            z = (d["context"] - d["mean"]) / d["scale"]
            reference_zero = independent_static(z, model["center"], model["covariance"])
            for index, name in enumerate(saved["adapted_methods"].tolist()):
                net = nets[name]
                transform = (
                    np.eye(17)
                    + net.amplitude
                    * np.tanh(net.correction.cpu().numpy())
                    @ net.projection.cpu().numpy().T
                )
                covariance = transform @ model["covariance"] @ transform.T
                np.testing.assert_allclose(
                    covariance, saved["covariances"][index], rtol=1e-12, atol=1e-12
                )
                if np.linalg.eigvalsh(covariance).min() <= 0:
                    raise ValueError("an adapted conditional covariance is not positive definite")
                reference = independent_static(z, model["center"], covariance)
                expected = g["anchor"].cpu().numpy() + (reference - reference_zero).astype(
                    np.float32
                )
                observed = np.isfinite(z)
                expected[observed] = z.astype(np.float32)[observed]
                context = repaired_context(net, g)[:, g["keep"]]
                canonical = context.T.contiguous().cpu().numpy()
                np.testing.assert_allclose(
                    canonical, expected[:, d["keep"]].T, rtol=1e-6, atol=1e-6
                )
                max_conditional_difference = max(
                    max_conditional_difference,
                    float(abs(canonical - expected[:, d["keep"]].T).max()),
                )
                np.testing.assert_array_equal(canonical, saved["contexts_z"][index])
                np.testing.assert_array_equal(
                    canonical.T[np.isfinite(z[:, d["keep"]])],
                    z.astype(np.float32)[:, d["keep"]][np.isfinite(z[:, d["keep"]])],
                )
                methods[name] = (
                    chronos_median(pipeline, context, row["horizon"], [0, 1]).cpu().numpy()
                )
                evaluation_calls += 1
        methods["half_static_var"] = (
            0.5 * methods["full_static_point"] + 0.5 * methods["linear_var_direct"]
        )
        for name, records_fit in (
            ("covariance_portfolio_global", fixed["global"]),
            ("covariance_portfolio_station", fixed["stations"][row["station"]]),
        ):
            prediction = np.zeros((row["horizon"], 2))
            for j, method in enumerate(fixed["methods"]):
                weights = np.asarray([record["weights"][j] for record in records_fit])
                prediction += methods[method] * weights[None, :]
            methods[name] = prediction
        names = saved["methods"].tolist()
        if len(names) != 87 or set(names) != set(methods):
            raise ValueError("a registered covariance output is missing")
        np.testing.assert_array_equal(saved["points"], np.stack([methods[n] for n in names]))
        if scores is not None:
            t, h = row["origin"], row["horizon"]
            truth = original_records[row["station"]]["values"][t : t + h, :2]
            for name in names:
                metrics = []
                for slot in (0, 1):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        methods[name][valid, slot]
                        - (truth[valid, slot] - d["mean"][slot]) / d["scale"][slot]
                    )
                    value = [float(abs(error).mean()), float(np.square(error).mean())]
                    actual = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        value, actual[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if actual["observed_count"] != valid.sum():
                        raise ValueError("outcome support changed")
                    metrics.append(value)
                value = np.mean(metrics, 0)
                rebuilt_scores.append(
                    {
                        "case_id": row["case_id"],
                        "panel": row["panel"],
                        "station": row["station"],
                        "method": name,
                        "mae": value[0],
                        "mse": value[1],
                    }
                )
    if rebuilt_scores:
        frame = pd.DataFrame(rebuilt_scores)
        actual = pd.read_parquet(result / "case_scores.parquet")
        np.testing.assert_allclose(
            frame.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            actual.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
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
        any(m["parameter_sha256"] != digest for m in (sfm, tm, fm))
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("the fixed forecasting backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "source_cases": len(source_data),
            "evaluation_cases": len(fm["cases"]),
            "source_queries_replayed": source_calls,
            "evaluation_queries_replayed": evaluation_calls,
            "zero_adapter_forecasts_replayed": baseline_calls,
            "optimizer_replays": replays,
            "prediction_difference": 0,
            "maximum_independent_input_difference": max_conditional_difference,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
