"""Independent posterior and input-block moment checks with exact prediction replay."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_dynamic_posterior import reference_mixture
from forecast_calibration_core import ROOT, load_npz, read_json
from normalization_interface import constant_statistics
from peer_outage_core import sources
from posterior_token_core import fixed_embedding
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from scipy.special import ndtr

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_encoding(block, patches, mean, variance, loc, scale, mode, point_output):
    size = patches.shape[-1] // 3
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    weights = weights / np.sqrt(np.pi)
    center = (mean.astype(float) - loc.astype(float)) / scale.astype(float)
    spread = variance / scale.astype(float) ** 2
    transformed = np.arcsinh(center[..., None] + np.sqrt(2 * spread)[..., None] * nodes)
    average = (transformed * weights).sum(-1)
    var = ((transformed - average[..., None]) ** 2 * weights).sum(-1)
    point_values = patches[..., size : 2 * size].reshape(mean.shape)
    average = np.where(variance > 0, average, point_values)
    var = np.where(variance > 0, var, 0)
    features = patches.copy()
    if mode in ("posterior_mean", "posterior_moment", "unconditional_moment"):
        features[..., size : 2 * size] = average.reshape(*patches.shape[:-1], size).astype(
            patches.dtype
        )
    feature_matrix = features.reshape(-1, features.shape[-1]).astype(float)
    w1 = block.hidden_layer.weight.detach().cpu().numpy().astype(float)
    b1 = block.hidden_layer.bias.detach().cpu().numpy().astype(float)
    hidden_mean = feature_matrix @ w1.T + b1
    if mode == "posterior_mean":
        activation = np.maximum(hidden_mean, 0)
    else:
        hidden_variance = var.reshape(-1, size) @ (w1[:, size : 2 * size] ** 2).T
        positive = hidden_variance > 0
        s = np.sqrt(np.where(positive, hidden_variance, 1))
        standardized = hidden_mean / s
        activation = np.maximum(
            s * np.exp(-0.5 * standardized**2) / np.sqrt(2 * np.pi)
            + hidden_mean * ndtr(standardized),
            0,
        )
        activation = np.where(positive, activation, np.maximum(hidden_mean, 0))
    w2 = block.output_layer.weight.detach().cpu().numpy().astype(float)
    b2 = block.output_layer.bias.detach().cpu().numpy().astype(float)
    wr = block.residual_layer.weight.detach().cpu().numpy().astype(float)
    br = block.residual_layer.bias.detach().cpu().numpy().astype(float)
    expected = (activation @ w2.T + b2 + feature_matrix @ wr.T + br).reshape(point_output.shape)
    active = variance.reshape(*patches.shape[:-1], size).sum(-1) > 0
    expected[~active] = point_output[~active]
    return expected, active


def audit(base):
    inputs, forecasts, output = (
        base / n for n in ("moment-inputs-v001", "moment-forecasts-v001", "moment-audit-v001")
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed posterior-moment audits")
    started = perf_counter()
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["identity"]["input_sha256"] != file_sha256(
        inputs / "manifest.json"
    ):
        raise ValueError("moment predictions or their registered inputs are incomplete")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a posterior-moment definition changed")
    if fm["identity"]["script_sha256"] != file_sha256(
        ROOT / "scripts/posterior_token_experiment.py"
    ) or fm["identity"]["core_sha256"] != file_sha256(ROOT / "scripts/posterior_token_core.py"):
        raise ValueError("a posterior-moment runtime changed")
    parent_root = ROOT / "artifacts/iclr27-r36"
    parent_inputs = read_json(parent_root / "covariance-inputs-v001/manifest.json")
    models = {
        r["station"]: load_npz(parent_root / "covariance-inputs-v001" / r["path"], r["sha256"])
        for r in parent_inputs["evaluation_models"]
    }
    old_entries = {
        r["case_id"]: r
        for r in read_json(parent_root / "covariance-forecasts-v001/manifest.json")["cases"]
    }
    static_root = ROOT / "artifacts/iclr27-r35"
    static_entries = {
        r["case_id"]: r
        for r in read_json(static_root / "dynamic-forecasts-v001/manifest.json")["cases"]
    }
    base_names = read_json(parent_root / "covariance-training-v001/fixed_portfolios.json")[
        "methods"
    ]
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if set(metadata) != {r["case_id"] for r in fm["cases"]} or (
        not fm["smoke"] and len(metadata) != 231
    ):
        raise ValueError("the registered posterior-moment population changed")
    scores = None
    if not fm["smoke"]:
        result = base / "moment-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
    records, _ = sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    block, norm = backbone.input_patch_embedding, backbone.instance_norm
    if backbone.config.dense_act_fn != "relu" or block.use_layer_norm or not norm.use_arcsinh:
        raise ValueError("the registered posterior block architecture changed")
    mid, calls, restored_calls, max_embedding_difference, max_stat_difference = (
        pipeline.quantiles.index(0.5),
        0,
        0,
        0.0,
        0.0,
    )
    reconstructed_scores = []
    for number, entry in enumerate(fm["cases"]):
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        parent = load_npz(
            parent_root / "covariance-inputs-v001" / row["parent_path"], row["parent_sha256"]
        )
        model = models[row["station"]]
        selected = np.flatnonzero(parent["keep"])
        expected_mean = np.array(
            (
                (parent["base_values"][:, selected] - parent["mean"][selected])
                / parent["scale"][selected]
            ).T,
            dtype=np.float32,
            order="C",
        )
        observed = np.isfinite(parent["context"][:, selected]).T
        np.testing.assert_array_equal(d["mean_z"], expected_mean)
        np.testing.assert_array_equal(d["observed"], observed)
        np.testing.assert_array_equal(d["mean"], parent["mean"])
        np.testing.assert_array_equal(d["scale"], parent["scale"])
        prior = model["covariance"] - 1e-6 * np.eye(17)
        for t, values in enumerate(parent["context"]):
            obs, missing = np.flatnonzero(np.isfinite(values)), np.flatnonzero(~np.isfinite(values))
            expected = np.zeros((17, 17))
            if len(missing):
                conditional = prior[np.ix_(missing, missing)].copy()
                if len(obs):
                    cross = prior[np.ix_(missing, obs)]
                    conditional -= cross @ np.linalg.solve(
                        model["covariance"][np.ix_(obs, obs)], cross.T
                    )
                expected[np.ix_(missing, missing)] = conditional
            expected = expected[np.ix_(selected, selected)]
            np.testing.assert_allclose(d["covariance"][t], expected, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(
                d["roots"][t] @ d["roots"][t].T, expected, rtol=1e-9, atol=1e-10
            )
            np.testing.assert_array_equal(
                d["roots"][t, observed[:, t]], np.zeros((observed[:, t].sum(), len(selected)))
            )
        np.testing.assert_array_equal(
            d["variance"], np.diagonal(d["covariance"], axis1=1, axis2=2).T
        )
        expected_prior_var = np.maximum(np.diag(prior)[selected], 0)[:, None] * (~observed)
        np.testing.assert_array_equal(d["unconditional_variance"], expected_prior_var)
        seed = hashlib.sha256(("r37|6103|" + row["case_id"]).encode()).hexdigest()[:16]
        if str(d["seed"]) != seed:
            raise ValueError("static conditional sampling seed changed")
        noise = np.random.default_rng(int(seed, 16)).standard_normal((8, 192, len(selected)))
        delta = np.zeros_like(noise)
        for t in range(192):
            delta[:, t] = noise[:, t] @ d["roots"][t].T
        delta[:, observed.T] = 0
        differences = (
            np.stack([delta, -delta], axis=1).reshape(16, 192, len(selected)).transpose(0, 2, 1)
        )
        samples = d["mean_z"].astype(float)[None] + differences
        samples[:, observed] = d["mean_z"][observed]
        np.testing.assert_array_equal(
            np.ascontiguousarray(samples, dtype=np.float32), d["samples_z"]
        )
        for sample in d["samples_z"]:
            np.testing.assert_array_equal(sample[observed], d["mean_z"][observed])
        old_entry = old_entries[row["case_id"]]
        old = load_npz(
            parent_root / "covariance-forecasts-v001" / old_entry["path"], old_entry["sha256"]
        )
        methods = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        var_point = methods["linear_var_direct"]
        control_names = [
            n for n in base_names if n not in ("full_static_point", "linear_var_direct")
        ]
        control_names += [
            "forecast_covariance",
            "imputation_covariance_targets",
            "imputation_covariance_all",
        ]
        for name in control_names:
            methods["half_var_" + name] = 0.5 * methods[name] + 0.5 * var_point
        static_entry = static_entries[row["case_id"]]
        static = load_npz(
            static_root / "dynamic-forecasts-v001" / static_entry["path"], static_entry["sha256"]
        )
        static_query = next(
            q for q in json.loads(str(static["queries"])) if q["name"] == "full_static"
        )
        old_query = load_npz(
            static_root / "dynamic-forecasts-v001" / static_query["path"], static_query["sha256"]
        )
        np.testing.assert_array_equal(d["mean_z"], old_query["context_z"])
        horizon = row["horizon"]
        static_q = old_query["quantiles"][:2, :, :horizon].transpose(1, 2, 0)
        mean_q, median_q = reference_mixture(static_q[None], pipeline.quantiles)
        for label, value in (("mean", mean_q), ("median", median_q)):
            methods["static_quantile_" + label] = value
            methods["hybrid_static_quantile_" + label] = 0.5 * value + 0.5 * var_point
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        queries = json.loads(str(saved["queries"]))
        if len(queries) != 22:
            raise ValueError("a moment query or Monte Carlo sample is missing")
        mean_tensor = torch.tensor(d["mean_z"], device="cuda")
        with torch.inference_mode():
            _, ordinary = norm.forward(mean_tensor)
        sample_outputs = {}
        for query in queries:
            name = query["name"]
            qdata = load_npz(forecasts / query["path"], query["sha256"])
            v = d["unconditional_variance"] if name == "unconditional_moment" else d["variance"]
            np.testing.assert_array_equal(qdata["loc"], ordinary[0].cpu().numpy())
            positive = v.mean(-1, keepdims=True) > 0
            reference_scale = np.sqrt(
                d["mean_z"].astype(float).var(-1, keepdims=True)
                + v.mean(-1, keepdims=True) * (191 / 192)
            )
            np.testing.assert_allclose(
                qdata["scale"][positive], reference_scale[positive], rtol=2e-6, atol=1e-6
            )
            np.testing.assert_array_equal(
                qdata["scale"][~positive], ordinary[1].cpu().numpy()[~positive]
            )
            if positive.any():
                max_stat_difference = max(
                    max_stat_difference,
                    float(abs(qdata["scale"][positive] - reference_scale[positive]).max()),
                )
            loc, scale = (
                torch.tensor(qdata["loc"], device="cuda"),
                torch.tensor(qdata["scale"], device="cuda"),
            )
            replacement = (
                None
                if qdata["replacement"].size == 0
                else torch.tensor(qdata["replacement"], device="cuda")
            )
            if replacement is not None:
                with torch.inference_mode(), constant_statistics(norm, loc, scale):
                    patches, _, _ = backbone._prepare_patched_context(mean_tensor)
                    point_embedding = block.forward(patches)
                size = backbone.chronos_config.input_patch_size
                active = v.reshape(len(selected), -1, size).sum(-1) > 0
                if name == "mc_embedding":
                    reference_samples = []
                    with torch.inference_mode():
                        for sample in d["samples_z"]:
                            normalized, _ = norm.forward(
                                torch.tensor(sample, device="cuda"), (loc, scale)
                            )
                            fields = patches.clone()
                            fields[..., size : 2 * size] = normalized.reshape(
                                *patches.shape[:-1], size
                            ).to(patches.dtype)
                            reference_samples.append(block.forward(fields).cpu().numpy())
                    reference = np.stack(reference_samples).astype(float).mean(0)
                    reference[~active] = point_embedding.cpu().numpy()[~active]
                else:
                    reference, active = reference_encoding(
                        block,
                        patches.cpu().numpy(),
                        d["mean_z"],
                        v,
                        qdata["loc"],
                        qdata["scale"],
                        name,
                        point_embedding.cpu().numpy(),
                    )
                np.testing.assert_array_equal(
                    qdata["replacement"][~active], point_embedding.cpu().numpy()[~active]
                )
                np.testing.assert_allclose(
                    qdata["replacement"][active], reference[active], rtol=2e-5, atol=2e-5
                )
                if active.any():
                    max_embedding_difference = max(
                        max_embedding_difference,
                        float(abs(qdata["replacement"][active] - reference[active]).max()),
                    )
            elif name not in ("posterior_norm", *[f"mc_sample_{i:02d}" for i in range(16)]):
                raise ValueError("a registered encoding replacement was omitted")
            if name.startswith("mc_sample_"):
                context = torch.tensor(d["samples_z"][int(name.rsplit("_", 1)[1])], device="cuda")
            else:
                context = mean_tensor
            captured = {"patches": []}

            def before_patch(_block, args, *, target=captured):
                target["patches"].append(args[0].detach().clone())

            def before_encoder(_encoder, _args, kwargs, *, target=captured):
                target["attention"] = kwargs["attention_mask"].detach().clone()

            handles = [
                block.register_forward_pre_hook(before_patch),
                backbone.encoder.register_forward_pre_hook(before_encoder, with_kwargs=True),
            ]
            from contextlib import nullcontext

            try:
                with (
                    torch.inference_mode(),
                    constant_statistics(norm, loc, scale),
                    (
                        fixed_embedding(block, replacement)
                        if replacement is not None
                        else nullcontext()
                    ),
                ):
                    actual = (
                        backbone(
                            context=context,
                            group_ids=torch.zeros(len(context), device="cuda", dtype=torch.long),
                            num_output_patches=int(
                                np.ceil(horizon / pipeline.model_output_patch_size)
                            ),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
            finally:
                for handle in handles:
                    handle.remove()
            np.testing.assert_array_equal(actual, qdata["quantiles"])
            if len(captured["patches"]) != 2:
                raise ValueError("the context/future embedding call count changed")
            patch_size = backbone.chronos_config.input_patch_size
            context_fields, future_fields = captured["patches"]
            torch.testing.assert_close(
                context_fields[..., 2 * patch_size :],
                torch.ones_like(context_fields[..., 2 * patch_size :]),
                rtol=0,
                atol=0,
            )
            time = (
                torch.arange(-192, 0, device="cuda", dtype=torch.float32)
                .reshape(1, -1, patch_size)
                .expand(len(context), -1, -1)
                / backbone.chronos_config.time_encoding_scale
            )
            torch.testing.assert_close(
                context_fields[..., :patch_size], time.to(backbone.dtype), rtol=0, atol=0
            )
            torch.testing.assert_close(
                future_fields[..., pipeline.model_output_patch_size :],
                torch.zeros_like(future_fields[..., pipeline.model_output_patch_size :]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                captured["attention"], torch.ones_like(captured["attention"]), rtol=0, atol=0
            )
            calls += 1
            if name.startswith("mc_sample_"):
                sample_outputs[name] = actual[:2, :, :horizon].transpose(1, 2, 0)
            else:
                value = actual[:2, mid, :horizon].T.astype(float)
                methods[name + "_component"] = value
                methods["hybrid_" + name] = 0.5 * value + 0.5 * var_point
        samples_q = np.stack([sample_outputs[f"mc_sample_{i:02d}"] for i in range(16)])
        mean_q, median_q = reference_mixture(samples_q, pipeline.quantiles)
        for name, value in (
            ("mc_point_mean16", samples_q[:, mid].mean(0, dtype=np.float64)),
            ("mc_point_median16", np.median(samples_q[:, mid], 0).astype(float)),
            ("mc_quantile_mean16", mean_q),
            ("mc_quantile_median16", median_q),
        ):
            methods[name + "_component"] = value
            methods["hybrid_" + name] = 0.5 * value + 0.5 * var_point
        with torch.inference_mode():
            ordinary_q = (
                backbone(
                    context=mean_tensor,
                    group_ids=torch.zeros(len(mean_tensor), device="cuda", dtype=torch.long),
                    num_output_patches=int(np.ceil(horizon / pipeline.model_output_patch_size)),
                )
                .quantile_preds.float()
                .cpu()
                .numpy()
            )
        np.testing.assert_array_equal(ordinary_q, old_query["quantiles"])
        restored_calls += 1
        names = saved["methods"].tolist()
        if len(names) != 140 or set(names) != set(methods):
            raise ValueError("a moment output or matched hybrid control is missing")
        np.testing.assert_array_equal(saved["points"], np.stack([methods[n] for n in names]))
        if scores is not None:
            t = row["origin"]
            truth = records[row["station"]]["values"][t : t + horizon, :2]
            for name in names:
                metric = []
                for slot in (0, 1):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        methods[name][valid, slot]
                        - (truth[valid, slot] - d["mean"][slot]) / d["scale"][slot]
                    )
                    values = [float(abs(error).mean()), float(np.square(error).mean())]
                    expected = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        values, expected[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if expected["observed_count"] != valid.sum():
                        raise ValueError("the outcome support changed")
                    metric.append(values)
                values = np.mean(metric, 0)
                reconstructed_scores.append(
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
    if reconstructed_scores:
        frame = pd.DataFrame(reconstructed_scores)
        case = pd.read_parquet(result / "case_scores.parquet")
        np.testing.assert_allclose(
            frame.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
            case.set_index(["case_id", "method"]).sort_index()[["mae", "mse"]],
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
        or parameter_digest(backbone) != digest
        or calls != fm["new_method_calls"]
        or restored_calls != fm["ordinary_verification_calls"]
    ):
        raise ValueError("backbone parameters or call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(fm["cases"]),
            "queries_replayed": calls,
            "ordinary_queries_restored": restored_calls,
            "prediction_difference": 0,
            "maximum_independent_embedding_difference": max_embedding_difference,
            "maximum_statistic_reference_difference": max_stat_difference,
            "score_rows": len(reconstructed_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
