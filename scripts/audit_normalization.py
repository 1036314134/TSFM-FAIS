"""Verify observed statistics, encoding invariants and all normalization outputs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import PARENT, ROOT, load_npz, read_json
from normalization_interface import constant_statistics, numpy_statistics
from peer_outage_core import sources
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def audit(base):
    forecast_root, output = base / "normalization-forecasts-v001", base / "normalization-audit-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed normalization audits")
    started = perf_counter()
    fm = read_json(forecast_root / "manifest.json")
    if fm["status"] != "completed":
        raise ValueError("finish all predictions before their audit")
    for path, sha in fm["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a registered definition or model implementation changed")
    inputs = Path(fm["input_root"])
    metadata = {r["case_id"]: r for r in read_json(inputs / "manifest.json")["cases"]}
    parent = read_json(PARENT / "peer-forecasts-v001/manifest.json")
    parent_entries = {r["case_id"]: r for r in parent["cases"]}
    if not fm["smoke"] and (
        len(fm["cases"]) != 231 or set(metadata) != {r["case_id"] for r in fm["cases"]}
    ):
        raise ValueError("the full registered population was not predicted")
    scores = None
    if not fm["smoke"]:
        result = base / "normalization-results-v001"
        study = read_json(result / "manifest.json")
        if study["forecast_sha256"] != file_sha256(forecast_root / "manifest.json"):
            raise ValueError("the score provenance changed")
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
    norm, mid = backbone.instance_norm, pipeline.quantiles.index(0.5)
    calls, restored_calls, fallback_rows, max_stat_difference = 0, 0, 0, 0.0
    reconstructed_scores = []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(inputs / row["path"], row["sha256"])
        saved = load_npz(forecast_root / entry["path"], entry["sha256"])
        old_entry = parent_entries[entry["case_id"]]
        old = load_npz(PARENT / "peer-forecasts-v001" / old_entry["path"], old_entry["sha256"])
        old_queries = {q["name"]: q for q in json.loads(str(old["queries"]))}
        rebuilt = dict(zip(old["methods"].tolist(), old["points"], strict=True))
        queries = json.loads(str(saved["queries"]))
        if len(queries) != 21:
            raise ValueError("a normalization intervention is missing")
        h = row["horizon"]
        for query in queries:
            p = query["parent_query"]
            if (
                p != old_queries[p["name"]]
                or p["name"].startswith("native_")
                or p["columns"][:2] != [0, 1]
            ):
                raise ValueError("a normalization query changed its original information set")
            raw = load_npz(PARENT / "peer-forecasts-v001/queries" / f"{p['key']}.npz", p["sha256"])
            new = load_npz(forecast_root / query["path"], query["sha256"])
            canonical, columns = raw["context_z"], p["columns"]
            original = np.array(
                ((d["context"][:, columns] - d["mean"][columns]) / d["scale"][columns]).T,
                dtype=np.float32,
                order="C",
            )
            mask = np.isfinite(original)
            np.testing.assert_array_equal(new["original_mask"], mask)
            np.testing.assert_array_equal(canonical[mask], original[mask])
            mode = query["mode"]
            expected_name = mode + "_norm_" + p["name"]
            if query["name"] != expected_name or (mode != "observed" and p["name"] != "peer_ridge"):
                raise ValueError("a statistical intervention changed identity")
            if float(new["eps"]) != norm.eps or bool(new["use_arcsinh"]) != norm.use_arcsinh:
                raise ValueError("normalizer configuration changed")
            loc_ref, scale_ref, effective, fallback = numpy_statistics(
                canonical, mask, mode, norm.eps
            )
            np.testing.assert_array_equal(new["effective_mask"], effective)
            np.testing.assert_array_equal(new["count"], effective.sum(-1, keepdims=True))
            np.testing.assert_array_equal(new["fallback"], fallback)
            context = torch.tensor(canonical, device="cuda")
            with torch.inference_mode():
                _, ordinary = norm(context)
            np.testing.assert_array_equal(new["ordinary_loc"], ordinary[0].cpu().numpy())
            np.testing.assert_array_equal(new["ordinary_scale"], ordinary[1].cpu().numpy())
            for key, reference, default, active in (
                ("chosen_loc", loc_ref, new["ordinary_loc"], ~fallback & (mode != "scale")),
                (
                    "chosen_scale",
                    scale_ref,
                    new["ordinary_scale"],
                    ~fallback & (mode != "location"),
                ),
            ):
                np.testing.assert_allclose(
                    new[key][active], reference[active], rtol=2e-6, atol=1e-6
                )
                np.testing.assert_array_equal(new[key][~active], default[~active])
                if active.any():
                    max_stat_difference = max(
                        max_stat_difference, float(abs(new[key][active] - reference[active]).max())
                    )
            fallback_rows += int(fallback.sum())
            loc = torch.tensor(new["chosen_loc"], device="cuda")
            scale = torch.tensor(new["chosen_scale"], device="cuda")
            captures = {"patches": []}

            def before_patch(_module, args, *, target=captures):
                target["patches"].append(args[0].detach().clone())

            def before_encoder(_module, _args, kwargs, *, target=captures):
                target["attention"] = kwargs["attention_mask"].detach().clone()
                target["groups"] = kwargs["group_ids"].detach().clone()

            handles = [
                backbone.input_patch_embedding.register_forward_pre_hook(before_patch),
                backbone.encoder.register_forward_pre_hook(before_encoder, with_kwargs=True),
            ]
            try:
                with torch.inference_mode(), constant_statistics(norm, loc, scale):
                    q = (
                        backbone(
                            context=context,
                            group_ids=torch.zeros(len(context), dtype=torch.long, device="cuda"),
                            num_output_patches=int(np.ceil(h / pipeline.model_output_patch_size)),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
            finally:
                for handle in handles:
                    handle.remove()
            np.testing.assert_array_equal(q, new["quantiles"])
            patch = backbone.chronos_config.input_patch_size
            if context.shape[-1] % patch or len(captures["patches"]) != 2:
                raise ValueError("the registered fixed-length patch interface changed")
            scaled = (context - loc) / scale
            if norm.use_arcsinh:
                scaled = torch.arcsinh(scaled)
            scaled = scaled.to(backbone.dtype)
            available = torch.isfinite(context).reshape(len(context), -1, patch)
            value = torch.where(available, scaled.reshape_as(available), 0.0)
            time = torch.arange(-context.shape[-1], 0, device="cuda", dtype=torch.float32).reshape(
                1, -1, patch
            )
            time = (
                time.expand(len(context), -1, -1) / backbone.chronos_config.time_encoding_scale
            ).to(backbone.dtype)
            expected_patch = torch.cat([time, value, available.to(backbone.dtype)], dim=-1)
            torch.testing.assert_close(captures["patches"][0], expected_patch, rtol=0, atol=0)
            output_patches = int(np.ceil(h / pipeline.model_output_patch_size))
            future_time = torch.arange(
                output_patches * pipeline.model_output_patch_size,
                device="cuda",
                dtype=torch.float32,
            ).reshape(1, output_patches, -1)
            future_time = (
                future_time.expand(len(context), -1, -1)
                / backbone.chronos_config.time_encoding_scale
            ).to(backbone.dtype)
            future_patch = torch.cat(
                [future_time, torch.zeros_like(future_time), torch.zeros_like(future_time)], dim=-1
            )
            torch.testing.assert_close(captures["patches"][1], future_patch, rtol=0, atol=0)
            extra = int(backbone.chronos_config.use_reg_token) + output_patches
            expected_attention = torch.cat(
                [
                    available.any(-1).to(backbone.dtype),
                    torch.ones((len(context), extra), device="cuda", dtype=backbone.dtype),
                ],
                dim=-1,
            )
            torch.testing.assert_close(captures["attention"], expected_attention, rtol=0, atol=0)
            torch.testing.assert_close(
                captures["groups"],
                torch.zeros(len(context), device="cuda", dtype=torch.long),
                rtol=0,
                atol=0,
            )
            rebuilt[query["name"]] = q[:2, mid, :h].T
            calls += 1
        for prefix in ("peer", "target"):
            values = [rebuilt["native_peer"].astype(np.float32)]
            values.extend(
                rebuilt["observed_norm_" + prefix + "_" + a] for a in d["actions"].tolist()
            )
            bank = np.stack(values).astype(np.float32)
            rebuilt["observed_norm_" + prefix + "_mean8"] = bank.mean(0, dtype=np.float64)
            rebuilt["observed_norm_" + prefix + "_median8"] = np.median(bank, 0).astype(float)
        primary = old_queries["peer_ridge"]
        raw = load_npz(
            PARENT / "peer-forecasts-v001/queries" / f"{primary['key']}.npz", primary["sha256"]
        )
        with torch.inference_mode():
            x = torch.tensor(raw["context_z"], device="cuda")
            ordinary_q = (
                backbone(
                    context=x,
                    group_ids=torch.zeros(len(x), device="cuda", dtype=torch.long),
                    num_output_patches=int(np.ceil(h / pipeline.model_output_patch_size)),
                )
                .quantile_preds.float()
                .cpu()
                .numpy()
            )
        np.testing.assert_array_equal(ordinary_q, raw["quantiles"])
        restored_calls += 1
        names = saved["methods"].tolist()
        if len(names) != 62 or set(names) != set(rebuilt):
            raise ValueError("a registered output is missing")
        np.testing.assert_array_equal(saved["points"], np.stack([rebuilt[n] for n in names]))
        if scores is not None:
            truth = records[row["station"]]["values"][row["origin"] : row["origin"] + h, :2]
            for name in names:
                metrics = []
                for slot in (0, 1):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        rebuilt[name][valid, slot]
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
        if restored_calls % 25 == 0:
            _write_json(output / "progress.json", {"cases": restored_calls, "queries": calls})
            print(json.dumps({"audited_cases": restored_calls, "queries": calls}), flush=True)
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
        or digest != parent["parameter_sha256"]
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("forecasting parameters changed")
    if calls != fm["new_method_calls"] or restored_calls != fm["ordinary_verification_calls"]:
        raise ValueError("model call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": restored_calls,
            "queries_replayed": calls,
            "ordinary_queries_restored": restored_calls,
            "prediction_difference": 0,
            "encoding_difference": 0,
            "attention_difference": 0,
            "maximum_statistic_reference_difference": max_stat_difference,
            "fallback_rows": fallback_rows,
            "score_rows": len(reconstructed_scores),
            "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
