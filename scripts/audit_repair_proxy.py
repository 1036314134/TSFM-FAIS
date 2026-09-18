"""Independent proxy layout/readout audit and exact frozen-backbone replay."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import ROOT, load_npz, read_json
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from reliability_attention_experiment import case_truth, truth_sources
from repair_proxy_core import POINT_NAMES
from repair_proxy_experiment import INPUTS, PARENT, POOL, captured_forward, proxy_input

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_queries(data):
    native, pool = data["native"], data["pool"]
    count, _, targets = pool.shape
    if np.isfinite(native[:targets]).all():
        return {}
    rows = np.stack(
        [pool[candidate, :, target] for candidate in range(count) for target in range(targets)]
    )
    extras = {
        "raw_plus_pool": rows,
        "raw_plus_gaussian": np.stack(
            [pool[int(data["gaussian_index"]), :, j] for j in range(targets)]
        ),
        "raw_plus_knn": np.stack([pool[int(data["knn_index"]), :, j] for j in range(targets)]),
        "raw_plus_pool_median": np.stack(
            [np.median(pool[:, :, j], axis=0) for j in range(targets)]
        ),
        "raw_plus_duplicate_pool": np.stack(
            [native[j] for _ in range(count) for j in range(targets)]
        ),
        "raw_plus_duplicate_single": native[:targets].copy(),
    }
    result = {
        name: np.vstack([native, values]).astype(np.float32) for name, values in extras.items()
    }
    result["proxy_pool_only"] = rows.astype(np.float32)
    return result


def reference_points(data, raw, horizon, mid):
    count, _, targets = data["pool"].shape
    points = {
        name: value[:targets, mid, :horizon].T.astype(float)
        for name, value in raw.items()
        if name != "proxy_pool_only"
    }
    for name, prefix, start in (
        ("raw_plus_pool", "raw_plus_pool_proxy", len(data["native"])),
        ("proxy_pool_only", "proxy_pool_only", 0),
    ):
        candidates = np.stack(
            [
                np.stack(
                    [
                        raw[name][start + c * targets + j, mid, :horizon].astype(float)
                        for j in range(targets)
                    ],
                    axis=1,
                )
                for c in range(count)
            ]
        )
        points[prefix + "_mean"] = candidates.mean(0)
        points[prefix + "_median"] = np.median(candidates, axis=0)
    return points


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "proxy-inputs-v001",
        base / "proxy-forecasts-v001",
        base / "proxy-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair-proxy audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("proxy prediction provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a frozen repair-proxy definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered proxy runtime changed")
    pools = {r["case_id"]: r for r in read_json(POOL / "manifest.json")["cases"]}
    original = {r["case_id"]: r for r in read_json(INPUTS / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(original)
    ):
        raise ValueError("registered proxy population changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "proxy-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("proxy score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 91825:
            raise ValueError("target score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, calls, ordinary, fallbacks, old_count, anchor_checks, rebuilt_scores = (
        pipeline.quantiles.index(0.5),
        0,
        0,
        0,
        0,
        0,
        [],
    )
    for number, entry in enumerate(fm["cases"]):
        row, source_row = metadata[entry["case_id"]], original[entry["case_id"]]
        for key in (
            "dataset",
            "panel",
            "station",
            "origin",
            "horizon",
            "target_count",
            "source_column",
            "native_name",
        ):
            if row[key] != source_row[key]:
                raise ValueError("evaluation metadata changed")
        if (
            row["source_attention_sha256"] != source_row["sha256"]
            or row["parent_sha256"] != parents[row["case_id"]]["sha256"]
        ):
            raise ValueError("original input or control identity changed")
        data = load_npz(inputs / row["path"], row["sha256"])
        rebuilt, bindings, fallback = proxy_input(source_row, pools)
        for key in rebuilt:
            np.testing.assert_array_equal(data[key], rebuilt[key])
        for key, value in bindings.items():
            if row[key] != value:
                raise ValueError("upstream candidate identity or status changed")
        if (
            fallback != row["complete_target_fallback"]
            or fallback != entry["complete_target_fallback"]
        ):
            raise ValueError("complete-target fallback changed")
        fallbacks += int(fallback)
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        old_count += len(points)
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        requests = json.loads(str(saved["queries"]))
        contexts = reference_queries(data)
        if {r["name"] for r in requests} != set(contexts) | {"ordinary"} or len(requests) != len(
            contexts
        ) + 1:
            raise ValueError("a registered query or fallback is missing")
        native_request = next(r for r in requests if r["name"] == "ordinary")
        native = load_npz(forecasts / native_request["path"], native_request["sha256"])
        qvalues = {}
        for request in requests:
            name = request["name"]
            query = load_npz(forecasts / request["path"], request["sha256"])
            expected_context = data["native"] if name == "ordinary" else contexts[name]
            np.testing.assert_array_equal(query["context_z"], expected_context)
            if request["input_rows"] != len(expected_context):
                raise ValueError("query size accounting changed")
            anchors = 0 if name == "proxy_pool_only" else len(data["native"])
            raw, captures = captured_forward(
                backbone, pipeline, query["context_z"], row["horizon"], anchors
            )
            np.testing.assert_array_equal(raw, query["quantiles"])
            for key, value in captures.items():
                np.testing.assert_array_equal(value, query[key])
                np.testing.assert_array_equal(value, native[key])
            if anchors and name != "ordinary":
                anchor_checks += 1
            if name == "ordinary":
                np.testing.assert_array_equal(
                    raw[: row["target_count"], mid, : row["horizon"]].T.astype(float),
                    points[row["native_name"]],
                )
                ordinary += 1
            else:
                qvalues[name] = raw
                calls += 1
        additions = (
            {name: points[row["native_name"]].copy() for name in POINT_NAMES}
            if fallback
            else reference_points(data, qvalues, row["horizon"], mid)
        )
        if set(additions) != set(POINT_NAMES):
            raise ValueError("proxy point rules changed")
        for name, value in additions.items():
            points[name] = value
            points["half_var_" + name] = 0.5 * value + 0.5 * points["linear_var_direct"]
        names = sorted(points)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([points[n] for n in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in points.items():
                values = []
                for slot in range(row["target_count"]):
                    valid = np.isfinite(truth[:, slot])
                    errors = (
                        point[valid, slot]
                        - (truth[valid, slot] - data["mean"][slot]) / data["scale"][slot]
                    )
                    metrics = [float(np.abs(errors).mean()), float(np.square(errors).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metrics, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("observed future support changed")
                    values.append(metrics)
                mae, mse = np.mean(values, axis=0)
                rebuilt_scores.append(
                    {
                        **{k: row[k] for k in ("case_id", "panel", "station")},
                        "method": name,
                        "mae": mae,
                        "mse": mse,
                    }
                )
        if (number + 1) % 25 == 0:
            _write_json(
                output / "progress.json",
                {"cases_audited": number + 1, "proxy_queries_replayed": calls},
            )
            print(
                json.dumps({"cases_audited": number + 1, "proxy_queries_replayed": calls}),
                flush=True,
            )
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        parameter_digest(backbone) != digest
        or digest != fm["parameter_sha256"]
        or calls != fm["proxy_calls"]
        or ordinary != fm["ordinary_calls"]
        or fallbacks != fm["fallback_cases"]
    ):
        raise ValueError("model identity or proxy query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "proxy_queries_replayed": calls,
            "ordinary_restorations": ordinary,
            "complete_target_fallbacks": fallbacks,
            "original_anchor_statistic_and_field_checks": anchor_checks,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "heldout_value_analysis": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
