"""Verify candidate isolation, variable scope, joint-query identity and score readouts."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import ROOT, load_npz, read_json
from group_scope_experiment import INPUTS, PARENT, grouped_forward
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_queries(data):
    count, _, targets = data["pool"].shape
    rows = [data["pool"][c, :, slot] for c in range(count) for slot in range(targets)]
    matrix = np.array(rows, dtype=np.float32, order="C")
    native = np.array(
        [data["native"][slot] for slot in range(targets)], dtype=np.float32, order="C"
    )
    return {
        "isolated_candidates": (
            matrix,
            np.asarray([c for c in range(count) for _ in range(targets)], dtype=np.int64),
        ),
        "joint_candidates": (matrix.copy(), np.asarray([0] * len(matrix), dtype=np.int64)),
        "native_targets": (native, np.asarray([0] * targets, dtype=np.int64)),
        "target_anchor_candidates": (
            np.vstack([native, matrix]),
            np.asarray([0] * (targets + len(matrix)), dtype=np.int64),
        ),
    }


def reference_points(data, raw, horizon, mid):
    count, _, targets = data["pool"].shape
    values = {}
    for name, offset in (
        ("isolated_candidates", 0),
        ("joint_candidates", 0),
        ("target_anchor_candidates", targets),
    ):
        values[name] = np.stack(
            [
                np.stack(
                    [
                        raw[name][offset + c * targets + slot, mid, :horizon].astype(float)
                        for slot in range(targets)
                    ],
                    axis=1,
                )
                for c in range(count)
            ]
        )
    isolated, joint, anchored = (
        values[name]
        for name in ("isolated_candidates", "joint_candidates", "target_anchor_candidates")
    )
    native = raw["native_targets"][:, mid, :horizon].T.astype(float)
    plus = np.stack([*isolated, native])
    result = {"isolated_" + name: isolated[c] for c, name in enumerate(data["pool_names"].tolist())}
    for label, array in (
        ("independent_proxy", isolated),
        ("joint_reference", joint),
        ("independent_plus_native", plus),
        ("target_anchor_proxy", anchored),
    ):
        result[label + "_mean"] = array.mean(0)
        result[label + "_median"] = np.median(array, axis=0)
    result["native_targets_only"] = native
    result["target_anchor_pool"] = raw["target_anchor_candidates"][
        :targets, mid, :horizon
    ].T.astype(float)
    return result


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "scope-inputs-v001",
        base / "scope-forecasts-v001",
        base / "scope-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed scope diagnostics")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("scope forecast provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a frozen scope diagnostic definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered scope runtime changed")
    originals = {r["case_id"]: r for r in read_json(INPUTS / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(originals)
    ):
        raise ValueError("scope diagnostic population changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "scope-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("scope score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 114809:
            raise ValueError("scope target score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, calls, serial_calls, restored, old_count, maximum_serial, rebuilt_scores = (
        pipeline.quantiles.index(0.5),
        0,
        0,
        0,
        0,
        0.0,
        [],
    )
    for number, entry in enumerate(fm["cases"]):
        row, original = metadata[entry["case_id"]], originals[entry["case_id"]]
        for key in (
            "dataset",
            "panel",
            "station",
            "origin",
            "horizon",
            "target_count",
            "source_column",
            "complete_target_fallback",
        ):
            if row[key] != original[key]:
                raise ValueError("scope case metadata changed")
        if (
            row["input_sha256"] != original["sha256"]
            or row["parent_sha256"] != parents[row["case_id"]]["sha256"]
        ):
            raise ValueError("scope inputs or baseline identity changed")
        data = load_npz(row["input_path"], row["input_sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        old_count += len(points)
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        definitions, queries = reference_queries(data), json.loads(str(saved["queries"]))
        if len(queries) != 4 or {q["name"] for q in queries} != set(definitions):
            raise ValueError("a matched scope query is missing")
        raw_queries, captured_queries = {}, {}
        for query in queries:
            name = query["name"]
            stored = load_npz(forecasts / query["path"], query["sha256"])
            context, groups = definitions[name]
            np.testing.assert_array_equal(stored["context_z"], context)
            np.testing.assert_array_equal(stored["group_ids"], groups)
            if query["input_rows"] != len(context) or query["logical_groups"] != len(
                np.unique(groups)
            ):
                raise ValueError("logical group or tensor size accounting changed")
            raw, fields = grouped_forward(backbone, pipeline, context, groups, row["horizon"])
            np.testing.assert_array_equal(raw, stored["quantiles"])
            for key, value in fields.items():
                np.testing.assert_array_equal(value, stored[key])
            raw_queries[name], captured_queries[name] = raw, fields
            calls += 1
        for key in captured_queries["joint_candidates"]:
            np.testing.assert_array_equal(
                captured_queries["joint_candidates"][key],
                captured_queries["isolated_candidates"][key],
            )
            np.testing.assert_array_equal(
                captured_queries["target_anchor_candidates"][key][: row["target_count"]],
                captured_queries["native_targets"][key],
            )
        if not row["complete_target_fallback"]:
            old = next(
                q for q in json.loads(str(parent["queries"])) if q["name"] == "proxy_pool_only"
            )
            reference = load_npz(PARENT / old["path"], old["sha256"])
            np.testing.assert_array_equal(
                reference["context_z"], definitions["joint_candidates"][0]
            )
            np.testing.assert_array_equal(reference["quantiles"], raw_queries["joint_candidates"])
            restored += 1
        if row["dataset"] == "hdb":
            np.testing.assert_array_equal(
                raw_queries["native_targets"][:1, mid, : row["horizon"]].T.astype(float),
                points["native_target192"],
            )
        serial_queries = json.loads(str(saved["serial_queries"]))
        expected_serial = len(data["pool"]) if fm["smoke"] else 0
        if len(serial_queries) != expected_serial:
            raise ValueError("serial preflight budget changed")
        for serial in serial_queries:
            candidate, target_count = serial["candidate"], row["target_count"]
            reference = load_npz(forecasts / serial["path"], serial["sha256"])
            start = candidate * target_count
            context = np.ascontiguousarray(
                definitions["isolated_candidates"][0][start : start + target_count]
            )
            np.testing.assert_array_equal(reference["context_z"], context)
            raw, _ = grouped_forward(
                backbone, pipeline, context, np.zeros(target_count, dtype=np.int64), row["horizon"]
            )
            np.testing.assert_array_equal(raw, reference["quantiles"])
            batched = raw_queries["isolated_candidates"][start : start + target_count]
            np.testing.assert_allclose(raw, batched, rtol=2e-5, atol=2e-5)
            maximum_serial = max(maximum_serial, float(abs(raw - batched).max()))
            serial_calls += 1
        additions = reference_points(data, raw_queries, row["horizon"], mid)
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
                    error = (
                        point[valid, slot]
                        - (truth[valid, slot] - data["mean"][slot]) / data["scale"][slot]
                    )
                    metric = [float(np.abs(error).mean()), float(np.square(error).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metric, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("target observation support changed")
                    values.append(metric)
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
                output / "progress.json", {"cases_audited": number + 1, "queries_replayed": calls}
            )
            print(json.dumps({"cases_audited": number + 1, "queries_replayed": calls}), flush=True)
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        parameter_digest(backbone) != digest
        or digest != fm["parameter_sha256"]
        or calls != fm["model_calls"]
        or serial_calls != fm["serial_verification_calls"]
        or restored != fm["old_joint_queries_restored"]
    ):
        raise ValueError("frozen model or diagnostic query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "queries_replayed": calls,
            "serial_queries_replayed": serial_calls,
            "old_joint_queries_restored": restored,
            "maximum_serial_difference": maximum_serial,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "heldout_value_analysis": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
