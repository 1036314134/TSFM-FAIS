"""Verify long-history causality, native patch coverage, input units and all scores."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import ROOT, load_npz, read_json
from group_scope_experiment import grouped_forward
from long_native_core import CONTEXT_LIMIT
from long_native_experiment import INPUTS, PARENT, query_names, source_banks
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_inputs(row, data, bank):
    source = load_npz(row["source_input_path"], row["source_input_sha256"])
    short = load_npz(row["source_attention_path"], row["source_attention_sha256"])
    t, length = row["origin"], min(row["origin"], CONTEXT_LIMIT)
    if row["history_start"] != t - length or row["actual_context_length"] != length:
        raise ValueError("long-history time bounds changed")
    if row["prefix_end"] > t or (row["dataset"] == "hdb" and t > 672):
        raise ValueError("future data entered a long-history definition")
    history = np.asarray(bank[np.arange(t - length, t)], dtype=float).copy()
    observed = np.isfinite(source["context"])
    np.testing.assert_array_equal(history[-192:][observed], source["context"][observed])
    history[-192:] = source["context"]
    selected = short["selected"].tolist()
    original = np.stack([history[:, column].astype(np.float32) for column in selected])
    standardized = np.stack(
        [
            ((history[:, column] - short["mean"][column]) / short["scale"][column]).astype(
                np.float32
            )
            for column in selected
        ]
    )
    expected = {
        "native_long_raw_peer": original,
        "native_long_raw_targets": original[: row["target_count"]].copy(),
        "native_long_prefix_peer": standardized,
        "native_long_prefix_targets": standardized[: row["target_count"]].copy(),
        "native_short_restoration": short["native"],
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(data[name], value)
    np.testing.assert_array_equal(standardized[:, -192:], short["native"])
    for field in ("mean", "scale", "selected"):
        np.testing.assert_array_equal(data[field], short[field])
    return expected


def check_native_fields(context, fields, horizon):
    n, length = context.shape
    padded = 16 * int(np.ceil(length / 16))
    patches = padded // 16
    if fields["context_fields"].shape != (n, patches, 48):
        raise ValueError("the full native context was not encoded")
    observed = np.pad(
        np.isfinite(context).astype(np.float32), ((0, 0), (padded - length, 0))
    ).reshape(n, patches, 16)
    np.testing.assert_array_equal(fields["context_fields"][..., 32:], observed)
    time = np.arange(-padded, 0, dtype=np.float32).reshape(1, patches, 16) / 8192
    np.testing.assert_array_equal(
        fields["context_fields"][..., :16], np.broadcast_to(time, (n, patches, 16))
    )
    np.testing.assert_array_equal(
        fields["context_fields"][..., 16:32][observed == 0],
        np.zeros(int((observed == 0).sum()), dtype=np.float32),
    )
    future_patches = int(np.ceil(horizon / 16))
    if fields["future_fields"].shape != (n, future_patches, 48):
        raise ValueError("forecast horizon patch count changed")
    np.testing.assert_array_equal(
        fields["future_fields"][..., 16:], np.zeros((n, future_patches, 32), dtype=np.float32)
    )


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "long-native-inputs-v001",
        base / "long-native-forecasts-v001",
        base / "long-native-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed default-context audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("long-native forecast provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a registered long-native definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a frozen long-native runtime changed")
    original = {r["case_id"]: r for r in read_json(INPUTS / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(original)
    ):
        raise ValueError("default-context population changed")
    banks = source_banks()
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "long-native-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("native score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 119169:
            raise ValueError("native target-score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    if pipeline.model_context_length != CONTEXT_LIMIT or fm["context_limit"] != CONTEXT_LIMIT:
        raise ValueError("the loaded model's context limit changed")
    mid, calls, restored, old_count, rebuilt_scores = pipeline.quantiles.index(0.5), 0, 0, 0, []
    for number, entry in enumerate(fm["cases"]):
        row, old_input = metadata[entry["case_id"]], original[entry["case_id"]]
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
            if row[key] != old_input[key]:
                raise ValueError("native case metadata changed")
        if (
            row["source_attention_sha256"] != old_input["sha256"]
            or row["parent_sha256"] != parents[row["case_id"]]["sha256"]
        ):
            raise ValueError("original input or output lineage changed")
        data = load_npz(inputs / row["path"], row["sha256"])
        expected_inputs = reference_inputs(row, data, banks[row["dataset"]][row["station"]])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        old_count += len(points)
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        requests = json.loads(str(saved["queries"]))
        if len(requests) != 5 or {q["name"] for q in requests} != set(query_names()):
            raise ValueError("a fixed native query is missing")
        for request in requests:
            name = request["name"]
            query = load_npz(forecasts / request["path"], request["sha256"])
            context = expected_inputs[name]
            np.testing.assert_array_equal(query["context_z"], context)
            if request["input_length"] != context.shape[1] or request["input_rows"] != len(context):
                raise ValueError("native resource accounting shape changed")
            if request["input_units"] != ("original" if "_raw_" in name else "prefix_standardized"):
                raise ValueError("input unit declaration changed")
            raw, fields = grouped_forward(
                backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), row["horizon"]
            )
            np.testing.assert_array_equal(raw, query["quantiles"])
            for key, value in fields.items():
                np.testing.assert_array_equal(value, query[key])
            check_native_fields(context, fields, row["horizon"])
            point = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
            if request["input_units"] == "original":
                point = np.stack(
                    [
                        (point[:, j] - data["mean"][j]) / data["scale"][j]
                        for j in range(row["target_count"])
                    ],
                    axis=1,
                )
            if name == "native_short_restoration":
                np.testing.assert_array_equal(point, points[row["native_name"]])
                restored += 1
            else:
                points[name] = point
                points["half_var_" + name] = 0.5 * point + 0.5 * points["linear_var_direct"]
            calls += 1
        names = sorted(points)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([points[name] for name in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in points.items():
                metrics = []
                for slot in range(row["target_count"]):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        point[valid, slot]
                        - (truth[valid, slot] - data["mean"][slot]) / data["scale"][slot]
                    )
                    values = [float(np.abs(error).mean()), float(np.square(error).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        values, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("target observation support changed")
                    metrics.append(values)
                mae, mse = np.mean(metrics, axis=0)
                rebuilt_scores.append(
                    {
                        **{k: row[k] for k in ("case_id", "panel", "station")},
                        "method": name,
                        "mae": mae,
                        "mse": mse,
                    }
                )
        if (number + 1) % 10 == 0:
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
        or restored != len(ids)
    ):
        raise ValueError("native base weights or query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "queries_replayed": calls,
            "short_queries_restored": restored,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "heldout_value_analysis": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
