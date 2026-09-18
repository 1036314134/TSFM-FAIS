"""Independent cached-repair overlay checks and exact baseline forecast replay."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from audit_long_native import check_native_fields
from forecast_calibration_core import ROOT, load_npz, read_json
from future_query_core import array_digest, field_digests
from group_scope_experiment import grouped_forward
from long_repair_core import selected_repairs
from long_repair_experiment import INPUTS, NATIVE, PARENT
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_queries(row, pool, native):
    original = native["peer"]["context_z"]
    targets, window = row["target_count"], 192
    np.testing.assert_array_equal(original[:, -window:], pool["native"])
    np.testing.assert_array_equal(native["targets"]["context_z"], original[:targets])
    choices = selected_repairs(row["dataset"], pool["pool_names"].tolist())
    if row["repairs"] != [{"name": name, "index": index} for name, index in choices]:
        raise ValueError("the registered repair membership changed")
    queries = {}
    for name, index in choices:
        repaired = original.copy()
        values = pool["pool"][index]
        for target in range(targets):
            for step in range(window):
                position = original.shape[1] - window + step
                if np.isfinite(original[target, position]):
                    if values[step, target] != original[target, position]:
                        raise ValueError("a cached repair changed an original observation")
                else:
                    repaired[target, position] = values[step, target]
        np.testing.assert_array_equal(repaired[:, :-window], original[:, :-window])
        np.testing.assert_array_equal(repaired[targets:], original[targets:])
        queries["long_repair_" + name + "_peer"] = np.ascontiguousarray(repaired)
        queries["long_repair_" + name + "_targets"] = np.ascontiguousarray(repaired[:targets])
        if row["dataset"] == "hdb":
            queries["short_repair_" + name + "_peer"] = np.ascontiguousarray(repaired[:, -window:])
    for scope in ("peer", "targets"):
        queries["ordinary_" + scope] = native[scope]["context_z"]
    return queries


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "long-repair-inputs-v001",
        base / "long-repair-forecasts-v001",
        base / "long-repair-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-repair audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("long-repair forecast provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a fixed baseline definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered baseline runtime changed")
    original = {r["case_id"]: r for r in read_json(INPUTS / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    native_entries = {r["case_id"]: r for r in read_json(NATIVE / "manifest.json")["cases"]}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(original)
    ):
        raise ValueError("baseline population changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "long-repair-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("baseline score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 136089:
            raise ValueError("baseline target score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    if pipeline.model_context_length != 8192:
        raise ValueError("the registered long-context capacity changed")
    mid, calls, ordinary, short_restores, old_count, rebuilt_scores = (
        pipeline.quantiles.index(0.5),
        0,
        0,
        0,
        0,
        [],
    )
    for number, entry in enumerate(fm["cases"]):
        row, old_row = metadata[entry["case_id"]], original[entry["case_id"]]
        for key in (
            "dataset",
            "panel",
            "station",
            "origin",
            "horizon",
            "target_count",
            "source_column",
        ):
            if row[key] != old_row[key]:
                raise ValueError("baseline task metadata changed")
        if (
            row["input_sha256"] != old_row["sha256"]
            or row["parent_sha256"] != parents[row["case_id"]]["sha256"]
        ):
            raise ValueError("baseline input or old-output identity changed")
        native_row = native_entries[row["case_id"]]
        native_case = load_npz(NATIVE / native_row["path"], native_row["sha256"])
        native_queries = json.loads(str(native_case["queries"]))
        native = {}
        for scope in ("peer", "targets"):
            selected = next(q for q in native_queries if q["name"] == "native_long_prefix_" + scope)
            binding = {"path": str(NATIVE / selected["path"]), "sha256": selected["sha256"]}
            if row["native_queries"][scope] != binding:
                raise ValueError("a fixed native history changed")
            native[scope] = load_npz(binding["path"], binding["sha256"])
        pool = load_npz(row["input_path"], row["input_sha256"])
        definitions = reference_queries(row, pool, native)
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        old_count += len(points)
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        requests = json.loads(str(saved["queries"]))
        if {q["name"] for q in requests} != set(definitions) or len(requests) != len(definitions):
            raise ValueError("a fixed baseline query is missing")
        for request in requests:
            name, context = request["name"], definitions[request["name"]]
            stored = load_npz(forecasts / request["path"], request["sha256"])
            if (
                str(stored["context_sha256"]) != array_digest(context)
                or request["input_length"] != context.shape[1]
                or request["input_rows"] != len(context)
            ):
                raise ValueError("baseline effective input changed")
            raw, fields = grouped_forward(
                backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), row["horizon"]
            )
            np.testing.assert_array_equal(raw, stored["quantiles"])
            if field_digests(fields) != json.loads(str(stored["field_hashes"])):
                raise ValueError("baseline input fields changed")
            check_native_fields(context, fields, row["horizon"])
            point = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
            if name.startswith("ordinary_"):
                np.testing.assert_array_equal(
                    raw, native[name.removeprefix("ordinary_")]["quantiles"]
                )
                ordinary += 1
            else:
                points[name] = point
                points["half_var_" + name] = 0.5 * point + 0.5 * points["linear_var_direct"]
                if name in ("short_repair_knn_peer", "short_repair_gaussian_peer"):
                    reference = (
                        "target_knn" if name == "short_repair_knn_peer" else "target_gaussian"
                    )
                    np.testing.assert_array_equal(point, points[reference])
                    short_restores += 1
                calls += 1
        names = sorted(points)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([points[name] for name in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in points.items():
                values = []
                for slot in range(row["target_count"]):
                    valid = np.isfinite(truth[:, slot])
                    error = (
                        point[valid, slot]
                        - (truth[valid, slot] - pool["mean"][slot]) / pool["scale"][slot]
                    )
                    metric = [float(np.abs(error).mean()), float(np.square(error).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        metric, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("target future support changed")
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
        if (number + 1) % 10 == 0:
            _write_json(
                output / "progress.json",
                {"cases_audited": number + 1, "repair_queries_replayed": calls},
            )
            print(
                json.dumps({"cases_audited": number + 1, "repair_queries_replayed": calls}),
                flush=True,
            )
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        parameter_digest(backbone) != digest
        or digest != fm["parameter_sha256"]
        or calls != fm["repair_calls"]
        or ordinary != fm["ordinary_calls"]
    ):
        raise ValueError("frozen model or baseline query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "repair_queries_replayed": calls,
            "ordinary_restorations": ordinary,
            "old_hdb_short_controls_restored": short_restores,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "heldout_value_analysis": False,
            "wall_seconds": perf_counter() - started,
        },
    )
