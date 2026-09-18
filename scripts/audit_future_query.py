"""Independent future-key mask definitions, frozen query replay and metric checks."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_forecast_correlation import check_metrics
from forecast_calibration_core import ROOT, load_npz, read_json
from future_query_core import DEFINITIONS, array_digest
from future_query_experiment import INPUTS, PARENT, SHORT, controlled_forward
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def check_masks(saved, context, horizon, targets, policy):
    rows, length = context.shape
    patches, future = int(np.ceil(length / 16)), int(np.ceil(horizon / 16))
    start = patches + 1
    seen = (
        np.pad(np.isfinite(context), ((0, 0), (patches * 16 - length, 0)))
        .reshape(rows, patches, 16)
        .any(-1)
    )
    valid = np.concatenate([seen, np.ones((rows, 1 + future), dtype=bool)], axis=1)
    dtype = saved["original_time_mask"].dtype
    minimum = np.finfo(dtype).min
    time = (1 - valid[:, None, None, :].astype(dtype)) * minimum
    group = (
        1
        - np.broadcast_to(valid.T[:, None, None, :], (start + future, 1, rows, rows)).astype(dtype)
    ) * minimum
    np.testing.assert_array_equal(time, saved["original_time_mask"])
    np.testing.assert_array_equal(group, saved["original_group_mask"])
    if int(saved["future_start"]) != start or int(saved["targets"]) != targets:
        raise ValueError("future/target boundary changed")
    if policy in ("aux_time", "aux_both"):
        time[targets:, ..., start:] = minimum
    if policy in ("aux_group", "aux_both"):
        group[start:, ..., targets:] = minimum
    if policy == "readonly":
        time[..., start:] = minimum
        for query in range(rows):
            for key in range(rows):
                if query != key:
                    group[start:, :, query, key] = minimum
    np.testing.assert_array_equal(time, saved["time_mask"])
    np.testing.assert_array_equal(group, saved["group_mask"])
    np.testing.assert_array_equal(saved["time_stride"], saved["original_time_stride"])
    np.testing.assert_array_equal(saved["group_stride"], saved["original_group_stride"])
    if str(saved["context_sha256"]) != array_digest(context):
        raise ValueError("canonical model input changed")


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "query-role-inputs-v001",
        base / "query-role-forecasts-v001",
        base / "query-role-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed future-query audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("future-query forecast provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a frozen future-query definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered future-query runtime changed")
    original = {r["case_id"]: r for r in read_json(INPUTS / "manifest.json")["cases"]}
    parents = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    shorts = {r["case_id"]: r for r in read_json(SHORT / "manifest.json")["cases"]}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(original)
    ):
        raise ValueError("future-query population changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "query-role-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("future-query score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 125709:
            raise ValueError("future-query score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, interventions, ordinary, property_calls, old_count, rebuilt_scores = (
        pipeline.quantiles.index(0.5),
        0,
        0,
        0,
        0,
        [],
    )
    property_results = []
    for number, entry in enumerate(fm["cases"]):
        row, reference = metadata[entry["case_id"]], original[entry["case_id"]]
        for key in (
            "dataset",
            "panel",
            "station",
            "origin",
            "horizon",
            "target_count",
            "source_column",
        ):
            if row[key] != reference[key]:
                raise ValueError("future-query case identity changed")
        if (
            row["source_attention_sha256"] != reference["sha256"]
            or row["parent_sha256"] != parents[row["case_id"]]["sha256"]
        ):
            raise ValueError("future-query original data lineage changed")
        data = load_npz(row["source_attention_path"], row["source_attention_sha256"])
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        old_count += len(points)
        short = load_npz(SHORT / shorts[row["case_id"]]["path"], shorts[row["case_id"]]["sha256"])
        short_queries, long_queries = (
            json.loads(str(short["queries"])),
            json.loads(str(parent["queries"])),
        )
        contexts, base_quantiles = {}, {}
        for name, choices, root, required in (
            ("native_short", short_queries, SHORT, "restore_native"),
            ("gaussian_short", short_queries, SHORT, "restore_gaussian"),
            ("native_long", long_queries, PARENT, "native_long_prefix_peer"),
        ):
            selected = next(q for q in choices if q["name"] == required)
            binding = {"path": str(root / selected["path"]), "sha256": selected["sha256"]}
            if row["base_queries"][name] != binding:
                raise ValueError("a baseline input was replaced")
            saved_base = load_npz(binding["path"], binding["sha256"])
            contexts[name], base_quantiles[name] = saved_base["context_z"], saved_base["quantiles"]
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        requests = json.loads(str(saved["queries"]))
        expected_names = set(DEFINITIONS) | {"ordinary_" + key for key in contexts}
        if len(requests) != 9 or {r["name"] for r in requests} != expected_names:
            raise ValueError("a future-query intervention or baseline is missing")
        output_quantiles, ordinary_fields = {}, {}
        h, targets = row["horizon"], row["target_count"]
        for request in requests:
            name = request["name"]
            context_name, policy = (
                DEFINITIONS[name]
                if name in DEFINITIONS
                else (name.removeprefix("ordinary_"), "none")
            )
            if request["context"] != context_name or request["policy"] != policy:
                raise ValueError("a future-query policy changed")
            context = contexts[context_name]
            stored = load_npz(forecasts / request["path"], request["sha256"])
            check_masks(stored, context, h, targets, policy)
            raw, trace, hashes = controlled_forward(backbone, pipeline, context, h, targets, policy)
            np.testing.assert_array_equal(raw, stored["quantiles"])
            for field, value in trace.items():
                np.testing.assert_array_equal(value, stored[field])
            if hashes != json.loads(str(stored["field_hashes"])):
                raise ValueError("input statistic or encoding field replay changed")
            output_quantiles[name] = raw
            if policy == "none":
                np.testing.assert_array_equal(raw, base_quantiles[context_name])
                ordinary_fields[context_name] = hashes
                ordinary += 1
            else:
                points[name] = raw[:targets, mid, :h].T.astype(float)
                points["half_var_" + name] = 0.5 * points[name] + 0.5 * points["linear_var_direct"]
                interventions += 1
        for request in requests:
            fields = json.loads(
                str(load_npz(forecasts / request["path"], request["sha256"])["field_hashes"])
            )
            if fields != ordinary_fields[request["context"]]:
                raise ValueError("the attention policy altered upstream inputs or statistics")
        properties = json.loads(str(saved["properties"]))
        expected_properties = (
            {
                "neutral_all_targets": ("aux_both", len(contexts["native_long"]), h, False),
                "ordinary_perturbed": ("none", targets, h, True),
                "aux_both_perturbed": ("aux_both", targets, h, True),
                "readonly_perturbed": ("readonly", targets, h, True),
                "readonly_extended": ("readonly", targets, h + 16, False),
                "ordinary_extended": ("none", targets, h + 16, False),
            }
            if row["property_case"]
            else {}
        )
        if {p["name"] for p in properties} != set(expected_properties) or len(properties) != len(
            expected_properties
        ):
            raise ValueError("the preflight property set changed")
        for prop in properties:
            policy, declared_targets, horizon, perturb = expected_properties[prop["name"]]
            if (prop["policy"], prop["targets"], prop["horizon"], prop["perturb"]) != (
                policy,
                declared_targets,
                horizon,
                perturb,
            ):
                raise ValueError("a preflight property definition changed")
            context = contexts["native_long"]
            stored = load_npz(forecasts / prop["path"], prop["sha256"])
            check_masks(stored, context, horizon, declared_targets, policy)
            raw, trace, hashes = controlled_forward(
                backbone, pipeline, context, horizon, declared_targets, policy, perturb
            )
            np.testing.assert_array_equal(raw, stored["quantiles"])
            for field, value in trace.items():
                np.testing.assert_array_equal(value, stored[field])
            if hashes != json.loads(str(stored["field_hashes"])):
                raise ValueError("property input fields changed")
            for field in ("context_fields", "loc", "scale"):
                if hashes[field] != ordinary_fields["native_long"][field]:
                    raise ValueError(
                        "a property test altered observed history or its normalization"
                    )
            baseline = output_quantiles["ordinary_native_long"]
            main = output_quantiles["query_aux_both_native_long"]
            readonly = output_quantiles["query_readonly_native_long"]
            name = prop["name"]
            if name == "neutral_all_targets":
                np.testing.assert_array_equal(raw, baseline)
                difference = 0.0
            elif name in ("aux_both_perturbed", "readonly_perturbed"):
                reference_q = main if name == "aux_both_perturbed" else readonly
                np.testing.assert_array_equal(raw[:targets], reference_q[:targets])
                difference = 0.0
            elif name == "readonly_extended":
                np.testing.assert_allclose(
                    raw[:targets, :, :h], readonly[:targets, :, :h], rtol=2e-5, atol=2e-5
                )
                difference = float(abs(raw[:targets, :, :h] - readonly[:targets, :, :h]).max())
            else:
                difference = float(abs(raw[:targets, :, :h] - baseline[:targets, :, :h]).max())
            if difference != prop["reference_difference"]:
                raise ValueError("property-difference accounting changed")
            property_results.append(
                {"case_id": row["case_id"], "name": name, "difference": difference}
            )
            property_calls += 1
        names = sorted(points)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([points[name] for name in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in points.items():
                values = []
                for slot in range(targets):
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
                        raise ValueError("observed outcome support changed")
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
                {"cases_audited": number + 1, "interventions_replayed": interventions},
            )
            print(
                json.dumps({"cases_audited": number + 1, "interventions_replayed": interventions}),
                flush=True,
            )
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        parameter_digest(backbone) != digest
        or digest != fm["parameter_sha256"]
        or interventions != fm["intervention_calls"]
        or ordinary != fm["ordinary_calls"]
        or property_calls != fm["property_calls"]
    ):
        raise ValueError("future-query weights or call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "interventions_replayed": interventions,
            "ordinary_restorations": ordinary,
            "property_queries_replayed": property_calls,
            "property_results": property_results,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "heldout_value_analysis": False,
            "wall_seconds": perf_counter() - started,
        },
    )
