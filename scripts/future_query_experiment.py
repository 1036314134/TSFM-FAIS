"""Target-role future-key inference with frozen short, repaired and long contexts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from forecast_calibration_core import ROOT, load_npz, read_json
from future_query_core import DEFINITIONS, array_digest, field_digests, future_key_policy
from group_scope_experiment import grouped_forward
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster
from readout_peer_outage import aggregate
from reliability_attention_experiment import case_truth, truth_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

PROTOCOL = ROOT / "docs/iclr2027/R46_FUTURE_QUERY_PROTOCOL.md"
INPUTS = ROOT / "artifacts/iclr27-r40/attention-inputs-v001"
SHORT = ROOT / "artifacts/iclr27-r40/attention-forecasts-v001"
PARENT = ROOT / "artifacts/iclr27-r45/long-native-forecasts-v001"


def prepare(base, smoke):
    output = base / "query-role-inputs-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed future-key input registration")
    short_entries = {r["case_id"]: r for r in read_json(SHORT / "manifest.json")["cases"]}
    parent_entries = {r["case_id"]: r for r in read_json(PARENT / "manifest.json")["cases"]}
    rows = read_json(INPUTS / "manifest.json")["cases"]
    if smoke:
        ids = {
            r["case_id"]
            for r in read_json(
                ROOT / "artifacts/iclr27-r45/smoke-v001/long-native-inputs-v001/manifest.json"
            )["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    entries, selected_properties = [], set()
    for row in rows:
        short_row, parent_row = short_entries[row["case_id"]], parent_entries[row["case_id"]]
        short = load_npz(SHORT / short_row["path"], short_row["sha256"])
        parent = load_npz(PARENT / parent_row["path"], parent_row["sha256"])
        short_queries, long_queries = (
            json.loads(str(short["queries"])),
            json.loads(str(parent["queries"])),
        )
        bindings = {}
        for name, pool, root, reference in (
            ("native_short", short_queries, SHORT, "restore_native"),
            ("gaussian_short", short_queries, SHORT, "restore_gaussian"),
            ("native_long", long_queries, PARENT, "native_long_prefix_peer"),
        ):
            query = next(q for q in pool if q["name"] == reference)
            if name == "native_long" and query["input_units"] != "prefix_standardized":
                raise ValueError("the registered long-input units changed")
            bindings[name] = {"path": str(root / query["path"]), "sha256": query["sha256"]}
        key = row["dataset"], row["horizon"]
        property_case = bool(smoke and key not in selected_properties)
        selected_properties.add(key)
        entries.append(
            {
                **row,
                "source_attention_path": str(INPUTS / row["path"]),
                "source_attention_sha256": row["sha256"],
                "parent_path": str(PARENT / parent_row["path"]),
                "parent_sha256": parent_row["sha256"],
                "base_queries": bindings,
                "property_case": property_case,
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": smoke,
            "cases": entries,
            "identity": {
                str(p): file_sha256(p)
                for p in (
                    PROTOCOL,
                    Path(__file__),
                    ROOT / "scripts/future_query_core.py",
                    INPUTS / "manifest.json",
                    SHORT / "manifest.json",
                    PARENT / "manifest.json",
                    ROOT / "artifacts/iclr27-r45/long-native-audit-v001/manifest.json",
                )
            },
            "heldout_value_analysis": False,
        },
    )


def controlled_forward(backbone, pipeline, context, horizon, targets, policy, perturb=False):
    trace = {}
    with future_key_policy(backbone, context, horizon, targets, policy, trace, perturb=perturb):
        raw, fields = grouped_forward(
            backbone, pipeline, context, np.zeros(len(context), dtype=np.int64), horizon
        )
    return raw, trace, field_digests(fields)


def save_query(path, context, raw, trace, hashes):
    _save_npz(
        path,
        quantiles=raw,
        context_sha256=np.asarray(array_digest(context)),
        field_hashes=np.asarray(json.dumps(hashes)),
        **trace,
    )


def check_property_outputs(name, raw, baseline, primary, readonly, targets, horizon):
    if name == "neutral_all_targets":
        np.testing.assert_array_equal(raw, baseline)
        return 0.0
    if name == "aux_both_perturbed":
        np.testing.assert_array_equal(raw[:targets], primary[:targets])
        return 0.0
    if name == "readonly_perturbed":
        np.testing.assert_array_equal(raw[:targets], readonly[:targets])
        return 0.0
    if name == "readonly_extended":
        np.testing.assert_allclose(
            raw[:targets, :, :horizon], readonly[:targets, :, :horizon], rtol=2e-5, atol=2e-5
        )
        return float(abs(raw[:targets, :, :horizon] - readonly[:targets, :, :horizon]).max())
    return float(abs(raw[:targets, :, :horizon] - baseline[:targets, :, :horizon]).max())


def forecast(base):
    inputs, output = base / "query-role-inputs-v001", base / "query-role-forecasts-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed future-role forecasts")
    prepared = read_json(inputs / "manifest.json")
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid, entries, interventions, ordinary_calls, property_calls = (
        pipeline.quantiles.index(0.5),
        [],
        0,
        0,
        0,
    )
    for number, row in enumerate(prepared["cases"]):
        parent = load_npz(row["parent_path"], row["parent_sha256"])
        points = dict(zip(parent["methods"].tolist(), parent["points"], strict=True))
        h, targets = row["horizon"], row["target_count"]
        contexts, original_outputs, original_hashes, requests, changed_outputs = {}, {}, {}, [], {}
        for context_name, binding in row["base_queries"].items():
            original = load_npz(binding["path"], binding["sha256"])
            context = original["context_z"]
            raw, trace, hashes = controlled_forward(backbone, pipeline, context, h, targets, "none")
            np.testing.assert_array_equal(raw, original["quantiles"])
            (
                contexts[context_name],
                original_outputs[context_name],
                original_hashes[context_name],
            ) = context, raw, hashes
            path = output / "queries" / f"{row['case_id']}-ordinary_{context_name}.npz"
            save_query(path, context, raw, trace, hashes)
            requests.append(
                {
                    "name": "ordinary_" + context_name,
                    "context": context_name,
                    "policy": "none",
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            ordinary_calls += 1
        for name, (context_name, policy) in DEFINITIONS.items():
            context = contexts[context_name]
            raw, trace, hashes = controlled_forward(backbone, pipeline, context, h, targets, policy)
            if hashes != original_hashes[context_name]:
                raise ValueError("the intervention changed original input statistics or encoding")
            path = output / "queries" / f"{row['case_id']}-{name}.npz"
            save_query(path, context, raw, trace, hashes)
            requests.append(
                {
                    "name": name,
                    "context": context_name,
                    "policy": policy,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            if name in points or "half_var_" + name in points:
                raise ValueError("a future-role output would overwrite a frozen control")
            points[name] = raw[:targets, mid, :h].T.astype(float)
            points["half_var_" + name] = 0.5 * points[name] + 0.5 * points["linear_var_direct"]
            changed_outputs[name] = raw
            interventions += 1
        properties = []
        if row["property_case"]:
            context = contexts["native_long"]
            definitions = (
                ("neutral_all_targets", "aux_both", len(context), h, False),
                ("ordinary_perturbed", "none", targets, h, True),
                ("aux_both_perturbed", "aux_both", targets, h, True),
                ("readonly_perturbed", "readonly", targets, h, True),
                ("readonly_extended", "readonly", targets, h + 16, False),
                ("ordinary_extended", "none", targets, h + 16, False),
            )
            for name, policy, declared_targets, horizon, perturb in definitions:
                raw, trace, hashes = controlled_forward(
                    backbone, pipeline, context, horizon, declared_targets, policy, perturb
                )
                difference = check_property_outputs(
                    name,
                    raw,
                    original_outputs["native_long"],
                    changed_outputs["query_aux_both_native_long"],
                    changed_outputs["query_readonly_native_long"],
                    targets,
                    h,
                )
                path = output / "properties" / f"{row['case_id']}-{name}.npz"
                save_query(path, context, raw, trace, hashes)
                properties.append(
                    {
                        "name": name,
                        "policy": policy,
                        "targets": declared_targets,
                        "horizon": horizon,
                        "perturb": perturb,
                        "reference_difference": difference,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                property_calls += 1
        names = sorted(points)
        if len(names) != (250 if row["dataset"] == "beijing" else 123):
            raise ValueError("registered future-role point set is incomplete")
        path = output / "cases" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([points[n] for n in names]),
            queries=np.asarray(json.dumps(requests)),
            properties=np.asarray(json.dumps(properties)),
        )
        entries.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "case_id",
                        "dataset",
                        "panel",
                        "station",
                        "origin",
                        "horizon",
                        "target_count",
                        "property_case",
                    )
                },
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        if (number + 1) % 10 == 0:
            _write_json(
                output / "progress.json",
                {"cases": number + 1, "interventions": interventions, "ordinary": ordinary_calls},
            )
            print(json.dumps({"cases": number + 1, "interventions": interventions}), flush=True)
    if (
        parameter_digest(backbone) != digest
        or interventions != 6 * len(entries)
        or ordinary_calls != 3 * len(entries)
    ):
        raise ValueError("base model identity or future-role query accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "cases": entries,
            "intervention_calls": interventions,
            "ordinary_calls": ordinary_calls,
            "property_calls": property_calls,
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "heldout_value_analysis": False,
        },
    )


def evaluate(base):
    inputs, forecasts, output = (
        base / "query-role-inputs-v001",
        base / "query-role-forecasts-v001",
        base / "query-role-results-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed future-role scores")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["smoke"] or len(fm["cases"]) != 314:
        raise ValueError("freeze the entire future-role population before scoring")
    records, hdb = truth_sources()
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    rows, targets = [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        data = load_npz(row["source_attention_path"], row["source_attention_sha256"])
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        truth = case_truth(row, records, hdb)
        valid = np.isfinite(truth)
        if (valid.sum(0) < row["horizon"] // 2).any():
            raise ValueError("future-role evaluation support changed")
        expected = (truth - data["mean"][: row["target_count"]]) / data["scale"][
            : row["target_count"]
        ]
        errors = np.where(valid[None], saved["points"] - expected[None], 0)
        mae, mse = abs(errors).sum(1) / valid.sum(0), np.square(errors).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "dataset", "panel", "station", "origin", "horizon")}
        for i, name in enumerate(saved["methods"].tolist()):
            rows.append(
                {**info, "method": name, "mae": float(mae[i].mean()), "mse": float(mse[i].mean())}
            )
            for slot in range(row["target_count"]):
                targets.append(
                    {
                        **info,
                        "method": name,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[i, slot]),
                        "mse": float(mse[i, slot]),
                    }
                )
    if len(rows) != 67959 or len(targets) != 125709:
        raise ValueError("registered future-role score counts changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    stations, summary = aggregate(frame)
    stations.to_csv(output / "stations.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    leave = []
    for (panel, excluded), _ in stations.groupby(["panel", "station"]):
        part = (
            stations.loc[(stations.panel == panel) & (stations.station != excluded)]
            .groupby(["panel", "method"])[["mae", "mse"]]
            .mean()
            .reset_index()
        )
        part["omitted_station"] = excluded
        leave.append(part)
    pd.concat(leave, ignore_index=True).to_csv(output / "leave_one_station_out.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": "query_aux_both_native_long",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "independent_confirmation": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "forecast", "evaluate", "audit"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    base, started = args.run_root.resolve(), perf_counter()
    if args.phase == "prepare":
        prepare(base, args.smoke)
    elif args.phase == "forecast":
        forecast(base)
    elif args.phase == "evaluate":
        evaluate(base)
    else:
        from audit_future_query import audit

        audit(base)
    print(
        json.dumps(
            {"phase": args.phase, "status": "completed", "seconds": perf_counter() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
