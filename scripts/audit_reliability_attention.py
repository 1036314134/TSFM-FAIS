"""Independent conditional-variance/mask checks and exact attention forecast replay."""

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
from reliability_attention_core import MODES, attention_intervention
from reliability_attention_experiment import (
    case_truth,
    forward,
    old_points,
    source_entries,
    truth_sources,
)

from tsfm_fais.utility_experiment import _write_json, file_sha256


def reference_weights(data, kind):
    cells = data["observed"].astype(float) if kind == "observed" else data["reliability"]
    weights = np.stack(
        [cells[:, i : i + 16].sum(1) / 16 for i in range(0, cells.shape[1], 16)], axis=1
    )
    if kind == "shifted":
        weights = np.concatenate([weights[:, -1:], weights[:, :-1]], axis=1)
    return weights.astype(np.float32)


def reference_mask(mask, key_weights):
    expected = mask.copy()
    weights = np.broadcast_to(key_weights, mask.shape)
    flat_mask, flat_weights, flat_result = (
        mask.reshape(-1, mask.shape[-1]),
        weights.reshape(-1, mask.shape[-1]),
        expected.reshape(-1, mask.shape[-1]),
    )
    fallback = 0
    for original, weight, result in zip(flat_mask, flat_weights, flat_result, strict=True):
        available = original == 0
        valid = available & (weight > 0)
        if not valid.any():
            fallback += int(available.any())
            continue
        result[available] = np.finfo(mask.dtype).min
        result[valid] = np.log(weight[valid].astype(float)).astype(mask.dtype)
    return expected, fallback


def check_masks(query, data, mode):
    _, kind, scope, provenance = MODES[mode]
    weights = reference_weights(data, kind)
    n = weights.shape[1]
    extended = np.concatenate(
        [
            weights,
            np.ones((len(weights), query["original_time_mask"].shape[-1] - n), dtype=weights.dtype),
        ],
        axis=1,
    )
    maximum = 0.0
    for label, shaped, active in (
        ("time", extended[:, None, None, :], scope in ("time", "both")),
        ("group", extended.T[:, None, None, :], scope in ("group", "both")),
    ):
        original, actual = query[f"original_{label}_mask"], query[f"{label}_mask"]
        expected, fallback = reference_mask(original, shaped) if active else (original, 0)
        np.testing.assert_array_equal(actual[original != 0], original[original != 0])
        hard = expected == np.finfo(expected.dtype).min
        np.testing.assert_array_equal(actual[hard], expected[hard])
        np.testing.assert_allclose(actual[~hard], expected[~hard], rtol=1e-6, atol=1e-6)
        if (~hard).any():
            maximum = max(maximum, float(abs(actual[~hard] - expected[~hard]).max()))
        if int(query[f"{label}_fallback_rows"]) != fallback:
            raise ValueError("no-evidence attention fallback accounting changed")
        if label == "time":
            np.testing.assert_array_equal(actual[..., n:], original[..., n:])
        else:
            np.testing.assert_array_equal(actual[n:], original[n:])
    if provenance:
        original, actual = query["original_embedding_fields"], query["embedding_fields"]
        np.testing.assert_array_equal(actual[..., :32], original[..., :32])
        np.testing.assert_array_equal(
            actual[..., 32:], data["observed"].reshape(len(weights), n, 16).astype(actual.dtype)
        )
    elif "embedding_fields" in query:
        raise ValueError("unexpected input encoding intervention")
    return maximum


def check_inputs(row, data):
    source = load_npz(row["source_input_path"], row["source_input_sha256"])
    model = load_npz(row["source_model_path"], row["source_model_sha256"])
    observed, covariance = np.isfinite(source["context"]), model["initial_covariance"]
    prior = covariance - 1e-6 * np.eye(len(covariance))
    prior_diagonal = np.diag(prior)
    reliability, variance = observed.astype(float), np.zeros_like(source["context"])
    for index, mask in enumerate(observed):
        known, unknown = np.flatnonzero(mask), np.flatnonzero(~mask)
        conditional = prior.copy()
        if len(known):
            conditional -= (
                prior[:, known] @ np.linalg.inv(covariance[np.ix_(known, known)]) @ prior[known, :]
            )
        variance[index, unknown] = np.maximum(np.diag(conditional)[unknown], 0)
        for column in unknown:
            if prior_diagonal[column] > 1e-12:
                reliability[index, column] = min(
                    1, max(0, 1 - conditional[column, column] / prior_diagonal[column])
                )
    selected = np.flatnonzero(source["keep"])
    np.testing.assert_array_equal(data["selected"], selected)
    np.testing.assert_array_equal(data["observed"], observed[:, selected].T)
    np.testing.assert_allclose(
        data["reliability"], reliability[:, selected].T, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        data["conditional_variance"], variance[:, selected].T, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_array_equal(
        data["reliability"][data["observed"]], np.ones(data["observed"].sum())
    )
    filled = (
        source["static_values"]
        if row["dataset"] == "beijing"
        else source["fills"][source["fill_names"].tolist().index("gaussian")]
    )
    for name, context in (("native", source["context"]), ("gaussian", filled)):
        expected = np.array(
            ((context[:, selected] - source["mean"][selected]) / source["scale"][selected]).T,
            dtype=np.float32,
            order="C",
        )
        np.testing.assert_array_equal(data[name], expected)
    np.testing.assert_array_equal(data["mean"], source["mean"])
    np.testing.assert_array_equal(data["scale"], source["scale"])
    return float(abs(data["reliability"] - reliability[:, selected].T).max())


def audit(base):
    started = perf_counter()
    inputs, forecasts, output = (
        base / "attention-inputs-v001",
        base / "attention-forecasts-v001",
        base / "attention-audit-v001",
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed attention audits")
    prepared, fm = read_json(inputs / "manifest.json"), read_json(forecasts / "manifest.json")
    if fm["status"] != "completed" or fm["input_sha256"] != file_sha256(inputs / "manifest.json"):
        raise ValueError("attention predictions or their input provenance changed")
    for path, sha in prepared["identity"].items():
        if file_sha256(Path(path)) != sha:
            raise ValueError("a frozen attention definition changed")
    if not fm["smoke"]:
        for row in read_json(base / "method_manifest.json")["files"]:
            if file_sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("a registered attention runtime changed")
    all_sources = {r["case_id"]: r for r in source_entries()}
    ids = {r["case_id"] for r in fm["cases"]}
    if ids != {r["case_id"] for r in prepared["cases"]} or (
        not fm["smoke"] and ids != set(all_sources)
    ):
        raise ValueError("the registered attention population changed")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, records, hdb = None, None, None
    if not fm["smoke"]:
        result = base / "attention-results-v001"
        if read_json(result / "manifest.json")["forecast_sha256"] != file_sha256(
            forecasts / "manifest.json"
        ):
            raise ValueError("attention score provenance changed")
        scores = pd.read_parquet(result / "target_scores.parquet").set_index(
            ["case_id", "method", "slot"]
        )
        if not scores.index.is_unique or len(scores) != 78745:
            raise ValueError("target score population changed")
        records, hdb = truth_sources()
    torch.set_num_threads(1)
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid = pipeline.quantiles.index(0.5)
    calls, ordinary, neutral, old_count, maximum_r, maximum_mask, rebuilt_scores = (
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        [],
    )
    for number, entry in enumerate(fm["cases"]):
        row = metadata[entry["case_id"]]
        for key, value in all_sources[row["case_id"]].items():
            if row[key] != value:
                raise ValueError("case identity or information scope changed")
        data = load_npz(inputs / row["path"], row["sha256"])
        maximum_r = max(maximum_r, check_inputs(row, data))
        saved = load_npz(forecasts / entry["path"], entry["sha256"])
        expected = old_points(row)
        old_count += len(expected)
        requests = json.loads(str(saved["queries"]))
        if {q["name"] for q in requests} != set(MODES) | {
            "restore_gaussian",
            "restore_native",
        } or len(requests) != 10:
            raise ValueError("a registered intervention or restoration is missing")
        for request in requests:
            name = request["name"]
            query = load_npz(forecasts / request["path"], request["sha256"])
            if name in MODES:
                kind = MODES[name][0]
                maximum_mask = max(maximum_mask, check_masks(query, data, name))
                captured = {}
                with attention_intervention(backbone, data, name, captured):
                    raw = forward(backbone, pipeline, query["context_z"], row["horizon"])
                for field in captured:
                    np.testing.assert_array_equal(captured[field], query[field])
                expected[name] = raw[: row["target_count"], mid, : row["horizon"]].T.astype(float)
                expected["half_var_" + name] = (
                    0.5 * expected[name] + 0.5 * expected["linear_var_direct"]
                )
                calls += 1
            else:
                kind = name.removeprefix("restore_")
                raw = forward(backbone, pipeline, query["context_z"], row["horizon"])
                baseline = row["gaussian_name"] if kind == "gaussian" else row["native_name"]
                np.testing.assert_array_equal(
                    raw[: row["target_count"], mid, : row["horizon"]].T.astype(float),
                    expected[baseline],
                )
                ordinary += 1
                if fm["smoke"] and kind == "gaussian":
                    with attention_intervention(
                        backbone, data, "gaussian_conditional_attention", neutral=True
                    ):
                        identity = forward(backbone, pipeline, query["context_z"], row["horizon"])
                    np.testing.assert_array_equal(identity, raw)
                    neutral += 1
            np.testing.assert_array_equal(query["context_z"], data[kind])
            np.testing.assert_array_equal(raw, query["quantiles"])
        names = sorted(expected)
        np.testing.assert_array_equal(saved["methods"], np.asarray(names))
        np.testing.assert_array_equal(saved["points"], np.stack([expected[n] for n in names]))
        if scores is not None:
            truth = case_truth(row, records, hdb)
            for name, point in expected.items():
                metrics = []
                for slot in range(row["target_count"]):
                    valid = np.isfinite(truth[:, slot])
                    errors = (
                        point[valid, slot]
                        - (truth[valid, slot] - data["mean"][slot]) / data["scale"][slot]
                    )
                    values = [float(np.abs(errors).mean()), float(np.square(errors).mean())]
                    target = scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        values, target[["mae", "mse"]].to_numpy(float), rtol=1e-12, atol=1e-12
                    )
                    if target["observed_count"] != valid.sum():
                        raise ValueError("score support changed")
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
        if (number + 1) % 25 == 0:
            _write_json(
                output / "progress.json",
                {"cases_audited": number + 1, "interventions_replayed": calls},
            )
            print(
                json.dumps({"cases_audited": number + 1, "interventions_replayed": calls}),
                flush=True,
            )
    if rebuilt_scores:
        check_metrics(pd.DataFrame(rebuilt_scores), result)
    if (
        digest != fm["parameter_sha256"]
        or parameter_digest(backbone) != digest
        or calls != fm["intervention_calls"]
        or ordinary != fm["ordinary_calls"]
        or neutral != fm["neutral_checks"]
    ):
        raise ValueError("backbone identity or call accounting changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": fm["smoke"],
            "cases": len(ids),
            "interventions_replayed": calls,
            "ordinary_restorations": ordinary,
            "neutral_identity_checks": neutral,
            "old_outputs_preserved": old_count,
            "prediction_difference": 0,
            "maximum_reliability_reference_difference": maximum_r,
            "maximum_logmask_reference_difference": maximum_mask,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )
