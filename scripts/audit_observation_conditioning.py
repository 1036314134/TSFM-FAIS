"""Reconstruct forecast origins, source values, conditional solves and downstream scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from audit_matched_replay import errors
from audit_tail_bridge import direct_guard
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources
from scipy.stats import norm

from tsfm_fais.utility_experiment import _write_json, file_sha256


def direct_conditioning(prior, quantiles, context):
    results = {
        name: prior[96:].copy()
        for name in (
            "prior_slice",
            "last_innovation",
            "mean_innovation",
            "unit_conditioner",
            "spread_conditioner",
        )
    }
    for slot in (0, 1):
        indices = [i for i in range(96) if np.isfinite(context[i, slot])]
        if not indices:
            continue
        residual = np.array([context[i, slot] - prior[i, slot] for i in indices])
        results["last_innovation"][:, slot] += residual[-1]
        results["mean_innovation"][:, slot] += sum(residual) / len(residual)
        spread = np.maximum(
            0.05, (quantiles[:, slot].max(1) - quantiles[:, slot].min(1)) / (2 * norm.ppf(0.9))
        )
        for name, scale in (("unit_conditioner", np.ones(192)), ("spread_conditioner", spread)):
            cov = np.array(
                [
                    [scale[i] * scale[j] * (0.5 + 0.5 * np.exp(-abs(i - j) / 96)) for j in indices]
                    for i in indices
                ]
            )
            cov += np.diag(0.1 * scale[indices] ** 2 + 1e-8)
            solved = np.linalg.solve(cov, residual)
            cross = np.array(
                [
                    [scale[i] * scale[j] * (0.5 + 0.5 * np.exp(-abs(i - j) / 96)) for j in indices]
                    for i in range(96, 192)
                ]
            )
            results[name][:, slot] += cross @ solved
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning audits")
    base = ROOT / "artifacts/iclr27-r23"
    input_root, forecast_root, study_root = [
        base / n
        for n in (
            "conditioning-inputs-v001",
            "conditioning-forecasts-v001",
            "conditioning-results-v001",
        )
    ]
    prepared = read_json(input_root / "manifest.json")
    forecasts = read_json(forecast_root / ("smoke.json" if args.smoke else "manifest.json"))
    if prepared["status"] != "completed" or forecasts["status"] != "completed":
        raise ValueError("finish the registered input and forecast records")
    for name, digest in forecasts["identity"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("a frozen definition changed")
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    parameters = {c["model_id"]: c["parameter_sha256"] for c in forecasts["costs"]}
    if parameters != {
        "chronos2": "ee1f6a9827cd88ee8b87b5aedbd9f2ae2f3c7123dcd629ee79f1f8fdd0595e9c",
        "timesfm2p5": "8c926884efa6844309ebf6a96c79cd56bc9834a0a679d1f26d2562a59b4c8f05",
    }:
        raise ValueError("the frozen forecasting parameters differ")
    if not args.smoke:
        study, frozen = (
            read_json(study_root / "manifest.json"),
            read_json(study_root / "predictions_frozen.json"),
        )
        if study["status"] != "completed" or study["predictions"] != frozen["predictions"]:
            raise ValueError("complete and freeze all conditional predictions first")
        for name, digest in study["identity"].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("a study definition changed")
        predictions = {(r["model_id"], r["case_id"]): r for r in study["predictions"]}
        old_root = ROOT / "artifacts/iclr27-r21/provenance-results-v001"
        old_study = read_json(old_root / "manifest.json")
        old_predictions = {(r["model_id"], r["case_id"]): r for r in old_study["predictions"]}
        old_scores = pd.read_parquet(old_root / "case_scores.parquet").set_index(
            ["model_id", "case_id", "method"]
        )
        scores = pd.read_parquet(study_root / "case_scores.parquet").set_index(
            ["model_id", "case_id", "method"]
        )
        target_scores = pd.read_parquet(study_root / "target_scores.parquet").set_index(
            ["model_id", "case_id", "method", "target_slot"]
        )
    keys, logical, maximum_error, maximum_prediction_error = set(), 0, 0.0, 0.0
    for entry in forecasts["cases"]:
        row, model_id = metadata[entry["case_id"]], entry["model_id"]
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        joint = model_id == "chronos2"
        if row["origin"] - 192 < source["prefix_end"]:
            raise ValueError("the scale prefix reaches the prior context")
        for path, digest in (
            (input_root / row["path"], row["sha256"]),
            (forecast_root / entry["path"], entry["sha256"]),
        ):
            if file_sha256(path) != digest:
                raise ValueError("a frozen input or forecast changed")
        with (
            np.load(input_root / row["path"], allow_pickle=False) as data,
            np.load(forecast_root / entry["path"], allow_pickle=False) as prior,
        ):
            full = source["values"][row["origin"] - 192 : row["origin"]]
            np.testing.assert_array_equal(data["direct_context"], full)
            np.testing.assert_array_equal(data["prior_context"], full[:96])
            np.testing.assert_array_equal(data["current_context"], full[96:])
            prefix = np.asarray(source["values"][: source["prefix_end"]], float)
            np.testing.assert_array_equal(data["mean"], np.nanmean(prefix, axis=0))
            scale = np.nanstd(prefix, axis=0, ddof=0)
            np.testing.assert_array_equal(data["scale"], np.where(scale <= 1e-12, 1.0, scale))
            np.testing.assert_array_equal(
                data["defaults"], np.nanmedian(source["values"][: source["prefix_end"]], axis=0)
            )
            queries = json.loads(str(prior["queries"]))
            if len(queries) != (3 if joint else 6):
                raise ValueError("the expected prior and control query count changed")
            for query in queries:
                role, slot = query["role"], query["slot"]
                raw = full[:96] if role == "prior" else full
                origin = row["origin"] - 96 if role == "prior" else row["origin"]
                horizon = 96 if role == "native192" else 192
                guarded = direct_guard(raw, data["defaults"], joint)
                effective = np.array(
                    guarded if joint else guarded[:, slot : slot + 1], dtype=np.float32, order="C"
                )
                effective[np.isnan(effective)] = np.nan
                binding = {
                    "identity_sha256": forecasts["identity_sha256"],
                    "model_id": model_id,
                    "parameter_sha256": parameters[model_id],
                    "case_id": row["case_id"],
                    "input_sha256": row["sha256"],
                    "role": role,
                    "forecast_origin": origin,
                    "history_start": origin - len(raw),
                    "horizon": horizon,
                    "target_slot": slot,
                    "quantile_levels": [0.1, 0.5, 0.9],
                }
                encoded = json.dumps(binding, sort_keys=True)
                key = hashlib.sha256(encoded.encode() + effective.tobytes()).hexdigest()
                if key != query["key"]:
                    raise ValueError("a predictive input, origin or output span differs")
                with np.load(
                    forecast_root / model_id / "queries" / f"{key}.npz", allow_pickle=False
                ) as saved:
                    np.testing.assert_array_equal(saved["effective"], effective)
                    if str(saved["binding"]) != encoded:
                        raise ValueError("a query binding changed")
                    destination = slice(None) if joint else slice(slot, slot + 1)
                    mean = data["mean"][:2] if joint else data["mean"][slot : slot + 1]
                    scale = data["scale"][:2] if joint else data["scale"][slot : slot + 1]
                    point = (saved["point"] - mean) / scale
                    np.testing.assert_array_equal(
                        prior[role][:, destination], point if role == "prior" else point[:96]
                    )
                    if role == "prior":
                        q = (saved["quantiles"] - mean[None, :, None]) / scale[None, :, None]
                        np.testing.assert_array_equal(prior["prior_quantiles"][:, destination], q)
                        np.testing.assert_array_equal(
                            np.arange(origin, origin + 192)[96:],
                            np.arange(row["origin"], row["origin"] + 96),
                        )
                keys.add((model_id, key))
                logical += 1
            context = (data["current_context"][:, :2] - data["mean"][:2]) / data["scale"][:2]
            rebuilt = direct_conditioning(prior["prior"], prior["prior_quantiles"], context)
            if args.smoke:
                from observation_conditioning import conditional_forecasts

                actual, _ = conditional_forecasts(prior["prior"], prior["prior_quantiles"], context)
                for name in rebuilt:
                    delta = float(abs(rebuilt[name] - actual[name]).max())
                    maximum_prediction_error = max(maximum_prediction_error, delta)
                    np.testing.assert_allclose(rebuilt[name], actual[name], rtol=1e-10, atol=1e-10)
                continue
            record = predictions[(model_id, row["case_id"])]
            old = old_predictions[(model_id, row["case_id"])]
            for path, digest in (
                (study_root / record["path"], record["sha256"]),
                (old_root / old["path"], old["sha256"]),
                (Path(row["original_path"]), row["original_sha256"]),
            ):
                if file_sha256(path) != digest:
                    raise ValueError("a prediction freeze, control or future source changed")
            rebuilt.update(native192=prior["native192"], budget_native192=prior["budget_native192"])
            with np.load(old_root / old["path"], allow_pickle=False) as older:
                for name, point in zip(older["methods"].tolist(), older["points"], strict=True):
                    if not name.startswith(("provenance_", "shifted_")):
                        rebuilt[name] = point
            with (
                np.load(study_root / record["path"], allow_pickle=False) as final,
                np.load(row["original_path"], allow_pickle=False) as original,
            ):
                truth = source["values"][row["origin"] : row["origin"] + 96, :2]
                np.testing.assert_array_equal(truth, original["future"][:96])
                np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96])
                for index, name in enumerate(final["methods"].tolist()):
                    delta = float(abs(rebuilt[name] - final["points"][index]).max())
                    maximum_prediction_error = max(maximum_prediction_error, delta)
                    np.testing.assert_allclose(
                        rebuilt[name], final["points"][index], rtol=1e-10, atol=1e-10
                    )
                loss = errors(final["points"], truth, data["mean"], data["scale"])
                for index, name in enumerate(final["methods"].tolist()):
                    actual = scores.loc[(model_id, row["case_id"], name), ["mae", "mse"]].to_numpy(
                        float
                    )
                    maximum_error = max(
                        maximum_error, float(abs(actual - loss[index].mean(0)).max())
                    )
                    np.testing.assert_allclose(actual, loss[index].mean(0), rtol=1e-10, atol=1e-10)
                    if (model_id, row["case_id"], name) in old_scores.index:
                        np.testing.assert_array_equal(
                            actual, old_scores.loc[(model_id, row["case_id"], name), ["mae", "mse"]]
                        )
                    for slot in (0, 1):
                        target = target_scores.loc[(model_id, row["case_id"], name, slot)]
                        np.testing.assert_allclose(
                            target[["mae", "mse"]].to_numpy(float),
                            loss[index, slot],
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        if target.observed_count != np.isfinite(truth[:, slot]).sum():
                            raise ValueError("a target scoring observation mask changed")
    if logical != sum(r["logical_scope_requests"] for r in forecasts["costs"]):
        raise ValueError("logical request accounting differs")
    for record in forecasts["costs"]:
        if (
            sum(model == record["model_id"] for model, _ in keys) != record["distinct_queries"]
            or record["new_queries"] + record["cache_hits"] != record["logical_scope_requests"]
        ):
            raise ValueError("cache accounting differs")
    if not args.smoke:
        if logical != 207 or len(scores) != 920 or len(target_scores) != 1840:
            raise ValueError("the full conditioning population changed")
        frame = scores.reset_index()
        groups = pd.read_csv(study_root / "groups.csv").set_index(
            ["model_id", "method", "group_id"]
        )
        for item in pd.read_csv(study_root / "summary.csv").itertuples(index=False):
            part = frame[(frame.model_id == item.model_id) & (frame.method == item.method)]
            values = []
            for group, rows in part.groupby("group_id"):
                datasets = []
                for _, dataset in rows.groupby("dataset_id"):
                    datasets.append(
                        np.mean(
                            [
                                series[["mae", "mse"]].to_numpy().mean(0)
                                for _, series in dataset.groupby("item_id")
                            ],
                            axis=0,
                        )
                    )
                value = np.mean(datasets, axis=0)
                np.testing.assert_allclose(
                    value,
                    groups.loc[(item.model_id, item.method, group), ["mae", "mse"]],
                    rtol=1e-12,
                    atol=1e-12,
                )
                values.append(value)
            np.testing.assert_allclose(
                np.mean(values, axis=0), [item.mae, item.mse], rtol=1e-12, atol=1e-12
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "smoke_only": args.smoke,
            "study_sha256": None if args.smoke else file_sha256(study_root / "manifest.json"),
            "verified_case_model_pairs": len(forecasts["cases"]),
            "verified_distinct_queries": len(keys),
            "verified_logical_requests": logical,
            "maximum_prediction_difference": maximum_prediction_error,
            "maximum_metric_difference": maximum_error,
            "current_future_scored": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
