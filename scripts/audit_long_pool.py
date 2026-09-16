"""Verify causal L192 imputer fitting, query inputs, all forecasts and observed-target readout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from audit_matched_replay import effective_input, errors
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook
from matched_replay_sources import native_sources, timestamp
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-pool audits")
    base = ROOT / "artifacts/iclr27-r25"
    plan = read_json(base / "long-plan-v001/manifest.json")
    input_root, forecast_root, result_root = [
        base / name for name in ("long-inputs-v001", "long-forecasts-v001", "long-results-v001")
    ]
    name = "smoke.json" if args.smoke else "manifest.json"
    prepared, forecasts = read_json(input_root / name), read_json(forecast_root / name)
    if prepared["status"] != "completed" or forecasts["status"] != "completed":
        raise ValueError("finish registered preparation and predictions")
    for mapping in (prepared["identity"]["files"], forecasts["identity"]):
        for filename, digest in mapping.items():
            if file_sha256(ROOT / filename) != digest:
                raise ValueError("a frozen long-pool definition changed")
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    eligible = {
        e["episode_id"]
        for s in sources
        for e in s["episodes"]
        if not np.isfinite(
            s["values"][int(e["window"]["origin"]) - 96 : int(e["window"]["origin"]), :2]
        ).all()
    }
    if eligible != {r["episode_id"] for r in plan["cases"]} or len(eligible) != 301:
        raise ValueError("the prespecified native target-missing population changed")
    failures = []
    for fit in prepared["fits"]:
        path = input_root / fit["training_batch_path"]
        if file_sha256(path) != fit["training_batch_sha256"]:
            raise ValueError("an L192 training batch changed")
        cutoff = []
        with np.load(path, allow_pickle=False) as training:
            if training["values"].shape[1] != 192 or len(training["values"]) > 64:
                raise ValueError("neural imputer window length or fitting budget changed")
            for index, identifier in enumerate(training["window_ids"].tolist()):
                item, suffix = identifier.rsplit("@", 1)
                start = int(suffix.split("|", 1)[0])
                source = source_map[(fit["cohort"], fit["dataset_id"], item)]
                if start + 192 > source["prefix_end"]:
                    raise ValueError("an imputer fit exceeded its original prefix")
                raw = source["values"][start : start + 192]
                observed = training["observed"][index]
                if (observed & ~np.isfinite(raw)).any():
                    raise ValueError("imputer fitting used a nonexistent observation")
                np.testing.assert_array_equal(training["values"][index][observed], raw[observed])
                cutoff.append(timestamp(source, start + 192))
        if max(cutoff) != pd.Timestamp(fit["latest_training_end"]):
            raise ValueError("the recorded fitting cutoff is incorrect")
        for row in prepared["cases"]:
            if (row["cohort"], row["dataset_id"]) == (fit["cohort"], fit["dataset_id"]):
                source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
                if max(cutoff) > timestamp(source, row["origin"]):
                    raise ValueError("an imputer fit reaches beyond a prediction origin")
        for entry in fit["deep_fits"]:
            marker = Path(entry["path"])
            if file_sha256(marker) != entry["sha256"]:
                raise ValueError("a deep imputer fitting record changed")
            record = read_json(marker)
            if record["status"] == "fitted":
                for saved in record["files"]:
                    if (
                        file_sha256(marker.parent / entry["candidate_id"] / saved["path"])
                        != saved["sha256"]
                    ):
                        raise ValueError("a fitted L192 neural imputer changed")
            else:
                failures.append(
                    {
                        "dataset_id": fit["dataset_id"],
                        "candidate_id": entry["candidate_id"],
                        "reason": record["reason"],
                    }
                )
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    training_root = ROOT / "artifacts/iclr27-r24/repair-training-v001"
    trained = read_json(training_root / "manifest.json")
    old = read_json(ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json")[
        "models"
    ]["chronos2"]
    matched = read_json(
        ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001/matched_controls.json"
    )["models"]["chronos2"]
    if not args.smoke:
        study = read_json(result_root / "manifest.json")
        for path, digest in study["files"].items():
            if file_sha256(ROOT / path) != digest:
                raise ValueError("a long-pool score definition changed")
        scores = pd.read_parquet(result_root / "case_scores.parquet").set_index(
            ["case_id", "method"]
        )
        target_scores = pd.read_parquet(result_root / "target_scores.parquet").set_index(
            ["case_id", "method", "target_slot"]
        )
    torch.set_num_threads(1)
    runner, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    modules = {}
    for condition in ("fraction", "pattern"):
        record = next(
            r
            for r in trained["models"]
            if r["model_id"] == "chronos2" and r["condition"] == condition
        )
        path = training_root / record["checkpoint_path"]
        if (
            file_sha256(path) != forecasts["repair_shas"][condition]
            or record["backbone_sha256"] != digest
        ):
            raise ValueError("the frozen R24 repair or its backbone changed")
        repair = PatchRepair(record["width"], record["patch_size"], record["rank"]).to("cuda")
        repair.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        modules[condition] = repair.eval().requires_grad_(False)
    spec = ForecastSpec(
        "chronos2", "joint_multivariate", 96, context_length=192, target_indices=[0, 1]
    )
    checked, maximum_metric, maximum_replay = set(), 0.0, 0.0
    for entry in forecasts["cases"]:
        row = metadata[entry["case_id"]]
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        for path, expected_sha in (
            (input_root / row["path"], row["sha256"]),
            (forecast_root / entry["path"], entry["sha256"]),
        ):
            if file_sha256(path) != expected_sha:
                raise ValueError("a fixed candidate or forecast changed")
        with (
            np.load(input_root / row["path"], allow_pickle=False) as data,
            np.load(forecast_root / entry["path"], allow_pickle=False) as prediction,
        ):
            context = source["values"][row["origin"] - 192 : row["origin"]]
            np.testing.assert_array_equal(context, data["context"])
            prefix = np.asarray(source["values"][: source["prefix_end"]], dtype=float)
            mean, scale = np.nanmean(prefix, axis=0), np.nanstd(prefix, axis=0, ddof=0)
            scale = np.where(scale <= 1e-12, 1.0, scale)
            np.testing.assert_array_equal(data["mean"], mean)
            np.testing.assert_array_equal(data["scale"], scale)
            candidates, ids = data["candidate_values"], data["candidate_ids"].tolist()
            observed = np.isfinite(context)
            for candidate in candidates:
                if not np.isfinite(candidate).all():
                    raise ValueError("an imputer failed its finite shared fallback")
                np.testing.assert_array_equal(candidate[observed], context[observed])
            names, actions = prediction["methods"].tolist(), prediction["actions"].tolist()
            queries = json.loads(str(prediction["queries"]))
            if (
                len(queries) != 10
                or len(names) != 18
                or actions != sorted([*ids, "guarded_direct"])
            ):
                raise ValueError("the fixed candidate/control population changed")
            for query in queries:
                mode, action = query["mode"], query["action"]
                values = np.array(
                    effective_input(context, candidates, ids, action, True),
                    dtype=np.float32,
                    order="C",
                )
                values[np.isnan(values)] = np.nan
                binding = {
                    "identity_sha256": forecasts["identity_sha256"],
                    "case_id": row["case_id"],
                    "input_sha256": row["sha256"],
                    "model_id": "chronos2",
                    "parameter_sha256": digest,
                    "mode": mode,
                    "repair_sha256": forecasts["repair_shas"].get(mode),
                    "origin": row["origin"],
                    "context_length": 192,
                    "horizon": 96,
                    "observation_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                }
                encoded = json.dumps(binding, sort_keys=True)
                key = hashlib.sha256(encoded.encode() + values.tobytes()).hexdigest()
                if key != query["key"]:
                    raise ValueError("a query used different values, masks or time bounds")
                with np.load(forecast_root / "queries" / f"{key}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["input"], values)
                    np.testing.assert_array_equal(saved["observed"], observed)
                    if str(saved["binding"]) != encoded:
                        raise ValueError("cached query metadata differs")
                    point = saved["point"]
                    if key not in checked:
                        if mode == "ordinary":
                            actual = runner.predict_missing(values[None], spec).point[0]
                        else:
                            with RepairHook(backbone, "chronos2", observed, modules[mode], mode):
                                actual = runner.predict_missing(values[None], spec).point[0]
                        maximum_replay = max(
                            maximum_replay, float((abs(actual - point) / scale[:2]).max())
                        )
                        np.testing.assert_array_equal(actual, point)
                normalized = (point - mean[:2]) / scale[:2]
                target = (
                    prediction["bank"][actions.index(action)]
                    if mode == "ordinary"
                    else prediction["points"][names.index(mode + "_repair")]
                )
                np.testing.assert_array_equal(normalized, target)
                checked.add(key)
            bank = prediction["bank"]
            for method, point in zip(names, prediction["points"], strict=True):
                if method in actions:
                    expected = bank[actions.index(method)]
                elif method.endswith("_repair"):
                    continue
                elif method == "mean8":
                    expected = sum(bank) / 8
                elif method == "median8":
                    ordered = np.sort(bank, axis=0)
                    expected = (ordered[3] + ordered[4]) / 2
                else:
                    control = matched if method.startswith("matched_") else old
                    if method.endswith("single_mae"):
                        expected = bank[control["single_index"]]
                    else:
                        weights = (
                            control["fixed_mae"]["weights"]
                            if method.endswith("fixed_mae")
                            else control.get(
                                "fixed_joint_weights", control.get("fixed_joint", {}).get("weights")
                            )
                        )
                        expected = sum(w * p for w, p in zip(weights, bank, strict=True))
                np.testing.assert_allclose(expected, point, rtol=1e-12, atol=1e-12)
            if not args.smoke:
                if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
                    raise ValueError("the original current outcome artifact changed")
                truth = source["values"][row["origin"] : row["origin"] + 96, :2]
                with np.load(row["original_path"], allow_pickle=False) as original:
                    np.testing.assert_array_equal(truth, original["future"][:96, :2])
                    np.testing.assert_array_equal(
                        np.isfinite(truth), original["future_observed"][:96, :2]
                    )
                loss = errors(prediction["points"], truth, mean, scale)
                for index, method in enumerate(names):
                    actual = scores.loc[(row["case_id"], method), ["mae", "mse"]].to_numpy(float)
                    maximum_metric = max(
                        maximum_metric, float(abs(actual - loss[index].mean(0)).max())
                    )
                    np.testing.assert_allclose(actual, loss[index].mean(0), rtol=1e-10, atol=1e-10)
                    for slot in (0, 1):
                        target = target_scores.loc[(row["case_id"], method, slot)]
                        np.testing.assert_allclose(
                            target[["mae", "mse"]].to_numpy(float),
                            loss[index, slot],
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        if target.observed_count != np.isfinite(truth[:, slot]).sum():
                            raise ValueError("target scoring did not use the original observations")
    if (
        parameter_digest(backbone) != digest
        or len(checked) != forecasts["counters"]["distinct_queries"]
    ):
        raise ValueError("backbone state or query coverage changed")
    if not args.smoke:
        if len(scores) != 5418 or len(target_scores) != 10836:
            raise ValueError("expanded score coverage changed")
        frame = scores.reset_index()
        group_table = pd.read_csv(result_root / "groups.csv").set_index(["method", "group_id"])
        for record in pd.read_csv(result_root / "summary.csv").itertuples(index=False):
            values = []
            for group, subset in frame[frame.method == record.method].groupby("group_id"):
                datasets = []
                for _, dataset in subset.groupby("dataset_id"):
                    datasets.append(
                        np.mean(
                            [
                                series[["mae", "mse"]].to_numpy().mean(0)
                                for _, series in dataset.groupby("item_id")
                            ],
                            axis=0,
                        )
                    )
                expected = np.mean(datasets, axis=0)
                np.testing.assert_allclose(
                    expected,
                    group_table.loc[(record.method, group), ["mae", "mse"]],
                    rtol=1e-12,
                    atol=1e-12,
                )
                values.append(expected)
            np.testing.assert_allclose(
                np.mean(values, axis=0), [record.mae, record.mse], rtol=1e-12, atol=1e-12
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke_only": args.smoke,
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": None if args.smoke else file_sha256(result_root / "manifest.json"),
            "verified_cases": len(forecasts["cases"]),
            "replayed_distinct_queries": len(checked),
            "maximum_prediction_difference": maximum_replay,
            "maximum_metric_difference": maximum_metric,
            "failed_deep_fits": failures,
            "all_deep_baselines_fitted": not failures,
            "evaluation_future_read": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
