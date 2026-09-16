"""Independently verify unchanged numerical features and fixed-method downstream errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from audit_matched_replay import effective_input, errors
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _write_json, file_sha256


def verify_fields(before, after, marker, model_id, role, call_index):
    joint = model_id == "chronos2"
    width = before.shape[-1] // (3 if joint else 2)
    expected = before.copy()
    if role != "ordinary":
        if joint and call_index == 0:
            expected[..., -width:] = marker.T.reshape(before.shape[0], -1, width)
        elif not joint:
            expected[0, :, -width:] = (~marker[:, 0]).reshape(-1, width)
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_array_equal(before[..., :-width], after[..., :-width])
    if not joint:
        np.testing.assert_array_equal(before[1:], after[1:])
    return int(np.count_nonzero(before != after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed marker audits")
    base = ROOT / "artifacts/iclr27-r21"
    input_root, forecast_root, study_root = [
        base / name
        for name in (
            "provenance-inputs-v001",
            "provenance-forecasts-v001",
            "provenance-results-v001",
        )
    ]
    prepared = read_json(input_root / "manifest.json")
    forecasts = read_json(forecast_root / ("smoke.json" if args.smoke else "manifest.json"))
    if prepared["status"] != "completed" or forecasts["status"] != "completed":
        raise ValueError("complete every input and forecast before audit")
    for name, sha in forecasts["identity"].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError("a frozen marker interface or protocol changed")
    metadata = {row["case_id"]: row for row in prepared["cases"]}
    parameters = {c["model_id"]: c["parameter_sha256"] for c in forecasts["costs"]}
    if parameters != {
        "chronos2": "ee1f6a9827cd88ee8b87b5aedbd9f2ae2f3c7123dcd629ee79f1f8fdd0595e9c",
        "timesfm2p5": "8c926884efa6844309ebf6a96c79cd56bc9834a0a679d1f26d2562a59b4c8f05",
    }:
        raise ValueError("the forecasting parameters changed")
    if not args.smoke:
        study = read_json(study_root / "manifest.json")
        freeze = read_json(study_root / "predictions_frozen.json")
        if study["status"] != "completed" or study["predictions"] != freeze["predictions"]:
            raise ValueError("complete and freeze all fixed-method outputs")
        for name, digest in study["identity"].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("the study sources or fixed method definition changed")
        default_path = ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json"
        if (
            file_sha256(default_path) != freeze["source_defaults_sha256"]
            or read_json(default_path.parent / "decisions_frozen.json")["source_defaults_sha256"]
            != freeze["source_defaults_sha256"]
        ):
            raise ValueError("source-fixed controls do not match the audited R19 freeze")
        defaults = read_json(default_path)
        selected = {(r["model_id"], r["case_id"]): r for r in study["predictions"]}
        scores = pd.read_parquet(study_root / "case_scores.parquet").set_index(
            ["model_id", "case_id", "method"]
        )
        target_scores = pd.read_parquet(study_root / "target_scores.parquet").set_index(
            ["model_id", "case_id", "method", "target_slot"]
        )
    keys, changed_features, pairs, maximum_error = set(), 0, 0, 0.0
    for entry in forecasts["cases"]:
        row, model = metadata[entry["case_id"]], entry["model_id"]
        joint = model == "chronos2"
        if (
            file_sha256(input_root / row["path"]) != row["sha256"]
            or file_sha256(forecast_root / entry["path"]) != entry["sha256"]
        ):
            raise ValueError("a prepared input or frozen case bank changed")
        with (
            np.load(input_root / row["path"], allow_pickle=False) as data,
            np.load(forecast_root / entry["path"], allow_pickle=False) as prediction,
        ):
            context, candidates, ids = (
                data["context"],
                data["candidate_values"],
                data["candidate_ids"].tolist(),
            )
            actions = sorted([*ids, "guarded_direct"])
            if prediction["actions"].tolist() != actions:
                raise ValueError("the fixed action order differs")
            queries = json.loads(str(prediction["queries"]))
            lookup = {(r["role"], r["action"], r["slot"]): r for r in queries}
            expected_count = 24 * (1 if joint else 2)
            if len(lookup) != expected_count or len(queries) != expected_count:
                raise ValueError("the registered query population changed")
            for query in queries:
                role, action, slot = query["role"], query["action"], query["slot"]
                canonical_role = "ordinary" if action == "guarded_direct" else role
                raw = effective_input(context, candidates, ids, action, joint)
                effective = np.array(
                    raw if joint else raw[:, slot : slot + 1], np.float32, order="C"
                )
                effective[np.isnan(effective)] = np.nan
                observed = (
                    np.isfinite(context) if joint else np.isfinite(context[:, slot : slot + 1])
                )
                marker = np.array(
                    np.roll(observed, 37, axis=0) if canonical_role == "shifted" else observed,
                    bool,
                    order="C",
                )
                np.testing.assert_array_equal(marker.sum(0), observed.sum(0))
                binding = {
                    "identity_sha256": forecasts["identity_sha256"],
                    "model_id": model,
                    "parameter_sha256": parameters[model],
                    "case_id": row["case_id"],
                    "input_sha256": row["sha256"],
                    "origin": row["origin"],
                    "context_length": 96,
                    "horizon": 96,
                    "target_slot": slot,
                    "role": canonical_role,
                    "marker_sha256": hashlib.sha256(marker.tobytes()).hexdigest(),
                }
                encoded = json.dumps(binding, sort_keys=True)
                key = hashlib.sha256(encoded.encode() + effective.tobytes()).hexdigest()
                if key != query["key"]:
                    raise ValueError("input values, marker positions or forecast times differ")
                path = forecast_root / model / "queries" / f"{key}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["binding"]) != encoded:
                        raise ValueError("the cached query metadata differs")
                    np.testing.assert_array_equal(saved["effective"], effective)
                    np.testing.assert_array_equal(saved["marker"], marker)
                    count = int(saved["tokenizer_calls"])
                    if count != (2 if np.isfinite(effective).all() else 0):
                        raise ValueError("unexpected tokenizer call count")
                    reference_key = lookup[("ordinary", action, slot)]["key"]
                    with np.load(
                        forecast_root / model / "queries" / f"{reference_key}.npz",
                        allow_pickle=False,
                    ) as reference:
                        for index in range(count):
                            before, after = (
                                saved[f"call_{index}_before"],
                                saved[f"call_{index}_after"],
                            )
                            np.testing.assert_array_equal(before, reference[f"call_{index}_before"])
                            change = verify_fields(
                                before, after, marker, model, canonical_role, index
                            )
                            if (model, key) not in keys:
                                changed_features += change
                    location = slice(None) if joint else slice(slot, slot + 1)
                    mean = data["mean"][:2] if joint else data["mean"][slot : slot + 1]
                    scale = data["scale"][:2] if joint else data["scale"][slot : slot + 1]
                    expected = (saved["point"] - mean) / scale
                    np.testing.assert_array_equal(
                        prediction["bank"][
                            ("ordinary", "provenance", "shifted").index(role),
                            actions.index(action),
                            :,
                            location,
                        ],
                        expected,
                    )
                keys.add((model, key))
            if not args.smoke:
                record = selected[(model, row["case_id"])]
                if (
                    file_sha256(study_root / record["path"]) != record["sha256"]
                    or file_sha256(Path(row["original_path"])) != row["original_sha256"]
                ):
                    raise ValueError("the prediction freeze or original target source changed")
                with (
                    np.load(row["original_path"], allow_pickle=False) as original,
                    np.load(study_root / record["path"], allow_pickle=False) as final,
                ):
                    truth = original["future"][:96]
                    np.testing.assert_array_equal(
                        np.isfinite(truth), original["future_observed"][:96]
                    )
                    control = defaults["models"][model]
                    for index, method in enumerate(final["methods"].tolist()):
                        bank_index = (
                            1
                            if method.startswith("provenance_")
                            else (2 if method.startswith("shifted_") else 0)
                        )
                        name = method.removeprefix("provenance_").removeprefix("shifted_")
                        bank = prediction["bank"][bank_index]
                        if name in actions:
                            expected = bank[actions.index(name)]
                        elif name == "median8":
                            order = np.sort(bank, axis=0)
                            expected = (order[3] + order[4]) / 2
                        elif name == "mean8":
                            expected = sum(bank) / 8
                        elif name == "source_single_mae":
                            expected = bank[control["single_index"]]
                        else:
                            weights = (
                                control["fixed_mae"]["weights"]
                                if name == "source_fixed_mae"
                                else control["fixed_joint_weights"]
                            )
                            expected = sum(w * p for w, p in zip(weights, bank, strict=True))
                        np.testing.assert_allclose(
                            final["points"][index], expected, rtol=1e-12, atol=1e-12
                        )
                    loss = errors(final["points"], truth, data["mean"], data["scale"])
                    for index, method in enumerate(final["methods"].tolist()):
                        actual = scores.loc[
                            (model, row["case_id"], method), ["mae", "mse"]
                        ].to_numpy(float)
                        maximum_error = max(
                            maximum_error, float(abs(actual - loss[index].mean(0)).max())
                        )
                        np.testing.assert_allclose(
                            actual, loss[index].mean(0), rtol=1e-10, atol=1e-10
                        )
                        for slot in (0, 1):
                            target = target_scores.loc[(model, row["case_id"], method, slot)]
                            np.testing.assert_allclose(
                                target[["mae", "mse"]].to_numpy(float),
                                loss[index, slot],
                                rtol=1e-10,
                                atol=1e-10,
                            )
                            if target.observed_count != np.isfinite(truth[:, slot]).sum():
                                raise ValueError("target scoring observations differ")
        pairs += 1
    for cost in forecasts["costs"]:
        if (
            sum(model == cost["model_id"] for model, _ in keys) != cost["distinct_queries"]
            or cost["new_queries"] + cost["cache_hits"] != cost["logical_scope_requests"]
        ):
            raise ValueError("the query accounting differs")
    if not args.smoke:
        if (
            pairs != 46
            or len(scores) != 1794
            or len(target_scores) != 3588
            or sum(c["logical_scope_requests"] for c in forecasts["costs"]) != 1518
        ):
            raise ValueError("the registered full population changed")
        frame = scores.reset_index()
        grouped = pd.read_csv(study_root / "groups.csv").set_index(
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
                    grouped.loc[(item.model_id, item.method, group), ["mae", "mse"]],
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
            "verified_case_model_pairs": pairs,
            "verified_distinct_queries": len(keys),
            "verified_changed_marker_features": changed_features,
            "maximum_metric_difference": maximum_error,
            "accuracy_values_read": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
