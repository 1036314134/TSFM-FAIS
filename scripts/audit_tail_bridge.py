"""Reconstruct tail eligibility, source chronology, forecast times and observed losses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from audit_matched_replay import effective_input, errors
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources, timestamp
from prepare_tail_bridge import smoke_cases

from tsfm_fais.utility_experiment import _write_json, file_sha256

BASE = ROOT / "artifacts/iclr27-r20"


def direct_gaps(context, joint):
    result = []
    for slot in (0, 1):
        gap = 0
        for row in context[::-1]:
            available = np.isfinite(row).any() if joint else np.isfinite(row[slot])
            if available:
                break
            gap += 1
        result.append(gap if 1 <= gap <= 48 else 0)
    return result


def direct_guard(raw, defaults, joint):
    result = raw.copy()
    empty = ~np.isfinite(raw).any(0)
    columns = (
        range(raw.shape[1])
        if joint and empty.any()
        else ([slot for slot in (0, 1) if empty[slot]] if not joint else [])
    )
    for column in columns:
        last = defaults[column]
        for index in range(len(raw)):
            if np.isfinite(raw[index, column]):
                last = raw[index, column]
            else:
                result[index, column] = last
    return result


def fit_cutoff(source, sources, entry):
    peers = {
        s["item_id"]: s
        for s in sources
        if (s["cohort"], s["dataset_id"]) == (source["cohort"], source["dataset_id"])
    }
    path = source["root"] / "imputers" / source["dataset_id"] / "training_batch.npz"
    if file_sha256(path) != source["dataset"]["training_batch_sha256"]:
        raise ValueError("the original imputer training batch changed")
    boundaries = []
    with np.load(path, allow_pickle=False) as saved:
        for index, identifier in enumerate(saved["window_ids"].tolist()):
            item, rest = identifier.rsplit("@", 1)
            start = int(rest.split("|", 1)[0])
            peer = peers[item]
            if start + 96 > peer["prefix_end"]:
                raise ValueError("training used observations beyond the prefix")
            observed = saved["observed"][index]
            raw = peer["values"][start : start + 96]
            if (observed & ~np.isfinite(raw)).any():
                raise ValueError("training treats an originally missing value as observed")
            np.testing.assert_array_equal(observed, np.isfinite(saved["values"][index]))
            np.testing.assert_array_equal(saved["values"][index][observed], raw[observed])
            boundaries.append(timestamp(peer, start + 96))
    for record in entry["frozen_neural_imputers"]:
        marker = Path(record["path"])
        if file_sha256(marker) != record["sha256"]:
            raise ValueError("a frozen imputer marker changed")
        fitted = read_json(marker)
        if fitted["status"] != "fitted":
            raise ValueError("the frozen neural imputer has no fit")
        for item in fitted["files"]:
            if file_sha256(marker.parent / record["candidate_id"] / item["path"]) != item["sha256"]:
                raise ValueError("a frozen imputer parameter file changed")
    return max(boundaries)


def reconstruct(row, data, entry, model_id, identity_sha, parameter_sha):
    joint = model_id == "chronos2"
    context, candidates = data["context"], data["candidate_values"]
    ids = data["candidate_ids"].tolist()
    actions = sorted([*ids, "guarded_direct"])
    mean, scale = data["mean"], data["scale"]
    keys, records = set(), []
    provenance = {name: row[name] for name in ("cohort", "dataset_id", "item_id", "prefix_end")}
    provenance["imputers"] = entry["frozen_neural_imputers"]

    def query(raw, original, origin, horizon, slot):
        effective = np.array(raw if joint else raw[:, slot : slot + 1], np.float32, order="C")
        effective[np.isnan(effective)] = np.nan
        mask = ~np.isfinite(original)
        location = slice(0, 2) if joint else slice(slot, slot + 1)
        binding = {
            "identity_sha256": identity_sha,
            "model_id": model_id,
            "parameter_sha256": parameter_sha,
            "provenance": provenance,
            "forecast_origin": origin,
            "history_start": origin - len(raw),
            "horizon": horizon,
            "target_slot": slot,
            "input_shape": list(effective.shape),
            "mean": mean[location].tolist(),
            "scale": scale[location].tolist(),
            "source_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(mask, bool).tobytes()
            ).hexdigest(),
        }
        encoded = json.dumps(binding, sort_keys=True)
        key = hashlib.sha256(encoded.encode() + effective.tobytes()).hexdigest()
        path = BASE / "tail-forecasts-v001" / model_id / "queries" / f"{key}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["binding"]) != encoded:
                raise ValueError("a forecast time or parameter binding changed")
            np.testing.assert_array_equal(saved["effective_input"], effective)
            np.testing.assert_array_equal(saved["source_mask"], mask)
            point = saved["point"]
            if point.shape != (horizon, 2 if joint else 1) or not np.isfinite(point).all():
                raise ValueError("cached forecast output support changed")
        keys.add((model_id, key))
        return (point - mean[location]) / scale[location], key

    normal, budget, bridge, longs = (
        np.empty((8, 96, 2)),
        np.empty((8, 96, 2)),
        np.empty((96, 2)),
        np.empty((2, 96, 2)),
    )
    for slot in [-1] if joint else [0, 1]:
        gap = direct_gaps(context, joint)[0 if joint else slot]
        destination = slice(None) if joint else slice(slot, slot + 1)
        for index, action in enumerate(actions):
            selected = effective_input(context, candidates, ids, action, joint)
            for role, horizon, bank in (("normal", 96, normal), ("budget", 96 + gap, budget)):
                point, key = query(selected, context, row["origin"], horizon, slot)
                bank[index, :, destination] = point[:96]
                records.append(
                    {
                        "role": role,
                        "action": action,
                        "slot": slot,
                        "key": key,
                        "gap": gap,
                        "slice_start": 0,
                    }
                )
        cropped = context[: 96 - gap]
        preserved = cropped if joint else cropped[:, slot : slot + 1]
        original = context if joint else context[:, slot : slot + 1]
        if np.isfinite(preserved).sum() != np.isfinite(original).sum():
            raise ValueError("moving the origin discarded available observations")
        point, key = query(
            direct_guard(cropped, data["defaults"], joint),
            cropped,
            row["origin"] - gap,
            96 + gap,
            slot,
        )
        # The first requested timestamp is t-g; output g must therefore be time t.
        times = np.arange(row["origin"] - gap, row["origin"] + 96)
        np.testing.assert_array_equal(times[gap:], np.arange(row["origin"], row["origin"] + 96))
        bridge[:, destination] = point[gap:]
        records.append({"role": "bridge", "slot": slot, "key": key, "gap": gap, "slice_start": gap})
        if gap == 0:
            np.testing.assert_array_equal(
                bridge[:, destination], normal[actions.index("guarded_direct"), :, destination]
            )
        for index, length in enumerate((1024, 4096)):
            original_long = data[f"long{length}"]
            point, key = query(
                direct_guard(original_long, data["defaults"], joint),
                original_long,
                row["origin"],
                96,
                slot,
            )
            longs[index, :, destination] = point
            records.append(
                {"role": f"long{length}", "slot": slot, "key": key, "gap": 0, "slice_start": 0}
            )
    return (
        {"normal": normal, "budget": budget, "bridge": bridge, "long": longs},
        actions,
        records,
        keys,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed tail audits")
    plan_root, input_root, forecast_root = [
        BASE / name for name in ("tail-plan-v001", "tail-inputs-v001", "tail-forecasts-v001")
    ]
    plan = read_json(plan_root / "manifest.json")
    inputs = read_json(input_root / ("smoke_preparation.json" if args.smoke else "manifest.json"))
    forecasts = read_json(forecast_root / ("smoke.json" if args.smoke else "manifest.json"))
    if any(row["status"] != "completed" for row in (plan, inputs, forecasts)):
        raise ValueError("registered inputs and forecasts must complete first")
    for record, checks in (
        (
            plan,
            {
                "script_sha256": ROOT / "scripts/plan_tail_bridge.py",
                "core_module_sha256": ROOT / "scripts/tail_bridge_core.py",
                "protocol_sha256": ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_PROTOCOL.md",
            },
        ),
        (
            inputs["identity"],
            {
                "script_sha256": ROOT / "scripts/prepare_tail_bridge.py",
                "pool_module_sha256": ROOT / "scripts/matched_replay_pool.py",
                "plan_sha256": plan_root / "manifest.json",
            },
        ),
        (
            forecasts["identity"],
            {
                "script_sha256": ROOT / "scripts/forecast_tail_bridge.py",
                "preparation_identity_sha256": input_root / "identity.json",
                "core_module_sha256": ROOT / "scripts/tail_bridge_core.py",
            },
        ),
    ):
        for key, path in checks.items():
            if record[key] != file_sha256(path):
                raise ValueError("a frozen tail experiment definition changed")
    sources = native_sources()
    source_map = {(s["cohort"], s["dataset_id"], s["item_id"]): s for s in sources}
    natural = set()
    for source in sources:
        for entry in source["episodes"]:
            origin = int(entry["window"]["origin"])
            raw = source["values"][origin - 96 : origin]
            if any(direct_gaps(raw, False)):
                natural.add(entry["episode_id"])
    if (
        natural
        != {row["original_episode_id"] for row in plan["cases"] if row["panel"] != "synthetic"}
        or len(natural) != 17
    ):
        raise ValueError("the complete natural tail eligibility population changed")
    r19 = read_json(ROOT / "artifacts/iclr27-r19/pilot-plan-v001/manifest.json")
    anchors = {
        (r["cohort"], r["dataset_id"], r["item_id"], u)
        for r in r19["cases"]
        for u in r["selected_anchors"]
    }
    synthetic = [row for row in plan["cases"] if row["panel"] == "synthetic"]
    if len({r["base_id"] for r in synthetic}) != 12 or len(synthetic) != 36:
        raise ValueError("the synthetic histories or shared gap panels changed")
    for base_id in {r["base_id"] for r in synthetic}:
        rows = [r for r in synthetic if r["base_id"] == base_id]
        if (
            sorted(r["gap"] for r in rows) != [8, 24, 48]
            or len({(r["cohort"], r["dataset_id"], r["item_id"], r["origin"]) for r in rows}) != 1
        ):
            raise ValueError("synthetic gaps do not share the same historical target")
    input_map = {r["case_id"]: r for r in inputs["cases"]}
    forecast_map = {(r["model_id"], r["case_id"]): r for r in forecasts["cases"]}
    parameters = {r["model_id"]: r["parameter_sha256"] for r in forecasts["costs"]}
    expected_parameters = {
        "chronos2": "ee1f6a9827cd88ee8b87b5aedbd9f2ae2f3c7123dcd629ee79f1f8fdd0595e9c",
        "timesfm2p5": "8c926884efa6844309ebf6a96c79cd56bc9834a0a679d1f26d2562a59b4c8f05",
    }
    if parameters != expected_parameters:
        raise ValueError("the frozen forecasting parameters changed")
    decisions = {}
    if not args.smoke:
        study_root = BASE / "tail-results-v001"
        study = read_json(study_root / "manifest.json")
        frozen = read_json(study_root / "predictions_frozen.json")
        if study["status"] != "completed" or study["predictions"] != frozen["predictions"]:
            raise ValueError("complete the frozen tail readout first")
        for name, digest in study["identity"]["files"].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("a study source, script or protocol changed")
        default_path = ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json"
        default_sha = file_sha256(default_path)
        old_freeze = read_json(default_path.parent / "decisions_frozen.json")
        old_audit = read_json(ROOT / "artifacts/iclr27-r19/replay-audit-v001/manifest.json")
        if (
            default_sha != frozen["source_defaults_sha256"]
            or default_sha != old_freeze["source_defaults_sha256"]
            or old_audit["study_sha256"] != file_sha256(default_path.parent / "manifest.json")
        ):
            raise ValueError("source defaults no longer match the completed R19 audit")
        defaults = read_json(default_path)
        decisions = {(r["model_id"], r["case_id"]): r for r in study["predictions"]}
        scores = pd.read_parquet(study_root / "case_scores.parquet").set_index(
            ["model_id", "case_id", "method"]
        )
        target_scores = pd.read_parquet(study_root / "target_scores.parquet").set_index(
            ["model_id", "case_id", "method", "target_slot"]
        )
    all_keys, fits, verified_pairs, maximum_error, logical = set(), {}, 0, 0.0, 0
    cases = smoke_cases(plan["cases"]) if args.smoke else plan["cases"]
    for row in cases:
        key = row["cohort"], row["dataset_id"], row["item_id"]
        source, entry = source_map[key], input_map[row["case_id"]]
        fit_key = key[:2]
        if fit_key not in fits:
            fits[fit_key] = fit_cutoff(source, sources, entry)
        if (
            fits[fit_key] != pd.Timestamp(entry["latest_neural_training_boundary"])
            or timestamp(source, row["origin"] - 96) < fits[fit_key]
        ):
            raise ValueError("frozen imputer training chronology is invalid")
        path = input_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen prepared tail input changed")
        context = source["values"][row["origin"] - 96 : row["origin"]].copy()
        if row["panel"] == "synthetic":
            if (*key, row["origin"]) not in anchors or not np.isfinite(context).all():
                raise ValueError("synthetic context is not a complete registered R19 history")
            context[-row["gap"] :] = np.nan
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["context"], context)
            prefix = source["values"][: source["prefix_end"]]
            numeric_prefix = np.asarray(prefix, dtype=float)
            mean, scale = (
                np.nanmean(numeric_prefix, axis=0),
                np.nanstd(numeric_prefix, axis=0, ddof=0),
            )
            scale = np.where(scale <= 1e-12, 1.0, scale)
            np.testing.assert_array_equal(mean, data["mean"])
            np.testing.assert_array_equal(scale, data["scale"])
            np.testing.assert_array_equal(np.nanmedian(prefix, axis=0), data["defaults"])
            for candidate in data["candidate_values"]:
                if not np.isfinite(candidate).all():
                    raise ValueError("an imputation candidate is incomplete")
                np.testing.assert_array_equal(
                    candidate[np.isfinite(context)], context[np.isfinite(context)]
                )
            for length in (1024, 4096):
                raw = source["values"][
                    max(source["prefix_end"], row["origin"] - length) : row["origin"]
                ].copy()
                raw[-96:] = context
                np.testing.assert_array_equal(raw, data[f"long{length}"])
            for model_id in row["models"]:
                if direct_gaps(context, model_id == "chronos2") != row["gaps"][model_id]:
                    raise ValueError("a model-specific eligible tail changed")
                rebuilt, actions, records, keys = reconstruct(
                    row, data, entry, model_id, forecasts["identity_sha256"], parameters[model_id]
                )
                all_keys.update(keys)
                logical += len(records)
                verified_pairs += 1
                if args.smoke:
                    continue
                record = forecast_map[(model_id, row["case_id"])]
                pred_path = forecast_root / record["path"]
                if file_sha256(pred_path) != record["sha256"]:
                    raise ValueError("a frozen case forecast changed")
                with np.load(pred_path, allow_pickle=False) as saved:
                    for name, value in rebuilt.items():
                        np.testing.assert_array_equal(saved[name], value)
                    if (
                        saved["actions"].tolist() != actions
                        or json.loads(str(saved["query_records"])) != records
                    ):
                        raise ValueError("the query request or action inventory differs")
                decision = decisions[(model_id, row["case_id"])]
                selected_path = study_root / decision["path"]
                if file_sha256(selected_path) != decision["sha256"]:
                    raise ValueError("a frozen fixed-method prediction changed")
                control = defaults["models"][model_id]
                truth = source["values"][row["origin"] : row["origin"] + 96, :2]
                if row["current_artifact"] is not None:
                    if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
                        raise ValueError("the previously scored natural outcome file changed")
                    with np.load(row["current_artifact"], allow_pickle=False) as old:
                        np.testing.assert_array_equal(truth, old["future"][:96])
                        np.testing.assert_array_equal(
                            np.isfinite(truth), old["future_observed"][:96]
                        )
                with np.load(selected_path, allow_pickle=False) as selected:
                    methods = selected["methods"].tolist()
                    for index, method in enumerate(methods):
                        if method == "bridge_native":
                            expected = rebuilt["bridge"]
                        elif method.startswith("native_long"):
                            expected = rebuilt["long"][0 if method.endswith("1024") else 1]
                        else:
                            bank = rebuilt["budget" if method.startswith("budget_") else "normal"]
                            name = method.removeprefix("budget_")
                            if name in actions:
                                expected = bank[actions.index(name)]
                            elif name == "median8":
                                ordered = np.sort(bank, axis=0)
                                expected = (ordered[3] + ordered[4]) / 2
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
                                expected = sum(
                                    weight * point
                                    for weight, point in zip(weights, bank, strict=True)
                                )
                        np.testing.assert_allclose(
                            expected, selected["points"][index], rtol=1e-12, atol=1e-12
                        )
                    loss = errors(selected["points"], truth, mean, scale)
                    for index, method in enumerate(methods):
                        scored = scores.loc[
                            (model_id, row["case_id"], method), ["mae", "mse"]
                        ].to_numpy(float)
                        delta = abs(scored - loss[index].mean(0))
                        maximum_error = max(maximum_error, float(delta.max()))
                        np.testing.assert_allclose(
                            scored, loss[index].mean(0), rtol=1e-10, atol=1e-10
                        )
                        for slot in (0, 1):
                            target = target_scores.loc[(model_id, row["case_id"], method, slot)]
                            np.testing.assert_allclose(
                                target[["mae", "mse"]].to_numpy(float),
                                loss[index, slot],
                                rtol=1e-10,
                                atol=1e-10,
                            )
                            if target.observed_count != np.isfinite(truth[:, slot]).sum():
                                raise ValueError("target loss used a different observed mask")
    for cost in forecasts["costs"]:
        count = sum(model == cost["model_id"] for model, _ in all_keys)
        if (
            count != cost["distinct_query_keys"]
            or cost["new_effective_inputs"] + cost["cache_hits"] != cost["logical_scope_requests"]
        ):
            raise ValueError("forecast cache or request accounting is inconsistent")
    if logical != sum(row["logical_scope_requests"] for row in forecasts["costs"]):
        raise ValueError("logical scope counts differ from reconstructed requests")
    if not args.smoke:
        if (
            verified_pairs != 100
            or logical != 2907
            or len(scores) != 2900
            or len(target_scores) != 5800
        ):
            raise ValueError("the registered full experiment population changed")
        # Explicit nested averages independently check every reported panel and source group.
        frame = scores.reset_index()
        summary = pd.read_csv(study_root / "summary.csv")
        group_table = pd.read_csv(study_root / "groups.csv").set_index(
            ["evaluation_panel", "model_id", "method", "group_id"]
        )
        for item in summary.itertuples(index=False):
            part = frame[(frame.model_id == item.model_id) & (frame.method == item.method)]
            if item.evaluation_panel == "native_target_all":
                part = part[part.panel != "synthetic"]
            elif item.evaluation_panel.startswith("synthetic"):
                part = part[part.panel == "synthetic"]
                if item.evaluation_panel != "synthetic_all":
                    part = part[part.gap == int(item.evaluation_panel.rsplit("g", 1)[1])]
            else:
                part = part[part.panel == item.evaluation_panel]
            group_values = []
            for group, grouped in part.groupby("group_id"):
                dataset_values = []
                for _, dataset in grouped.groupby("dataset_id"):
                    series_values = []
                    for _, series in dataset.groupby("item_id"):
                        histories = [
                            history[["mae", "mse"]].to_numpy().mean(0)
                            for _, history in series.groupby("base_id")
                        ]
                        series_values.append(np.mean(histories, axis=0))
                    dataset_values.append(np.mean(series_values, axis=0))
                value = np.mean(dataset_values, axis=0)
                np.testing.assert_allclose(
                    value,
                    group_table.loc[
                        (item.evaluation_panel, item.model_id, item.method, group), ["mae", "mse"]
                    ],
                    rtol=1e-12,
                    atol=1e-12,
                )
                group_values.append(value)
            np.testing.assert_allclose(
                np.mean(group_values, axis=0), [item.mae, item.mse], rtol=1e-12, atol=1e-12
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke_only": args.smoke,
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": None if args.smoke else file_sha256(study_root / "manifest.json"),
            "verified_case_model_pairs": verified_pairs,
            "verified_distinct_queries": len(all_keys),
            "verified_logical_scope_requests": logical,
            "natural_eligible_events": len(natural),
            "maximum_metric_difference": maximum_error,
            "current_future_values_scored": not args.smoke,
        },
    )


if __name__ == "__main__":
    main()
