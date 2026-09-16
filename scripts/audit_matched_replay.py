"""Independently reconstruct historical masks, query inputs, decisions and pilot scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401 - initialize Arrow before Torch-dependent audit helpers.
from aligned_portfolio_io import decision_truth
from audit_metric_source_gates import direct_control
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources, timestamp
from pool_gate_inputs import load_pool_inputs

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _write_json, file_sha256


def errors(predictions, truth, mean, scale):
    result = np.empty((len(predictions), 2, 2))
    for slot in (0, 1):
        available = np.isfinite(truth[:, slot])
        if available.sum() < 48:
            raise ValueError("common original target observations are insufficient")
        raw = predictions[:, :, slot] * scale[slot] + mean[slot]
        delta = (raw[:, available] - truth[available, slot]) / scale[slot]
        result[:, slot, 0] = abs(delta).mean(1)
        result[:, slot, 1] = (delta**2).mean(1)
    return result


def runs(mask):
    result = []
    for column in mask.T:
        lengths, length = [], 0
        for missing in [*column, False]:
            if missing:
                length += 1
            elif length:
                lengths.append(length)
                length = 0
        result.append(sorted(lengths))
    return result


def effective_input(context, candidates, ids, action, joint):
    locf = candidates[ids.index("locf")]
    if action != "guarded_direct":
        values = candidates[ids.index(action)]
    elif joint:
        values = locf if (~np.isfinite(context).any(0)).any() else context
    else:
        values = context.copy()
        for slot in (0, 1):
            if not np.isfinite(context[:, slot]).any():
                values[:, slot] = locf[:, slot]
    return values if joint else values[:, :2]


def independent_choices(loss, joint):
    paired = loss[:, :8] - loss[:, 8:9]
    if joint:
        paired = (paired[:, :, 0] + paired[:, :, 1])[:, :, None] / 2
    mean = np.sum(paired, axis=0) / 8
    uncertainty = np.sqrt(np.sum((paired - mean[None]) ** 2, axis=0) / 7 / 8)
    chosen = []
    for penalty in (None, 0.0, 1.0):
        score = mean if penalty is None else mean + penalty * uncertainty
        selections = []
        for slot in range(score.shape[1]):
            minimum = min(score[:, slot])
            index = next(i for i in range(8) if score[i, slot] <= minimum + 1e-10)
            selections.append(index if penalty is None or minimum < -1e-10 else 8)
        chosen.append(selections * 2 if joint else selections)
    return np.asarray(chosen), mean, uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed replay audits")
    base = ROOT / "artifacts/iclr27-r19"
    plan = read_json(base / "pilot-plan-v001/manifest.json")
    prepared = read_json(base / "replay-inputs-v001/manifest.json")
    forecasts = read_json(base / "replay-forecasts-v001/manifest.json")
    study_root = base / "replay-results-v001"
    study = read_json(study_root / "manifest.json")
    frozen = read_json(study_root / "decisions_frozen.json")
    if (
        any(row["status"] != "completed" for row in (plan, prepared, forecasts, study))
        or frozen["decisions"] != study["decisions"]
    ):
        raise ValueError("complete and freeze every registered decision before auditing")
    for key, path in (
        ("script_sha256", ROOT / "scripts/evaluate_matched_replay.py"),
        ("core_module_sha256", ROOT / "scripts/matched_replay_core.py"),
        ("plan_sha256", base / "pilot-plan-v001/manifest.json"),
        ("prepared_sha256", base / "replay-inputs-v001/manifest.json"),
        ("forecast_sha256", base / "replay-forecasts-v001/manifest.json"),
        ("protocol_sha256", ROOT / "docs/iclr2027/R19_MATCHED_REPLAY_PROTOCOL.md"),
    ):
        if study["identity"][key] != file_sha256(path):
            raise ValueError("a replay definition changed")
    sources = native_sources()
    source_map = {(s["cohort"], s["dataset_id"], s["item_id"]): s for s in sources}
    metadata = {row["case_id"]: row for row in plan["cases"]}
    input_map = {row["case_id"]: row for row in prepared["cases"]}
    forecast_map = {(row["model_id"], row["case_id"]): row for row in forecasts["cases"]}
    defaults_path = study_root / "source_defaults.json"
    if file_sha256(defaults_path) != frozen["source_defaults_sha256"]:
        raise ValueError("source defaults changed after decision freeze")
    defaults = read_json(defaults_path)
    for model_id in ("chronos2", "timesfm2p5"):
        _, frame, arrays = load_pool_inputs(
            ROOT / "artifacts/iclr27-r12/motm-pool-inputs-v001", model_id
        )
        training = np.flatnonzero(frame.split.to_numpy() == "train")
        points = arrays["vectors"][training]
        truth = decision_truth(
            frame.iloc[training],
            np.load(
                ROOT / "artifacts/iclr27-r4/accuracy-development-v002/truth_z.npy", mmap_mode="r"
            ),
        )
        weight = _family_weights(frame.iloc[training])
        record = defaults["models"][model_id]
        actual = np.sum(
            abs(points - truth[:, None]).mean(2) * (weight / weight.sum())[:, None], axis=0
        )
        if int(actual.argmin()) != record["single_index"]:
            raise ValueError("the source MAE default used different outcomes")
        probability = np.asarray(record["fixed_mae"]["weights"])
        value, gradient = direct_control(points, truth, weight, probability, "mae")
        gap = (gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0)
        if gap > 1e-7 or probability.min() < 0 or abs(probability.sum() - 1) > 1e-10:
            raise ValueError("source fixed MAE optimality failed")
        np.testing.assert_allclose(value, record["fixed_mae"]["objective"], rtol=1e-12, atol=1e-12)
    checked_queries, rows, checked_fits = set(), [], {}
    maximum_risk_difference = 0.0
    for decision in study["decisions"]:
        model_id, case_id = decision["model_id"], decision["case_id"]
        row, input_record = metadata[case_id], input_map[case_id]
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        values = source["values"]
        fit_key = row["cohort"], row["dataset_id"]
        if fit_key not in checked_fits:
            item_map = {
                s["item_id"]: s
                for s in sources
                if s["cohort"] == row["cohort"] and s["dataset_id"] == row["dataset_id"]
            }
            training_path = source["root"] / "imputers" / row["dataset_id"] / "training_batch.npz"
            if file_sha256(training_path) != source["dataset"]["training_batch_sha256"]:
                raise ValueError("a registered native imputer batch changed")
            cutoffs = []
            with np.load(training_path, allow_pickle=False) as training_batch:
                for index, identifier in enumerate(training_batch["window_ids"].tolist()):
                    item, suffix = identifier.rsplit("@", 1)
                    start = int(suffix.split("|", 1)[0])
                    fitted_source = item_map[item]
                    if start + 96 > fitted_source["prefix_end"]:
                        raise ValueError("an imputer training interval exceeds its prefix")
                    observed = training_batch["observed"][index]
                    np.testing.assert_array_equal(
                        training_batch["values"][index][observed],
                        fitted_source["values"][start : start + 96][observed],
                    )
                    cutoffs.append(timestamp(fitted_source, start + 96))
            for fit in input_record["frozen_neural_imputers"]:
                path = Path(fit["path"])
                if file_sha256(path) != fit["sha256"]:
                    raise ValueError("a frozen neural imputer record changed")
                fitted = read_json(path)
                if fitted["status"] != "fitted":
                    raise ValueError("a registered imputer was not fitted")
                for item in fitted["files"]:
                    if (
                        file_sha256(path.parent / fit["candidate_id"] / item["path"])
                        != item["sha256"]
                    ):
                        raise ValueError("a frozen neural imputer artifact changed")
            checked_fits[fit_key] = max(cutoffs)
        if checked_fits[fit_key] != pd.Timestamp(input_record["latest_neural_training_boundary"]):
            raise ValueError("the claimed imputer training cutoff is incorrect")
        input_path = base / "replay-inputs-v001" / input_record["path"]
        forecast_record = forecast_map[(model_id, case_id)]
        forecast_path = base / "replay-forecasts-v001" / forecast_record["path"]
        decision_path = study_root / decision["path"]
        for path, digest in (
            (input_path, input_record["sha256"]),
            (forecast_path, forecast_record["sha256"]),
            (decision_path, decision["sha256"]),
        ):
            if file_sha256(path) != digest:
                raise ValueError("a frozen case input, prediction or decision changed")
        with (
            np.load(input_path, allow_pickle=False) as data,
            np.load(forecast_path, allow_pickle=False) as predicted,
            np.load(decision_path, allow_pickle=False) as selected,
        ):
            context = values[row["origin"] - 96 : row["origin"]]
            np.testing.assert_array_equal(context, data["context"])
            mask = ~np.isfinite(context)
            np.testing.assert_array_equal(data["masks"][2], mask)
            valid_shifts = [
                amount
                for amount in range(1, 96)
                if runs(np.roll(mask, amount, axis=0)) == runs(mask)
                and not np.array_equal(np.roll(mask, amount, axis=0), mask)
            ]
            seed = int(
                hashlib.sha256(("r19-mask|" + row["episode_id"]).encode()).hexdigest()[:8], 16
            )
            rng = np.random.default_rng(seed)
            if (
                valid_shifts[int(rng.integers(len(valid_shifts)))]
                != row["mask_definition"]["shift"]
            ):
                raise ValueError("the position control did not use its registered random draw")
            generic = np.zeros_like(mask)
            for column in range(mask.shape[1]):
                generic[rng.choice(96, int(mask[:, column].sum()), replace=False), column] = True
            np.testing.assert_array_equal(generic, data["masks"][0])
            shifted = np.roll(mask, row["mask_definition"]["shift"], axis=0)
            np.testing.assert_array_equal(data["masks"][1], shifted)
            if runs(shifted) != runs(mask) or np.array_equal(shifted, mask):
                raise ValueError("the position control changes block sizes or is degenerate")
            np.testing.assert_array_equal(
                shifted.astype(int).T @ shifted.astype(int), mask.astype(int).T @ mask.astype(int)
            )
            for alternative in data["masks"]:
                np.testing.assert_array_equal(alternative.sum(0), mask.sum(0))
            eligible = []
            for past in range(row["origin"] - 96, 0, -96):
                if past - 96 < max(row["prefix_end"], row["origin"] - 4096):
                    break
                if (
                    np.isfinite(values[past - 96 : past]).all()
                    and (np.isfinite(values[past : past + 96, :2]).sum(0) >= 48).all()
                ):
                    eligible.append(past)
            if eligible[:8] != row["selected_anchors"]:
                raise ValueError("history selection used a different support rule")
            if timestamp(source, min(eligible[:8]) - 96) < pd.Timestamp(
                input_record["latest_neural_training_boundary"]
            ):
                raise ValueError("a calibration context precedes the recorded training cutoff")
            mean, scale = data["mean"], data["scale"]
            prefix = values[: row["prefix_end"]]
            np.testing.assert_allclose(mean, np.nanmean(prefix, axis=0), rtol=1e-12, atol=1e-12)
            actual_scale = np.nanstd(prefix, axis=0)
            actual_scale = np.where(actual_scale > 1e-12, actual_scale, 1.0)
            np.testing.assert_allclose(scale, actual_scale, rtol=1e-12, atol=1e-12)
            ids, actions = data["candidate_ids"].tolist(), predicted["actions"].tolist()
            for candidate in data["current_candidates"]:
                np.testing.assert_array_equal(
                    candidate[np.isfinite(context)], context[np.isfinite(context)]
                )
            joint = model_id == "chronos2"
            query_specs = []
            for i, action in enumerate(actions):
                raw = effective_input(context, data["current_candidates"], ids, action, joint)
                query_specs.append(
                    (predicted["current_keys"][i], raw, mask, False, predicted["current"][i])
                )
            for rule in range(3):
                historical = []
                for anchor, past in enumerate(eligible[:8]):
                    masked = values[past - 96 : past].copy()
                    masked[data["masks"][rule]] = np.nan
                    np.testing.assert_array_equal(masked, data["historical_contexts"][rule, anchor])
                    np.testing.assert_array_equal(
                        values[past : past + 96, :2], data["historical_future"][anchor]
                    )
                    candidates = data["historical_candidates"][rule, anchor]
                    observed = np.isfinite(masked)
                    for candidate in candidates:
                        np.testing.assert_array_equal(candidate[observed], masked[observed])
                    for i, action in enumerate(actions):
                        raw = effective_input(masked, candidates, ids, action, joint)
                        query_specs.append(
                            (
                                predicted["historical_keys"][rule, anchor, i],
                                raw,
                                ~observed,
                                False,
                                predicted["historical"][rule, anchor, i],
                            )
                        )
                    p = predicted["historical"][rule, anchor]
                    bank = np.concatenate([p, np.median(p, axis=0)[None]])
                    historical.append(errors(bank, values[past : past + 96, :2], mean, scale))
                historical = np.stack(historical)
                for metric, name in ((0, "historical_mae"), (1, "historical_mse")):
                    delta = float(abs(historical[..., metric] - selected[name][rule]).max())
                    maximum_risk_difference = max(maximum_risk_difference, delta)
                    np.testing.assert_allclose(
                        historical[..., metric], selected[name][rule], rtol=1e-12, atol=1e-12
                    )
                # Replay decisions from the verified stored loss representation to retain its declared arithmetic.
                choices, paired, uncertainty = independent_choices(
                    selected["historical_mae"][rule], joint
                )
                np.testing.assert_array_equal(choices, selected["choices"][rule])
                np.testing.assert_allclose(
                    paired, selected["paired_means"][rule], rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    uncertainty, selected["uncertainties"][rule], rtol=1e-12, atol=1e-12
                )
            if row["long_start"] != max(row["prefix_end"], row["origin"] - 4096):
                raise ValueError("the long-history control changed its available time range")
            long = np.asarray(values[row["long_start"] : row["origin"]]).copy()
            np.testing.assert_array_equal(long, data["long_context"])
            filled = long.copy()
            for column in range(long.shape[1]):
                last = data["defaults"][column]
                for time in range(len(long)):
                    if np.isfinite(filled[time, column]):
                        last = filled[time, column]
                    else:
                        filled[time, column] = last
            raw_long = filled if joint and (~np.isfinite(long).any(0)).any() else long.copy()
            if not joint:
                for slot in (0, 1):
                    if not np.isfinite(long[:, slot]).any():
                        raw_long[:, slot] = filled[:, slot]
                raw_long = raw_long[:, :2]
            for name, key, point in zip(
                predicted["extra_names"].tolist(),
                predicted["extra_keys"],
                predicted["extras"],
                strict=True,
            ):
                raw = (
                    effective_input(
                        context, data["current_candidates"], ids, "guarded_direct", joint
                    )
                    if name == "native_prefix96"
                    else raw_long
                )
                source_mask = mask if name == "native_prefix96" else ~np.isfinite(long)
                query_specs.append((key, raw, source_mask, name != "native_long_raw", point))
            for key, raw, source_mask, normalized, point in query_specs:
                effective = np.asarray(
                    (raw - (mean if joint else mean[:2])) / (scale if joint else scale[:2])
                    if normalized
                    else raw,
                    np.float32,
                ).copy(order="C")
                effective[np.isnan(effective)] = np.nan
                path = base / "replay-forecasts-v001" / model_id / "queries" / f"{key}.npz"
                with np.load(path, allow_pickle=False) as query:
                    np.testing.assert_array_equal(query["effective_input"], effective)
                    np.testing.assert_array_equal(query["source_mask"], source_mask)
                    binding_text = str(query["binding"])
                    binding = json.loads(binding_text)
                    if hashlib.sha256(
                        binding_text.encode() + str(effective.shape).encode() + effective.tobytes()
                    ).hexdigest() != str(key):
                        raise ValueError("a query identity cannot be reconstructed")
                    if (
                        binding["input_normalized"] != normalized
                        or binding["context_length"] != len(effective)
                        or binding["horizon"] != 96
                    ):
                        raise ValueError("a query changed its temporal or normalization definition")
                    np.testing.assert_array_equal(binding["mean"], mean)
                    np.testing.assert_array_equal(binding["scale"], scale)
                    if binding["fit_binding"]["neural"] != input_record["frozen_neural_imputers"]:
                        raise ValueError("a query used different imputer provenance")
                    parameter = next(
                        cost["parameter_sha256"]
                        for cost in forecasts["costs"]
                        if cost["model_id"] == model_id
                    )
                    if binding["parameter_sha256"] != parameter:
                        raise ValueError("a query used different forecasting weights")
                    rebuilt = (
                        query["point"] if normalized else (query["point"] - mean[:2]) / scale[:2]
                    )
                    np.testing.assert_array_equal(rebuilt, point)
                checked_queries.add((model_id, str(key)))
            current = predicted["current"]
            bank = np.concatenate([current, np.median(current, axis=0)[None]])
            np.testing.assert_array_equal(bank, selected["current_bank"])
            point_methods = dict(zip(selected["methods"].tolist(), selected["points"], strict=True))
            for index, action in enumerate(actions):
                np.testing.assert_array_equal(point_methods[action], current[index])
            np.testing.assert_array_equal(point_methods["median8"], bank[8])
            np.testing.assert_array_equal(point_methods["mean8"], current.mean(0))
            default = defaults["models"][model_id]
            np.testing.assert_array_equal(
                point_methods["source_single_mae"], current[default["single_index"]]
            )
            for method, weights in (
                ("source_fixed_mae", default["fixed_mae"]["weights"]),
                ("source_fixed_joint", default["fixed_joint_weights"]),
            ):
                direct = np.einsum("a,ahk->hk", weights, current)
                np.testing.assert_allclose(point_methods[method], direct, rtol=1e-12, atol=1e-12)
            for name, point in zip(
                predicted["extra_names"].tolist(), predicted["extras"], strict=True
            ):
                np.testing.assert_array_equal(point_methods[name], point)
            if len(query_specs) != 203:
                raise ValueError("the registered per-case logical forecast coverage changed")
            for rule_index, rule in enumerate(("generic", "shuffled", "matched")):
                for mode_index, mode in enumerate(("forced", "erm", "conservative")):
                    choices = selected["choices"][rule_index, mode_index]
                    point = np.column_stack([bank[int(choices[slot]), :, slot] for slot in (0, 1)])
                    np.testing.assert_array_equal(point, point_methods[f"{rule}_{mode}"])
            if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
                raise ValueError("the original current scoring data changed")
            with np.load(row["current_artifact"], allow_pickle=False) as actual:
                truth = actual["future"][:96]
                np.testing.assert_array_equal(truth, values[row["origin"] : row["origin"] + 96, :2])
                np.testing.assert_array_equal(actual["future_observed"][:96], np.isfinite(truth))
            losses = errors(selected["points"], truth, mean, scale).mean(1)
            results = dict(zip(selected["methods"].tolist(), losses, strict=True))
            base_losses = errors(bank, truth, mean, scale)
            for name, probability in zip(
                selected["random_methods"].tolist(), selected["random_probabilities"], strict=True
            ):
                if (probability < 0).any() or not np.allclose(probability.sum(0), 1):
                    raise ValueError("a random-action comparator has invalid probabilities")
                if name == "random_forced":
                    choices = [0, 0]
                else:
                    rule, mode, _, _ = name.split("_")
                    choices = selected["choices"][
                        ("generic", "shuffled", "matched").index(rule),
                        ("forced", "erm", "conservative").index(mode),
                    ]
                expected_probability = np.zeros((9, 2))
                for slot in (0, 1):
                    if choices[slot] == 8:
                        expected_probability[8, slot] = 1
                    else:
                        expected_probability[:8, slot] = 0.125
                np.testing.assert_array_equal(probability, expected_probability)
                results[name] = np.sum(probability[:, :, None] * base_losses, axis=0).mean(0)
            for method, loss in results.items():
                rows.append(
                    {
                        **{
                            name: row[name]
                            for name in (
                                "group_id",
                                "family_id",
                                "dataset_id",
                                "item_id",
                                "episode_id",
                                "origin_id",
                            )
                        },
                        "model_id": model_id,
                        "case_id": case_id,
                        "method": method,
                        "mae": loss[0],
                        "mse": loss[1],
                    }
                )
    frame = pd.DataFrame(rows)
    saved = pd.read_parquet(study_root / "case_scores.parquet")
    keys = ["model_id", "case_id", "method"]
    pd.testing.assert_frame_equal(
        frame.sort_values(keys).reset_index(drop=True),
        saved.sort_values(keys).reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    series = (
        frame.groupby(["model_id", "method", "group_id", "dataset_id", "item_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    datasets = (
        series.groupby(["model_id", "method", "group_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    groups = (
        datasets.groupby(["model_id", "method", "group_id"])[["mae", "mse"]].mean().reset_index()
    )
    summary = groups.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    pd.testing.assert_frame_equal(
        summary,
        pd.read_csv(study_root / "summary.csv", float_precision="round_trip"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if len(frame) != 1536 or len(study["decisions"]) != 48:
        raise ValueError("the registered replay audit coverage changed")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "study_sha256": file_sha256(study_root / "manifest.json"),
            "verified_queries": len(checked_queries),
            "verified_decisions": 48,
            "verified_scores": len(frame),
            "maximum_past_risk_difference": maximum_risk_difference,
            "new_forecaster_calls": 0,
            "limits": "support-qualified used-data pilot; no independent confirmation",
        },
    )


if __name__ == "__main__":
    main()
