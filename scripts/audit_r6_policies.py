"""Replay frozen R6 decisions and recompute every method, horizon and source score."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import load_frozen_selector, result_panels  # noqa: E402
from audit_followup_policies import explicit_votes  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from r6_policy_inputs import decision_inputs, pack_gate_features  # noqa: E402

from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "forecast-root",
        "policy-root",
        "method-freeze",
        "source-future-control",
        "pairwise-bundle",
        "legacy-bundle",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed R6 audits")
    binding = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    future_bundle = json.loads(
        (args.source_future_control / "manifest.json").read_text(encoding="utf-8")
    )
    pairwise = json.loads((args.pairwise_bundle / "manifest.json").read_text(encoding="utf-8"))
    legacy = json.loads((args.legacy_bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_controls = json.loads(Path(binding["controls_path"]).read_text(encoding="utf-8"))
    old_controls = json.loads(
        (args.legacy_bundle / legacy["controls_file"]).read_text(encoding="utf-8")
    )
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.prepared_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    torch.set_num_threads(1)
    total, weight_rows, feature_cases = 0, 0, 0
    maximum_prediction, maximum_metric = 0.0, 0.0
    model_records, summaries = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.policy_root / model_id
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        forecast_dir = args.forecast_root / model_id
        forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["status"] != "completed"
            or manifest["methods"] != 23
            or manifest["input_tasks"] != len(prep["episodes"])
        ):
            raise ValueError("the complete registered policy bank is required")
        for key, path in (
            ("method_freeze_sha256", args.method_freeze),
            ("prepared_sha256", args.prepared_root / "manifest.json"),
            ("source_future_control_sha256", args.source_future_control / "manifest.json"),
            ("forecast_sha256", forecast_dir / "manifest.json"),
            ("pairwise_bundle_sha256", args.pairwise_bundle / "manifest.json"),
            ("legacy_bundle_sha256", args.legacy_bundle / "manifest.json"),
        ):
            if manifest["identity"][key] != file_sha256(path):
                raise ValueError("a policy input identity changed")
        joint, actions = model_id == "chronos2", gate_controls[model_id]["actions"]
        saved_summary = pd.read_csv(root / "summary.csv").set_index(["horizon", "panel", "method"])
        for entry in manifest["horizons"]:
            horizon = entry["horizon"]
            directory = root / entry["directory"]
            frozen_path = directory / "predictions_frozen.json"
            if file_sha256(frozen_path) != entry["marker_sha256"]:
                raise ValueError("the pre-score prediction record changed")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            for name, sha in frozen["files_sha256"].items():
                if file_sha256(directory / name) != sha:
                    raise ValueError("a frozen policy input or decision changed")
            if file_sha256(directory / "policy_predictions.npz") != frozen["prediction_sha256"]:
                raise ValueError("policy predictions changed after scoring")
            with np.load(directory / "policy_predictions.npz", allow_pickle=False) as saved:
                bank, methods, ids = (
                    saved["point_z"],
                    saved["methods"].tolist(),
                    saved["episode_ids"].tolist(),
                )
            if (
                len(set(methods)) != 23
                or ids != [row["episode_id"] for row in prep["episodes"]]
                or bank.shape
                != (
                    len(ids),
                    23,
                    horizon,
                    2,
                )
            ):
                raise ValueError("prediction coverage, horizon or target axes changed")
            individual = pd.read_parquet(directory / "individual_features.parquet")
            portfolios = pd.read_parquet(directory / "portfolio_features.parquet")
            decisions = pd.read_parquet(directory / "decisions.parquet")
            order = pd.MultiIndex.from_product(
                [decisions.episode_id, actions], names=["episode_id", "candidate_id"]
            )
            features = (
                individual.set_index(["episode_id", "candidate_id"])
                .loc[order, list(FORECAST_FEATURES)]
                .to_numpy(np.float32)
                .reshape(len(decisions), 7, 33)
            )
            features = pack_gate_features(features)
            weights = {}
            for objective in ("ensemble", "member", "future"):
                source_entries = [
                    row
                    for row in (
                        future_bundle["models"]
                        if objective == "future"
                        else binding["source_models"]
                    )
                    if row["model_id"] == model_id
                    and (objective == "future" or row["objective"] == objective)
                ]
                if [row["seed"] for row in source_entries] != [5101, 5102, 5103]:
                    raise ValueError("source seed selection changed")
                replayed = []
                for source in source_entries:
                    path = (
                        args.source_future_control / source["path"]
                        if objective == "future"
                        else Path(source["path"])
                    )
                    if file_sha256(path) != source["sha256"]:
                        raise ValueError("a source gate changed")
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    replayed.append(replay_network(checkpoint["state_dict"], features))
                with np.load(directory / f"{objective}_weights.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["seed_weights"], np.stack(replayed))
                    averaged = np.mean(replayed, axis=0)
                    np.testing.assert_array_equal(saved["mean_weights"], averaged)
                weights["source_future_gate" if objective == "future" else objective + "_gate"] = (
                    averaged
                )
                weight_rows += len(decisions)
            choices = {}
            for objective in ("median_risk", "member_risk"):
                learner, _ = load_frozen_selector(
                    args.pairwise_bundle, pairwise, model_id, objective, "label_kind"
                )
                replayed = (
                    explicit_votes(learner, portfolios)
                    .set_index("episode_id")
                    .candidate_id.sort_index()
                )
                expected_choices = (
                    pd.read_parquet(directory / f"{objective}_choices.parquet")
                    .set_index("episode_id")
                    .candidate_id.sort_index()
                )
                pd.testing.assert_series_equal(replayed, expected_choices)
                choices[objective] = expected_choices.to_dict()
                if (
                    objective == "median_risk"
                    and frozen["source_fixed_choices"]["source_fixed_median_risk"]
                    != learner.baseline_id
                ):
                    raise ValueError("the source-fixed triple changed")
                del learner
            learner, _ = load_frozen_selector(
                args.legacy_bundle, legacy, model_id, "clean_forecast_mse", "objective"
            )
            episode_ids, margins, pairs = learner.pair_scores(individual)
            ranks = rank_pairwise_candidates(
                margins, pairs, learner.candidate_ids, learner.baseline_id
            )
            ranked = {
                identifier: [learner.candidate_ids[i] for i in row[:3]]
                for identifier, row in zip(episode_ids, ranks, strict=True)
            }
            if ranked != json.loads(
                (directory / "old_teacher_choices.json").read_text(encoding="utf-8")
            ):
                raise ValueError("the original teacher rule did not replay")
            del learner
            forecast_entry = next(row for row in forecast["horizons"] if row["horizon"] == horizon)
            horizon_manifest_path = forecast_dir / forecast_entry["path"]
            if file_sha256(horizon_manifest_path) != forecast_entry["sha256"]:
                raise ValueError("candidate forecasts changed")
            candidate_records = {
                row["episode_id"]: row
                for row in json.loads(horizon_manifest_path.read_text(encoding="utf-8"))[
                    "predictions"
                ]
            }
            score_path = directory / "episode_results.parquet"
            if file_sha256(score_path) != entry["scores_sha256"]:
                raise ValueError("recorded scores changed")
            scores = pd.read_parquet(score_path)
            if len(scores) != len(ids) * 23 or scores.duplicated(["episode_id", "method"]).any():
                raise ValueError("a method lost or duplicated evaluation windows")
            indexed_scores = scores.set_index(["episode_id", "method"])
            decision_positions = {
                identifier: index for index, identifier in enumerate(decisions.episode_id)
            }
            sampled = set()
            for index, record in enumerate(prep["episodes"]):
                row = candidate_records[record["episode_id"]]
                path = horizon_manifest_path.parent / row["path"]
                if file_sha256(path) != row["sha256"]:
                    raise ValueError("a candidate prediction artifact changed")
                with np.load(path, allow_pickle=False) as saved:
                    raw_points, raw_order = saved["point_z"], saved["candidate_ids"].tolist()
                points = raw_points[[raw_order.index(name) for name in actions]]
                expected = dict(zip(raw_order, raw_points, strict=True))
                expected.update(
                    forecast_median_guarded=np.median(raw_points[:7], axis=0),
                    forecast_mean_guarded=raw_points[:7].mean(0),
                    forecast_median_finite=np.median(raw_points[:6], axis=0),
                    forecast_mean_finite=raw_points[:6].mean(0),
                    forecast_median_with_motm=np.median(raw_points, axis=0),
                )
                for name in (
                    *weights,
                    "gate_source_fixed_convex",
                    "gate_source_fixed_single",
                    *choices,
                    "source_fixed_median_risk",
                    "old_teacher_rank3",
                    "old_source_fixed3",
                ):
                    result = []
                    for target in (0, 1):
                        identifier = (
                            record["episode_id"]
                            if joint
                            else record["episode_id"] + f"|target={target}"
                        )
                        position = decision_positions[identifier]
                        if name in weights or name == "gate_source_fixed_convex":
                            probability = (
                                weights[name][position]
                                if name in weights
                                else np.asarray(gate_controls[model_id]["convex_weights"])
                            )
                            probability = probability / probability.sum()
                            value = points[0, :, target] + (
                                probability[:, None]
                                * (points[:, :, target] - points[:1, :, target])
                            ).sum(0)
                        elif name == "gate_source_fixed_single":
                            value = points[gate_controls[model_id]["single_index"], :, target]
                        else:
                            if name in choices:
                                selected = choices[name][identifier][len("median:") :].split("+")
                            elif name == "source_fixed_median_risk":
                                selected = frozen["source_fixed_choices"][name][
                                    len("median:") :
                                ].split("+")
                            elif name == "old_teacher_rank3":
                                selected = ranked[identifier]
                            else:
                                selected = old_controls[model_id]["fixed3_clean_forecast_mse"][
                                    target
                                ]
                            if len(selected) != 3 or len(set(selected)) != 3:
                                raise ValueError("a triple has invalid membership")
                            value = np.median(
                                points[[actions.index(name) for name in selected], :, target],
                                axis=0,
                            )
                        result.append(value)
                    expected[name] = np.stack(result, axis=1)
                rebuilt = np.stack([expected[name] for name in methods])
                maximum_prediction = max(
                    maximum_prediction, float(abs(rebuilt - bank[index]).max())
                )
                np.testing.assert_allclose(rebuilt, bank[index], rtol=1e-12, atol=1e-12)
                total += 23
                path = args.prepared_root / record["path"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("an audited source input changed")
                with np.load(path, allow_pickle=False) as saved:
                    future, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
                    context = saved["context"]
                    key = (
                        record["dataset_id"],
                        record["mechanism"],
                        bool((~np.isfinite(context).any(0)).any()),
                    )
                    if key not in sampled:
                        candidates, coverage, candidate_ids = (
                            saved["candidate_values"],
                            saved["native_coverage"],
                            saved["candidate_ids"].tolist(),
                        )
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
                if key not in sampled:
                    generated = decision_inputs(
                        context,
                        candidates,
                        candidate_ids,
                        coverage,
                        raw_points[:7],
                        mean,
                        scale,
                        joint=joint,
                        period=record["period"],
                        metadata={
                            **record,
                            "model_id": model_id,
                            "episode_index": index,
                            "split": "r6_confirmation",
                        },
                    )
                    for table, regenerated in (
                        (individual, generated["individual"]),
                        (portfolios, generated["portfolios"]),
                    ):
                        old_rows = table[
                            table.source_episode_id == record["episode_id"]
                        ].reset_index(drop=True)
                        pd.testing.assert_frame_equal(
                            old_rows,
                            regenerated.reset_index(drop=True),
                            check_dtype=False,
                            check_exact=True,
                        )
                    sampled.add(key)
                    feature_cases += 1
                np.testing.assert_array_equal(observed, np.isfinite(future))
                counts = observed.sum(0)
                if (counts < horizon // 2).any():
                    raise ValueError("a method scored an ineligible future")
                residual = bank[index] - (future - mean[:2]) / scale[:2]
                for position, method in enumerate(methods):
                    saved_row = indexed_scores.loc[(record["episode_id"], method)]
                    if (
                        saved_row.native_missing_context != record["window"]["context_has_missing"]
                        or [saved_row.future_observed_target0, saved_row.future_observed_target1]
                        != counts.tolist()
                    ):
                        raise ValueError("a missingness panel or future observation count changed")
                    if (
                        saved_row.horizon != horizon
                        or saved_row.model_id != model_id
                        or any(
                            saved_row[key] != record[key]
                            for key in (
                                "origin_id",
                                "dataset_id",
                                "family_id",
                                "item_id",
                                "panel",
                                "mechanism",
                                "missing_rate",
                                "mask_seed",
                            )
                        )
                    ):
                        raise ValueError("a score was assigned to the wrong horizon or condition")
                    for metric, values in (
                        ("mae", abs(residual)),
                        ("mse", residual**2),
                        ("raw_mae", abs(residual * scale[:2])),
                        ("raw_mse", (residual * scale[:2]) ** 2),
                    ):
                        value = np.mean(
                            [
                                values[position, observed[:, target], target].mean()
                                for target in (0, 1)
                            ]
                        )
                        maximum_metric = max(maximum_metric, abs(value - saved_row[metric]))
                        np.testing.assert_allclose(value, saved_row[metric], rtol=1e-10, atol=1e-10)
            for panel, frame in result_panels(scores):
                keys, metrics = (
                    ["method", "family_id", "dataset_id", "item_id"],
                    ["mae", "mse", "raw_mae", "raw_mse"],
                )
                items = frame.groupby(keys)[metrics].mean()
                datasets = items.groupby(keys[:-1])[metrics].mean()
                families = datasets.groupby(keys[:-2])[metrics].mean()
                summary = families.groupby("method")[metrics].mean()
                np.testing.assert_allclose(
                    saved_summary.loc[(horizon, panel)].loc[summary.index, metrics],
                    summary,
                    rtol=1e-12,
                    atol=1e-12,
                )
                summaries.append(
                    summary.reset_index().assign(
                        model_id=model_id,
                        horizon=horizon,
                        panel=panel,
                        families=frame.family_id.nunique(),
                    )
                )
        model_records.append(
            {"model_id": model_id, "manifest_sha256": file_sha256(root / "manifest.json")}
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "models": model_records,
            "verified_method_vectors": total,
            "verified_gate_decisions": weight_rows,
            "raw_feature_replay_cases": feature_cases,
            "maximum_prediction_difference": maximum_prediction,
            "maximum_metric_difference": maximum_metric,
            "limits": "prediction, decision, feature and metric audit; conclusions still require the prespecified paired source-level readout",
        },
    )


if __name__ == "__main__":
    main()
