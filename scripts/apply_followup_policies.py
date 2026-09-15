"""Apply frozen selectors to the follow-up forecast bank before scoring any futures."""

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_preforecast_student import extend_forecast_features  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.routing.followup_portfolio import (  # noqa: E402
    portfolio_feature_frame,
    selected_portfolio_points,
)
from tsfm_fais.routing.forecast_response import forecast_response_inputs  # noqa: E402
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def load_frozen_selector(root, manifest, model, label, key):
    for entry in manifest["models"]:
        path = root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen source model record changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["model_id"] != model or record[key] != label:
            continue
        path = root / record["model_path"]
        if file_sha256(path) != record["model_sha256"]:
            raise ValueError("a frozen source model file changed")
        return joblib.load(path), record
    raise ValueError("required frozen source selector is missing")


def ranked_prediction(points, actions, ranking, decision_ids, *, joint):
    if joint:
        return np.median(
            points[[actions.index(name) for name in ranking[decision_ids[0]][:3]]], axis=0
        )
    return np.stack(
        [
            np.median(
                points[[actions.index(name) for name in ranking[identifier][:3]], :, slot], axis=0
            )
            for slot, identifier in enumerate(decision_ids)
        ],
        axis=1,
    )


def result_panels(frame):
    for panel, base in frame.groupby("panel", sort=True):
        yield panel + "_all", base
        if panel != "new_synthetic":
            for missing, name in ((True, "missing"), (False, "complete")):
                subset = base[base.native_missing_context == missing]
                if not subset.empty:
                    yield panel + "_" + name, subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "forecast-root",
        "source-bundle",
        "old-bundle",
        "runtime-replay-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed follow-up policy results")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    forecast_dir = args.forecast_root / args.model
    forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((args.old_bundle / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((args.runtime_replay_root / "manifest.json").read_text(encoding="utf-8"))
    runtime_path = ROOT / "src/tsfm_fais/routing/followup_portfolio.py"
    if any(row["status"] != "completed" for row in (prep, forecast, bundle, old, replay)):
        raise ValueError("finish source fitting, feature replay and candidate forecasting first")
    if replay["runtime_sha256"] != file_sha256(runtime_path) or replay[
        "source_bundle_sha256"
    ] != file_sha256(args.source_bundle / "manifest.json"):
        raise ValueError("the tested deployment representation or source selector changed")
    if forecast["identity"]["prepared_manifest_sha256"] != file_sha256(
        args.prepared_root / "manifest.json"
    ):
        raise ValueError("candidate predictions use different inputs")
    for name, sha in bundle["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError("the source method definition changed")
    control_path = args.old_bundle / old["controls_file"]
    if file_sha256(control_path) != old["controls_sha256"]:
        raise ValueError("old source-fixed controls changed")
    controls = json.loads(control_path.read_text(encoding="utf-8"))[args.model]
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("prefix standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(forecast_dir / "manifest.json"),
        "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
        "old_bundle_sha256": file_sha256(args.old_bundle / "manifest.json"),
        "runtime_replay_sha256": file_sha256(args.runtime_replay_root / "manifest.json"),
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/followup_portfolio.py",
                "src/tsfm_fais/routing/preforecast_replay.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/budgeted_portfolio.py",
                "src/tsfm_fais/forecasting/observed_accuracy.py",
                "scripts/replay_preforecast_student.py",
                "scripts/run_native_confirmation.py",
            )
        },
        "primary_method": "median_risk",
        "primary_comparator": "forecast_median_guarded",
        "fitting_performed": False,
        "metrics": "training-prefix-standardized MAE/MSE; raw errors auxiliary",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial follow-up policy identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    policy_path = output / "policy_predictions.npz"
    policy_marker = output / "policy_predictions.json"
    joint = args.model == "chronos2"
    if not policy_marker.exists():
        forecast_map = {row["episode_id"]: row for row in forecast["predictions"]}
        if len(forecast_map) != len(prep["episodes"]) or set(forecast_map) != {
            row["episode_id"] for row in prep["episodes"]
        }:
            raise ValueError("candidate prediction coverage changed")
        new_frames, old_frames, vector_banks, decisions, point_banks = [], [], [], [], []
        started = monotonic()
        for index, record in enumerate(prep["episodes"]):
            source = args.prepared_root / record["path"]
            prediction = forecast_map[record["episode_id"]]
            path = forecast_dir / prediction["path"]
            if file_sha256(source) != record["sha256"] or file_sha256(path) != prediction["sha256"]:
                raise ValueError("a prepared context or candidate prediction changed")
            with np.load(source, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            with np.load(path, allow_pickle=False) as saved:
                points, order = saved["point_z"], saved["candidate_ids"].tolist()
            if order != [*actions, "guarded_direct"]:
                raise ValueError("candidate forecast order disagrees with input order")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            frame = candidate_feature_frame(
                context,
                candidates,
                actions,
                coverage,
                mean,
                scale,
                [0, 1],
                joint=joint,
                period=record["period"],
                metadata={
                    **record,
                    "episode_index": index,
                    "model_id": args.model,
                    "split": "followup",
                },
            )
            last = (candidates[actions.index("locf"), -1, :2] - mean[:2]) / scale[:2]
            old_frames.append(
                forecast_response_inputs(extend_forecast_features(frame, points, actions, last))
            )
            generated, vectors, decision, option_names = portfolio_feature_frame(
                frame, points, order, last, joint=joint
            )
            new_frames.append(generated)
            vector_banks.append(vectors)
            decisions.append(decision)
            point_banks.append(points)
        features = pd.concat(new_frames, ignore_index=True)
        old_features = pd.concat(old_frames, ignore_index=True)
        features.to_parquet(output / "portfolio_features.parquet", index=False)
        old_features.to_parquet(output / "individual_features.parquet", index=False)
        feature_seconds = monotonic() - started
        choices, timings, fixed = {}, {}, None
        for kind in ("median_risk", "member_risk"):
            learner, _ = load_frozen_selector(
                args.source_bundle, bundle, args.model, kind, "label_kind"
            )
            started = monotonic()
            choices[kind] = learner.select(features)[
                ["episode_id", "candidate_id", "pairwise_wins"]
            ]
            timings[kind] = monotonic() - started
            choices[kind].to_parquet(output / f"{kind}_choices.parquet", index=False)
            if kind == "median_risk":
                fixed = learner.baseline_id
            del learner
        learner, _ = load_frozen_selector(
            args.old_bundle, old, args.model, "clean_forecast_mse", "objective"
        )
        started = monotonic()
        ids, preferences, pairs = learner.pair_scores(old_features)
        ranked = rank_pairwise_candidates(
            preferences, pairs, learner.candidate_ids, learner.baseline_id
        )
        rankings = {
            identifier: [learner.candidate_ids[position] for position in row]
            for identifier, row in zip(ids, ranked, strict=True)
        }
        timings["old_teacher_rank3"] = monotonic() - started
        _write_json(output / "old_teacher_rankings.json", rankings)
        del learner
        all_predictions = []
        for index in range(len(prep["episodes"])):
            points, vectors, decision = point_banks[index], vector_banks[index], decisions[index]
            methods = dict(zip(order, points, strict=True))
            for kind in ("median_risk", "member_risk"):
                local_choices = choices[kind][choices[kind].episode_id.isin(decision.episode_id)]
                methods[kind] = selected_portfolio_points(
                    local_choices, vectors, decision, option_names, joint=joint
                )
            fixed_choices = decision[["episode_id"]].assign(candidate_id=fixed)
            methods["source_fixed_median_risk"] = selected_portfolio_points(
                fixed_choices, vectors, decision, option_names, joint=joint
            )
            decision_ids = decision.episode_id.tolist()
            methods["old_teacher_rank3"] = ranked_prediction(
                points, order, rankings, decision_ids, joint=joint
            )
            fixed_by_target = controls["fixed3_clean_forecast_mse"]
            if joint and fixed_by_target[0] != fixed_by_target[1]:
                raise ValueError("the source-fixed Chronos control must choose jointly")
            local_ranking = {
                identifier: fixed_by_target[0 if joint else slot]
                for slot, identifier in enumerate(decision_ids)
            }
            methods["old_source_fixed3"] = ranked_prediction(
                points, order, local_ranking, decision_ids, joint=joint
            )
            methods["forecast_median_guarded"] = np.median(points, axis=0)
            methods["forecast_mean_guarded"] = np.mean(points, axis=0)
            methods["forecast_median_finite"] = np.median(points[:6], axis=0)
            methods["forecast_mean_finite"] = np.mean(points[:6], axis=0)
            all_predictions.append(np.stack(list(methods.values())))
        bank = np.stack(all_predictions)
        if bank.shape != (len(prep["episodes"]), 16, 96, 2) or not np.isfinite(bank).all():
            raise ValueError("the registered policies produced an invalid prediction bank")
        _save_npz(
            policy_path,
            point_z=bank,
            methods=np.asarray(list(methods)),
            episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
            identity_sha256=np.asarray(identity_sha),
        )
        files = [
            "portfolio_features.parquet",
            "individual_features.parquet",
            "median_risk_choices.parquet",
            "member_risk_choices.parquet",
            "old_teacher_rankings.json",
        ]
        _write_json(
            policy_marker,
            {
                "status": "predictions_frozen_before_scoring",
                "identity_sha256": identity_sha,
                "prediction_sha256": file_sha256(policy_path),
                "source_fixed_median_risk": fixed,
                "files_sha256": {name: file_sha256(output / name) for name in files},
                "feature_build_seconds": feature_seconds,
                "batch_selection_seconds": timings,
                "followup_future_arrays_read_at_freeze": False,
            },
        )
    policy = json.loads(policy_marker.read_text(encoding="utf-8"))
    if (
        policy["identity_sha256"] != identity_sha
        or file_sha256(policy_path) != policy["prediction_sha256"]
    ):
        raise ValueError("the frozen policy predictions changed")
    for name, sha in policy["files_sha256"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("a frozen policy decision artifact changed")
    with np.load(policy_path, allow_pickle=False) as saved:
        bank, methods = saved["point_z"], saved["methods"].tolist()
        if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
            raise ValueError("policy prediction episode order changed")
    # All method choices and predictions are now saved; score the original observed futures.
    rows = []
    for index, record in enumerate(prep["episodes"]):
        with np.load(args.prepared_root / record["path"], allow_pickle=False) as saved:
            future, observed = saved["future"], saved["future_observed"]
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
        errors, counts = observed_future_errors(bank[index] * scale + mean, future, observed, scale)
        for method_index, method in enumerate(methods):
            rows.append(
                {
                    **{
                        key: record[key]
                        for key in (
                            "episode_id",
                            "origin_id",
                            "family_id",
                            "dataset_id",
                            "item_id",
                            "panel",
                            "mechanism",
                            "missing_rate",
                            "mask_seed",
                        )
                    },
                    "model_id": args.model,
                    "method": method,
                    "native_missing_context": record["window"]["context_has_missing"],
                    "future_observed_target0": int(counts[0]),
                    "future_observed_target1": int(counts[1]),
                    **{name: float(value[method_index].mean()) for name, value in errors.items()},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summaries = []
    for name, panel in result_panels(frame):
        items, families, summary = hierarchical_metrics(panel)
        items.to_csv(output / f"{name}_item_metrics.csv", index=False)
        families.to_csv(output / f"{name}_family_metrics.csv", index=False)
        summaries.append(
            summary.assign(
                panel=name, families=panel.family_id.nunique(), origins=panel.origin_id.nunique()
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "policy_record_sha256": file_sha256(policy_marker),
            "scores_sha256": file_sha256(output / "episode_results.parquet"),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "scored_episodes": len(prep["episodes"]),
            "methods": methods,
            "new_forecaster_calls": 0,
            "limits": "follow-up outcomes are now used evaluation data; independent replay and readout are still required",
        },
    )


if __name__ == "__main__":
    main()
