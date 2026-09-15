"""Apply every frozen R6 policy before scoring either new confirmation horizon."""

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
from audit_shared_forecast_gate import replay_network  # noqa: E402
from r6_policy_inputs import decision_inputs, pack_gate_features  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def restore(values, decisions, count, horizon, joint):
    values = np.asarray(values, float)
    result = np.full((count, horizon, 2), np.nan)
    indices = decisions.episode_index.to_numpy(int)
    if joint:
        if len(decisions) != count or set(decisions.target_slot) != {-1}:
            raise ValueError("joint target coverage changed")
        result[indices] = values.reshape(count, horizon, 2)
    else:
        if len(decisions) != 2 * count or set(decisions.target_slot) != {0, 1}:
            raise ValueError("independent target coverage changed")
        result[indices, :, decisions.target_slot.to_numpy(int)] = values
    if not np.isfinite(result).all():
        raise ValueError("a decision output lost a forecast target")
    return result


def triple_vectors(vectors, decisions, choices, actions, *, named_portfolio):
    values = []
    for index, row in enumerate(decisions.itertuples(index=False)):
        selected = choices[row.episode_id]
        if named_portfolio:
            if not selected.startswith("median:"):
                raise ValueError("unknown triple identity")
            selected = selected[len("median:") :].split("+")
        if len(selected) != 3 or len(set(selected)) != 3 or not set(selected) <= set(actions):
            raise ValueError("a frozen triple contains an invalid candidate")
        values.append(np.median(vectors[index, [actions.index(name) for name in selected]], axis=0))
    return np.stack(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "forecast-root",
        "method-freeze",
        "source-future-control",
        "pairwise-bundle",
        "legacy-bundle",
        "runtime-check",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed independent confirmation scores")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    binding = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    forecast_dir = args.forecast_root / args.model
    forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
    pairwise = json.loads((args.pairwise_bundle / "manifest.json").read_text(encoding="utf-8"))
    legacy = json.loads((args.legacy_bundle / "manifest.json").read_text(encoding="utf-8"))
    future_control = json.loads(
        (args.source_future_control / "manifest.json").read_text(encoding="utf-8")
    )
    runtime_check = json.loads((args.runtime_check / "manifest.json").read_text(encoding="utf-8"))
    if runtime_check["status"] != "completed" or runtime_check["tests_passed"] != 4:
        raise ValueError("complete the new policy assembly checks first")
    for name, digest in runtime_check["runtime_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the checked policy runtime changed")
    if any(
        row["status"] != "completed"
        for row in (prep, binding, forecast, pairwise, legacy, future_control)
    ):
        raise ValueError("finish all registered input, source-fit and forecasting stages first")
    if forecast["identity"]["method_freeze_sha256"] != file_sha256(args.method_freeze) or forecast[
        "identity"
    ]["prepared_sha256"] != file_sha256(args.prepared_root / "manifest.json"):
        raise ValueError("the confirmed method or cohort inputs changed")
    gate_manifest_path = Path(binding["source_gate_manifest_path"])
    if file_sha256(gate_manifest_path) != binding["source_gate_manifest_sha256"]:
        raise ValueError("the final source-gate bundle changed")
    gate_bundle = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if (
        file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py")
        != gate_bundle["identity"]["module_sha256"]
    ):
        raise ValueError("the frozen gate implementation changed")
    if (
        future_control["identity"]["method_freeze_sha256"] != file_sha256(args.method_freeze)
        or not future_control["normalization_and_training_population_match"]
    ):
        raise ValueError("the future-supervised comparator does not match this method freeze")
    for name, digest in pairwise["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the frozen pairwise reference implementation changed")
    gate_control_path = Path(binding["controls_path"])
    if file_sha256(gate_control_path) != binding["controls_sha256"]:
        raise ValueError("source-fixed gate controls changed")
    gate_controls = json.loads(gate_control_path.read_text(encoding="utf-8"))[args.model]
    legacy_control_path = args.legacy_bundle / legacy["controls_file"]
    if file_sha256(legacy_control_path) != legacy["controls_sha256"]:
        raise ValueError("original source-fixed controls changed")
    legacy_controls = json.loads(legacy_control_path.read_text(encoding="utf-8"))[args.model]
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("target-prefix scoring statistics changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(forecast_dir / "manifest.json"),
        "method_freeze_sha256": file_sha256(args.method_freeze),
        "source_future_control_sha256": file_sha256(args.source_future_control / "manifest.json"),
        "pairwise_bundle_sha256": file_sha256(args.pairwise_bundle / "manifest.json"),
        "legacy_bundle_sha256": file_sha256(args.legacy_bundle / "manifest.json"),
        "representation_sha256": file_sha256(ROOT / "scripts/r6_policy_inputs.py"),
        "runtime_check_sha256": file_sha256(args.runtime_check / "manifest.json"),
        "primary_method": "ensemble_gate",
        "primary_horizon": 96,
        "secondary_horizon": 192,
        "source_models_refitted": False,
        "current_outcomes_used_for_decisions": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial confirmation policy identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    horizon_records = []
    for horizon in (96, 192):
        directory = output / f"h{horizon}"
        directory.mkdir(exist_ok=True)
        marker = directory / "predictions_frozen.json"
        points_path = directory / "policy_predictions.npz"
        forecast_entry = next(row for row in forecast["horizons"] if row["horizon"] == horizon)
        manifest_path = forecast_dir / forecast_entry["path"]
        if file_sha256(manifest_path) != forecast_entry["sha256"]:
            raise ValueError("a candidate horizon manifest changed")
        horizon_forecast = json.loads(manifest_path.read_text(encoding="utf-8"))
        forecast_map = {row["episode_id"]: row for row in horizon_forecast["predictions"]}
        if len(forecast_map) != len(prep["episodes"]) or set(forecast_map) != {
            row["episode_id"] for row in prep["episodes"]
        }:
            raise ValueError("candidate horizon coverage changed")
        if not marker.exists():
            (
                individual_frames,
                portfolio_frames,
                decision_frames,
                feature_arrays,
                vector_arrays,
                point_arrays,
            ) = [], [], [], [], [], []
            for index, record in enumerate(prep["episodes"]):
                source_path = args.prepared_root / record["path"]
                entry = forecast_map[record["episode_id"]]
                path = manifest_path.parent / entry["path"]
                if (
                    file_sha256(source_path) != record["sha256"]
                    or file_sha256(path) != entry["sha256"]
                ):
                    raise ValueError("a registered context or candidate forecast changed")
                with np.load(source_path, allow_pickle=False) as saved:
                    context, candidates, coverage = (
                        saved["context"],
                        saved["candidate_values"],
                        saved["native_coverage"],
                    )
                    candidate_ids = saved["candidate_ids"].tolist()
                with np.load(path, allow_pickle=False) as saved:
                    points, order = saved["point_z"], saved["candidate_ids"].tolist()
                if order != [
                    *candidate_ids,
                    "guarded_direct",
                    "motm_reference",
                ] or points.shape != (8, horizon, 2):
                    raise ValueError("candidate forecast axes or identities changed")
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                inputs = decision_inputs(
                    context,
                    candidates,
                    candidate_ids,
                    coverage,
                    points[:7],
                    np.asarray(scaler["mean"]),
                    np.asarray(scaler["scale"]),
                    joint=args.model == "chronos2",
                    period=record["period"],
                    metadata={
                        **record,
                        "model_id": args.model,
                        "episode_index": index,
                        "split": "r6_confirmation",
                    },
                )
                if list(inputs["actions"]) != gate_controls["actions"]:
                    raise ValueError("the source and deployment candidate order differs")
                individual_frames.append(inputs["individual"])
                portfolio_frames.append(inputs["portfolios"])
                decision_frames.append(inputs["decisions"])
                feature_arrays.append(inputs["gate_features"])
                vector_arrays.append(inputs["vectors"])
                point_arrays.append(points)
            individual, portfolios = (
                pd.concat(individual_frames, ignore_index=True),
                pd.concat(portfolio_frames, ignore_index=True),
            )
            decisions = pd.concat(decision_frames, ignore_index=True)
            features, vectors, points = (
                pack_gate_features(np.concatenate(feature_arrays)),
                np.concatenate(vector_arrays),
                np.stack(point_arrays),
            )
            actions, joint = gate_controls["actions"], args.model == "chronos2"
            individual.to_parquet(directory / "individual_features.parquet", index=False)
            portfolios.to_parquet(directory / "portfolio_features.parquet", index=False)
            decisions.to_parquet(directory / "decisions.parquet", index=False)
            methods = {name: points[:, index] for index, name in enumerate(order)}
            methods.update(
                forecast_median_guarded=np.median(points[:, :7], axis=1),
                forecast_mean_guarded=points[:, :7].mean(1),
                forecast_median_finite=np.median(points[:, :6], axis=1),
                forecast_mean_finite=points[:, :6].mean(1),
                forecast_median_with_motm=np.median(points, axis=1),
            )
            weight_files, choice_records = [], {}
            for objective in ("ensemble", "member", "future"):
                entries = [
                    row
                    for row in (
                        future_control["models"]
                        if objective == "future"
                        else binding["source_models"]
                    )
                    if row["model_id"] == args.model
                    and (objective == "future" or row["objective"] == objective)
                ]
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("the three frozen seeds must all be retained")
                weights = []
                for entry in entries:
                    path = (
                        args.source_future_control / entry["path"]
                        if objective == "future"
                        else Path(entry["path"])
                    )
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a frozen source gate changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    expected_identity = (
                        future_control["identity_sha256"]
                        if objective == "future"
                        else gate_bundle["identity_sha256"]
                    )
                    if saved["identity_sha256"] != expected_identity:
                        raise ValueError("a gate checkpoint has another fitting identity")
                    model = SharedForecastGate()
                    model.load_state_dict(saved["state_dict"])
                    model.eval()
                    predicted_weights = predict_weights(model, features)
                    np.testing.assert_array_equal(
                        predicted_weights, replay_network(saved["state_dict"], features)
                    )
                    weights.append(predicted_weights)
                weights = np.stack(weights)
                mean_weights = weights.mean(0)
                name = "source_future_gate" if objective == "future" else objective + "_gate"
                methods[name] = restore(
                    compose_forecasts(vectors, mean_weights),
                    decisions,
                    len(prep["episodes"]),
                    horizon,
                    joint,
                )
                path = directory / f"{objective}_weights.npz"
                _save_npz(path, seed_weights=weights, mean_weights=mean_weights)
                weight_files.append(path.name)
            fixed = np.repeat(
                np.asarray(gate_controls["convex_weights"])[None], len(decisions), axis=0
            )
            methods["gate_source_fixed_convex"] = restore(
                compose_forecasts(vectors, fixed), decisions, len(prep["episodes"]), horizon, joint
            )
            methods["gate_source_fixed_single"] = restore(
                vectors[:, gate_controls["single_index"]],
                decisions,
                len(prep["episodes"]),
                horizon,
                joint,
            )
            for objective in ("median_risk", "member_risk"):
                learner, _ = load_frozen_selector(
                    args.pairwise_bundle, pairwise, args.model, objective, "label_kind"
                )
                choices = learner.select(portfolios)[
                    ["episode_id", "candidate_id", "pairwise_wins"]
                ]
                choice_map = choices.set_index("episode_id").candidate_id.to_dict()
                methods[objective] = restore(
                    triple_vectors(vectors, decisions, choice_map, actions, named_portfolio=True),
                    decisions,
                    len(prep["episodes"]),
                    horizon,
                    joint,
                )
                choices.to_parquet(directory / f"{objective}_choices.parquet", index=False)
                if objective == "median_risk":
                    fixed_choices = {
                        identifier: learner.baseline_id for identifier in decisions.episode_id
                    }
                    methods["source_fixed_median_risk"] = restore(
                        triple_vectors(
                            vectors, decisions, fixed_choices, actions, named_portfolio=True
                        ),
                        decisions,
                        len(prep["episodes"]),
                        horizon,
                        joint,
                    )
                    choice_records["source_fixed_median_risk"] = learner.baseline_id
                del learner
            learner, _ = load_frozen_selector(
                args.legacy_bundle, legacy, args.model, "clean_forecast_mse", "objective"
            )
            ids, margins, pairs = learner.pair_scores(individual)
            rankings = rank_pairwise_candidates(
                margins, pairs, learner.candidate_ids, learner.baseline_id
            )
            ranked = {
                identifier: [learner.candidate_ids[i] for i in row[:3]]
                for identifier, row in zip(ids, rankings, strict=True)
            }
            methods["old_teacher_rank3"] = restore(
                triple_vectors(vectors, decisions, ranked, actions, named_portfolio=False),
                decisions,
                len(prep["episodes"]),
                horizon,
                joint,
            )
            _write_json(directory / "old_teacher_choices.json", ranked)
            fixed = legacy_controls["fixed3_clean_forecast_mse"]
            if joint and fixed[0] != fixed[1]:
                raise ValueError("Chronos fixed control must use one shared triple")
            fixed_choices = {
                row.episode_id: fixed[0 if joint else row.target_slot]
                for row in decisions.itertuples(index=False)
            }
            methods["old_source_fixed3"] = restore(
                triple_vectors(vectors, decisions, fixed_choices, actions, named_portfolio=False),
                decisions,
                len(prep["episodes"]),
                horizon,
                joint,
            )
            bank = np.stack(list(methods.values()), axis=1)
            if (
                len(methods) != 23
                or bank.shape != (len(prep["episodes"]), 23, horizon, 2)
                or not np.isfinite(bank).all()
            ):
                raise ValueError("the registered 23-method output bank changed")
            _save_npz(
                points_path,
                point_z=bank,
                methods=np.asarray(list(methods)),
                episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                identity_sha256=np.asarray(identity_sha),
            )
            names = [
                "individual_features.parquet",
                "portfolio_features.parquet",
                "decisions.parquet",
                "median_risk_choices.parquet",
                "member_risk_choices.parquet",
                "old_teacher_choices.json",
                *weight_files,
            ]
            _write_json(
                marker,
                {
                    "status": "predictions_frozen_before_scoring",
                    "identity_sha256": identity_sha,
                    "prediction_sha256": file_sha256(points_path),
                    "files_sha256": {name: file_sha256(directory / name) for name in names},
                    "source_fixed_choices": choice_records,
                    "future_arrays_read": False,
                },
            )
        frozen = json.loads(marker.read_text(encoding="utf-8"))
        if (
            frozen["identity_sha256"] != identity_sha
            or file_sha256(points_path) != frozen["prediction_sha256"]
        ):
            raise ValueError("frozen confirmation policy predictions changed")
        for name, digest in frozen["files_sha256"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("a saved confirmation decision artifact changed")
        horizon_records.append(
            {"horizon": horizon, "directory": directory, "marker_sha256": file_sha256(marker)}
        )
    # Both horizons' complete policy banks are fixed before any new future arrays are scored.
    summaries = []
    for item in horizon_records:
        horizon, directory = item["horizon"], item["directory"]
        with np.load(directory / "policy_predictions.npz", allow_pickle=False) as saved:
            bank, names = saved["point_z"], saved["methods"].tolist()
            if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                raise ValueError("policy prediction order changed")
        rows = []
        for index, record in enumerate(prep["episodes"]):
            with np.load(args.prepared_root / record["path"], allow_pickle=False) as saved:
                future, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            errors, counts = observed_future_errors(
                bank[index] * scale + mean, future, observed, scale, minimum_observed=horizon // 2
            )
            for position, method in enumerate(names):
                rows.append(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "episode_id",
                                "origin_id",
                                "dataset_id",
                                "family_id",
                                "item_id",
                                "panel",
                                "mechanism",
                                "missing_rate",
                                "mask_seed",
                            )
                        },
                        "model_id": args.model,
                        "horizon": horizon,
                        "method": method,
                        "native_missing_context": record["window"]["context_has_missing"],
                        "future_observed_target0": int(counts[0]),
                        "future_observed_target1": int(counts[1]),
                        **{name: float(value[position].mean()) for name, value in errors.items()},
                    }
                )
        scores = pd.DataFrame(rows)
        scores.to_parquet(directory / "episode_results.parquet", index=False)
        for name, panel in result_panels(scores):
            items, families, summary = hierarchical_metrics(panel)
            items.to_csv(directory / f"{name}_item_metrics.csv", index=False)
            families.to_csv(directory / f"{name}_family_metrics.csv", index=False)
            summaries.append(
                summary.assign(
                    panel=name,
                    horizon=horizon,
                    families=panel.family_id.nunique(),
                    origins=panel.origin_id.nunique(),
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
            "horizons": [
                {
                    "horizon": row["horizon"],
                    "directory": str(row["directory"].relative_to(output)),
                    "marker_sha256": row["marker_sha256"],
                    "scores_sha256": file_sha256(row["directory"] / "episode_results.parquet"),
                }
                for row in horizon_records
            ],
            "summary_sha256": file_sha256(output / "summary.csv"),
            "input_tasks": len(prep["episodes"]),
            "methods": 23,
            "limits": "new outcomes have now been scored; independent replay and paired source-level readout are required before interpreting gains",
        },
    )


if __name__ == "__main__":
    main()
