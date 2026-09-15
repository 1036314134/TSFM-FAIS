"""Replay observable student decisions as actual inputs and compare input mixtures."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    RESPONSE_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.preforecast import STATIC_FEATURES  # noqa: E402
from tsfm_fais.routing.preforecast_replay import (  # noqa: E402
    assemble_selected_context,
    candidate_feature_frame,
    unique_forecaster_inputs,
)
from tsfm_fais.routing.structured_preforecast import (  # noqa: E402
    ALL_FEATURES,
    EXTRA_FEATURES,
    input_change_features,
    structured_inputs,
)
from tsfm_fais.routing.utility import response_features  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def extend_candidate_features(frame, context, candidates, actions, mean, scale, targets, *, joint):
    """Rebuild each candidate's structured features from the input it will serve."""
    context_z = (context - mean) / scale
    reference_z = (candidates[actions.index("locf")] - mean) / scale
    rows = []
    for row in frame.itertuples(index=False):
        selected_targets = targets if joint else [targets[row.target_slot]]
        effective = assemble_selected_context(
            context,
            candidates,
            actions,
            [row.candidate_id] if joint else [row.candidate_id] * len(selected_targets),
            selected_targets,
            joint=joint,
        )
        rows.append(
            input_change_features(
                context_z, (effective - mean) / scale, reference_z, selected_targets, joint=joint
            )
        )
    return pd.concat([frame, pd.DataFrame(rows, index=frame.index)], axis=1)


def rank_context_actions(learner, frame, budget):
    """Return one action list per context, with independent targets in slot order."""
    episodes, preferences, pairs = learner.pair_scores(frame)
    ranked = rank_pairwise_candidates(
        preferences, pairs, learner.candidate_ids, learner.baseline_id
    )
    slots = frame.drop_duplicates("episode_id").set_index("episode_id").loc[episodes].target_slot
    names = np.asarray(learner.candidate_ids)[ranked[np.argsort(slots.to_numpy()), :budget]]
    return names.T.tolist()


def extend_forecast_features(frame, point_z, actions, last_z):
    """Use queried forecasts; match the export's finite-candidate median reference."""
    all_actions = [*actions, "guarded_direct"]
    reference = point_z[actions.index("locf")]
    pool = np.median(point_z[: len(actions)], axis=0)
    rows = []
    for row in frame.itertuples(index=False):
        slots = list(range(point_z.shape[2])) if row.target_slot == -1 else [row.target_slot]
        rows.append(
            response_features(
                point_z[all_actions.index(row.candidate_id)][:, slots],
                reference[:, slots],
                pool[:, slots],
                last_z[slots],
                np.ones(len(slots)),
                None,
            )
        )
    return pd.concat([frame, pd.DataFrame(rows, index=frame.index)], axis=1)


def query_candidate_points(
    runner, spec, context, candidates, actions, targets, mean, scale, *, joint
):
    inputs = [
        assemble_selected_context(
            context,
            candidates,
            actions,
            [action] if joint else [action] * len(targets),
            targets,
            joint=joint,
        )
        for action in [*actions, "guarded_direct"]
    ]
    unique, reverse = unique_forecaster_inputs(inputs, targets, joint=joint)
    point = runner.predict_missing(unique, spec).point[reverse]
    return (point - mean[targets]) / scale[targets], len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "student-root", "plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    parser.add_argument("--structured-root", type=Path)
    parser.add_argument(
        "--query-budget",
        "--aggregation-size",
        dest="query_budget",
        type=int,
        choices=(1, 3),
        default=1,
        help="Selected forecasts to combine; forecast-response selection queries all seven candidates first",
    )
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed input replays")
    output.mkdir(parents=True, exist_ok=True)
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    students = json.loads((args.student_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    feature_view = students["identity"].get("feature_view", "base_static")
    if feature_view not in {
        "base_static",
        "target_temporal",
        "full_dependency",
        "forecast_response",
    }:
        raise ValueError("unknown replay feature view")
    input_scope = students["identity"].get("input_scope", "preforecast")
    uses_forecasts = input_scope == "candidate_forecasts"
    if input_scope not in {"preforecast", "candidate_forecasts"} or uses_forecasts != (
        feature_view == "forecast_response"
    ):
        raise ValueError("the selector's input scope and feature view do not agree")
    if (feature_view in {"target_temporal", "full_dependency"}) != (
        args.structured_root is not None
    ):
        raise ValueError("supply structured features only for a matching structured student")
    feature_manifest = None
    if args.structured_root is not None:
        feature_manifest = json.loads(
            (args.structured_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            feature_manifest["status"] != "completed"
            or file_sha256(args.structured_root / "manifest.json")
            != students["identity"]["structured_manifest_sha256"]
            or file_sha256(args.structured_root / feature_manifest["feature_file"])
            != feature_manifest["feature_sha256"]
        ):
            raise ValueError("the structured feature source changed")
    expected_features = (
        FORECAST_FEATURES
        if uses_forecasts
        else ALL_FEATURES
        if feature_manifest is not None
        else STATIC_FEATURES
    )
    if set(students["identity"]["static_features"]) != set(expected_features):
        raise ValueError("the student feature inventory differs from this replay")
    if (
        students["status"] != "completed"
        or accuracy["source_episode_manifest_sha256"] != file_sha256(source_path)
        or students["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or students["identity"]["candidate_table_sha256"]
        != file_sha256(args.accuracy_root / "candidate_accuracy.parquet")
        or plan["source_manifest_sha256"] != file_sha256(source_path)
    ):
        raise ValueError("student, source and screening plan have different provenance")
    config = source["identity"]["config"]
    records = {row["episode_id"]: (index, row) for index, row in enumerate(source["episodes"])}
    selected = [records[episode] for episode in plan["decision_episode_ids"]]
    if len(selected) != 90 or any(row["split"] != "validation" for _, row in selected):
        raise ValueError("use the fixed 90-task screening panel")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    fold_index = {}
    for record in students["folds"]:
        path = args.student_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("student fold metadata changed")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if fold["model_id"] == args.model:
            fold_index[(fold["held_family"], fold["objective"])] = fold
    objectives = students["identity"]["objectives"]
    if len(fold_index) != 15 * len(objectives):
        raise ValueError("the selected model must have all 15 family folds per objective")
    identity = {
        "source_manifest_sha256": file_sha256(source_path),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "student_manifest_sha256": file_sha256(args.student_root / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "query_budget": 7 if uses_forecasts else args.query_budget,
        "aggregation_size": args.query_budget,
        "aggregation_sizes_reported": [1, 3]
        if uses_forecasts and args.query_budget == 3
        else [args.query_budget],
        "input_scope": input_scope,
        "selected_context_calls_are_additional_validation": uses_forecasts,
        "objectives": objectives,
        "feature_view": feature_view,
        "teacher_kind": students["identity"].get("teacher_kind", "candidate_forecast_median"),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/preforecast_replay.py",
                "src/tsfm_fais/routing/budgeted_portfolio.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/utility.py",
                "src/tsfm_fais/forecasting/runner.py",
                "src/tsfm_fais/forecasting/adapters/chronos.py",
                "src/tsfm_fais/forecasting/adapters/timesfm.py",
                "scripts/evaluate_timesfm_vendor_missing.py",
            )
        },
        "prediction_tolerance": {"rtol": 2e-4, "atol": 2e-4},
        "input_recipe": "original R4 raw inputs; shared prefix-standardized scores",
        "timing_scope": "diagnostic post-imputation calls, fixed execution order; excludes candidate generation and model loading; no speedup claim",
    }
    if feature_manifest is not None:
        identity["structured_manifest_sha256"] = file_sha256(args.structured_root / "manifest.json")
        name = "src/tsfm_fais/routing/structured_preforecast.py"
        identity["source_sha256"][name] = file_sha256(ROOT / name)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("input replay identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "assembly_snapshot.py").write_bytes(
        (ROOT / "src/tsfm_fais/routing/preforecast_replay.py").read_bytes()
    )
    torch.set_num_threads(1)
    registry = default_forecast_registry()
    joint = registry.get(args.model).mode == "joint_multivariate"
    targets = config["target_indices"]
    adapter = (
        TimesFMVendorMissingAdapter(
            model_name=config["forecaster_artifacts"][args.model], device="cuda", batch_size=8
        )
        if not joint
        else registry.build(
            args.model,
            model_name=config["forecaster_artifacts"][args.model],
            device="cuda",
            batch_size=8,
        )
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    before = parameter_digest(backbone)
    spec = ForecastSpec(
        args.model,
        registry.get(args.model).mode,
        config["horizon"],
        context_length=config["context_length"],
        target_indices=targets,
    )
    learner_cache, rows, files = {}, [], []
    point_path = args.accuracy_root / f"{args.model}_point_z.npy"
    if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
        raise ValueError("cached reference forecasts changed")
    old_points = np.load(point_path, mmap_mode="r")
    static_reference = pd.read_parquet(
        args.accuracy_root / "candidate_accuracy.parquet",
        columns=[
            "episode_id",
            "candidate_id",
            "target_slot",
            "model_id",
            *STATIC_FEATURES,
            *(RESPONSE_FEATURES if uses_forecasts else ()),
        ],
    )
    static_reference = static_reference[static_reference.model_id == args.model]
    structured_reference = None
    if feature_manifest is not None:
        structured_reference = pd.read_parquet(
            args.structured_root / feature_manifest["feature_file"],
            columns=["episode_id", "candidate_id", "target_slot", "model_id", *EXTRA_FEATURES],
            filters=[("model_id", "==", args.model)],
        )
    for number, (source_index, record) in enumerate(selected):
        source_file = args.source_root / record["path"]
        if file_sha256(source_file) != record["sha256"]:
            raise ValueError("cached imputation inputs changed")
        cache = (
            output
            / "predictions"
            / (hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24] + ".npz")
        )
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        if not cache.exists():
            with np.load(source_file, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            calls_before = runner.call_count
            fresh, unique_count, response_difference = None, None, 0.0
            generated = candidate_feature_frame(
                context,
                candidates,
                actions,
                coverage,
                mean,
                scale,
                targets,
                joint=joint,
                period=record["period"],
                metadata={**record, "model_id": args.model, "episode_index": source_index},
            )
            original = static_reference[
                (static_reference.episode_id == record["episode_id"])
                & static_reference.target_slot.isin([-1] if joint else [0, 1])
                & static_reference.candidate_id.isin([*actions, "guarded_direct"])
            ]
            np.testing.assert_allclose(
                generated.set_index(["candidate_id", "target_slot"]).sort_index()[
                    list(STATIC_FEATURES)
                ],
                original.set_index(["candidate_id", "target_slot"]).sort_index()[
                    list(STATIC_FEATURES)
                ],
                rtol=0,
                atol=1e-12,
            )
            if structured_reference is not None:
                generated = extend_candidate_features(
                    generated, context, candidates, actions, mean, scale, targets, joint=joint
                )
                reference = structured_reference[
                    structured_reference.episode_id == record["episode_id"]
                ]
                np.testing.assert_allclose(
                    generated.set_index(["candidate_id", "target_slot"]).sort_index()[
                        list(EXTRA_FEATURES)
                    ],
                    reference.set_index(["candidate_id", "target_slot"]).sort_index()[
                        list(EXTRA_FEATURES)
                    ],
                    rtol=0,
                    atol=1e-12,
                )
                generated = structured_inputs(generated, view=feature_view)
            if uses_forecasts:
                fresh, unique_count = query_candidate_points(
                    runner, spec, context, candidates, actions, targets, mean, scale, joint=joint
                )
                last_z = (candidates[actions.index("locf"), -1, targets] - mean[targets]) / scale[
                    targets
                ]
                generated = extend_forecast_features(generated, fresh, actions, last_z)
                actual = generated.set_index(["candidate_id", "target_slot"]).sort_index()[
                    list(RESPONSE_FEATURES)
                ]
                cached = original.set_index(["candidate_id", "target_slot"]).sort_index()[
                    list(RESPONSE_FEATURES)
                ]
                np.testing.assert_allclose(actual, cached, **identity["prediction_tolerance"])
                response_difference = float(np.max(np.abs(actual.to_numpy() - cached.to_numpy())))
                generated = forecast_response_inputs(generated)
            inputs, choice_records, selected_seconds = {}, {}, {}
            for objective in objectives:
                key = (record["family_id"], objective)
                if key not in learner_cache:
                    fold = fold_index[key]
                    for kind in ("model", "choices"):
                        if (
                            file_sha256(args.student_root / fold[kind + "_path"])
                            != fold[kind + "_sha256"]
                        ):
                            raise ValueError("a student model or recorded choice changed")
                    learner_cache[key] = (
                        joblib.load(args.student_root / fold["model_path"]),
                        pd.read_parquet(args.student_root / fold["choices_path"]),
                    )
                learner, saved_choices = learner_cache[key]
                if set(learner.feature_names) != set(expected_features):
                    raise ValueError("a fitted model contains features outside its declared view")
                started = monotonic()
                choice_records[objective] = rank_context_actions(
                    learner, generated, args.query_budget
                )
                expected = saved_choices[
                    saved_choices.source_episode_id == record["episode_id"]
                ].sort_values("target_slot")
                if choice_records[objective][0] != expected.candidate_id.tolist():
                    raise ValueError(
                        "regenerated pre-forecast choices differ from the cached student"
                    )
                inputs[objective] = np.stack(
                    [
                        assemble_selected_context(
                            context, candidates, actions, selected, targets, joint=joint
                        )
                        for selected in choice_records[objective]
                    ]
                )
                selected_seconds[objective] = monotonic() - started
            calls_before_choices = runner.call_count - calls_before
            if calls_before_choices != int(uses_forecasts):
                raise ValueError("the actual queries differ from the declared decision input scope")
            predictions, call_seconds = {}, {}
            for objective, values in inputs.items():
                started = monotonic()
                raw = runner.predict_missing(values, spec).point
                point = (raw - mean[targets]) / scale[targets]
                expected = []
                reference_points = fresh if uses_forecasts else old_points[source_index]
                reference_actions = (
                    [*actions, "guarded_direct"]
                    if uses_forecasts
                    else accuracy["action_orders"][args.model]
                )
                for selected in choice_records[objective]:
                    expected.append(
                        reference_points[reference_actions.index(selected[0])]
                        if joint
                        else np.stack(
                            [
                                reference_points[reference_actions.index(action), :, slot]
                                for slot, action in enumerate(selected)
                            ],
                            axis=1,
                        )
                    )
                expected = np.stack(expected)
                np.testing.assert_allclose(point, expected, **identity["prediction_tolerance"])
                method = "student_" + objective + ("_rank3" if args.query_budget == 3 else "")
                predictions[method] = np.median(expected if uses_forecasts else point, axis=0)
                if uses_forecasts and args.query_budget == 3:
                    predictions["student_" + objective] = expected[0]
                call_seconds[objective] = monotonic() - started
            if fresh is None:
                fresh, unique_count = query_candidate_points(
                    runner, spec, context, candidates, actions, targets, mean, scale, joint=joint
                )
            expected = old_points[
                source_index,
                [
                    accuracy["action_orders"][args.model].index(action)
                    for action in [*actions, "guarded_direct"]
                ],
            ]
            np.testing.assert_allclose(fresh, expected, **identity["prediction_tolerance"])
            predictions.update(dict(zip([*actions, "guarded_direct"], fresh, strict=True)))
            predictions["forecast_median_guarded"] = np.median(fresh, axis=0)
            observed = np.isfinite(context)
            for name, mixed in (
                ("input_mean_finite", candidates.mean(axis=0)),
                ("input_median_finite", np.median(candidates, axis=0)),
            ):
                mixed[observed] = context[observed]
                raw = runner.predict(mixed[None], spec).point[0]
                predictions[name] = (raw - mean[targets]) / scale[targets]
            _save_npz(
                cache,
                methods=np.asarray(list(predictions)),
                point_z=np.stack(list(predictions.values())),
                metadata=np.asarray(
                    json.dumps(
                        {
                            "student_choices": choice_records,
                            "selection_seconds": selected_seconds,
                            "student_forecast_seconds": call_seconds,
                            "student_current_forecast_calls_before_choices": calls_before_choices,
                            "distinct_contexts_before_choices": unique_count
                            if uses_forecasts
                            else 0,
                            "response_feature_maximum_difference": response_difference,
                            "student_input_contexts_each": args.query_budget,
                            "student_input_calls_are_additional_validation": uses_forecasts,
                            "unique_baseline_inputs": unique_count,
                        }
                    )
                ),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(before),
            )
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != before
            ):
                raise ValueError("cached replay belongs to another run")
            methods, point = saved["methods"].tolist(), saved["point_z"]
        with np.load(source_file, allow_pickle=False) as saved:
            truth = (saved["future"][:, targets] - mean[targets]) / scale[targets]
        for method, forecast in zip(methods, point, strict=True):
            error = forecast - truth
            rows.append(
                {
                    "model_id": args.model,
                    "method": method,
                    "family_id": record["family_id"],
                    "episode_id": record["episode_id"],
                    "mae": float(np.abs(error).mean()),
                    "mse": float((error**2).mean()),
                }
            )
        files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        print(
            json.dumps({"model": args.model, "completed_replays": number + 1, "total": 90}),
            flush=True,
        )
    if before != parameter_digest(backbone):
        raise ValueError("forecaster parameters changed")
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family = frame.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "parameters_unchanged": True,
            "parameter_sha256": before,
            "predictions": files,
            "summary": summary.to_dict("records"),
            "raw_input_feature_and_choice_parity": True,
            "selected_input_prediction_parity": True,
            "single_input_prediction_parity": None if uses_forecasts else args.query_budget == 1,
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": "candidate imputations were cached; call timings have fixed order and are not end-to-end performance comparisons",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
