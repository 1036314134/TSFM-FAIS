"""Evaluate the frozen source policy on the registered natural-missing cohort."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from replay_preforecast_student import (  # noqa: E402
    extend_forecast_features,
    query_candidate_points,
    rank_context_actions,
)

from tsfm_fais.contracts import ForecastSpec, SeriesBatch  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.routing.forecast_response import forecast_response_inputs  # noqa: E402
from tsfm_fais.routing.preforecast_replay import (  # noqa: E402
    assemble_selected_context,
    candidate_feature_frame,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def chosen_prediction(points, actions, choices, *, joint):
    if joint:
        if len(choices) != 1:
            raise ValueError("a joint context must use one candidate")
        return points[actions.index(choices[0])]
    if len(choices) != points.shape[2]:
        raise ValueError("each independent target needs one candidate")
    return np.stack(
        [points[actions.index(name), :, slot] for slot, name in enumerate(choices)], axis=1
    )


def hierarchical_metrics(rows):
    keys = ["model_id", "method", "family_id", "dataset_id", "item_id"]
    metrics = ["mae", "mse", "raw_mae", "raw_mse"]
    # A failed method is not silently averaged over its successful windows.
    items = (
        rows.groupby(keys)[metrics]
        .agg(lambda values: values.mean() if values.notna().all() else np.nan)
        .reset_index()
    )
    datasets = (
        items.groupby(keys[:-1])[metrics]
        .agg(lambda values: values.mean() if values.notna().all() else np.nan)
        .reset_index()
    )
    families = (
        datasets.groupby(keys[:-2])[metrics]
        .agg(lambda values: values.mean() if values.notna().all() else np.nan)
        .reset_index()
    )
    summary = (
        families.groupby(keys[:2])[metrics]
        .agg(lambda values: values.mean() if values.notna().all() else np.nan)
        .reset_index()
    )
    return items, families, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "motm-root", "source-bundle", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed independent confirmation")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    motm = json.loads((args.motm_root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    if (
        any(item["status"] != "completed" for item in (prepared, motm, bundle))
        or len(prepared["episodes"]) != 373
    ):
        raise ValueError("finish source fitting and registered input preparation first")
    if prepared["identity"]["cohort_sha256"] != bundle["identity"]["cohort_sha256"] or motm[
        "identity"
    ]["prepared_manifest_sha256"] != file_sha256(args.prepared_root / "manifest.json"):
        raise ValueError("confirmation artifacts use different cohorts")
    if Path(bundle["identity"]["confirmation_output_root"]).resolve() != args.output_root.resolve():
        raise ValueError("use the output location named in the source freeze")
    for name, expected in bundle["identity"]["runtime_code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("the frozen forecast runtime changed before confirmation")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prepared["standardizers_sha256"]:
        raise ValueError("target-prefix standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    motm_cases = {record["episode_id"]: record for record in motm["episodes"]}
    control_path = args.source_bundle / bundle["controls_file"]
    if file_sha256(control_path) != bundle["controls_sha256"]:
        raise ValueError("source-only fixed controls changed")
    controls = json.loads(control_path.read_text(encoding="utf-8"))[args.model]
    learners = {}
    for item in bundle["models"]:
        path = args.source_bundle / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a frozen source-model record changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["model_id"] == args.model:
            if file_sha256(args.source_bundle / record["model_path"]) != record["model_sha256"]:
                raise ValueError("a frozen source selector changed")
            learners[record["objective"]] = joblib.load(args.source_bundle / record["model_path"])
    if set(learners) != {"clean_forecast_mse", "future_mse"}:
        raise ValueError("both frozen source-supervision comparisons are required")
    identity = {
        "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "motm_manifest_sha256": file_sha256(args.motm_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/replay_preforecast_student.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/preforecast_replay.py",
                "src/tsfm_fais/forecasting/observed_accuracy.py",
                "src/tsfm_fais/imputers/classical.py",
            )
        },
        "primary_method": "teacher_rank3",
        "primary_input_length": 96,
        "horizon": 96,
        "primary_input_scale": "raw",
        "main_metrics": ["training_prefix_z_mae", "training_prefix_z_mse"],
        "primary_query_budget": 7,
        "additional_history_control": "prefix-z native/guarded 1024",
        "all_methods_require_same_original_future_mask": True,
        "raw_native_failure_policy": "retain failed windows; no complete aggregate metric if any window fails",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("confirmation identity changed after predictions may have been generated")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    registry = default_forecast_registry()
    joint = registry.get(args.model).mode == "joint_multivariate"
    model_path = bundle["identity"]["forecaster_artifacts"][args.model]
    adapter = (
        registry.build(args.model, model_name=model_path, device="cuda", batch_size=8)
        if joint
        else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    parameter_sha = parameter_digest(backbone)
    targets = [0, 1]
    spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=96, target_indices=targets
    )
    long_spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=1024, target_indices=targets
    )
    rows, files = [], []
    for index, record in enumerate(prepared["episodes"]):
        source = args.prepared_root / record["path"]
        motm_record = motm_cases[record["episode_id"]]
        motm_path = args.motm_root / motm_record["path"]
        if (
            file_sha256(source) != record["sha256"]
            or file_sha256(motm_path) != motm_record["sha256"]
        ):
            raise ValueError("a prepared confirmation input changed")
        cache = output / "predictions" / source.name
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale, defaults = (
            np.asarray(scaler["mean"]),
            np.asarray(scaler["scale"]),
            np.asarray(scaler["fallback_medians"]),
        )
        if not cache.exists():
            with np.load(source, allow_pickle=False) as saved:
                context, long_context, candidates = (
                    saved["context"],
                    saved["long_context"],
                    saved["candidate_values"],
                )
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            if context.shape[0] != 96 or long_context.shape != (1024, context.shape[1]):
                raise ValueError("a prepared history has an undeclared context length")
            if bool((~np.isfinite(context)).any()) != record["window"]["context_has_missing"]:
                raise ValueError("the registered native-missing context label changed")
            all_actions = [*actions, "guarded_direct"]
            points, distinct = query_candidate_points(
                runner, spec, context, candidates, actions, targets, mean, scale, joint=joint
            )
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
                metadata={
                    **record,
                    "model_id": args.model,
                    "episode_index": index,
                    "split": "confirmation",
                },
            )
            last_z = (candidates[actions.index("locf"), -1, targets] - mean[targets]) / scale[
                targets
            ]
            generated = forecast_response_inputs(
                extend_forecast_features(generated, points, actions, last_z)
            )
            predictions = dict(zip(all_actions, points, strict=True))
            choice_records, errors = {}, {}
            for objective, learner in learners.items():
                choices = rank_context_actions(learner, generated, 3)
                choice_records[objective] = choices
                selected = np.stack(
                    [
                        chosen_prediction(points, all_actions, choice, joint=joint)
                        for choice in choices
                    ]
                )
                name = "teacher" if objective == "clean_forecast_mse" else "future_supervised"
                predictions[name + "_rank1"] = selected[0]
                predictions[name + "_rank3"] = np.median(selected, axis=0)
            predictions["forecast_mean_guarded"] = np.mean(points, axis=0)
            predictions["forecast_median_guarded"] = np.median(points, axis=0)
            predictions["forecast_mean_finite"] = np.mean(points[: len(actions)], axis=0)
            predictions["forecast_median_finite"] = np.median(points[: len(actions)], axis=0)
            fixed = controls["fixed1_joint"]
            predictions["source_fixed1_joint"] = chosen_prediction(
                points, all_actions, fixed[:1] if joint else fixed, joint=joint
            )
            for name in ("fixed3_clean_forecast_mse", "fixed3_future_mse"):
                by_target = controls[name]
                if joint and by_target[0] != by_target[1]:
                    raise ValueError("joint fixed portfolios have inconsistent target choices")
                choices = [
                    [by_target[0][slot]]
                    if joint
                    else [by_target[target][slot] for target in targets]
                    for slot in range(3)
                ]
                predictions["source_" + name] = np.median(
                    np.stack(
                        [
                            chosen_prediction(points, all_actions, choice, joint=joint)
                            for choice in choices
                        ]
                    ),
                    axis=0,
                )
            observed = np.isfinite(context)
            for name, values in (
                ("input_mean_finite", np.mean(candidates, axis=0)),
                ("input_median_finite", np.median(candidates, axis=0)),
            ):
                values[observed] = context[observed]
                predictions[name] = (
                    runner.predict(values[None], spec).point[0] - mean[targets]
                ) / scale[targets]
            with np.load(motm_path, allow_pickle=False) as saved:
                values = saved["values"]
            np.testing.assert_array_equal(values[observed], context[observed])
            predictions["motm_reference"] = (
                runner.predict(values[None], spec).point[0] - mean[targets]
            ) / scale[targets]
            predictions["forecast_median_with_motm"] = np.median(
                np.concatenate([points, predictions["motm_reference"][None]], axis=0), axis=0
            )
            guard_choices = ["guarded_direct"] if joint else ["guarded_direct"] * len(targets)
            guarded = assemble_selected_context(
                context, candidates, actions, guard_choices, targets, joint=joint
            )
            predictions["native_guarded_prefix_z96"] = runner.predict_missing(
                ((guarded - mean) / scale)[None], spec
            ).point[0]
            long_batch = SeriesBatch(long_context[None], np.isfinite(long_context[None]))
            long_result = CandidateRunner().run("locf", long_batch)
            long_fallback = long_result.values[0].copy()
            long_fallback[~long_result.native_valid_mask[0]] = np.broadcast_to(
                defaults, long_context.shape
            )[~long_result.native_valid_mask[0]]
            long_guarded = assemble_selected_context(
                long_context, long_fallback[None], ["locf"], guard_choices, targets, joint=joint
            )
            predictions["native_guarded_prefix_z1024"] = runner.predict_missing(
                ((long_guarded - mean) / scale)[None], long_spec
            ).point[0]
            empty = ~np.isfinite(context).any(axis=0)
            fallback = bool(empty.any()) if joint else bool(empty[targets].any())
            if not fallback:
                predictions["raw_native"] = points[-1].copy()
            else:
                try:
                    value = (
                        runner.predict_missing(context[None], spec).point[0] - mean[targets]
                    ) / scale[targets]
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite raw-native prediction")
                    predictions["raw_native"] = value
                except Exception as error:
                    errors["raw_native"] = f"{type(error).__name__}: {error}"
                    predictions["raw_native"] = np.full((96, 2), np.nan)
            for name, point in predictions.items():
                if name not in errors and (point.shape != (96, 2) or not np.isfinite(point).all()):
                    raise ValueError("a declared complete forecast method failed")
            _save_npz(
                cache,
                methods=np.asarray(list(predictions)),
                point_z=np.stack(list(predictions.values())),
                metadata=np.asarray(
                    json.dumps(
                        {
                            "choices": choice_records,
                            "errors": errors,
                            "distinct_primary_contexts": distinct,
                            "native_guard_fallback": fallback,
                            "decisions_use_current_outcomes": False,
                        }
                    )
                ),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(parameter_sha),
            )
        # Read the original incomplete future only after the prediction artifact exists.
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != parameter_sha
            ):
                raise ValueError("a confirmation prediction cache belongs to another run")
            methods, points = saved["methods"].tolist(), saved["point_z"]
            diagnostics = json.loads(str(saved["metadata"]))
        with np.load(source, allow_pickle=False) as saved:
            future, future_mask = saved["future"], saved["future_observed"]
        for method, point in zip(methods, points, strict=True):
            if method in diagnostics["errors"]:
                errors = {
                    name: np.full((1, 2), np.nan) for name in ("mae", "mse", "raw_mae", "raw_mse")
                }
                counts = future_mask.sum(axis=0)
            else:
                errors, counts = observed_future_errors(
                    (point * scale[targets] + mean[targets])[None],
                    future,
                    future_mask,
                    scale[targets],
                )
            rows.append(
                {
                    "model_id": args.model,
                    "method": method,
                    "episode_id": record["episode_id"],
                    "family_id": record["family_id"],
                    "dataset_id": record["dataset_id"],
                    "item_id": record["item_id"],
                    "origin_id": record["origin_id"],
                    "native_missing_context": record["window"]["context_has_missing"],
                    "future_observed_target0": int(counts[0]),
                    "future_observed_target1": int(counts[1]),
                    "failed": method in diagnostics["errors"],
                    **{name: float(value.mean()) for name, value in errors.items()},
                }
            )
        files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        _write_json(
            output / "progress.json",
            {"status": "forecasting", "completed_episodes": index + 1, "total_episodes": 373},
        )
        print(
            json.dumps(
                {"model_id": args.model, "completed_episodes": index + 1, "total_episodes": 373}
            ),
            flush=True,
        )
    if parameter_digest(backbone) != parameter_sha:
        raise ValueError("the frozen forecasting model changed")
    frame = pd.DataFrame(rows)
    for _, group in frame.groupby("method"):
        if (
            len(group) != 373
            or group.episode_id.nunique() != 373
            or group.native_missing_context.sum() != 164
        ):
            raise ValueError("a confirmation method has incomplete or changed cohort coverage")
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summaries = []
    for panel, data in (
        ("all_registered", frame),
        ("naturally_missing", frame[frame.native_missing_context]),
    ):
        items, families, summary = hierarchical_metrics(data)
        items.to_csv(output / f"{panel}_item_metrics.csv", index=False)
        families.to_csv(output / f"{panel}_family_metrics.csv", index=False)
        summaries.append(summary.assign(panel=panel, families=data.family_id.nunique()))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "predictions": files,
            "parameters_unchanged": True,
            "parameter_sha256": parameter_sha,
            "summary": summary.astype(object).where(pd.notna(summary), None).to_dict("records"),
            "failed_windows": frame.groupby("method").failed.sum().astype(int).to_dict(),
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": [
                "nine prespecified families; seven with naturally missing contexts",
                "original future observation masks are preserved",
                "longer-history and changed-input-scale controls have separately declared information conditions",
                "cached imputation inputs do not establish end-to-end deployment timing",
            ],
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_episodes": len(files), "total_episodes": 373},
    )


if __name__ == "__main__":
    main()
