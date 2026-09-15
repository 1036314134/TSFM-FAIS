"""Compare declared selector inputs using outcome and forecast-teacher supervision."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector  # noqa: E402
from tsfm_fais.routing.preforecast import (  # noqa: E402
    METADATA,
    STATIC_FEATURES,
    consensus_projection_costs,
    decision_keys,
    preforecast_inputs,
    projection_row_costs,
)
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def read_costs(path, *, model, split, family=None):
    filters = [("model_id", "==", model), ("split", "==", split)]
    if family is not None:
        filters.append(("family_id", "==", family))
    return pd.read_parquet(
        path,
        columns=[
            "episode_id",
            "candidate_id",
            "target_slot",
            "family_id",
            "dataset_id",
            "mae",
            "mse",
        ],
        filters=filters,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--structured-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument(
        "--input-scope", choices=("preforecast", "candidate_forecasts"), default="preforecast"
    )
    parser.add_argument(
        "--feature-view", choices=("target_temporal", "full_dependency"), default="full_dependency"
    )
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    parser.add_argument("--objectives", default="consensus_mse,future_joint")
    args = parser.parse_args()
    uses_forecasts = args.input_scope == "candidate_forecasts"
    if uses_forecasts and args.structured_root is not None:
        parser.error("the candidate-forecast comparison uses the original static representation")
    models, objectives = args.models.split(","), args.objectives.split(",")
    if (
        not models
        or len(set(models)) != len(models)
        or set(models) - {"chronos2", "timesfm2p5"}
        or not objectives
        or len(set(objectives)) != len(objectives)
        or set(objectives) - {"consensus_mse", "future_joint", "future_mse", "clean_forecast_mse"}
    ):
        parser.error("distinct supported models and objectives are required")
    clean_teacher = "clean_forecast_mse" in objectives
    if clean_teacher != (args.teacher_root is not None) or (
        clean_teacher and objectives != ["clean_forecast_mse"]
    ):
        parser.error(
            "complete-history supervision requires its own run and an explicit teacher root"
        )
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed static-student results")
    output.mkdir(parents=True, exist_ok=True)
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    teacher_manifest = None
    if clean_teacher:
        teacher_manifest = json.loads(
            (args.teacher_root / "manifest.json").read_text(encoding="utf-8")
        )
        if teacher_manifest["status"] != "completed" or teacher_manifest["identity"][
            "accuracy_manifest_sha256"
        ] != file_sha256(root / "manifest.json"):
            raise ValueError("complete-history teacher does not share the accuracy protocol")
    source_path = root / "candidate_accuracy.parquet"
    schema = pq.read_schema(source_path)
    if {name for name in schema.names if name.startswith("static.")} != set(STATIC_FEATURES):
        raise ValueError("the static feature inventory differs from the audited export")
    # Outcome columns are excluded from the decision inputs in every comparison.
    input_features = STATIC_FEATURES
    select_inputs = preforecast_inputs
    if uses_forecasts:
        from tsfm_fais.routing.forecast_response import FORECAST_FEATURES, forecast_response_inputs

        input_features, select_inputs = FORECAST_FEATURES, forecast_response_inputs
        if {name for name in schema.names if name.startswith(("static.", "response."))} != set(
            input_features
        ):
            raise ValueError("candidate forecast features differ from the audited inventory")
    feature_columns = [name for name in (*METADATA, *input_features) if name in schema.names]
    structured_manifest = None
    if args.structured_root is None:
        static = pd.read_parquet(source_path, columns=feature_columns)
    else:
        from tsfm_fais.routing.structured_preforecast import ALL_FEATURES, structured_inputs

        structured_path = args.structured_root / "manifest.json"
        structured_manifest = json.loads(structured_path.read_text(encoding="utf-8"))
        feature_path = args.structured_root / structured_manifest["feature_file"]
        if (
            structured_manifest["status"] != "completed"
            or structured_manifest["identity"]["accuracy_manifest_sha256"]
            != file_sha256(root / "manifest.json")
            or file_sha256(feature_path) != structured_manifest["feature_sha256"]
        ):
            raise ValueError("structured features do not match the prediction source")
        static = pd.read_parquet(feature_path)
        input_features = ALL_FEATURES

        def select_inputs(frame):
            return structured_inputs(frame, view=args.feature_view)

        static = select_inputs(static)
    static = static[~static.candidate_id.isin(["native_missing", "vendor_missing"])]
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "candidate_table_sha256": file_sha256(source_path),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/utility.py",
                "scripts/export_downstream_accuracy.py",
                "src/tsfm_fais/forecasting/accuracy.py",
            )
        },
        "static_features": list(input_features),
        "input_scope": args.input_scope,
        "nominal_forecast_contexts_required_before_decision": 7 if uses_forecasts else 0,
        "models": models,
        "objectives": objectives,
        "teacher_kind": "complete_source_history" if clean_teacher else "candidate_forecast_median",
        "feature_view": "forecast_response"
        if uses_forecasts
        else args.feature_view
        if args.structured_root
        else "base_static",
        "classifier": "cost-sensitive pairwise classification; 160 trees, 15 leaves, rate .05, minimum leaf 25, L2 5",
        "model_seed": 5101,
        "folds": "leave one family out",
        "candidate_pool_size": 7,
        "input_recipe": "original R4 raw forecaster inputs, scores in common prefix units",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lightgbm", "numpy", "pandas", "pyarrow", "joblib")
        },
    }
    if clean_teacher:
        identity["teacher_manifest_sha256"] = file_sha256(args.teacher_root / "manifest.json")
        identity["source_sha256"]["src/tsfm_fais/routing/forecast_teacher.py"] = file_sha256(
            ROOT / "src/tsfm_fais/routing/forecast_teacher.py"
        )
    if uses_forecasts:
        name = "src/tsfm_fais/routing/forecast_response.py"
        identity["source_sha256"][name] = file_sha256(ROOT / name)
    if structured_manifest is not None:
        identity["structured_manifest_sha256"] = file_sha256(args.structured_root / "manifest.json")
        identity["source_sha256"]["src/tsfm_fais/routing/structured_preforecast.py"] = file_sha256(
            ROOT / "src/tsfm_fais/routing/structured_preforecast.py"
        )
    total_folds = len(models) * len(objectives) * 15
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("static-student identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name, expected in identity["source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("a student implementation changed")
        (output / (Path(name).stem + "_snapshot.py")).write_bytes((ROOT / name).read_bytes())
    results, files, projection_files = [], [], []
    for model in models:
        actions = [
            name
            for name in accuracy["action_orders"][model]
            if name not in {"native_missing", "vendor_missing"}
        ]
        positions = [accuracy["action_orders"][model].index(name) for name in actions]
        point_path = root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("a source forecast array changed")
        points = np.load(point_path, mmap_mode="r")[:, positions]
        teacher_sha = None
        if clean_teacher:
            from tsfm_fais.routing.forecast_teacher import forecast_reference_costs

            teacher_record = next(
                row for row in teacher_manifest["models"] if row["model_id"] == model
            )
            if teacher_record["actions"] != actions:
                raise ValueError("teacher candidate order differs from the student pool")
            teacher_path = args.teacher_root / teacher_record["teacher_file"]
            stored_cost_path = args.teacher_root / teacher_record["cost_file"]
            if (
                file_sha256(teacher_path) != teacher_record["teacher_sha256"]
                or file_sha256(stored_cost_path) != teacher_record["cost_sha256"]
            ):
                raise ValueError("a complete-history teacher array changed")
            costs = forecast_reference_costs(points, np.load(teacher_path, mmap_mode="r"))
            np.testing.assert_allclose(
                costs, np.load(stored_cost_path, mmap_mode="r"), rtol=0, atol=1e-12
            )
            teacher_sha = teacher_record["teacher_sha256"]
        else:
            costs = consensus_projection_costs(points)
        del points
        cost_path = output / f"{model}_projection_costs.npy"
        with cost_path.open("wb") as handle:
            np.save(handle, costs, allow_pickle=False)
        projection_files.append(
            {
                "model_id": model,
                "path": cost_path.name,
                "sha256": file_sha256(cost_path),
                "source_sha256": accuracy["prediction_arrays"][point_path.name],
                "teacher_kind": identity["teacher_kind"],
                "teacher_reference_sha256": teacher_sha,
            }
        )
        view = static[
            (static.model_id == model)
            & static.target_slot.isin([-1] if model == "chronos2" else [0, 1])
        ]
        source_outcomes = None
        for objective in identity["objectives"]:
            if objective in {"future_joint", "future_mse"}:
                source_outcomes = read_costs(source_path, model=model, split="train")
            for family in sorted(view.family_id.unique()):
                name = hashlib.sha256(f"{model}|{objective}|{family}".encode()).hexdigest()[:24]
                directory = output / "folds"
                directory.mkdir(exist_ok=True)
                cache = directory / f"{name}.json"
                if cache.exists():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    if saved["identity_sha256"] != identity_sha:
                        raise ValueError("a static-student fold belongs to another experiment")
                    for kind in ("model", "choices", "scores"):
                        if file_sha256(output / saved[kind + "_path"]) != saved[kind + "_sha256"]:
                            raise ValueError("a static-student artifact changed")
                else:
                    training_base = view[(view.split == "train") & (view.family_id != family)]
                    evaluation_base = view[
                        (view.split == "validation") & (view.family_id == family)
                    ]
                    if family in set(training_base.family_id) or set(training_base.origin_id) & set(
                        evaluation_base.origin_id
                    ):
                        raise ValueError("source histories or families overlap evaluation")
                    training = select_inputs(decision_keys(training_base))
                    denominators = None
                    if objective in {"consensus_mse", "clean_forecast_mse"}:
                        absolute, floor = projection_row_costs(training, costs, actions)
                        training["loss"] = absolute - floor
                    else:
                        if objective == "future_joint":
                            reference = source_outcomes[
                                (source_outcomes.family_id != family)
                                & (source_outcomes.candidate_id == "locf")
                                & (source_outcomes.target_slot == -1)
                            ]
                            denominators = {
                                key: max(family_macro(reference, key), 1e-12)
                                for key in ("mae", "mse")
                            }
                        outcomes = source_outcomes.rename(
                            columns={"episode_id": "source_episode_id"}
                        )
                        training = training.merge(
                            outcomes[
                                ["source_episode_id", "candidate_id", "target_slot", "mae", "mse"]
                            ],
                            on=["source_episode_id", "candidate_id", "target_slot"],
                            validate="one_to_one",
                        )
                        loss = (
                            training.mse
                            if objective == "future_mse"
                            else 0.5
                            * (
                                training.mae / denominators["mae"]
                                + training.mse / denominators["mse"]
                            )
                        )
                        training = select_inputs(training).assign(loss=loss)
                    if any(
                        (name.startswith("response.") and not uses_forecasts)
                        or name in {"mae", "mse", "teacher_mse"}
                        for name in training
                    ):
                        raise ValueError(
                            "forbidden response or outcome column in the student fitting table"
                        )
                    started = monotonic()
                    selector = PairwiseUtilitySelector(
                        mode="classification", seed=5101, n_estimators=160, n_jobs=1
                    ).fit(training)
                    if set(selector.feature_names) != set(input_features):
                        raise ValueError(
                            "selector features do not match the declared decision boundary"
                        )
                    fitted = monotonic()
                    evaluation = select_inputs(decision_keys(evaluation_base))
                    selected = selector.select(evaluation)
                    choice_columns = [
                        "episode_id",
                        "source_episode_id",
                        "episode_index",
                        "origin_id",
                        "model_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "candidate_id",
                        "target_slot",
                        "pairwise_wins",
                    ]
                    choices = selected[choice_columns]
                    choice_path = directory / f"{name}.choices.parquet"
                    choices.to_parquet(choice_path, index=False)
                    # Forecast labels enter only after the input-only student has decided.
                    outcomes = read_costs(
                        source_path, model=model, split="validation", family=family
                    ).rename(columns={"episode_id": "source_episode_id"})
                    scores = choices.merge(
                        outcomes[
                            ["source_episode_id", "candidate_id", "target_slot", "mae", "mse"]
                        ],
                        on=["source_episode_id", "candidate_id", "target_slot"],
                        validate="one_to_one",
                    )
                    absolute, floor = projection_row_costs(scores, costs, actions)
                    (
                        scores["teacher_mse"],
                        scores["projection_floor"],
                        scores["projection_regret"],
                    ) = absolute, floor, absolute - floor
                    windows = (
                        scores.groupby(["source_episode_id", "family_id", "dataset_id", "item_id"])[
                            ["mae", "mse", "teacher_mse", "projection_floor", "projection_regret"]
                        ]
                        .mean()
                        .reset_index()
                    )
                    score_path = directory / f"{name}.scores.parquet"
                    scores.to_parquet(score_path, index=False)
                    model_path = directory / f"{name}.joblib"
                    joblib.dump(selector, model_path, compress=3)
                    saved = {
                        "identity_sha256": identity_sha,
                        "model_id": model,
                        "held_family": family,
                        "objective": objective,
                        "training_origins": sorted(training.origin_id.unique()),
                        "training_feature_names": selector.feature_names,
                        "source_outcome_supervision": objective in {"future_joint", "future_mse"},
                        "source_complete_history_supervision": clean_teacher,
                        "teacher_kind": identity["teacher_kind"],
                        "normalizers": denominators,
                        "fit_seconds": fitted - started,
                        "predict_score_save_seconds": monotonic() - fitted,
                        "pair_fit_records": selector.fit_records,
                        "baseline_id": selector.baseline_id,
                        "windows": len(windows),
                        **{
                            key: family_macro(windows, key)
                            for key in (
                                "mae",
                                "mse",
                                "teacher_mse",
                                "projection_floor",
                                "projection_regret",
                            )
                        },
                        "model_path": str(model_path.relative_to(output)),
                        "model_sha256": file_sha256(model_path),
                        "choices_path": str(choice_path.relative_to(output)),
                        "choices_sha256": file_sha256(choice_path),
                        "scores_path": str(score_path.relative_to(output)),
                        "scores_sha256": file_sha256(score_path),
                    }
                    _write_json(cache, saved)
                results.append(
                    {
                        key: saved[key]
                        for key in (
                            "model_id",
                            "held_family",
                            "objective",
                            "mae",
                            "mse",
                            "teacher_mse",
                            "projection_floor",
                            "projection_regret",
                            "windows",
                            "fit_seconds",
                            "predict_score_save_seconds",
                        )
                    }
                )
                files.append({"path": str(cache.relative_to(output)), "sha256": file_sha256(cache)})
                _write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "completed_folds": len(files),
                        "total_folds": total_folds,
                        "model": model,
                        "objective": objective,
                        "family": family,
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_folds": len(files),
                            "total_folds": total_folds,
                            "model": model,
                            "objective": objective,
                            "family": family,
                        }
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(results)
    frame.to_csv(output / "family_metrics.csv", index=False)
    summary = (
        frame.groupby(["model_id", "objective"])[
            ["mae", "mse", "teacher_mse", "projection_floor", "projection_regret"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": files,
            "projection_files": projection_files,
            "summary": summary.to_dict("records"),
            "candidate_forecast_queries_added": 0,
            "decision_input_boundary": list(input_features),
            "preforecast_input_boundary": None if uses_forecasts else list(input_features),
            "deployment_replay_status": "not_performed",
            "information": "projection objectives fit source prediction distances; complete-history supervision uses source values before artificial masking; current inputs exclude hidden histories and outcomes; candidate forecasts are allowed only in the explicitly declared candidate_forecasts scope",
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_folds": len(files), "total_folds": total_folds},
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
