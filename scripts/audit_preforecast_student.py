"""Audit static student inputs, selected costs and omitted-covariate dependence."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.preforecast import (  # noqa: E402
    STATIC_FEATURES,
    decision_keys,
    preforecast_inputs,
)
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "input-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--structured-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed static-student audits")
    output.mkdir(parents=True, exist_ok=True)
    root = args.input_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    input_scope = manifest["identity"].get("input_scope", "preforecast")
    if input_scope not in {"preforecast", "candidate_forecasts"}:
        raise ValueError("unknown selector input scope")
    uses_forecasts = input_scope == "candidate_forecasts"
    if uses_forecasts and args.structured_root is not None:
        raise ValueError("the declared forecast-response comparison uses original static inputs")
    clean_teacher = manifest["identity"].get("teacher_kind") == "complete_source_history"
    if clean_teacher != (args.teacher_root is not None):
        raise ValueError(
            "the complete-history teacher must be supplied only for its matching audit"
        )
    teacher_manifest = None
    if clean_teacher:
        teacher_manifest = json.loads(
            (args.teacher_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            teacher_manifest["status"] != "completed"
            or file_sha256(args.teacher_root / "manifest.json")
            != manifest["identity"]["teacher_manifest_sha256"]
        ):
            raise ValueError("complete-history teacher provenance changed")
    models = manifest["identity"].get("models", ["chronos2", "timesfm2p5"])
    expected_folds = len(models) * len(manifest["identity"]["objectives"]) * 15
    if manifest["status"] != "completed" or len(manifest["folds"]) != expected_folds:
        raise ValueError("complete all static-student folds")
    table_path = args.accuracy_root / "candidate_accuracy.parquet"
    if file_sha256(table_path) != manifest["identity"]["candidate_table_sha256"]:
        raise ValueError("the static feature and cost source changed")
    columns = [
        "episode_id",
        "episode_index",
        "origin_id",
        "model_id",
        "candidate_id",
        "target_slot",
        "family_id",
        "dataset_id",
        "item_id",
        "split",
        "mae",
        "mse",
        *STATIC_FEATURES,
    ]
    expected_features = STATIC_FEATURES
    select_inputs = preforecast_inputs
    if uses_forecasts:
        from tsfm_fais.routing.forecast_response import (
            FORECAST_FEATURES,
            RESPONSE_FEATURES,
            forecast_response_inputs,
        )

        columns.extend(RESPONSE_FEATURES)
        expected_features, select_inputs = FORECAST_FEATURES, forecast_response_inputs
        if manifest["identity"].get("nominal_forecast_contexts_required_before_decision") != 7:
            raise ValueError("the candidate-query cost was not declared")
    source = pd.read_parquet(table_path, columns=columns)
    origin_map = (
        source[["origin_id", "family_id", "split"]].drop_duplicates().set_index("origin_id")
    )
    structured_manifest_sha = None
    if args.structured_root is not None:
        from tsfm_fais.routing.structured_preforecast import (
            ALL_FEATURES,
            EXTRA_FEATURES,
            structured_inputs,
        )

        feature_manifest_path = args.structured_root / "manifest.json"
        features = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        structured_manifest_sha = file_sha256(feature_manifest_path)
        feature_path = args.structured_root / features["feature_file"]
        if (
            features["status"] != "completed"
            or structured_manifest_sha != manifest["identity"]["structured_manifest_sha256"]
            or file_sha256(feature_path) != features["feature_sha256"]
        ):
            raise ValueError("structured audit features have different provenance")
        keys = ["model_id", "episode_id", "candidate_id", "target_slot"]
        extra = pd.read_parquet(feature_path, columns=[*keys, *EXTRA_FEATURES])
        source = source.merge(extra, on=keys, validate="one_to_one")
        expected_features = ALL_FEATURES

        def select_inputs(frame):
            return structured_inputs(frame, view=manifest["identity"]["feature_view"])

    bank, projection, actions, dependence = {}, {}, {}, []
    for model in models:
        source_array = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(source_array) != accuracy["prediction_arrays"][source_array.name]:
            raise ValueError("source forecasts changed")
        actions[model] = [
            name
            for name in accuracy["action_orders"][model]
            if name not in {"native_missing", "vendor_missing"}
        ]
        positions = [accuracy["action_orders"][model].index(name) for name in actions[model]]
        bank[model] = np.load(source_array, mmap_mode="r")[:, positions]
        if clean_teacher:
            teacher_record = next(
                row for row in teacher_manifest["models"] if row["model_id"] == model
            )
            teacher_path = args.teacher_root / teacher_record["teacher_file"]
            if (
                file_sha256(teacher_path) != teacher_record["teacher_sha256"]
                or teacher_record["actions"] != actions[model]
            ):
                raise ValueError("the complete-history reference or candidate order changed")
            teacher = np.load(teacher_path, mmap_mode="r")
        else:
            teacher = np.sort(bank[model], axis=1)[:, len(positions) // 2]
        projection[model] = np.square(bank[model] - teacher[:, None]).sum(axis=2) / 96
        saved_costs = next(row for row in manifest["projection_files"] if row["model_id"] == model)
        if file_sha256(root / saved_costs["path"]) != saved_costs["sha256"]:
            raise ValueError("saved projection costs changed")
        np.testing.assert_allclose(
            np.load(root / saved_costs["path"]), projection[model], rtol=0, atol=1e-12
        )
        if args.structured_root is not None or clean_teacher:
            continue
        # This diagnostic uses fully observed target histories and missing covariates.
        selected = source[
            (source.model_id == model)
            & (source.split == "validation")
            & (source.target_slot == -1)
            & (source.candidate_id == "locf")
            & (source["static.target_missing_fraction"] == 0)
            & (source["static.missing_fraction"] > 0)
        ]
        finite = [index for index, name in enumerate(actions[model]) if name != "guarded_direct"]
        maximum_target_changes = (
            source[
                (source.model_id == model)
                & (source.target_slot == -1)
                & source.candidate_id.isin([actions[model][index] for index in finite])
            ]
            .groupby("episode_index")["static.max_fill_change"]
            .max()
        )
        for record in selected.itertuples(index=False):
            index = record.episode_index
            pool = bank[model][index, finite]
            delta = pool - pool[0]
            if maximum_target_changes.loc[index] != 0:
                raise ValueError("a finite candidate changed a fully observed target history")
            dependence.append(
                {
                    "model_id": model,
                    "episode_id": record.episode_id,
                    "origin_id": record.origin_id,
                    "family_id": record.family_id,
                    "dataset_id": record.dataset_id,
                    "max_target_forecast_change_z": float(np.abs(delta).max()),
                    "mean_target_forecast_change_squared_z": float((delta**2).mean()),
                }
            )
    rows, decisions, maximum_difference, fitted_pairs = [], 0, 0.0, 0
    for item in manifest["folds"]:
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a static fold record changed")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if fold["identity_sha256"] != manifest["identity_sha256"]:
            raise ValueError("a static fold identity changed")
        for kind in ("model", "choices", "scores"):
            if file_sha256(root / fold[kind + "_path"]) != fold[kind + "_sha256"]:
                raise ValueError("a static fold artifact changed")
        classifier = joblib.load(root / fold["model_path"])
        if set(classifier.feature_names) != set(expected_features) or set(
            fold["training_feature_names"]
        ) != set(expected_features):
            raise ValueError("a model uses features outside its declared decision boundary")
        if fold["objective"] in {"consensus_mse", "clean_forecast_mse"} and (
            fold["source_outcome_supervision"] or fold["normalizers"] is not None
        ):
            raise ValueError("projection supervision unexpectedly uses actual source outcomes")
        if (
            fold["objective"] in {"future_joint", "future_mse"}
            and not fold["source_outcome_supervision"]
        ):
            raise ValueError("actual-future supervision was not declared")
        if fold["objective"] == "future_mse" and fold["normalizers"] is not None:
            raise ValueError("the matched MSE control must use the shared prefix units directly")
        if clean_teacher and (
            not fold.get("source_complete_history_supervision")
            or fold.get("teacher_kind") != "complete_source_history"
        ):
            raise ValueError("the fold did not record its privileged source-history supervision")
        train = origin_map.loc[fold["training_origins"]]
        if (train.family_id == fold["held_family"]).any() or set(train.split) != {"train"}:
            raise ValueError("training used an evaluation family or time split")
        choices = pd.read_parquet(root / fold["choices_path"])
        if any(
            name.startswith("response.") or name in {"mae", "mse", "loss", "teacher_mse"}
            for name in choices
        ):
            raise ValueError("the pre-scoring choices contain unavailable outcomes")
        evaluation_base = source[
            (source.model_id == fold["model_id"])
            & (source.split == "validation")
            & (source.family_id == fold["held_family"])
            & source.target_slot.isin([-1] if fold["model_id"] == "chronos2" else [0, 1])
            & source.candidate_id.isin(classifier.candidate_ids)
        ]
        rebuilt = classifier.select(select_inputs(decision_keys(evaluation_base)))
        choice_index = ["source_episode_id", "target_slot"]
        pd.testing.assert_series_equal(
            choices.set_index(choice_index).candidate_id.sort_index(),
            rebuilt.set_index(choice_index).candidate_id.sort_index(),
        )
        evaluation = evaluation_base.rename(columns={"episode_id": "source_episode_id"})
        expected = evaluation[["source_episode_id", "target_slot"]].drop_duplicates()
        if len(choices) != len(expected) or choices.duplicated(choice_index).any():
            raise ValueError("a static policy duplicated an evaluation decision")
        if set(map(tuple, choices[["source_episode_id", "target_slot"]].to_numpy())) != set(
            map(tuple, expected.to_numpy())
        ):
            raise ValueError("a static policy lacks evaluation decisions")
        joined = choices.merge(
            evaluation[["source_episode_id", "candidate_id", "target_slot", "mae", "mse"]],
            on=["source_episode_id", "candidate_id", "target_slot"],
            validate="one_to_one",
        )
        cost = projection[fold["model_id"]]
        absolute, floor = [], []
        for row in choices.itertuples(index=False):
            possible = (
                cost[row.episode_index].mean(axis=1)
                if row.target_slot == -1
                else cost[row.episode_index, :, row.target_slot]
            )
            absolute.append(possible[actions[fold["model_id"]].index(row.candidate_id)])
            floor.append(possible.min())
        joined["teacher_mse"], joined["projection_floor"] = absolute, floor
        joined["projection_regret"] = np.asarray(absolute) - floor
        stored_scores = pd.read_parquet(root / fold["scores_path"])
        score_index = ["source_episode_id", "candidate_id", "target_slot"]
        score_columns = ["mae", "mse", "teacher_mse", "projection_floor", "projection_regret"]
        if len(stored_scores) != len(joined) or stored_scores.duplicated(score_index).any():
            raise ValueError("stored student scores have different coverage")
        np.testing.assert_allclose(
            stored_scores.set_index(score_index).sort_index()[score_columns],
            joined.set_index(score_index).sort_index()[score_columns],
            rtol=0,
            atol=1e-10,
        )
        windows = (
            joined.groupby(["source_episode_id", "family_id", "dataset_id", "item_id"])[
                ["mae", "mse", "teacher_mse", "projection_floor", "projection_regret"]
            ]
            .mean()
            .reset_index()
        )
        measured = {
            key: family_macro(windows, key)
            for key in ("mae", "mse", "teacher_mse", "projection_floor", "projection_regret")
        }
        for key, value in measured.items():
            maximum_difference = max(maximum_difference, abs(value - fold[key]))
            np.testing.assert_allclose(value, fold[key], rtol=0, atol=1e-10)
        rows.append(
            {
                "model_id": fold["model_id"],
                "objective": fold["objective"],
                "family_id": fold["held_family"],
                **measured,
            }
        )
        fitted_pairs += sum(not row["constant"] for row in fold["pair_fit_records"])
        decisions += len(choices)
    pd.DataFrame(rows).to_csv(output / "recomputed_family_metrics.csv", index=False)
    dependency_frame = pd.DataFrame(
        dependence,
        columns=[
            "model_id",
            "episode_id",
            "origin_id",
            "family_id",
            "dataset_id",
            "max_target_forecast_change_z",
            "mean_target_forecast_change_squared_z",
        ],
    )
    dependency_frame.to_csv(output / "observed_target_covariate_effects.csv", index=False)
    dependency_summary = []
    for model, group in dependency_frame.groupby("model_id"):
        dependency_summary.append(
            {
                "model_id": model,
                "episodes": len(group),
                "origins": group.origin_id.nunique(),
                "families": group.family_id.nunique(),
                "max_forecast_change_z": float(group.max_target_forecast_change_z.max()),
                "episodes_with_change_over_1e_4_z": int(
                    (group.max_target_forecast_change_z > 1e-4).sum()
                ),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_folds": len(rows),
            "verified_decisions": decisions,
            "fitted_tree_models": fitted_pairs,
            "maximum_metric_difference": maximum_difference,
            "input_scope": input_scope,
            "decision_features_verified": list(expected_features),
            "preforecast_features_verified": None if uses_forecasts else list(expected_features),
            "structured_manifest_sha256": structured_manifest_sha,
            "source_outcome_free_projection_supervision": all(
                objective in {"consensus_mse", "clean_forecast_mse"}
                for objective in manifest["identity"]["objectives"]
            ),
            "source_outcome_supervised_objectives": [
                objective
                for objective in manifest["identity"]["objectives"]
                if objective in {"future_joint", "future_mse"}
            ],
            "teacher_kind": manifest["identity"].get("teacher_kind", "candidate_forecast_median"),
            "source_complete_history_supervision": clean_teacher,
            "covariate_dependence_diagnostic": dependency_summary,
            "interpretation": "parameter-input and metric audit; joint-model covariate changes can alter observed-target forecasts; no global cause or accuracy guarantee",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "status": "completed",
                "verified_folds": len(rows),
                "verified_decisions": decisions,
                "maximum_metric_difference": maximum_difference,
                "covariate_dependence_diagnostic": dependency_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
