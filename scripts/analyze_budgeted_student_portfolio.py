"""Evaluate a fixed three-query student portfolio and matched source-only controls."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.budgeted_portfolio import (  # noqa: E402
    median_portfolio,
    rank_pairwise_candidates,
)
from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.preforecast import (  # noqa: E402
    METADATA,
    STATIC_FEATURES,
    decision_keys,
    preforecast_inputs,
)
from tsfm_fais.routing.structured_preforecast import structured_inputs  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def checked_json(path, expected=None):
    if expected is not None and file_sha256(path) != expected:
        raise ValueError(f"changed source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "teacher-root", "structured-root", "spec", "plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed portfolio studies")
    output.mkdir(parents=True, exist_ok=True)
    root = args.accuracy_root.resolve()
    accuracy = checked_json(root / "manifest.json")
    source_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    source = checked_json(source_path, accuracy["source_episode_manifest_sha256"])
    spec, plan = checked_json(args.spec), checked_json(args.plan)
    if spec["budget"] != 3 or plan["source_manifest_sha256"] != file_sha256(source_path):
        raise ValueError("the fixed budget or screening source changed")
    metadata = pd.DataFrame(source["episodes"]).assign(
        episode_index=np.arange(len(source["episodes"]))
    )
    validation = metadata[metadata.split == "validation"].copy()
    train = metadata[metadata.split == "train"].copy()
    if (
        len(validation) != 1872
        or len(train) != 5940
        or set(train.origin_id) & set(validation.origin_id)
    ):
        raise ValueError("the development history split changed")
    val_ids, train_ids = validation.episode_index.to_numpy(), train.episode_index.to_numpy()
    position = {index: slot for slot, index in enumerate(val_ids)}
    screening_ids = set(plan["decision_episode_ids"])
    if len(screening_ids) != 90 or not screening_ids <= set(validation.episode_id):
        raise ValueError("the screening panel changed")
    teacher_manifest = checked_json(args.teacher_root / "manifest.json")
    structured_manifest = checked_json(args.structured_root / "manifest.json")
    accuracy_sha = file_sha256(root / "manifest.json")
    for manifest in (teacher_manifest, structured_manifest):
        if (
            manifest["status"] != "completed"
            or manifest["identity"]["accuracy_manifest_sha256"] != accuracy_sha
        ):
            raise ValueError("teacher or features use another forecast source")
    student_sources = {}
    for study_id, settings in spec["models"].items():
        for objective, value in settings["students"].items():
            student_sources[f"{study_id}/{objective}"] = file_sha256(ROOT / value / "manifest.json")
    identity = {
        "accuracy_manifest_sha256": accuracy_sha,
        "teacher_manifest_sha256": file_sha256(args.teacher_root / "manifest.json"),
        "structured_manifest_sha256": file_sha256(args.structured_root / "manifest.json"),
        "spec_sha256": file_sha256(args.spec),
        "plan_sha256": file_sha256(args.plan),
        "student_manifests": student_sources,
        "budget": 3,
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/budgeted_portfolio.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/structured_preforecast.py",
                "src/tsfm_fais/routing/forecast_response.py",
            )
        },
        "input_recipe": "R4 raw forecaster inputs; shared prefix-standardized MAE/MSE",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise ValueError("portfolio study identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name in identity["source_sha256"]:
        (output / (Path(name).stem + "_snapshot.py")).write_bytes((ROOT / name).read_bytes())
    all_results, completed = [], []
    for study_id, settings in spec["models"].items():
        model = settings.get("model_id", study_id)
        if model not in {"chronos2", "timesfm2p5"}:
            raise ValueError("the study must identify a supported forecast model")
        destination = output / study_id
        destination.mkdir(exist_ok=True)
        marker = destination / "manifest.json"
        if marker.exists():
            cached = checked_json(marker)
            if cached["identity_sha256"] != identity_sha:
                raise ValueError("a model checkpoint belongs to another study")
            for record in cached["files"]:
                if file_sha256(destination / record["path"]) != record["sha256"]:
                    raise ValueError("a completed model checkpoint changed")
            all_results.append(pd.read_parquet(destination / "episode_results.parquet"))
            completed.append(
                {"study_id": study_id, "model_id": model, "sha256": file_sha256(marker)}
            )
            continue
        joint = model == "chronos2"
        actions = sorted(
            set(accuracy["action_orders"][model]) - {"native_missing", "vendor_missing"}
        )
        ranks, student_references = {}, {}
        uses_forecasts = settings["representation"] == "forecast_response"
        for objective, value in settings["students"].items():
            student_root = ROOT / value
            students = checked_json(
                student_root / "manifest.json", student_sources[f"{study_id}/{objective}"]
            )
            student_references[objective] = []
            if (
                students["status"] != "completed"
                or students["identity"]["accuracy_manifest_sha256"] != accuracy_sha
                or students["identity"].get("feature_view", "base_static")
                != settings["representation"]
                or students["identity"].get("input_scope", "preforecast")
                != ("candidate_forecasts" if uses_forecasts else "preforecast")
            ):
                raise ValueError("student representation or source changed")
            if settings["representation"] in {"base_static", "forecast_response"}:
                path = root / "candidate_accuracy.parquet"
                if file_sha256(path) != students["identity"]["candidate_table_sha256"]:
                    raise ValueError("the original candidate features changed")
                schema_names = pq.read_schema(path).names
                feature_names = FORECAST_FEATURES if uses_forecasts else STATIC_FEATURES
                columns = [name for name in (*METADATA, *feature_names) if name in schema_names]
                select_inputs = forecast_response_inputs if uses_forecasts else preforecast_inputs
                features = select_inputs(
                    pd.read_parquet(path, columns=columns, filters=[("model_id", "==", model)])
                )
            else:
                if (
                    students["identity"]["structured_manifest_sha256"]
                    != identity["structured_manifest_sha256"]
                ):
                    raise ValueError("student structured features changed")
                path = args.structured_root / structured_manifest["feature_file"]
                if file_sha256(path) != structured_manifest["feature_sha256"]:
                    raise ValueError("structured input values changed")
                features = structured_inputs(
                    pd.read_parquet(path, filters=[("model_id", "==", model)]),
                    view=settings["representation"],
                )
            features = features[
                (features.split == "validation")
                & features.target_slot.isin([-1] if joint else [0, 1])
                & features.candidate_id.isin(actions)
            ]
            rankings = np.full((len(validation), 2, 7), -1, dtype=int)
            fold_count = 0
            for record in students["folds"]:
                fold = checked_json(student_root / record["path"], record["sha256"])
                if fold["model_id"] != model or fold["objective"] != objective:
                    continue
                student_references[objective].append(
                    {key: fold[key] for key in ("held_family", "mae", "mse")}
                )
                training_origins = train[train.origin_id.isin(fold["training_origins"])]
                if (training_origins.family_id == fold["held_family"]).any() or set(
                    training_origins.origin_id
                ) != set(fold["training_origins"]):
                    raise ValueError(
                        "student source histories contain an evaluation family or time"
                    )
                for kind in ("model", "choices"):
                    if file_sha256(student_root / fold[kind + "_path"]) != fold[kind + "_sha256"]:
                        raise ValueError("a student model or recorded decision changed")
                learner = joblib.load(student_root / fold["model_path"])
                if set(learner.feature_names) != set(students["identity"]["static_features"]):
                    raise ValueError("the student uses undeclared inputs")
                view = decision_keys(features[features.family_id == fold["held_family"]])
                episodes, preferences, pairs = learner.pair_scores(view)
                ordered = rank_pairwise_candidates(
                    preferences, pairs, learner.candidate_ids, learner.baseline_id
                )
                names = np.asarray(learner.candidate_ids)[ordered]
                expected = (
                    pd.read_parquet(student_root / fold["choices_path"])
                    .set_index("episode_id")
                    .loc[episodes]
                )
                np.testing.assert_array_equal(names[:, 0], expected.candidate_id.to_numpy())
                indexed = view.drop_duplicates("episode_id").set_index("episode_id").loc[episodes]
                numeric = np.vectorize(actions.index, otypes=[int])(names)
                for row_number, row in enumerate(indexed.itertuples()):
                    target_slots = [0, 1] if joint else [row.target_slot]
                    rankings[position[row.episode_index], target_slots] = numeric[row_number]
                fold_count += 1
            if fold_count != 15 or (rankings < 0).any():
                raise ValueError("student rankings have incomplete coverage")
            ranks[objective] = rankings
        # All observable rankings are recorded before actual futures are opened.
        _save_npz(
            destination / "student_rankings.npz", **ranks, identity_sha256=np.asarray(identity_sha)
        )
        point_path = root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("candidate forecasts changed")
        points = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(action) for action in actions]
        ]
        if points.shape != (len(metadata), 7, 96, 2):
            raise ValueError("the supported forecast bank changed shape")
        truth_path = root / "truth_z.npy"
        if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
            raise ValueError("standardized outcomes changed")
        truth = np.load(truth_path, mmap_mode="r")
        teacher_record = next(row for row in teacher_manifest["models"] if row["model_id"] == model)
        teacher_path = args.teacher_root / teacher_record["teacher_file"]
        if file_sha256(teacher_path) != teacher_record["teacher_sha256"]:
            raise ValueError("the complete-history teacher changed")
        teacher = np.load(teacher_path, mmap_mode="r")
        forecasts = {"forecast_median_guarded": np.median(points[val_ids], axis=1)}
        for objective, rankings in ranks.items():
            for budget in (1, 3, 7):
                prediction = median_portfolio(points[val_ids], rankings, budget, joint=joint)
                if budget == 7:
                    np.testing.assert_array_equal(prediction, forecasts["forecast_median_guarded"])
                else:
                    forecasts[f"{objective}_rank{budget}"] = prediction
        subsets = np.array(list(combinations(range(7), 3)))
        training_costs = {
            name: np.empty((len(train), len(subsets), 2))
            for name in ("future_mse", "clean_forecast_mse")
        }
        subset_predictions = []
        uniform_mae, uniform_mse = np.zeros((len(validation), 2)), np.zeros((len(validation), 2))
        for number, subset in enumerate(subsets):
            prediction = np.median(points[:, subset], axis=1)
            training_costs["future_mse"][:, number] = np.square(
                prediction[train_ids] - truth[train_ids]
            ).mean(axis=1)
            training_costs["clean_forecast_mse"][:, number] = np.square(
                prediction[train_ids] - teacher[train_ids]
            ).mean(axis=1)
            subset_predictions.append(prediction[val_ids])
            error = prediction[val_ids] - truth[val_ids]
            uniform_mae += np.abs(error).mean(axis=1) / len(subsets)
            uniform_mse += np.square(error).mean(axis=1) / len(subsets)
        fixed_records = []
        for objective, costs in training_costs.items():
            result = np.full_like(forecasts["forecast_median_guarded"], np.nan)
            for family in sorted(validation.family_id.unique()):
                source_families = sorted(set(train.family_id) - {family})
                mean_cost = np.stack(
                    [
                        costs[train.family_id.to_numpy() == name].mean(axis=0)
                        for name in source_families
                    ]
                ).mean(axis=0)
                selected_subsets = (
                    np.repeat(mean_cost.mean(axis=1).argmin(), 2)
                    if joint
                    else mean_cost.argmin(axis=0)
                )
                held = validation.family_id.to_numpy() == family
                for slot, index in enumerate(selected_subsets):
                    result[held, :, slot] = subset_predictions[index][held, :, slot]
                fixed_records.append(
                    {
                        "objective": objective,
                        "held_family": family,
                        "source_families": source_families,
                        "actions_by_target": [
                            [actions[index] for index in subsets[subset]]
                            for subset in selected_subsets
                        ],
                    }
                )
            forecasts[f"source_fixed3_{objective}"] = result
        _write_json(destination / "source_fixed_choices.json", fixed_records)
        records = []
        errors = {
            name: (
                np.abs(point - truth[val_ids]).mean(axis=(1, 2)),
                np.square(point - truth[val_ids]).mean(axis=(1, 2)),
            )
            for name, point in forecasts.items()
        }
        errors["uniform_random3_expected"] = (uniform_mae.mean(axis=1), uniform_mse.mean(axis=1))
        columns = ["episode_id", "origin_id", "family_id", "dataset_id", "item_id"]
        for method, (mae, mse) in errors.items():
            aggregation_size = (
                7 if method == "forecast_median_guarded" else 1 if method.endswith("_rank1") else 3
            )
            forecast_contexts = 7 if uses_forecasts and "_rank" in method else aggregation_size
            frame = (
                validation[columns]
                .copy()
                .assign(
                    model_id=model,
                    representation=settings["representation"],
                    method=method,
                    aggregation_size=aggregation_size,
                    forecast_contexts_required=forecast_contexts,
                    mae=mae,
                    mse=mse,
                )
            )
            if not np.isfinite(frame[["mae", "mse"]].to_numpy()).all():
                raise ValueError("nonfinite portfolio scores")
            records.append(frame)
        frame = pd.concat(records, ignore_index=True)
        family_scores = frame.groupby(["method", "family_id"])[["mae", "mse"]].mean()
        for objective, source_scores in student_references.items():
            reference = pd.DataFrame(source_scores).set_index("held_family").sort_index()
            np.testing.assert_allclose(
                family_scores.loc[f"{objective}_rank1"].sort_index()[["mae", "mse"]],
                reference[["mae", "mse"]],
                rtol=0,
                atol=1e-10,
            )
        reference = pd.read_csv(root / "analysis-joint-mae-mse-v001/summary.csv")
        reference = reference[
            (reference.model_id == model)
            & (reference.scope == "unseen_family")
            & (reference.method == "forecast_median_guarded")
        ]
        np.testing.assert_allclose(
            family_scores.loc["forecast_median_guarded"].mean().to_numpy(),
            reference[["mae", "mse"]].to_numpy()[0],
            rtol=0,
            atol=1e-10,
        )
        frame.to_parquet(destination / "episode_results.parquet", index=False)
        _write_json(
            marker,
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "actions": actions,
                "top1_parity": True,
                "top7_parity": True,
                "files": [
                    {"path": name, "sha256": file_sha256(destination / name)}
                    for name in (
                        "student_rankings.npz",
                        "source_fixed_choices.json",
                        "episode_results.parquet",
                    )
                ],
            },
        )
        all_results.append(frame)
        completed.append({"study_id": study_id, "model_id": model, "sha256": file_sha256(marker)})
        print(
            json.dumps({"study_id": study_id, "model_id": model, "status": "completed"}), flush=True
        )
    frame = pd.concat(all_results, ignore_index=True)
    frames = [
        frame.assign(panel="full_development"),
        frame[frame.episode_id.isin(screening_ids)].assign(panel="screening_90"),
    ]
    strata = [
        "panel",
        "model_id",
        "representation",
        "method",
        "aggregation_size",
        "forecast_contexts_required",
    ]
    family = pd.concat(frames).groupby([*strata, "family_id"])[["mae", "mse"]].mean().reset_index()
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(strata)[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": completed,
            "summary": summary.to_dict("records"),
            "new_forecaster_calls": 0,
            "limits": [
                "development results only",
                "candidate-forecast selectors require all seven contexts even when combining fewer outputs; TimesFM processes two target series per context",
                "source imputations were cached; no end-to-end speed claim",
                "uniform_random3_expected averages policy losses, not forecast values",
                "no new-method or significance claim",
            ],
        },
    )
    print(summary[summary.panel == "full_development"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
