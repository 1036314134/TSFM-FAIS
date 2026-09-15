"""Freeze source-trained selectors and fixed controls before native confirmation."""

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
from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.forecast_teacher import forecast_reference_costs  # noqa: E402
from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector  # noqa: E402
from tsfm_fais.routing.preforecast import (  # noqa: E402
    METADATA,
    decision_keys,
    projection_row_costs,
)
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def source_macro(values, metadata):
    families = []
    for family in sorted(metadata.family_id.unique()):
        datasets = []
        for dataset in sorted(metadata[metadata.family_id == family].dataset_id.unique()):
            selected = (metadata.family_id == family) & (metadata.dataset_id == dataset)
            datasets.append(values[selected.to_numpy()].mean(axis=0))
        families.append(np.mean(datasets, axis=0))
    return np.mean(families, axis=0)


def fixed_source_controls(points, truth, teacher, metadata, actions, *, joint):
    mae = source_macro(np.abs(points - truth[:, None]).mean(axis=2), metadata)
    mse = source_macro(np.square(points - truth[:, None]).mean(axis=2), metadata)
    locf = actions.index("locf")
    denominators = {
        "mae": max(float(mae[locf].mean()), 1e-12),
        "mse": max(float(mse[locf].mean()), 1e-12),
    }
    score = 0.5 * (mae / denominators["mae"] + mse / denominators["mse"])
    selected = np.repeat(score.mean(axis=1).argmin(), 2) if joint else score.argmin(axis=0)
    output = {
        "fixed1_joint": [actions[index] for index in selected],
        "joint_denominators": denominators,
    }
    subsets = list(combinations(range(len(actions)), 3))
    teacher_costs, future_costs = [], []
    for subset in subsets:
        forecast = np.median(points[:, subset], axis=1)
        teacher_costs.append(source_macro(np.square(forecast - teacher).mean(axis=1), metadata))
        future_costs.append(source_macro(np.square(forecast - truth).mean(axis=1), metadata))
    for name, values in (
        ("fixed3_clean_forecast_mse", teacher_costs),
        ("fixed3_future_mse", future_costs),
    ):
        values = np.asarray(values)
        selected = np.repeat(values.mean(axis=1).argmin(), 2) if joint else values.argmin(axis=0)
        output[name] = [[actions[index] for index in subsets[number]] for number in selected]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "teacher-root", "cohort", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the frozen source bundle")
    output.mkdir(parents=True, exist_ok=True)
    if args.confirmation_root.exists() and any(args.confirmation_root.glob("*/predictions/*.npz")):
        raise ValueError("confirmation predictions already exist; do not refreeze this cohort")
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    teachers = json.loads((args.teacher_root / "manifest.json").read_text(encoding="utf-8"))
    source_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"] or teachers[
        "identity"
    ]["accuracy_manifest_sha256"] != file_sha256(root / "manifest.json"):
        raise ValueError("source forecasts and complete-history teachers changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    if cohort["forecast_status"] != "not_started" or cohort["task_count"] != 373:
        raise ValueError("freeze against the unevaluated registered cohort")
    table_path = root / "candidate_accuracy.parquet"
    schema = pq.read_schema(table_path).names
    features = pd.read_parquet(
        table_path,
        columns=[name for name in (*METADATA, *FORECAST_FEATURES) if name in schema],
        filters=[("split", "==", "train")],
    )
    features = forecast_response_inputs(
        features[~features.candidate_id.isin(["native_missing", "vendor_missing"])]
    )
    if set(features.family_id) & {row["family_id"] for row in cohort["sources"]}:
        raise ValueError("confirmation families are present in source fitting")
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "teacher_manifest_sha256": file_sha256(args.teacher_root / "manifest.json"),
        "candidate_table_sha256": file_sha256(table_path),
        "cohort_sha256": file_sha256(args.cohort),
        "confirmation_output_root": str(args.confirmation_root.resolve()),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/pairwise_utility.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/forecast_teacher.py",
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/utility.py",
            )
        },
        "source_families": sorted(features.family_id.unique()),
        "source_origins": sorted(features.origin_id.unique()),
        "primary_method": "complete-history forecast teacher; cost-sensitive pairwise ranking; median of top three candidate forecasts",
        "aggregation_size": 3,
        "query_budget": 7,
        "input_scope": "candidate_forecasts",
        "input_recipe": "impute raw history, query frozen forecasters on raw inputs, score using common observed training-prefix mean and population std",
        "context_length": 96,
        "horizon": 96,
        "target_indices": [0, 1],
        "forecaster_artifacts": source["identity"]["config"]["forecaster_artifacts"],
        "objectives": ["clean_forecast_mse", "future_mse"],
        "feature_names": list(FORECAST_FEATURES),
        "seed": 5101,
        "runtime_code_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/replay_preforecast_student.py",
                "scripts/evaluate_timesfm_vendor_missing.py",
                "src/tsfm_fais/forecasting/runner.py",
                "src/tsfm_fais/forecasting/adapters/chronos.py",
                "src/tsfm_fais/forecasting/adapters/timesfm.py",
            )
        },
    }
    if len(identity["source_families"]) != 15 or len(identity["source_origins"]) != 165:
        raise ValueError("the final source cohort changed")
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("the source fitting identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    truth_path = root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("source outcomes changed")
    records, controls = [], {}
    for model in ("chronos2", "timesfm2p5"):
        actions = [
            name
            for name in accuracy["action_orders"][model]
            if name not in {"native_missing", "vendor_missing"}
        ]
        points_path = root / f"{model}_point_z.npy"
        if file_sha256(points_path) != accuracy["prediction_arrays"][points_path.name]:
            raise ValueError("source candidate predictions changed")
        points = np.load(points_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(action) for action in actions]
        ]
        teacher_record = next(row for row in teachers["models"] if row["model_id"] == model)
        teacher_path = args.teacher_root / teacher_record["teacher_file"]
        if (
            file_sha256(teacher_path) != teacher_record["teacher_sha256"]
            or teacher_record["actions"] != actions
        ):
            raise ValueError("source complete-history predictions changed")
        complete = np.load(teacher_path, mmap_mode="r")
        costs = forecast_reference_costs(points, complete)
        data = decision_keys(
            features[
                (features.model_id == model)
                & features.target_slot.isin([-1] if model == "chronos2" else [0, 1])
            ]
        )
        metadata = data.drop_duplicates("episode_index")
        if len(metadata) != 5940:
            raise ValueError("the source training decisions changed")
        for objective in identity["objectives"]:
            label = output / f"{model}_{objective}.json"
            if label.exists():
                record = json.loads(label.read_text(encoding="utf-8"))
                if (
                    record["identity_sha256"] != identity_sha
                    or file_sha256(output / record["model_path"]) != record["model_sha256"]
                ):
                    raise ValueError("a final source model changed")
            else:
                if objective == "clean_forecast_mse":
                    absolute, floor = projection_row_costs(data, costs, actions)
                    fitting = data.assign(loss=absolute - floor)
                else:
                    outcomes = pd.read_parquet(
                        table_path,
                        columns=["episode_id", "target_slot", "candidate_id", "mse"],
                        filters=[("model_id", "==", model), ("split", "==", "train")],
                    ).rename(columns={"episode_id": "source_episode_id"})
                    joined = data.merge(
                        outcomes,
                        on=["source_episode_id", "target_slot", "candidate_id"],
                        validate="one_to_one",
                    )
                    fitting = forecast_response_inputs(joined).assign(loss=joined.mse)
                learner = PairwiseUtilitySelector(
                    mode="classification", seed=5101, n_estimators=160, n_jobs=1
                ).fit(fitting)
                if set(learner.feature_names) != set(FORECAST_FEATURES):
                    raise ValueError("a final selector learned from undeclared features")
                path = output / f"{model}_{objective}.joblib"
                joblib.dump(learner, path, compress=3)
                record = {
                    "identity_sha256": identity_sha,
                    "model_id": model,
                    "objective": objective,
                    "model_path": path.name,
                    "model_sha256": file_sha256(path),
                    "candidate_ids": list(learner.candidate_ids),
                    "baseline_id": learner.baseline_id,
                    "pair_fit_records": learner.fit_records,
                    "source_outcome_supervision": objective == "future_mse",
                    "source_complete_history_supervision": objective == "clean_forecast_mse",
                }
                _write_json(label, record)
            records.append({"path": label.name, "sha256": file_sha256(label)})
        indices = metadata.episode_index.to_numpy(int)
        controls[model] = fixed_source_controls(
            points[indices],
            np.load(truth_path, mmap_mode="r")[indices],
            complete[indices],
            metadata,
            actions,
            joint=model == "chronos2",
        )
        print(json.dumps({"model_id": model, "source_fitting": "completed"}), flush=True)
    _write_json(output / "controls.json", controls)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": records,
            "controls_file": "controls.json",
            "controls_sha256": file_sha256(output / "controls.json"),
            "confirmation_outcomes_read": False,
            "confirmation_forecasts_generated": False,
            "new_forecaster_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
