"""Compare direct teacher-risk regression with normalized residual projection."""

import argparse
import hashlib
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
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    ForecastProjectionRegressor,
    compose_from_estimates,
    projection_targets,
)
from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.preforecast import METADATA, decision_keys  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def vectors_for_decisions(decisions, bank):
    return np.stack(
        [
            bank[row.episode_index].reshape(bank.shape[1], -1)
            if row.target_slot == -1
            else bank[row.episode_index, :, :, row.target_slot]
            for row in decisions.itertuples(index=False)
        ]
    )


def row_positions(frame, decisions, actions):
    positions = {episode: index for index, episode in enumerate(decisions.episode_id)}
    if (
        frame.duplicated(["episode_id", "candidate_id"]).any()
        or not (frame.groupby("episode_id").candidate_id.nunique() == len(actions)).all()
    ):
        raise ValueError("each decision must contain every candidate exactly once")
    return (
        frame.episode_id.map(positions).to_numpy(int),
        frame.candidate_id.map({name: i for i, name in enumerate(actions)}).to_numpy(int),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "teacher-root", "plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed forecast projection experiments")
    output.mkdir(parents=True, exist_ok=True)
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source_manifest_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if file_sha256(source_manifest_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("the episode source changed")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    teacher = json.loads((args.teacher_root / "manifest.json").read_text(encoding="utf-8"))
    if teacher["status"] != "completed" or teacher["identity"][
        "accuracy_manifest_sha256"
    ] != file_sha256(root / "manifest.json"):
        raise ValueError("teacher and candidates have different forecast provenance")
    source_path = root / "candidate_accuracy.parquet"
    schema = pq.read_schema(source_path).names
    if {name for name in schema if name.startswith(("static.", "response."))} != set(
        FORECAST_FEATURES
    ):
        raise ValueError("the original candidate forecast feature inventory changed")
    features = pd.read_parquet(
        source_path, columns=[name for name in (*METADATA, *FORECAST_FEATURES) if name in schema]
    )
    features = forecast_response_inputs(
        features[~features.candidate_id.isin(["native_missing", "vendor_missing"])]
    )
    source_columns = [
        "episode_id",
        "family_id",
        "dataset_id",
        "item_id",
        "origin_id",
        "split",
        "episode_index",
    ]
    declared_episodes = pd.DataFrame(source_manifest["episodes"]).assign(
        episode_index=np.arange(len(source_manifest["episodes"]))
    )
    pd.testing.assert_frame_equal(
        features[source_columns]
        .drop_duplicates()
        .sort_values("episode_index")
        .reset_index(drop=True),
        declared_episodes[source_columns].sort_values("episode_index").reset_index(drop=True),
        check_dtype=False,
    )
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "teacher_manifest_sha256": file_sha256(args.teacher_root / "manifest.json"),
        "candidate_table_sha256": file_sha256(source_path),
        "plan_sha256": file_sha256(args.plan),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/forecast_projection.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/utility.py",
            )
        },
        "target_kinds": ["direct_risk", "raw_projection", "unit_projection"],
        "source_supervision": "complete source-history forecast; no actual source outcomes",
        "anchor": "coordinate median of the seven queried candidate forecasts",
        "learner": "one shared 160-tree regressor per fold, 33 declared features plus seven action indicators; 15 leaves, rate .05, L2 5, minimum leaf 25, seed 5101",
        "query_budget": 7,
        "input_recipe": "R4 raw inputs; shared training-prefix-standardized scores",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("forecast projection experiment identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name in identity["source_sha256"]:
        (output / (Path(name).stem + "_snapshot.py")).write_bytes((ROOT / name).read_bytes())
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    screening = set(plan["decision_episode_ids"])
    if (
        len(screening) != 90
        or plan["source_manifest_sha256"] != accuracy["source_episode_manifest_sha256"]
    ):
        raise ValueError("use the original fixed screening panel")
    all_scores, folds = [], []
    for model in ("chronos2", "timesfm2p5"):
        actions = sorted(
            set(accuracy["action_orders"][model]) - {"native_missing", "vendor_missing"}
        )
        path = root / f"{model}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("cached candidate forecasts changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in actions]
        ]
        record = next(row for row in teacher["models"] if row["model_id"] == model)
        teacher_path = args.teacher_root / record["teacher_file"]
        if file_sha256(teacher_path) != record["teacher_sha256"]:
            raise ValueError("complete-history teacher predictions changed")
        teacher_bank = np.load(teacher_path, mmap_mode="r")[:, None]
        view = features[
            (features.model_id == model)
            & features.target_slot.isin([-1] if model == "chronos2" else [0, 1])
        ]
        for family in sorted(view.family_id.unique()):
            training = decision_keys(view[(view.split == "train") & (view.family_id != family)])
            evaluation = decision_keys(
                view[(view.split == "validation") & (view.family_id == family)]
            )
            if set(training.origin_id) & set(evaluation.origin_id) or family in set(
                training.family_id
            ):
                raise ValueError("training used an evaluation history or family")
            train_decisions, eval_decisions = (
                training.drop_duplicates("episode_id"),
                evaluation.drop_duplicates("episode_id"),
            )
            train_rows, train_actions = row_positions(training, train_decisions, actions)
            eval_rows, eval_actions = row_positions(evaluation, eval_decisions, actions)
            labels = projection_targets(
                vectors_for_decisions(train_decisions, bank),
                vectors_for_decisions(train_decisions, teacher_bank)[:, 0],
            )
            current_points = vectors_for_decisions(eval_decisions, bank)
            for target_kind in identity["target_kinds"]:
                name = hashlib.sha256(f"{model}|{family}|{target_kind}".encode()).hexdigest()[:24]
                directory = output / "folds"
                directory.mkdir(exist_ok=True)
                marker = directory / f"{name}.json"
                if marker.exists():
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    if saved["identity_sha256"] != identity_sha:
                        raise ValueError("a forecast projection checkpoint changed identity")
                    for kind in ("model", "predictions", "scores"):
                        if file_sha256(output / saved[kind + "_path"]) != saved[kind + "_sha256"]:
                            raise ValueError("a forecast projection checkpoint changed")
                else:
                    started = monotonic()
                    learner = ForecastProjectionRegressor().fit(
                        training, labels[target_kind][train_rows, train_actions]
                    )
                    estimates = np.empty((len(eval_decisions), len(actions)))
                    estimates[eval_rows, eval_actions] = learner.predict(evaluation)
                    point, weights, gap, scale = compose_from_estimates(
                        current_points, estimates, target_kind=target_kind
                    )
                    model_path, prediction_path = (
                        directory / f"{name}.joblib",
                        directory / f"{name}.npz",
                    )
                    joblib.dump(learner, model_path, compress=3)
                    _save_npz(
                        prediction_path,
                        decision_ids=eval_decisions.episode_id.to_numpy(dtype=str),
                        episode_indices=eval_decisions.episode_index.to_numpy(int),
                        target_slots=eval_decisions.target_slot.to_numpy(int),
                        point_z=point,
                        weights=weights,
                        estimates=estimates,
                        normalized_duality_gap=gap,
                        objective_scale=scale,
                        identity_sha256=np.asarray(identity_sha),
                    )
                    # True futures are used only after all weights and predictions have been saved.
                    truth_path = root / "truth_z.npy"
                    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                        raise ValueError("cached standardized outcomes changed")
                    truth = vectors_for_decisions(
                        eval_decisions, np.load(truth_path, mmap_mode="r")[:, None]
                    )[:, 0]
                    metadata_columns = [
                        "episode_id",
                        "source_episode_id",
                        "origin_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "target_slot",
                    ]
                    records = []
                    for method, values in (
                        (target_kind, point),
                        ("forecast_median_guarded", np.median(current_points, axis=1)),
                        ("forecast_mean_guarded", np.mean(current_points, axis=1)),
                    ):
                        error = values - truth
                        records.append(
                            eval_decisions[metadata_columns]
                            .copy()
                            .assign(
                                model_id=model,
                                method=method,
                                mae=np.abs(error).mean(axis=1),
                                mse=np.square(error).mean(axis=1),
                            )
                        )
                    scores = pd.concat(records, ignore_index=True)
                    score_path = directory / f"{name}.scores.parquet"
                    scores.to_parquet(score_path, index=False)
                    saved = {
                        "identity_sha256": identity_sha,
                        "model_id": model,
                        "held_family": family,
                        "target_kind": target_kind,
                        "training_origins": sorted(training.origin_id.unique()),
                        "feature_names": list(learner.feature_names),
                        "candidate_ids": list(learner.candidate_ids),
                        "source_outcome_supervision": False,
                        "source_complete_history_supervision": True,
                        "maximum_normalized_duality_gap": float(gap.max()),
                        "maximum_absolute_duality_gap": float((gap * scale).max()),
                        "windows": eval_decisions.source_episode_id.nunique(),
                        "seconds": monotonic() - started,
                        **{
                            kind + suffix: value
                            for kind, file in (
                                ("model", model_path),
                                ("predictions", prediction_path),
                                ("scores", score_path),
                            )
                            for suffix, value in (
                                ("_path", str(file.relative_to(output))),
                                ("_sha256", file_sha256(file)),
                            )
                        },
                    }
                    _write_json(marker, saved)
                scored = pd.read_parquet(output / saved["scores_path"])
                # Fixed controls are shared by both target kinds; store them once per fold.
                all_scores.append(
                    scored
                    if target_kind == identity["target_kinds"][0]
                    else scored[scored.method == target_kind]
                )
                folds.append(
                    {"path": str(marker.relative_to(output)), "sha256": file_sha256(marker)}
                )
                _write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "completed_folds": len(folds),
                        "total_folds": 90,
                        "model_id": model,
                        "family": family,
                        "target_kind": target_kind,
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_folds": len(folds),
                            "total_folds": 90,
                            "model_id": model,
                            "target_kind": target_kind,
                        }
                    ),
                    flush=True,
                )
    scores = pd.concat(all_scores, ignore_index=True)
    windows = (
        scores.groupby(
            [
                "model_id",
                "method",
                "source_episode_id",
                "origin_id",
                "family_id",
                "dataset_id",
                "item_id",
            ]
        )[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    for _, group in windows.groupby(["model_id", "method"]):
        if len(group) != 1872 or group.source_episode_id.nunique() != 1872:
            raise ValueError("forecast projection results lack complete development coverage")
    windows.to_parquet(output / "episode_results.parquet", index=False)
    panels = [
        windows.assign(panel="full_development"),
        windows[windows.source_episode_id.isin(screening)].assign(panel="screening_90"),
    ]
    family = (
        pd.concat(panels)
        .groupby(["panel", "model_id", "method", "family_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(["panel", "model_id", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    prior = pd.read_csv(root / "analysis-joint-mae-mse-v001/summary.csv")
    for model in ("chronos2", "timesfm2p5"):
        for method in ("forecast_median_guarded", "forecast_mean_guarded"):
            measured = summary[
                (summary.panel == "full_development")
                & (summary.model_id == model)
                & (summary.method == method)
            ][["mae", "mse"]].to_numpy()
            reference = prior[
                (prior.scope == "unseen_family")
                & (prior.model_id == model)
                & (prior.method == method)
            ][["mae", "mse"]].to_numpy()
            if measured.shape != (1, 2) or reference.shape != (1, 2):
                raise ValueError("fixed forecast controls must have unique matched references")
            np.testing.assert_allclose(measured, reference, rtol=0, atol=1e-10)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": folds,
            "summary": summary.to_dict("records"),
            "new_forecaster_calls": 0,
            "deployment_forecast_contexts": 7,
            "limits": [
                "development results only",
                "teacher outputs are privileged source supervision",
                "convex combinations include the already-computed forecast median, not one imputed history",
                "numeric optimizer certificates do not certify real forecast risk or model novelty",
            ],
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_folds": len(folds), "total_folds": 90},
    )
    print(summary[summary.panel == "full_development"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
