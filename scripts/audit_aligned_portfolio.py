"""Recompute portfolio labels, replay every held-family model, and check returned forecasts."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import (  # noqa: E402
    current_options,
    decision_truth,
    decision_vectors,
    load_prepared_model,
    option_rows,
)

from tsfm_fais.routing.aligned_portfolio import option_catalog  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "accuracy-root",
        "teacher-root",
        "study-root",
        "portfolio-root",
        "screen-plan",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed portfolio audits")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    teachers = json.loads((args.teacher_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        prep["status"] != "completed"
        or prep["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or prep["identity"]["teacher_manifest_sha256"]
        != file_sha256(args.teacher_root / "manifest.json")
    ):
        raise ValueError("the prepared portfolio sources changed")
    for name, expected in prep["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("feature or prediction code changed after preparation")
    studies, source_hashes = {}, {}
    screen_plan = json.loads(args.screen_plan.read_text(encoding="utf-8"))
    for kind in ("unit_projection", "direct_risk"):
        path = args.study_root / kind / "manifest.json"
        studies[kind] = json.loads(path.read_text(encoding="utf-8"))
        if (
            studies[kind]["status"] != "completed"
            or len(studies[kind]["folds"]) != 30
            or studies[kind]["identity"]["prepared_manifest_sha256"]
            != file_sha256(args.prepared_root / "manifest.json")
        ):
            raise ValueError("complete the two registered 30-fold studies")
        source_hashes[kind] = file_sha256(path)
        if studies[kind]["identity"]["screen_plan_sha256"] != file_sha256(args.screen_plan):
            raise ValueError("the fixed screening panel changed")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("validation outcomes changed")
    truth = np.load(truth_path, mmap_mode="r")
    max_metric_difference, max_label_difference = 0.0, 0.0
    verified_models, verified_decisions, verified_label_rows = 0, 0, 0
    verified_frames = {kind: [] for kind in studies}
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.prepared_root, prep, model)
        names, members = option_catalog(info["actions"])
        if (
            list(names) != info["option_names"]
            or [list(group) for group in members] != info["members"]
        ):
            raise ValueError("the registered option catalog changed")
        point_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("candidate predictions changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in info["actions"]]
        ]
        teacher_record = next(row for row in teachers["models"] if row["model_id"] == model)
        teacher_path = args.teacher_root / teacher_record["teacher_file"]
        if file_sha256(teacher_path) != teacher_record["teacher_sha256"]:
            raise ValueError("source teacher forecasts changed")
        teacher = np.load(teacher_path, mmap_mode="r")
        for start in range(0, len(decisions), 256):
            stop = min(start + 256, len(decisions))
            current = decisions.iloc[start:stop]
            options = current_options(current, bank, info["actions"])
            reference = np.median(decision_vectors(current, bank), axis=1)
            expected = decision_truth(current, teacher)
            option_loss = np.square(options - expected[:, None]).mean(axis=2)
            reference_loss = np.square(reference - expected).mean(axis=1)
            exact = option_loss - reference_loss[:, None]
            norm = np.sqrt(np.square(options - reference[:, None]).mean(axis=2))
            np.testing.assert_allclose(norm, arrays["norm"][start:stop], rtol=0, atol=1e-12)
            reconstructed = norm**2 - 2 * norm * arrays["unit_projection"][start:stop]
            tolerance = 1e-10 + 64 * np.finfo(float).eps * (option_loss + reference_loss[:, None])
            for values in (arrays["direct_risk"][start:stop], reconstructed):
                difference = np.abs(values - exact)
                if np.any(difference > tolerance):
                    raise ValueError("a teacher label no longer matches the actual option risk")
                max_label_difference = max(max_label_difference, float(difference.max()))
            if np.any(arrays["unit_projection"][start:stop][norm <= 1e-12] != 0):
                raise ValueError("zero directions require zero projection labels")
            verified_label_rows += (stop - start) * 43
        for kind, study in studies.items():
            directory = args.study_root / kind
            for fold_entry in study["folds"]:
                path = directory / fold_entry["path"]
                if file_sha256(path) != fold_entry["sha256"]:
                    raise ValueError("a fitted-fold record changed")
                fold = json.loads(path.read_text(encoding="utf-8"))
                if fold["model_id"] != model:
                    continue
                if (
                    fold["identity_sha256"] != study["identity_sha256"]
                    or fold["source_outcome_supervision"]
                ):
                    raise ValueError("fold identity or supervision changed")
                for name in ("model", "prediction", "scores", "choices"):
                    if file_sha256(directory / fold[name + "_path"]) != fold[name + "_sha256"]:
                        raise ValueError("a model, choice, prediction, or score artifact changed")
                family = fold["held_family"]
                training = decisions[(decisions.split == "train") & (decisions.family_id != family)]
                indices = np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                current = decisions.iloc[indices]
                if (
                    sorted(training.origin_id.unique()) != fold["training_origins"]
                    or sorted(training.family_id.unique()) != fold["training_families"]
                    or set(training.origin_id) & set(current.origin_id)
                ):
                    raise ValueError("a fold used another source population")
                learner = joblib.load(directory / fold["model_path"])
                matrix = option_rows(decisions, arrays["features"], info, indices)
                if list(learner._matrix(matrix).columns) != info["feature_names"]:
                    raise ValueError("the fitted input whitelist changed")
                parameters = learner.model.get_params()
                for key, value in {
                    "n_estimators": 160,
                    "num_leaves": 15,
                    "learning_rate": 0.05,
                    "min_child_samples": 25,
                    "reg_lambda": 5.0,
                    "random_state": 5101,
                    "n_jobs": 1,
                }.items():
                    if parameters[key] != value:
                        raise ValueError("the fixed learner settings changed")
                estimates = learner.predict(matrix).reshape(len(indices), 43)
                options = current_options(current, bank, info["actions"])
                reference = np.median(decision_vectors(current, bank), axis=1)
                energy = np.square(options - reference[:, None]).mean(axis=2)
                scores = (
                    energy - 2 * np.sqrt(energy) * estimates
                    if kind == "unit_projection"
                    else estimates.copy()
                )
                scores[energy <= 1e-24] = 0.0
                reported = pd.read_parquet(directory / fold["scores_path"])
                if (
                    len(reported) != len(indices) * 5
                    or reported.duplicated(["episode_id", "method"]).any()
                ):
                    raise ValueError("a decision score is duplicated")
                expected_future = decision_truth(current, truth)
                with np.load(directory / fold["prediction_path"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(indices, saved["decision_indices"])
                    np.testing.assert_allclose(estimates, saved["estimates"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(scores, saved["scores"], rtol=0, atol=1e-12)
                    predictions = {"forecast_median_guarded": reference}
                    menus = {
                        "single": np.arange(7),
                        "triple": np.arange(7, 42),
                        "mixed": np.arange(42),
                        "full": np.array([42, *range(42)]),
                    }
                    for menu, allowed in menus.items():
                        selected = allowed[scores[:, allowed].argmin(axis=1)]
                        np.testing.assert_array_equal(selected, saved["choice_" + menu])
                        predictions[menu] = options[np.arange(len(indices)), selected]
                        np.testing.assert_array_equal(predictions[menu], saved["point_" + menu])
                    np.testing.assert_array_equal(reference, saved["reference"])
                for method, prediction in predictions.items():
                    expected_scores = np.column_stack(
                        [
                            np.abs(prediction - expected_future).mean(axis=1),
                            np.square(prediction - expected_future).mean(axis=1),
                        ]
                    )
                    selected = (
                        reported[reported.method == method]
                        .set_index("episode_id")
                        .loc[current.episode_id]
                    )
                    actual = selected[["mae", "mse"]].to_numpy()
                    np.testing.assert_allclose(actual, expected_scores, rtol=0, atol=1e-10)
                    max_metric_difference = max(
                        max_metric_difference, float(np.abs(actual - expected_scores).max())
                    )
                    verified_frames[kind].append(
                        current.copy().assign(
                            model_id=model,
                            method=method,
                            mae=expected_scores[:, 0],
                            mse=expected_scores[:, 1],
                        )
                    )
                verified_models += 1
                verified_decisions += len(indices)
    if verified_models != 60 or verified_decisions != 11232:
        raise ValueError("the audit did not cover all registered folds and decisions")
    summaries = []
    reference = pd.read_csv(args.portfolio_root / "summary.csv", float_precision="round_trip")
    for kind in studies:
        summary = pd.read_csv(args.study_root / kind / "summary.csv", float_precision="round_trip")
        verified = pd.concat(verified_frames[kind], ignore_index=True)
        window_keys = [
            "model_id",
            "method",
            "source_episode_id",
            "origin_id",
            "family_id",
            "dataset_id",
            "item_id",
        ]
        windows = verified.groupby(window_keys)[["mae", "mse"]].mean().reset_index()
        stored = pd.read_parquet(args.study_root / kind / "episode_results.parquet")
        pd.testing.assert_frame_equal(
            windows.sort_values(window_keys).reset_index(drop=True),
            stored.sort_values(window_keys).reset_index(drop=True),
            check_dtype=False,
        )
        panels = [
            windows.assign(panel="full_development"),
            windows[windows.source_episode_id.isin(screen_plan["decision_episode_ids"])].assign(
                panel="screening_90"
            ),
        ]
        keys = ["panel", "model_id", "method"]
        expected_summary = (
            pd.concat(panels)
            .groupby([*keys, "family_id"])[["mae", "mse"]]
            .mean()
            .groupby(keys)
            .mean()
            .reset_index()
        )
        pd.testing.assert_frame_equal(
            expected_summary.sort_values(keys).reset_index(drop=True),
            summary.sort_values(keys).reset_index(drop=True),
            check_dtype=False,
            rtol=0,
            atol=1e-10,
        )
        for model in ("chronos2", "timesfm2p5"):
            current = summary[
                (summary.panel == "full_development")
                & (summary.model_id == model)
                & (summary.method == "forecast_median_guarded")
            ]
            old = reference[
                (reference.panel == "full_development")
                & (reference.model_id == model)
                & (reference.method == "forecast_median_guarded")
            ]
            np.testing.assert_allclose(
                current[["mae", "mse"]], old[["mae", "mse"]], rtol=0, atol=1e-10
            )
        summary["target_kind"] = kind
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": source_hashes,
            "script_sha256": file_sha256(Path(__file__)),
            "verified_models": verified_models,
            "verified_decisions": verified_decisions,
            "verified_teacher_label_rows": verified_label_rows,
            "maximum_metric_difference": max_metric_difference,
            "maximum_teacher_label_difference": max_label_difference,
            "information_boundary": "current forecasts and descriptors only at choice time; complete-history teacher labels only in source fitting",
            "limits": "source development; no new independent confirmation or superiority claim",
        },
    )
    print(
        summary[
            (summary.panel == "full_development")
            & summary.method.isin(["full", "forecast_median_guarded"])
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
