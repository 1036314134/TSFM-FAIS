"""Compare forecast-objective selection with and without target-history labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.routing.utility import UtilitySelector, family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def attach_objective(frame, objective, denominators):
    result = frame.copy()
    result["source_episode_id"] = result.episode_id
    if result.target_slot.min() >= 0:
        result["episode_id"] = result.episode_id + "|target=" + result.target_slot.astype(str)
    result["loss"] = (
        result[objective]
        if objective in {"mae", "mse"}
        else 0.5 * (result.mae / denominators["mae"] + result.mse / denominators["mse"])
    )
    return result


def choose_scores(frame, scores):
    scored = frame.assign(
        score=np.asarray(scores), reference_priority=(frame.candidate_id != "locf").astype(int)
    )
    return scored.sort_values(
        ["episode_id", "score", "reference_priority", "candidate_id"]
    ).drop_duplicates("episode_id")


def blend_history(selector, history, evaluation, weight=0.5):
    keys = ["candidate_id", "target_slot"]
    means = history.groupby(keys).loss.mean()
    reference = history[history.candidate_id == "locf"].groupby("target_slot").loss.mean()
    adjustment = np.array(
        [
            means.loc[(row.candidate_id, row.target_slot)] - reference.loc[row.target_slot]
            for row in evaluation.itertuples()
        ]
    )
    return choose_scores(
        evaluation, (1 - weight) * selector.scores(evaluation) + weight * adjustment
    )


def mse_ensemble_weights(errors):
    """Fit nonnegative, sum-one forecast weights using historical residuals."""
    matrix = np.asarray(errors, float)
    gram = matrix.T @ matrix / len(matrix)
    normalized = gram / max(float(np.max(np.diag(gram))), 1e-12)
    n = gram.shape[0]
    result = minimize(
        lambda w: float(w @ normalized @ w),
        np.full(n, 1 / n),
        jac=lambda w: 2 * normalized @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={
            "type": "eq",
            "fun": lambda w: float(w.sum() - 1),
            "jac": lambda w: np.ones(n),
        },
        options={"ftol": 1e-10, "maxiter": 300},
    )
    alternatives = [np.full(n, 1 / n), np.eye(n)[int(np.argmin(np.diag(gram)))]]
    if np.isfinite(result.x).all() and result.x.min() >= -1e-7 and abs(result.x.sum() - 1) < 1e-6:
        w = np.maximum(result.x, 0)
        alternatives.append(w / w.sum())
    weights = min(alternatives, key=lambda w: float(w @ gram @ w))
    return weights, bool(result.success)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--objectives", default="joint")
    args = parser.parse_args()
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[-1]])
    root = args.run_root.resolve()
    objectives = args.objectives.split(",")
    if set(objectives) - {"joint", "mae", "mse"}:
        parser.error("objectives must be joint, mae, or mse")
    output = root / ("analysis-" + "-".join(objectives) + "-v001")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("analysis already completed; preserve its evidence")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(root / "candidate_accuracy.parquet")
    controls = pd.read_parquet(root / "control_accuracy.parquet")
    results, folds, weight_records = [], [], []

    def record(selected, model, method, scope, objective):
        data = selected.copy()
        if "source_episode_id" in data:
            data["episode_id"] = data.source_episode_id
        grouped = (
            data.groupby(["episode_id", "origin_id", "episode_index", "family_id", "dataset_id"])[
                ["mae", "mse"]
            ]
            .mean()
            .reset_index()
        )
        grouped["model_id"], grouped["method"], grouped["scope"], grouped["objective"] = (
            model,
            method,
            scope,
            objective,
        )
        results.append(grouped)

    for model in sorted(frame.model_id.unique()):
        all_model = frame[frame.model_id == model]
        data = all_model[~all_model.candidate_id.isin(["native_missing", "vendor_missing"])]
        sequence = data[data.target_slot == -1]
        for candidate, group in all_model[
            (all_model.split == "validation") & (all_model.target_slot == -1)
        ].groupby("candidate_id"):
            record(group, model, "fixed_" + candidate, "unseen_family", "none")
        for method, group in controls[
            (controls.model_id == model)
            & (controls.split == "validation")
            & (controls.target_slot == -1)
        ].groupby("method"):
            scope = (
                "unavailable_clean_context"
                if method == "clean"
                else "superseded_layout_diagnostic"
                if method.endswith("_legacy")
                else "unseen_family"
            )
            record(group, model, method, scope, "none")
        for family in sorted(data.family_id.unique()):
            train_sequence = sequence[(sequence.family_id != family) & (sequence.split == "train")]
            denominator = {
                metric: max(
                    family_macro(train_sequence[train_sequence.candidate_id == "locf"], metric),
                    1e-12,
                )
                for metric in ("mae", "mse")
            }
            history_base = sequence[(sequence.family_id == family) & (sequence.split == "train")]
            evaluate_base = sequence[
                (sequence.family_id == family) & (sequence.split == "validation")
            ]
            if history_base.origin.max() >= evaluate_base.origin.min():
                raise ValueError("target history must precede all evaluation origins")
            for objective in objectives:
                training = attach_objective(train_sequence, objective, denominator)
                validation = attach_objective(evaluate_base, objective, denominator)
                history = attach_objective(history_base, objective, denominator)
                fixed = min(
                    training.candidate_id.unique(),
                    key=lambda action: family_macro(training[training.candidate_id == action]),
                )
                record(
                    validation[validation.candidate_id == fixed],
                    model,
                    "train_best_fixed",
                    "unseen_family",
                    objective,
                )
                local_fixed = history.groupby("candidate_id").loss.mean().idxmin()
                record(
                    validation[validation.candidate_id == local_fixed],
                    model,
                    "history_best_fixed",
                    "complete_target_history_labels",
                    objective,
                )
                for granularity, slots in [
                    ("sequence", [-1]),
                    *([("target", [0, 1])] if model == "timesfm2p5" else []),
                ]:
                    view = data[data.target_slot.isin(slots)]
                    training = attach_objective(
                        view[(view.family_id != family) & (view.split == "train")],
                        objective,
                        denominator,
                    )
                    calibration = attach_objective(
                        view[(view.family_id != family) & (view.split == "validation")],
                        objective,
                        denominator,
                    )
                    validation = attach_objective(
                        view[(view.family_id == family) & (view.split == "validation")],
                        objective,
                        denominator,
                    )
                    history = attach_objective(
                        view[(view.family_id == family) & (view.split == "train")],
                        objective,
                        denominator,
                    )
                    if set(training.origin_id) & set(calibration.origin_id) or set(
                        history.origin_id
                    ) & set(validation.origin_id):
                        raise ValueError("training/calibration/evaluation origins overlap")
                    if granularity == "target":
                        chosen = pd.concat(
                            [
                                group[
                                    group.candidate_id
                                    == history[history.target_slot == slot]
                                    .groupby("candidate_id")
                                    .loss.mean()
                                    .idxmin()
                                ]
                                for slot, group in validation.groupby("target_slot")
                            ]
                        )
                        record(
                            chosen,
                            model,
                            "history_target_fixed",
                            "complete_target_history_labels",
                            objective,
                        )
                    for use_response in (False, True):
                        name = granularity + ("_response" if use_response else "_static")
                        selector = (
                            UtilitySelector(
                                use_response=use_response,
                                objective="regression",
                                reference_id="locf",
                                n_jobs=1,
                            )
                            .fit(training)
                            .calibrate(calibration)
                        )
                        record(selector.select(validation), model, name, "unseen_family", objective)
                        record(
                            selector.select(validation, gated=True),
                            model,
                            name + "_gated",
                            "unseen_family",
                            objective,
                        )
                        record(
                            blend_history(selector, history, validation),
                            model,
                            name + "_history_blend",
                            "complete_target_history_labels",
                            objective,
                        )
                        if use_response:
                            local = UtilitySelector(
                                use_response=True,
                                objective="regression",
                                reference_id="locf",
                                n_jobs=1,
                            ).fit(history)
                            record(
                                local.select(validation),
                                model,
                                name + "_history_only",
                                "complete_target_history_labels",
                                objective,
                            )
                        folds.append(
                            {
                                "model_id": model,
                                "held_family": family,
                                "objective": objective,
                                "method": name,
                                "source_training_families": sorted(training.family_id.unique()),
                                "history_origins": int(history.origin_id.nunique()),
                                "evaluation_origins": int(validation.origin_id.nunique()),
                                "normalizers": denominator,
                                "calibration": selector.calibration,
                            }
                        )
            print(json.dumps({"model": model, "family": family, "status": "selected"}), flush=True)
        # A strong comparator with the same target-history labels. Its forecast
        # combination is evaluated separately from imputation-level selection.
        point = np.load(root / f"{model}_point_z.npy", mmap_mode="r")
        truth = np.load(root / "truth_z.npy", mmap_mode="r")
        orders = manifest["action_orders"][model]
        for family in sorted(sequence.family_id.unique()):
            history = sequence[
                (sequence.family_id == family) & (sequence.split == "train")
            ].drop_duplicates("episode_id")
            evaluate = sequence[
                (sequence.family_id == family) & (sequence.split == "validation")
            ].drop_duplicates("episode_id")
            train_indices, evaluate_indices = (
                history.episode_index.to_numpy(),
                evaluate.episode_index.to_numpy(),
            )
            for pool_name, actions in [
                (
                    "finite",
                    [
                        name
                        for name in orders
                        if name not in {"native_missing", "vendor_missing", "guarded_direct"}
                    ],
                ),
                (
                    "guarded",
                    [name for name in orders if name not in {"native_missing", "vendor_missing"}],
                ),
            ]:
                positions = [orders.index(action) for action in actions]
                residual = point[train_indices][:, positions] - truth[train_indices, None]
                residual = residual.transpose(0, 2, 3, 1).reshape(-1, len(actions))
                weights, success = mse_ensemble_weights(residual)
                prediction = np.einsum(
                    "nahk,a->nhk", point[evaluate_indices][:, positions], weights
                )
                errors = prediction - truth[evaluate_indices]
                selected = evaluate.assign(
                    mae=np.mean(np.abs(errors), axis=(1, 2)), mse=np.mean(errors**2, axis=(1, 2))
                )
                record(
                    selected,
                    model,
                    "history_weighted_forecast_" + pool_name,
                    "complete_target_history_labels",
                    "mse",
                )
                weight_records.append(
                    {
                        "model_id": model,
                        "family_id": family,
                        "pool": pool_name,
                        "weights": dict(zip(actions, weights.tolist(), strict=True)),
                        "solver_success": success,
                        "train_origin_count": int(history.origin_id.nunique()),
                    }
                )
    result = pd.concat(results, ignore_index=True)
    keys = ["model_id", "scope", "objective", "method"]
    if result.duplicated(keys + ["episode_id"]).any():
        raise ValueError("duplicate selected episode results")
    family = (
        result.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    result.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "folds.json", folds)
    _write_json(output / "history_ensemble_weights.json", weight_records)
    _write_json(
        output / "manifest.json",
        {
            "evidence_role": "development",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "selector_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/utility.py"),
            "objectives": objectives,
            "history_blend_weight": 0.5,
            "primary_metrics": ["mae", "mse"],
            "scope_note": "unseen_family uses supervised source-family losses but no target-family loss labels; complete_target_history_labels additionally uses all earlier clean target-family outcomes, including values hidden in the synthetic observation trajectory, and is an information-rich reference rather than an observed-feedback deployment method; target-level cached assembly is valid only for independent TimesFM predictions; clean-context and old-layout diagnostics are labeled separately",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
