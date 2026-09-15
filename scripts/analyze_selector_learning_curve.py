"""Test fixed utility-selector learning curves using cached independent histories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_downstream_accuracy import attach_objective  # noqa: E402

from tsfm_fais.routing.origin_sampling import nested_origin_ids  # noqa: E402
from tsfm_fais.routing.utility import UtilitySelector, family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def scored_selection(selector, evaluation):
    # Explicitly remove current outcomes from the decision input.
    choices = selector.select(evaluation.drop(columns=["mae", "mse", "loss"]))[
        ["episode_id", "candidate_id", "predicted_delta"]
    ]
    return choices.merge(evaluation, on=["episode_id", "candidate_id"], validate="one_to_one")


def reduce_targets(selected):
    return (
        selected.groupby(
            ["source_episode_id", "origin_id", "family_id", "dataset_id", "item_id"], sort=False
        )[["mae", "mse"]]
        .mean()
        .reset_index()
    )


def metric_row(selected, metadata, *, scope, method):
    windows = reduce_targets(selected)
    return {
        **metadata,
        "scope": scope,
        "method": method,
        "windows": len(windows),
        "mae": family_macro(windows, "mae"),
        "mse": family_macro(windows, "mse"),
    }


def verify_full_reference(rows, reference):
    results = pd.DataFrame(rows)
    full = results[
        (results.fraction == 1)
        & (results.scope == "held_family")
        & (results.method == "response_selector")
    ]
    comparisons = []
    for model, method in (("chronos2", "sequence_response"), ("timesfm2p5", "target_response")):
        model_rows = full[full.model_id == model]
        if len(model_rows) != 15 or model_rows.held_family.nunique() != 15:
            raise ValueError("full-source reference requires all 15 held-out families")
        expected = reference[
            (reference.model_id == model)
            & (reference.scope == "unseen_family")
            & (reference.objective == "joint")
            & (reference.method == method)
        ]
        if len(expected) != 1:
            raise ValueError("the prior full-source reference is not unique")
        actual = model_rows[["mae", "mse"]].mean().to_numpy()
        target = expected[["mae", "mse"]].to_numpy()[0]
        np.testing.assert_allclose(actual, target, rtol=0, atol=1e-9)
        comparisons.append(
            {
                "model_id": model,
                "mae": actual[0],
                "mse": actual[1],
                "maximum_difference": float(np.abs(actual - target).max()),
            }
        )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed learning curves")
    output.mkdir(parents=True, exist_ok=True)
    reference_root = root / "analysis-joint-mae-mse-v001"
    reference_manifest = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
    selector_path = ROOT / "src/tsfm_fais/routing/utility.py"
    if reference_manifest["selector_sha256"] != file_sha256(selector_path):
        raise ValueError("the fixed selector differs from its reference implementation")
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "candidate_table_sha256": file_sha256(root / "candidate_accuracy.parquet"),
        "control_table_sha256": file_sha256(root / "control_accuracy.parquet"),
        "reference_summary_sha256": file_sha256(reference_root / "summary.csv"),
        "script_sha256": file_sha256(Path(__file__)),
        "selector_sha256": file_sha256(selector_path),
        "sampling_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/origin_sampling.py"),
        "objective_source_sha256": file_sha256(ROOT / "scripts/analyze_downstream_accuracy.py"),
        "fractions": [0.25, 0.5, 1.0],
        "sampling_seeds": [6101, 6102, 6103],
        "model_seed": 5101,
        "n_estimators": 160,
        "reference_id": "locf",
        "objective": "regression",
        "decision_granularity": {"chronos2": "whole context", "timesfm2p5": "independent target"},
        "metric": "prefix-standardized downstream MAE/MSE; original R4 raw-input forecast recipes",
        "normalizers": "two full-source LOCF macro errors fixed across fractions; fit-row sample-size study, not total-label-budget study",
        "evaluation": "leave one family out plus source-family later-time diagnostic; no threshold calibration",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lightgbm", "numpy", "pandas", "scikit-learn")
        },
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("learning-curve identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "selector_snapshot.py").write_bytes(selector_path.read_bytes())
    (output / "sampling_snapshot.py").write_bytes(
        (ROOT / "src/tsfm_fais/routing/origin_sampling.py").read_bytes()
    )
    frame = pd.read_parquet(root / "candidate_accuracy.parquet")
    frame = frame[~frame.candidate_id.isin(["native_missing", "vendor_missing"])]
    metadata = frame[
        ["origin_id", "family_id", "dataset_id", "item_id", "origin", "split"]
    ].drop_duplicates()
    if (
        metadata.origin_id.duplicated().any()
        or metadata[metadata.split == "train"].origin_id.nunique() != 165
    ):
        raise ValueError("source histories have unexpected identities or coverage")
    for _, group in metadata.groupby(["dataset_id", "item_id"]):
        if (
            group[group.split == "train"].origin.max() + 96
            > group[group.split == "validation"].origin.min() - 96
        ):
            raise ValueError("training future overlaps a validation context")
    reference = pd.read_csv(reference_root / "summary.csv")
    rows, files, budgets = [], [], []
    conditions = [
        (1.0, 6101),
        *[(fraction, seed) for fraction in (0.25, 0.5) for seed in identity["sampling_seeds"]],
    ]
    done = 0
    for fraction, seed in conditions:
        for model in ("chronos2", "timesfm2p5"):
            model_data = frame[frame.model_id == model]
            sequence = model_data[model_data.target_slot == -1]
            view = model_data[model_data.target_slot.isin([-1] if model == "chronos2" else [0, 1])]
            for held_family in sorted(view.family_id.unique()):
                tag = f"{model}|{held_family}|{fraction}|{seed}"
                name = hashlib.sha256(tag.encode()).hexdigest()[:24]
                cache = output / "folds" / f"{name}.json"
                if cache.exists():
                    cached = json.loads(cache.read_text(encoding="utf-8"))
                    if cached["identity_sha256"] != identity_sha:
                        raise ValueError("a completed fold has different provenance")
                    for kind in ("model", "predictions"):
                        if file_sha256(output / cached[kind + "_path"]) != cached[kind + "_sha256"]:
                            raise ValueError("a completed fold artifact changed")
                else:
                    source_sequence = sequence[
                        (sequence.split == "train") & (sequence.family_id != held_family)
                    ]
                    denominators = {
                        key: max(
                            family_macro(
                                source_sequence[source_sequence.candidate_id == "locf"], key
                            ),
                            1e-12,
                        )
                        for key in ("mae", "mse")
                    }
                    full_training = attach_objective(
                        view[(view.split == "train") & (view.family_id != held_family)],
                        "joint",
                        denominators,
                    )
                    chosen = nested_origin_ids(full_training, fraction, seed)
                    training = full_training[full_training.origin_id.isin(chosen)]
                    held = attach_objective(
                        view[(view.split == "validation") & (view.family_id == held_family)],
                        "joint",
                        denominators,
                    )
                    source_validation = attach_objective(
                        view[(view.split == "validation") & (view.family_id != held_family)],
                        "joint",
                        denominators,
                    )
                    if held_family in set(training.family_id) or set(chosen) & (
                        set(held.origin_id) | set(source_validation.origin_id)
                    ):
                        raise ValueError("training origins or families leaked into evaluation")
                    started = monotonic()
                    selector = UtilitySelector(
                        use_response=True,
                        objective="regression",
                        reference_id="locf",
                        n_jobs=1,
                        seed=5101,
                        n_estimators=160,
                    ).fit(training)
                    fixed = min(
                        training.candidate_id.unique(),
                        key=lambda action: family_macro(training[training.candidate_id == action]),
                    )
                    info = {
                        "model_id": model,
                        "held_family": held_family,
                        "fraction": fraction,
                        "sampling_seed": seed,
                        "fit_origins": len(chosen),
                        "available_source_origins": full_training.origin_id.nunique(),
                        "fit_decisions": training.episode_id.nunique(),
                        "fit_candidate_rows": len(training),
                    }
                    fold_rows = []
                    held_selected = None
                    for scope, evaluate in (
                        ("held_family", held),
                        ("source_temporal_validation", source_validation),
                    ):
                        selected = scored_selection(selector, evaluate)
                        fold_rows.append(
                            metric_row(selected, info, scope=scope, method="response_selector")
                        )
                        fold_rows.append(
                            metric_row(
                                evaluate[evaluate.candidate_id == fixed],
                                info,
                                scope=scope,
                                method="fit_best_fixed",
                            )
                        )
                        if scope == "held_family":
                            held_selected = selected[
                                [
                                    "episode_id",
                                    "source_episode_id",
                                    "origin_id",
                                    "family_id",
                                    "dataset_id",
                                    "item_id",
                                    "candidate_id",
                                    "target_slot",
                                    "predicted_delta",
                                    "mae",
                                    "mse",
                                ]
                            ]
                    prediction_path = output / "folds" / f"{name}.parquet"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    held_selected.to_parquet(prediction_path, index=False)
                    model_path = output / "folds" / f"{name}.model.txt"
                    selector.model.booster_.save_model(str(model_path))
                    cached = {
                        "identity_sha256": identity_sha,
                        "metadata": info,
                        "rows": fold_rows,
                        "training_origins": chosen,
                        "normalizers": denominators,
                        "fixed_action": fixed,
                        "feature_names": selector.feature_names,
                        "candidate_ids": selector.candidate_ids,
                        "reference_id": selector.baseline_id,
                        "fit_and_score_seconds": monotonic() - started,
                        "model_path": str(model_path.relative_to(output)),
                        "model_sha256": file_sha256(model_path),
                        "predictions_path": str(prediction_path.relative_to(output)),
                        "predictions_sha256": file_sha256(prediction_path),
                    }
                    _write_json(cache, cached)
                rows.extend(cached["rows"])
                budgets.append(
                    {**cached["metadata"], "fit_and_score_seconds": cached["fit_and_score_seconds"]}
                )
                files.append({"path": str(cache.relative_to(output)), "sha256": file_sha256(cache)})
                done += 1
                _write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "completed_fits": done,
                        "total_fits": 210,
                        "last_model": model,
                        "last_family": held_family,
                        "fraction": fraction,
                        "sampling_seed": seed,
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_fits": done,
                            "total_fits": 210,
                            "model": model,
                            "fraction": fraction,
                            "seed": seed,
                            "family": held_family,
                        }
                    ),
                    flush=True,
                )
        if fraction == 1:
            _write_json(
                output / "full_reference_check.json",
                {"status": "passed", "comparisons": verify_full_reference(rows, reference)},
            )
    results = pd.DataFrame(rows)
    results.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(budgets).to_csv(output / "fit_budgets.csv", index=False)
    summary = (
        results.groupby(["model_id", "scope", "method", "fraction", "sampling_seed"])[
            ["mae", "mse", "fit_origins"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "learning_curve.csv", index=False)
    controls = pd.read_parquet(root / "control_accuracy.parquet")
    controls = controls[
        (controls.split == "validation")
        & (controls.target_slot == -1)
        & (controls.method == "forecast_median_guarded")
    ]
    baselines = [
        {
            "model_id": model,
            "method": "forecast_median_guarded",
            "mae": family_macro(group, "mae"),
            "mse": family_macro(group, "mse"),
        }
        for model, group in controls.groupby("model_id")
    ]
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "fitted_models": len(files),
            "folds": files,
            "strong_baselines": baselines,
            "full_reference": verify_full_reference(rows, reference),
            "limitations": [
                "fit-row fractions share two full-source scalar loss normalizers",
                "100 percent is identical across sampling seeds and fitted once per fold",
                "source validation overlaps across held-family folds; descriptive diagnostic only",
                "uses R4 raw-input recipes; cannot be combined with R5 input-standardized method rows",
                "no extra GPU inference and no independent confirmation",
            ],
        },
    )
    _write_json(
        output / "progress.json", {"status": "completed", "completed_fits": done, "total_fits": 210}
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
