"""Compare pairwise regression and cost-sensitive voting on frozen forecast caches."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_downstream_accuracy import attach_objective  # noqa: E402
from analyze_selector_learning_curve import reduce_targets  # noqa: E402

from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector  # noqa: E402
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed pairwise results")
    output.mkdir(parents=True, exist_ok=True)
    module_path = ROOT / "src/tsfm_fais/routing/pairwise_utility.py"
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "candidate_table_sha256": file_sha256(root / "candidate_accuracy.parquet"),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(module_path),
        "helpers_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/analyze_downstream_accuracy.py",
                "scripts/analyze_selector_learning_curve.py",
                "src/tsfm_fais/routing/utility.py",
            )
        },
        "seed": 5101,
        "modes": ["classification", "regression"],
        "n_estimators": 160,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 25,
        "reg_lambda": 5.0,
        "vote": "hard pairwise wins; ties use the source-training best fixed action",
        "information": "same R4 raw-input forecasts, candidates, observable features, source labels and prefix-standardized outcomes",
        "scope": "established algorithm-selection references; no new-method or confirmation claim",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lightgbm", "numpy", "pandas", "joblib")
        },
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("pairwise experiment identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "module_snapshot.py").write_bytes(module_path.read_bytes())
    frame = pd.read_parquet(root / "candidate_accuracy.parquet")
    frame = frame[~frame.candidate_id.isin(["native_missing", "vendor_missing"])]
    results, files = [], []
    for model in ("chronos2", "timesfm2p5"):
        model_data = frame[frame.model_id == model]
        sequence = model_data[model_data.target_slot == -1]
        view = model_data[model_data.target_slot.isin([-1] if model == "chronos2" else [0, 1])]
        for family in sorted(view.family_id.unique()):
            train_sequence = sequence[(sequence.split == "train") & (sequence.family_id != family)]
            denominator = {
                key: max(
                    family_macro(train_sequence[train_sequence.candidate_id == "locf"], key), 1e-12
                )
                for key in ("mae", "mse")
            }
            training = attach_objective(
                view[(view.split == "train") & (view.family_id != family)], "joint", denominator
            )
            evaluation = attach_objective(
                view[(view.split == "validation") & (view.family_id == family)],
                "joint",
                denominator,
            )
            if set(training.origin_id) & set(evaluation.origin_id) or family in set(
                training.family_id
            ):
                raise ValueError("training histories or families leaked into evaluation")
            for mode in identity["modes"]:
                name = hashlib.sha256(f"{model}|{family}|{mode}".encode()).hexdigest()[:24]
                directory = output / "folds"
                directory.mkdir(exist_ok=True)
                cache = directory / f"{name}.json"
                if cache.exists():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    if saved["identity_sha256"] != identity_sha:
                        raise ValueError("a pairwise fold belongs to another experiment")
                    for kind in ("model", "predictions"):
                        if file_sha256(output / saved[kind + "_path"]) != saved[kind + "_sha256"]:
                            raise ValueError("a pairwise artifact changed")
                else:
                    started = monotonic()
                    selector = PairwiseUtilitySelector(
                        mode=mode, n_estimators=160, seed=5101, n_jobs=1
                    ).fit(training)
                    fitted = monotonic()
                    decisions = selector.select(evaluation.drop(columns=["mae", "mse", "loss"]))[
                        ["episode_id", "candidate_id", "pairwise_wins"]
                    ]
                    selected = decisions.merge(
                        evaluation, on=["episode_id", "candidate_id"], validate="one_to_one"
                    )
                    windows = reduce_targets(selected)
                    method = (
                        "cost_sensitive_pairwise"
                        if mode == "classification"
                        else "pairwise_regression"
                    )
                    windows["model_id"], windows["method"] = model, method
                    prediction_path = directory / f"{name}.parquet"
                    selected[
                        [
                            "episode_id",
                            "source_episode_id",
                            "origin_id",
                            "family_id",
                            "dataset_id",
                            "candidate_id",
                            "target_slot",
                            "pairwise_wins",
                            "mae",
                            "mse",
                        ]
                    ].to_parquet(prediction_path, index=False)
                    model_path = directory / f"{name}.joblib"
                    joblib.dump(selector, model_path, compress=3)
                    saved = {
                        "identity_sha256": identity_sha,
                        "model_id": model,
                        "held_family": family,
                        "method": method,
                        "training_origins": sorted(training.origin_id.unique()),
                        "normalizers": denominator,
                        "baseline_id": selector.baseline_id,
                        "pair_fit_records": selector.fit_records,
                        "fit_seconds": fitted - started,
                        "predict_and_save_seconds": monotonic() - fitted,
                        "mae": family_macro(windows, "mae"),
                        "mse": family_macro(windows, "mse"),
                        "windows": len(windows),
                        "model_path": str(model_path.relative_to(output)),
                        "model_sha256": file_sha256(model_path),
                        "predictions_path": str(prediction_path.relative_to(output)),
                        "predictions_sha256": file_sha256(prediction_path),
                    }
                    _write_json(cache, saved)
                results.append(
                    {
                        key: saved[key]
                        for key in (
                            "model_id",
                            "held_family",
                            "method",
                            "mae",
                            "mse",
                            "windows",
                            "fit_seconds",
                            "predict_and_save_seconds",
                        )
                    }
                )
                files.append({"path": str(cache.relative_to(output)), "sha256": file_sha256(cache)})
                _write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "completed_folds": len(files),
                        "total_folds": 60,
                        "model": model,
                        "family": family,
                        "method": saved["method"],
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_folds": len(files),
                            "total_folds": 60,
                            "model": model,
                            "family": family,
                            "mode": mode,
                        }
                    ),
                    flush=True,
                )
    pd.DataFrame(results).to_csv(output / "family_metrics.csv", index=False)
    summary = (
        pd.DataFrame(results).groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": files,
            "summary": summary.to_dict("records"),
            "decision_input_excludes_current_outcomes": True,
            "logical_forecaster_queries_added": 0,
            "pairwise_predictors_per_fold": 21,
            "comparison_scope": "regression and classification have the same pair-specific features and model count; shared original utility regression has a different capacity",
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_folds": len(files), "total_folds": 60},
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
