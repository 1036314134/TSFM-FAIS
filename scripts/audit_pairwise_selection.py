"""Verify pairwise selections against the original cached action costs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "input-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed pairwise audits")
    output.mkdir(parents=True, exist_ok=True)
    root = args.input_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or len(manifest["folds"]) != 60:
        raise ValueError("complete all pairwise folds")
    if (
        file_sha256(args.accuracy_root / "candidate_accuracy.parquet")
        != manifest["identity"]["candidate_table_sha256"]
    ):
        raise ValueError("the original action costs changed")
    columns = [
        "model_id",
        "episode_id",
        "origin_id",
        "candidate_id",
        "target_slot",
        "family_id",
        "dataset_id",
        "item_id",
        "split",
        "mae",
        "mse",
    ]
    source = pd.read_parquet(args.accuracy_root / "candidate_accuracy.parquet", columns=columns)
    origin_families = (
        source[["origin_id", "family_id", "split"]].drop_duplicates().set_index("origin_id")
    )
    validation = source[source.split == "validation"].rename(
        columns={"episode_id": "source_episode_id"}
    )
    rows, max_difference, pairs, fitted_pairs, decisions = [], 0.0, 0, 0, 0
    for record in manifest["folds"]:
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a fold record changed")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if fold["identity_sha256"] != manifest["identity_sha256"]:
            raise ValueError("a fold identity changed")
        for kind in ("model", "predictions"):
            if file_sha256(root / fold[kind + "_path"]) != fold[kind + "_sha256"]:
                raise ValueError("a fitted model or selection changed")
        train = origin_families.loc[fold["training_origins"]]
        if (train.family_id == fold["held_family"]).any() or set(train.split) != {"train"}:
            raise ValueError("held-family information entered the fitting origins")
        selected = pd.read_parquet(root / fold["predictions_path"])
        if set(selected.family_id) != {fold["held_family"]} or set(selected.origin_id) & set(
            fold["training_origins"]
        ):
            raise ValueError("the saved choices have invalid evaluation provenance")
        model_source = validation[validation.model_id == fold["model_id"]]
        available = model_source[
            (model_source.family_id == fold["held_family"])
            & model_source.target_slot.isin([-1] if fold["model_id"] == "chronos2" else [0, 1])
        ]
        expected = available[["source_episode_id", "target_slot"]].drop_duplicates()
        if set(map(tuple, selected[["source_episode_id", "target_slot"]].to_numpy())) != set(
            map(tuple, expected.to_numpy())
        ):
            raise ValueError("the selected targets do not cover the original evaluation panel")
        joined = selected.merge(
            model_source,
            on=["source_episode_id", "candidate_id", "target_slot"],
            suffixes=("_saved", ""),
            validate="one_to_one",
        )
        np.testing.assert_allclose(
            joined[["mae_saved", "mse_saved"]], joined[["mae", "mse"]], rtol=0, atol=1e-12
        )
        windows = (
            joined.groupby(["source_episode_id", "family_id", "dataset_id", "item_id"])[
                ["mae", "mse"]
            ]
            .mean()
            .reset_index()
        )
        actual = np.array([family_macro(windows, key) for key in ("mae", "mse")])
        reference = np.array([fold[key] for key in ("mae", "mse")])
        max_difference = max(max_difference, float(np.abs(actual - reference).max()))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-10)
        rows.append(
            {
                "model_id": fold["model_id"],
                "family_id": fold["held_family"],
                "method": fold["method"],
                "mae": actual[0],
                "mse": actual[1],
            }
        )
        pairs += len(fold["pair_fit_records"])
        fitted_pairs += sum(not item["constant"] for item in fold["pair_fit_records"])
        decisions += len(selected)
    family = pd.DataFrame(rows)
    prior_path = args.accuracy_root / "analysis-joint-mae-mse-v001/family_results.csv"
    prior = pd.read_csv(prior_path)
    references = []
    for model, method in (("chronos2", "sequence_response"), ("timesfm2p5", "target_response")):
        selected = prior[
            (prior.model_id == model)
            & (prior.scope == "unseen_family")
            & (
                ((prior.objective == "joint") & prior.method.isin([method, "train_best_fixed"]))
                | ((prior.objective == "none") & (prior.method == "forecast_median_guarded"))
            )
        ].copy()
        selected["method"] = selected.method.replace({method: "shared_utility_regression"})
        references.append(selected[["model_id", "family_id", "method", "mae", "mse"]])
    combined = pd.concat([family, *references], ignore_index=True)
    if len(combined) != 150 or combined.duplicated(["model_id", "family_id", "method"]).any():
        raise ValueError("the comparison must contain five methods for two models and 15 families")
    combined.to_csv(output / "family_metrics.csv", index=False)
    summary = combined.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    comparisons = []
    for model in family.model_id.unique():
        for method in family.method.unique():
            for baseline in (
                "shared_utility_regression",
                "train_best_fixed",
                "forecast_median_guarded",
                "pairwise_regression",
            ):
                if method == baseline:
                    continue
                local = combined[combined.model_id == model]
                paired = local[local.method == method].merge(
                    local[local.method == baseline],
                    on="family_id",
                    suffixes=("", "_baseline"),
                    validate="one_to_one",
                )
                mae, mse = paired.mae - paired.mae_baseline, paired.mse - paired.mse_baseline
                comparisons.append(
                    {
                        "model_id": model,
                        "method": method,
                        "baseline": baseline,
                        "delta_mae": float(mae.mean()),
                        "delta_mse": float(mse.mean()),
                        "both_metrics_win_families": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                        "families": len(paired),
                    }
                )
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "original_reference_sha256": file_sha256(prior_path),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_fold_count": len(rows),
            "verified_decisions": decisions,
            "pairwise_problems": pairs,
            "fitted_tree_models": fitted_pairs,
            "constant_pairwise_models": pairs - fitted_pairs,
            "maximum_metric_difference": max_difference,
            "summary": summary.to_dict("records"),
            "comparisons": comparisons,
            "scope": "R4 raw-input recipes; matched cached forecast units; development evaluation only",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
