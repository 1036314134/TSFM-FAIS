"""Test transfer across the two development forecasters and held-out families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from analyze_utility_anchor_ablation import anchored_fit, reanchor_features

from tsfm_fais.routing.utility import family_macro
from tsfm_fais.utility_experiment import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    original = {
        model: pd.read_parquet(root / model / "utility_rows.parquet")
        for model in ("chronos2", "timesfm2p5")
    }
    common_actions = sorted(
        set.intersection(*(set(frame.candidate_id) for frame in original.values()))
    )
    frames = {
        model: reanchor_features(
            root, frame[frame.candidate_id.isin(common_actions)], model, "locf"
        )
        for model, frame in original.items()
    }
    results, folds = [], []
    for source_model, target_model in (("chronos2", "timesfm2p5"), ("timesfm2p5", "chronos2")):
        source, target = frames[source_model], frames[target_model]
        for held_family in sorted(target.family_id.unique()):
            train = source[(source.family_id != held_family) & (source.split == "train")]
            calibrate = source[(source.family_id != held_family) & (source.split == "validation")]
            evaluate = target[(target.family_id == held_family) & (target.split == "validation")]
            if set(train.origin_id).intersection(evaluate.origin_id) or set(
                calibrate.origin_id
            ).intersection(evaluate.origin_id):
                raise ValueError(
                    "transfer training or calibration includes a target evaluation origin"
                )
            selected = {
                "fixed_locf": evaluate[evaluate.candidate_id == "locf"],
                "forecast_medoid": evaluate.sort_values(
                    ["episode_id", "response.pool_distance", "candidate_id"]
                ).drop_duplicates("episode_id"),
            }
            for name, response, objective in (
                ("static_l2", False, "regression"),
                ("response_l2", True, "regression"),
                ("response_bounded", True, "bounded_regression"),
            ):
                selector = anchored_fit(train, "locf", response, objective).calibrate(calibrate)
                selected[name] = selector.select(evaluate)
                selected[name + "_gated"] = selector.select(evaluate, gated=True)
                folds.append(
                    {
                        "source_model": source_model,
                        "target_model": target_model,
                        "held_family": held_family,
                        "method": name,
                        "training_families": sorted(train.family_id.unique()),
                        "calibration": selector.calibration,
                    }
                )
            for name, choices in selected.items():
                columns = [
                    "episode_id",
                    "origin_id",
                    "family_id",
                    "dataset_id",
                    "candidate_id",
                    "loss",
                ]
                result = choices[columns].copy()
                result["source_model"], result["target_model"], result["method"] = (
                    source_model,
                    target_model,
                    name,
                )
                results.append(result)
        print(
            json.dumps({"source": source_model, "target": target_model, "status": "completed"}),
            flush=True,
        )
    frame = pd.concat(results, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "source_model": source,
                "target_model": target,
                "method": method,
                "family_macro_mase": family_macro(group),
                "family_count": int(group.family_id.nunique()),
                "episode_count": int(group.episode_id.nunique()),
            }
            for (source, target, method), group in frame.groupby(
                ["source_model", "target_model", "method"]
            )
        ]
    )
    output = root / "analysis-forecaster-transfer-v001"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    frame.to_csv(output / "episode_results.csv", index=False)
    (output / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "evidence_role": "development",
                "common_actions": common_actions,
                "reference": "locf",
                "source_episode_manifest_sha256": file_sha256(root / "episodes_manifest.json"),
                "scripts_sha256": {
                    path.name: file_sha256(path)
                    for path in (
                        Path(__file__),
                        Path(__file__).with_name("analyze_utility_anchor_ablation.py"),
                    )
                },
                "interpretation": "within each fold, neither target forecaster labels nor target family labels are used for fitting/calibration; both forecasters have already been inspected in development, so this is not a new independent confirmation model",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
