"""Compare structured students with matched full-panel and 90-task controls."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-root",
        "temporal-root",
        "dependency-root",
        "accuracy-root",
        "replay-root",
        "plan",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-readout-root", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed structured-student comparisons")
    output.mkdir(parents=True, exist_ok=True)
    episode_ids = set(json.loads(args.plan.read_text(encoding="utf-8"))["decision_episode_ids"])
    if len(episode_ids) != 90:
        raise ValueError("the matched screening panel must have 90 distinct decisions")
    frames, screening, sources = [], [], {}
    expected_records = 0
    teacher_kinds = {}
    for representation, root in (
        ("base_static", args.base_root),
        ("target_temporal", args.temporal_root),
        ("full_dependency", args.dependency_root),
    ):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed":
            raise ValueError("complete all feature views before comparison")
        sources[representation] = file_sha256(root / "manifest.json")
        expected_records += len(manifest["folds"])
        teacher_kinds[representation] = manifest["identity"].get(
            "teacher_kind", "candidate_forecast_median"
        )
        family = pd.read_csv(root / "family_metrics.csv").rename(
            columns={"held_family": "family_id"}
        )
        family["representation"] = representation
        frames.append(family)
        rows = []
        for record in manifest["folds"]:
            path = root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a student fold changed")
            fold = json.loads(path.read_text(encoding="utf-8"))
            score_path = root / fold["scores_path"]
            if file_sha256(score_path) != fold["scores_sha256"]:
                raise ValueError("student scores changed")
            scored = pd.read_parquet(score_path)
            scored = scored[scored.source_episode_id.isin(episode_ids)]
            windows = (
                scored.groupby(["source_episode_id", "family_id"])[["mae", "mse"]]
                .mean()
                .reset_index()
            )
            windows["model_id"], windows["objective"], windows["representation"] = (
                fold["model_id"],
                fold["objective"],
                representation,
            )
            rows.append(windows)
        screen = pd.concat(rows, ignore_index=True)
        for _, group in screen.groupby(["model_id", "objective"]):
            if set(group.source_episode_id) != episode_ids or len(group) != 90:
                raise ValueError("student screening results have different coverage")
        screening.append(screen)
    all_family = pd.concat(frames, ignore_index=True)
    index = ["model_id", "representation", "objective", "family_id"]
    if len(all_family) != expected_records or all_family.duplicated(index).any():
        raise ValueError("the full-panel family comparison has incomplete or duplicate entries")
    all_family.to_csv(output / "full_family_metrics.csv", index=False)
    full_summary = (
        all_family.groupby(index[:-1])[
            ["mae", "mse", "teacher_mse", "projection_floor", "projection_regret"]
        ]
        .mean()
        .reset_index()
    )
    full_summary.to_csv(output / "full_summary.csv", index=False)
    supervision_changes = []
    if args.previous_readout_root is not None:
        previous = pd.read_csv(args.previous_readout_root / "full_family_metrics.csv")
        for (model, representation, objective), group in all_family.groupby(index[:-1]):
            for old_objective in ("consensus_mse", "future_joint"):
                prior_group = previous[
                    (previous.model_id == model)
                    & (previous.representation == representation)
                    & (previous.objective == old_objective)
                ]
                paired = group.merge(
                    prior_group, on="family_id", suffixes=("", "_prior"), validate="one_to_one"
                )
                if len(paired) != 15:
                    raise ValueError("supervision comparison requires the same 15 families")
                mae, mse = paired.mae - paired.mae_prior, paired.mse - paired.mse_prior
                supervision_changes.append(
                    {
                        "model_id": model,
                        "representation": representation,
                        "objective": objective,
                        "prior_objective": old_objective,
                        "delta_mae": float(mae.mean()),
                        "delta_mse": float(mse.mean()),
                        "both_metrics_win_families": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                        "families": len(paired),
                    }
                )
        pd.DataFrame(supervision_changes).to_csv(output / "supervision_changes.csv", index=False)
    changes = []
    for (model, representation, objective), group in all_family.groupby(index[:-1]):
        if representation == "base_static":
            continue
        for baseline in (
            ["base_static", "target_temporal"]
            if representation == "full_dependency"
            else ["base_static"]
        ):
            reference = all_family[
                (all_family.model_id == model)
                & (all_family.representation == baseline)
                & (all_family.objective == objective)
            ]
            paired = group.merge(
                reference, on="family_id", suffixes=("", "_reference"), validate="one_to_one"
            )
            mae, mse = paired.mae - paired.mae_reference, paired.mse - paired.mse_reference
            changes.append(
                {
                    "model_id": model,
                    "representation": representation,
                    "objective": objective,
                    "baseline": baseline,
                    "delta_mae": float(mae.mean()),
                    "delta_mse": float(mse.mean()),
                    "both_metrics_win_families": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                    "families": len(paired),
                }
            )
    pd.DataFrame(changes).to_csv(output / "full_paired_changes.csv", index=False)
    screen = pd.concat(screening, ignore_index=True)
    screen.to_parquet(output / "screening_episode_metrics.parquet", index=False)
    screen_family = (
        screen.groupby(["model_id", "representation", "objective", "family_id"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    screen_family.to_csv(output / "screening_family_metrics.csv", index=False)
    screen_summary = (
        screen_family.groupby(["model_id", "representation", "objective"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    screen_summary.to_csv(output / "screening_summary.csv", index=False)
    reference_rows = []
    for model in ("chronos2", "timesfm2p5"):
        reference = pd.read_csv(args.replay_root / model / "family_metrics.csv")
        reference = reference[
            reference.method.isin(
                [
                    "input_mean_finite",
                    "input_median_finite",
                    "forecast_median_guarded",
                    "guarded_direct",
                ]
            )
        ]
        reference_rows.append(reference)
    reference = pd.concat(reference_rows, ignore_index=True)
    reference.to_csv(output / "screening_reference_families.csv", index=False)
    screen_changes = []
    for (model, representation, objective), group in screen_family.groupby(
        ["model_id", "representation", "objective"]
    ):
        for baseline in ("input_median_finite", "forecast_median_guarded"):
            control = reference[(reference.model_id == model) & (reference.method == baseline)]
            paired = group.merge(
                control, on="family_id", suffixes=("", "_reference"), validate="one_to_one"
            )
            mae, mse = paired.mae - paired.mae_reference, paired.mse - paired.mse_reference
            screen_changes.append(
                {
                    "model_id": model,
                    "representation": representation,
                    "objective": objective,
                    "baseline": baseline,
                    "delta_mae": float(mae.mean()),
                    "delta_mse": float(mse.mean()),
                    "both_metrics_win_families": int(((mae < -1e-12) & (mse < -1e-12)).sum()),
                    "families": len(paired),
                }
            )
    pd.DataFrame(screen_changes).to_csv(output / "screening_paired_changes.csv", index=False)
    prior = pd.read_csv(args.accuracy_root / "analysis-joint-mae-mse-v001/summary.csv")
    prior[
        (prior.scope == "unseen_family")
        & (
            ((prior.objective == "joint") & (prior.method == "train_best_fixed"))
            | ((prior.objective == "none") & (prior.method == "forecast_median_guarded"))
        )
    ].to_csv(output / "full_reference_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": sources,
            "teacher_kinds": teacher_kinds,
            "supervision_changes": supervision_changes,
            "previous_readout_manifest_sha256": file_sha256(
                args.previous_readout_root / "manifest.json"
            )
            if args.previous_readout_root
            else None,
            "script_sha256": file_sha256(Path(__file__)),
            "full_summary": full_summary.to_dict("records"),
            "full_changes": changes,
            "screening_changes": screen_changes,
            "limits": [
                "all results are development estimates",
                "screening and full-panel populations are separately reported",
                "learner forecasts are cache-derived until structured-input replay",
                "screening controls use verified raw-input replays",
                "no significance or deployment-speed claim",
            ],
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(full_summary.to_string(index=False))
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()
