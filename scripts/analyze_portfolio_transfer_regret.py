"""Measure finite-class teacher regret across source fitting and transfer boundaries."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import load_prepared_model  # noqa: E402
from apply_followup_policies import load_frozen_selector  # noqa: E402
from train_pairwise_portfolio import triple_rows  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def selection_regret(decisions, risks, choices, scope):
    choices = np.asarray(choices, int)
    if (
        risks.shape != (len(decisions), 35)
        or choices.shape != (len(decisions),)
        or np.any((choices < 0) | (choices >= 35))
    ):
        raise ValueError("aligned 35-option costs and decisions are required")
    selected = risks[np.arange(len(decisions)), choices]
    optimum = risks.min(axis=1)
    regret = selected - optimum
    if not np.isfinite(risks).all() or np.any(regret < -1e-12):
        raise ValueError("selected cost cannot beat the actual finite-class minimum")
    return decisions[["episode_id", "source_episode_id", "origin_id", "family_id"]].assign(
        scope=scope,
        teacher_regret=regret,
        teacher_gain_over_median7=-selected,
        oracle_teacher_gain_over_median7=-optimum,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned-root",
        "source-bundle",
        "matched-root",
        "matched-audit-root",
        "followup-diagnostic-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed transfer diagnostics")
    prep = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    matched = json.loads((args.matched_root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.matched_audit_root / "manifest.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (args.followup_diagnostic_root / "manifest.json").read_text(encoding="utf-8")
    )
    if any(row["status"] != "completed" for row in (prep, bundle, matched, audit, diagnostic)):
        raise ValueError("completed frozen inputs and their audits are required")
    if audit["source_manifests"]["median_risk"] != file_sha256(args.matched_root / "manifest.json"):
        raise ValueError("the held-family comparison changed after its audit")
    output.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.aligned_root, prep, model)
        learner, _ = load_frozen_selector(
            args.source_bundle, bundle, model, "median_risk", "label_kind"
        )
        features = triple_rows(decisions, arrays["features"], info, np.arange(len(decisions)))
        selected = learner.select(features).set_index("episode_id").loc[decisions.episode_id]
        names = info["option_names"][7:42]
        choices = np.asarray([names.index(name) for name in selected.candidate_id])
        risks = arrays["direct_risk"][:, 7:42]
        for split, scope in (
            ("train", "source_in_sample"),
            ("validation", "source_later_seen_family"),
        ):
            indices = np.flatnonzero(decisions.split.to_numpy() == split)
            frame = selection_regret(
                decisions.iloc[indices], risks[indices], choices[indices], scope
            )
            rows.append(frame.assign(model_id=model))
        selected.reset_index()[["episode_id", "candidate_id", "pairwise_wins"]].to_parquet(
            output / f"{model}_source_choices.parquet", index=False
        )
        covered = np.zeros(len(decisions), bool)
        for entry in matched["folds"]:
            path = args.matched_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a held-family fold record changed")
            fold = json.loads(path.read_text(encoding="utf-8"))
            if fold["model_id"] != model:
                continue
            path = args.matched_root / fold["prediction_path"]
            if file_sha256(path) != fold["prediction_sha256"]:
                raise ValueError("a held-family prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                indices, choices = saved["decision_indices"], saved["choices"] - 7
            if covered[indices].any() or set(decisions.iloc[indices].family_id) != {
                fold["held_family"]
            }:
                raise ValueError("held-family decisions overlap or use the wrong family")
            covered[indices] = True
            rows.append(
                selection_regret(
                    decisions.iloc[indices], risks[indices], choices, "source_held_family"
                ).assign(model_id=model)
            )
        if not np.array_equal(covered, decisions.split.to_numpy() == "validation"):
            raise ValueError("held-family validation coverage is incomplete")
        predictions.append(
            {
                "model_id": model,
                "source_decisions": len(decisions),
                "choices_sha256": file_sha256(output / f"{model}_source_choices.parquet"),
            }
        )
        del learner, features, arrays
    source_rows = pd.concat(rows, ignore_index=True)
    # Average independent target decisions back to each actual history/mask episode first.
    metrics = ["teacher_regret", "teacher_gain_over_median7", "oracle_teacher_gain_over_median7"]
    source_rows = (
        source_rows.groupby(["model_id", "scope", "family_id", "origin_id", "source_episode_id"])[
            metrics
        ]
        .mean()
        .reset_index()
    )
    followup = pd.read_parquet(args.followup_diagnostic_root / "episode_metrics.parquet")
    for (model, family), group in followup.groupby(["model_id", "family_id"]):
        table = group.pivot(index="episode_id", columns="method", values="teacher_mse")
        metadata = group.drop_duplicates("episode_id").set_index("episode_id").loc[table.index]
        rows = pd.DataFrame(
            {
                "model_id": model,
                "scope": "used_followup_synthetic",
                "family_id": family,
                "origin_id": metadata.origin_id,
                "source_episode_id": table.index,
                "teacher_regret": table.median_risk - table.unavailable_teacher_best3,
                "teacher_gain_over_median7": table.forecast_median_seven - table.median_risk,
                "oracle_teacher_gain_over_median7": table.forecast_median_seven
                - table.unavailable_teacher_best3,
            }
        ).reset_index(drop=True)
        source_rows = pd.concat([source_rows, rows], ignore_index=True)
    source_rows.to_parquet(output / "episode_regret.parquet", index=False)
    families = source_rows.groupby(["model_id", "scope", "family_id"])[metrics].mean()
    families.to_csv(output / "family_regret.csv")
    summary = families.groupby(["model_id", "scope"]).mean()
    summary.to_csv(output / "summary.csv")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
            "followup_diagnostic_sha256": file_sha256(
                args.followup_diagnostic_root / "manifest.json"
            ),
            "predictions": predictions,
            "new_fits": 0,
            "new_forecaster_calls": 0,
            "limits": "post-evaluation auxiliary teacher-regret analysis; training and temporal evaluations share source families; oracle gaps include unavailable information and do not prove reducibility by more training",
        },
    )


if __name__ == "__main__":
    main()
