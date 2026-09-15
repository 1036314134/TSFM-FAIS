"""Fit source-only portfolio scores and evaluate fixed option menus by held-out family."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import (  # noqa: E402
    current_options,
    decision_truth,
    load_prepared_model,
    option_rows,
)

from tsfm_fais.routing.aligned_portfolio import (  # noqa: E402
    AlignedPortfolioRegressor,
    choose_option,
    option_scores,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "accuracy-root", "screen-plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-kind", choices=("unit_projection", "direct_risk"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.target_kind
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed aligned portfolio fits")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.screen_plan.read_text(encoding="utf-8"))
    if (
        prep["status"] != "completed"
        or prep["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or plan["source_manifest_sha256"] != accuracy["source_episode_manifest_sha256"]
    ):
        raise ValueError("the prepared inputs and evaluation panel have different sources")
    for name, expected in prep["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("portfolio feature or score code changed after preparation")
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "screen_plan_sha256": file_sha256(args.screen_plan),
        "target_kind": args.target_kind,
        "menus": prep["identity"]["menus"],
        "primary_menu": "full",
        "script_sha256": file_sha256(Path(__file__)),
        "io_sha256": file_sha256(ROOT / "scripts/aligned_portfolio_io.py"),
        "source_outcome_supervision": False,
        "source_complete_history_supervision": True,
        "query_budget": 7,
        "learner": "shared 160-tree regressor; inherited fixed projection-regression settings",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial fit changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    folds, results, selection_records = [], [], []
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.prepared_root, prep, model)
        point_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("the candidate forecast bank changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in info["actions"]]
        ]
        for family in sorted(decisions.family_id.unique()):
            train_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "train") & (decisions.family_id.to_numpy() != family)
            )
            eval_ids = np.flatnonzero(
                (decisions.split.to_numpy() == "validation")
                & (decisions.family_id.to_numpy() == family)
            )
            training, evaluation = decisions.iloc[train_ids], decisions.iloc[eval_ids]
            if (
                len(train_ids) == 0
                or len(eval_ids) == 0
                or family in set(training.family_id)
                or set(training.origin_id) & set(evaluation.origin_id)
            ):
                raise ValueError("a source fit used the held family or an evaluation history")
            key = hashlib.sha256(f"{model}|{family}|{args.target_kind}".encode()).hexdigest()[:24]
            directory = output / "folds"
            directory.mkdir(exist_ok=True)
            marker = directory / f"{key}.json"
            if marker.exists():
                record = json.loads(marker.read_text(encoding="utf-8"))
                if record["identity_sha256"] != identity_sha:
                    raise ValueError("a completed fold changed identity")
                for name in ("model", "prediction", "scores", "choices"):
                    if file_sha256(output / record[name + "_path"]) != record[name + "_sha256"]:
                        raise ValueError("a completed fold artifact changed")
            else:
                started = monotonic()
                train_frame = option_rows(decisions, arrays["features"], info, train_ids)
                learner = AlignedPortfolioRegressor(
                    member_names=tuple(
                        name for name in info["feature_names"] if name.startswith("member.")
                    )
                ).fit(train_frame, arrays[args.target_kind][train_ids].reshape(-1))
                del train_frame
                eval_frame = option_rows(decisions, arrays["features"], info, eval_ids)
                estimates = learner.predict(eval_frame).reshape(len(eval_ids), 43)
                vectors = current_options(evaluation, bank, info["actions"])
                scores = option_scores(vectors, estimates, target_kind=args.target_kind)
                choices = {menu: choose_option(scores, menu) for menu in identity["menus"]}
                predictions = {
                    menu: vectors[np.arange(len(eval_ids)), indices]
                    for menu, indices in choices.items()
                }
                model_path = directory / f"{key}.joblib"
                joblib.dump(learner, model_path)
                prediction_path = directory / f"{key}.predictions.npz"
                _save_npz(
                    prediction_path,
                    decision_indices=eval_ids,
                    estimates=estimates,
                    scores=scores,
                    reference=vectors[:, -1],
                    **{"choice_" + menu: value for menu, value in choices.items()},
                    **{"point_" + menu: value for menu, value in predictions.items()},
                )
                # The current choices and returned predictions are fixed before future scoring.
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("source validation outcomes changed")
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                records = []
                for method, values in [
                    *predictions.items(),
                    ("forecast_median_guarded", vectors[:, -1]),
                ]:
                    residual = values - truth
                    frame = evaluation.copy().assign(
                        model_id=model,
                        method=method,
                        mae=np.abs(residual).mean(axis=1),
                        mse=np.square(residual).mean(axis=1),
                    )
                    records.append(frame)
                score_path = directory / f"{key}.scores.parquet"
                pd.concat(records, ignore_index=True).to_parquet(score_path, index=False)
                selections = []
                for menu, indices in choices.items():
                    selections.append(
                        evaluation.copy().assign(
                            model_id=model,
                            menu=menu,
                            option_id=np.asarray(info["option_names"])[indices],
                            members=np.asarray([len(group) for group in info["members"]])[indices],
                        )
                    )
                choice_path = directory / f"{key}.choices.parquet"
                pd.concat(selections, ignore_index=True).to_parquet(choice_path, index=False)
                record = {
                    "status": "completed",
                    "identity_sha256": identity_sha,
                    "model_id": model,
                    "held_family": family,
                    "target_kind": args.target_kind,
                    "training_origins": sorted(training.origin_id.unique()),
                    "training_families": sorted(training.family_id.unique()),
                    "feature_names": info["feature_names"],
                    "option_names": info["option_names"],
                    "source_outcome_supervision": False,
                    "seconds": monotonic() - started,
                }
                for name, path in (
                    ("model", model_path),
                    ("prediction", prediction_path),
                    ("scores", score_path),
                    ("choices", choice_path),
                ):
                    record.update(
                        {
                            name + "_path": str(path.relative_to(output)),
                            name + "_sha256": file_sha256(path),
                        }
                    )
                _write_json(marker, record)
            results.append(pd.read_parquet(output / record["scores_path"]))
            selection_records.append(pd.read_parquet(output / record["choices_path"]))
            folds.append({"path": str(marker.relative_to(output)), "sha256": file_sha256(marker)})
            _write_json(
                output / "progress.json",
                {"status": "fitting", "completed_folds": len(folds), "total_folds": 30},
            )
            print(
                json.dumps({"model": model, "family": family, "completed_folds": len(folds)}),
                flush=True,
            )
    frame = pd.concat(results, ignore_index=True)
    windows = (
        frame.groupby(
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
            raise ValueError("an aligned portfolio has incomplete validation coverage")
    windows.to_parquet(output / "episode_results.parquet", index=False)
    selections = pd.concat(selection_records, ignore_index=True)
    selections.to_parquet(output / "choices.parquet", index=False)
    selections.groupby(["model_id", "menu", "members"]).size().rename(
        "decisions"
    ).reset_index().to_csv(output / "selection_sizes.csv", index=False)
    panels = [
        windows.assign(panel="full_development"),
        windows[windows.source_episode_id.isin(plan["decision_episode_ids"])].assign(
            panel="screening_90"
        ),
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
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "folds": folds,
            "new_forecaster_calls": 0,
            "new_selector_fits": len(folds),
            "primary_method": args.target_kind + "/full",
            "summary": summary.to_dict("records"),
            "limits": "source development only; no new native confirmation, no reduced forecast-query claim",
        },
    )
    print(summary[summary.panel == "full_development"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
