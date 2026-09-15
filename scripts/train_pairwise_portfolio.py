"""Match pairwise learner capacity while changing individual versus portfolio supervision."""

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

from tsfm_fais.routing.pairwise_utility import PairwiseUtilitySelector  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def triple_labels(direct_risk, members, kind):
    direct_risk = np.asarray(direct_risk, float)
    if direct_risk.ndim != 2 or direct_risk.shape[1] != 43 or len(members) != 43:
        raise ValueError("the fixed option-risk bank is required")
    if kind == "median_risk":
        return direct_risk[:, 7:42]
    if kind == "member_risk":
        return np.stack([direct_risk[:, group].mean(axis=1) for group in members[7:42]], axis=1)
    raise ValueError("unsupported matched supervision")


def triple_rows(decisions, features, info, indices):
    frame = option_rows(decisions, features, info, indices)
    frame = frame[frame.candidate_id.isin(info["option_names"][7:42])].copy()
    observed = {name for name in frame if name.startswith(("static.", "response.", "member."))}
    if observed != set(info["feature_names"]) or len(frame) != len(indices) * 35:
        raise ValueError("matched portfolio inputs changed coverage or feature inventory")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "accuracy-root", "screen-plan", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--label-kind", choices=("member_risk", "median_risk"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.label_kind
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed matched portfolio studies")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.screen_plan.read_text(encoding="utf-8"))
    if (
        prep["status"] != "completed"
        or prep["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
        or plan["source_manifest_sha256"] != accuracy["source_episode_manifest_sha256"]
    ):
        raise ValueError("the matched study sources disagree")
    for name, digest in prep["identity"]["source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("the prepared option feature definitions changed")
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "screen_plan_sha256": file_sha256(args.screen_plan),
        "script_sha256": file_sha256(Path(__file__)),
        "label_kind": args.label_kind,
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/aligned_portfolio_io.py",
                "src/tsfm_fais/routing/pairwise_utility.py",
            )
        },
        "options": 35,
        "comparators_per_fold": 595,
        "query_budget": 7,
        "source_outcome_supervision": False,
        "primary_label_kind": "median_risk",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial matched study changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    results, folds = [], []
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.prepared_root, prep, model)
        labels = triple_labels(arrays["direct_risk"], info["members"], args.label_kind)
        point_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("source forecasts changed")
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
            if family in set(training.family_id) or set(training.origin_id) & set(
                evaluation.origin_id
            ):
                raise ValueError("the held family or history entered training")
            key = hashlib.sha256(f"{model}|{family}|{args.label_kind}".encode()).hexdigest()[:24]
            directory = output / "folds"
            directory.mkdir(exist_ok=True)
            marker = directory / f"{key}.json"
            if marker.exists():
                record = json.loads(marker.read_text(encoding="utf-8"))
                if record["identity_sha256"] != identity_sha:
                    raise ValueError("a saved fold changed identity")
                for name in ("model", "prediction", "scores"):
                    if file_sha256(output / record[name + "_path"]) != record[name + "_sha256"]:
                        raise ValueError("a saved matched-fold artifact changed")
            else:
                started = monotonic()
                train_frame = triple_rows(decisions, arrays["features"], info, train_ids)
                train_frame["loss"] = labels[train_ids].reshape(-1)
                learner = PairwiseUtilitySelector(
                    feature_prefixes=("static.", "response.", "member.")
                ).fit(train_frame)
                del train_frame
                if len(learner.models) != 595 or set(learner.feature_names) != set(
                    info["feature_names"]
                ):
                    raise ValueError("matched classifier capacity or inputs changed")
                evaluation_frame = triple_rows(decisions, arrays["features"], info, eval_ids)
                selected = (
                    learner.select(evaluation_frame)
                    .set_index("episode_id")
                    .loc[evaluation.episode_id]
                )
                choice = np.array(
                    [info["option_names"].index(name) for name in selected.candidate_id]
                )
                if not np.all((choice >= 7) & (choice < 42)):
                    raise ValueError("a choice left the common triple catalog")
                options = current_options(evaluation, bank, info["actions"])
                point, reference = options[np.arange(len(eval_ids)), choice], options[:, -1]
                model_path = directory / f"{key}.joblib"
                joblib.dump(learner, model_path, compress=3)
                prediction_path = directory / f"{key}.predictions.npz"
                _save_npz(
                    prediction_path,
                    decision_indices=eval_ids,
                    choices=choice,
                    point=point,
                    reference=reference,
                    pairwise_wins=selected.pairwise_wins.to_numpy(),
                )
                # Fix the selected predictions before opening current future outcomes.
                truth_path = args.accuracy_root / "truth_z.npy"
                if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
                    raise ValueError("source validation outcomes changed")
                truth = decision_truth(evaluation, np.load(truth_path, mmap_mode="r"))
                records = []
                for method, values in (
                    (args.label_kind, point),
                    ("forecast_median_guarded", reference),
                ):
                    errors = values - truth
                    records.append(
                        evaluation.copy().assign(
                            model_id=model,
                            method=method,
                            mae=np.abs(errors).mean(axis=1),
                            mse=np.square(errors).mean(axis=1),
                        )
                    )
                score_path = directory / f"{key}.scores.parquet"
                pd.concat(records, ignore_index=True).to_parquet(score_path, index=False)
                record = {
                    "status": "completed",
                    "identity_sha256": identity_sha,
                    "model_id": model,
                    "label_kind": args.label_kind,
                    "held_family": family,
                    "training_origins": sorted(training.origin_id.unique()),
                    "training_families": sorted(training.family_id.unique()),
                    "feature_names": list(learner.feature_names),
                    "candidate_ids": list(learner.candidate_ids),
                    "comparators": len(learner.models),
                    "nonconstant_comparators": sum(
                        not isinstance(value, float) for value in learner.models.values()
                    ),
                    "source_outcome_supervision": False,
                    "seconds": monotonic() - started,
                }
                for name, path in (
                    ("model", model_path),
                    ("prediction", prediction_path),
                    ("scores", score_path),
                ):
                    record.update(
                        {
                            name + "_path": str(path.relative_to(output)),
                            name + "_sha256": file_sha256(path),
                        }
                    )
                _write_json(marker, record)
            results.append(pd.read_parquet(output / record["scores_path"]))
            folds.append({"path": str(marker.relative_to(output)), "sha256": file_sha256(marker)})
            _write_json(
                output / "progress.json",
                {"status": "fitting", "completed_folds": len(folds), "total_folds": 30},
            )
            print(
                json.dumps(
                    {
                        "model": model,
                        "family": family,
                        "completed_folds": len(folds),
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
    scores = pd.concat(results, ignore_index=True)
    keys = [
        "model_id",
        "method",
        "source_episode_id",
        "origin_id",
        "family_id",
        "dataset_id",
        "item_id",
    ]
    windows = scores.groupby(keys)[["mae", "mse"]].mean().reset_index()
    for _, group in windows.groupby(["model_id", "method"]):
        if len(group) != 1872 or group.source_episode_id.nunique() != 1872:
            raise ValueError("the matched study has incomplete validation coverage")
    windows.to_parquet(output / "episode_results.parquet", index=False)
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
            "summary": summary.to_dict("records"),
            "limits": "matched source-development objective control; no new independent confirmation or method superiority claim",
        },
    )
    print(summary[summary.panel == "full_development"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
