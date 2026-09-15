"""Replay matched portfolio classifiers, vote aggregation, and downstream summaries."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import current_options, decision_truth, load_prepared_model  # noqa: E402
from train_pairwise_portfolio import triple_labels, triple_rows  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "accuracy-root", "study-root", "screen-plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed matched audits")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.screen_plan.read_text(encoding="utf-8"))
    if prep["status"] != "completed" or prep["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("prepared source forecasts changed")
    studies, sources = {}, {}
    for kind in ("member_risk", "median_risk"):
        path = args.study_root / kind / "manifest.json"
        studies[kind] = json.loads(path.read_text(encoding="utf-8"))
        identity = studies[kind]["identity"]
        if (
            studies[kind]["status"] != "completed"
            or len(studies[kind]["folds"]) != 30
            or identity["prepared_manifest_sha256"]
            != file_sha256(args.prepared_root / "manifest.json")
            or identity["screen_plan_sha256"] != file_sha256(args.screen_plan)
            or identity["label_kind"] != kind
        ):
            raise ValueError("complete both registered matched-label studies")
        for name, expected in identity["source_sha256"].items():
            if file_sha256(ROOT / name) != expected:
                raise ValueError("pairwise inference code changed")
        sources[kind] = file_sha256(path)
    if (
        studies["member_risk"]["identity"]["protocol_sha256"]
        != studies["median_risk"]["identity"]["protocol_sha256"]
    ):
        raise ValueError("the two objectives follow different protocols")
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("source validation outcomes changed")
    truth = np.load(truth_path, mmap_mode="r")
    verified = {kind: [] for kind in studies}
    model_count, comparator_count, decision_count, maximum_difference = 0, 0, 0, 0.0
    population_records = {}
    for model in ("chronos2", "timesfm2p5"):
        info, decisions, arrays = load_prepared_model(args.prepared_root, prep, model)
        path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("the frozen forecast bank changed")
        bank = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in info["actions"]]
        ]
        for kind, study in studies.items():
            labels = triple_labels(arrays["direct_risk"], info["members"], kind)
            directory = args.study_root / kind
            for entry in study["folds"]:
                path = directory / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a matched-fold record changed")
                fold = json.loads(path.read_text(encoding="utf-8"))
                if fold["model_id"] != model:
                    continue
                if (
                    fold["identity_sha256"] != study["identity_sha256"]
                    or fold["label_kind"] != kind
                    or fold["source_outcome_supervision"]
                ):
                    raise ValueError("fold identity or supervision changed")
                for name in ("model", "prediction", "scores"):
                    if file_sha256(directory / fold[name + "_path"]) != fold[name + "_sha256"]:
                        raise ValueError("a matched-fold artifact changed")
                family = fold["held_family"]
                train_ids = np.flatnonzero(
                    (decisions.split.to_numpy() == "train")
                    & (decisions.family_id.to_numpy() != family)
                )
                eval_ids = np.flatnonzero(
                    (decisions.split.to_numpy() == "validation")
                    & (decisions.family_id.to_numpy() == family)
                )
                training, evaluation = decisions.iloc[train_ids], decisions.iloc[eval_ids]
                if (
                    sorted(training.origin_id.unique()) != fold["training_origins"]
                    or sorted(training.family_id.unique()) != fold["training_families"]
                    or set(training.origin_id) & set(evaluation.origin_id)
                ):
                    raise ValueError("a matched fold changed its source population")
                population_records[(model, family, kind)] = (
                    fold["training_origins"],
                    eval_ids.tolist(),
                    fold["feature_names"],
                    fold["candidate_ids"],
                )
                learner = joblib.load(directory / fold["model_path"])
                if (
                    learner.mode != "classification"
                    or len(learner.models) != 595
                    or set(learner.feature_names) != set(info["feature_names"])
                    or set(learner.candidate_ids) != set(info["option_names"][7:42])
                ):
                    raise ValueError("matched capacity, feature inventory, or catalog changed")
                ordered_labels = labels[train_ids][
                    :, [info["option_names"][7:42].index(name) for name in learner.candidate_ids]
                ]
                for (left, right), predictor in learner.models.items():
                    record = next(
                        row
                        for row in learner.fit_records
                        if (row["left"], row["right"])
                        == (learner.candidate_ids[left], learner.candidate_ids[right])
                    )
                    difference = ordered_labels[:, left] - ordered_labels[:, right]
                    positive = difference != 0
                    classes = difference[positive] < 0
                    constant = not positive.any() or np.unique(classes).size == 1
                    if (
                        record["examples"] != len(training)
                        or record["positive_weight_examples"] != int(positive.sum())
                        or record["constant"] != constant
                    ):
                        raise ValueError("the fitted comparator used another label population")
                    if constant:
                        expected = 0.5 if not positive.any() else float(classes[0])
                        if not isinstance(predictor, float) or predictor != expected:
                            raise ValueError(
                                "a constant comparator disagrees with its source costs"
                            )
                    else:
                        parameters = predictor.get_params()
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
                                raise ValueError("a comparator changed the fixed tree parameters")
                    comparator_count += 1
                label_frame = training.iloc[np.repeat(np.arange(len(training)), 35)][
                    ["family_id", "dataset_id"]
                ].reset_index(drop=True)
                label_frame["candidate_id"] = np.tile(learner.candidate_ids, len(training))
                label_frame["loss"] = ordered_labels.reshape(-1)
                candidate_costs = (
                    label_frame.groupby(["family_id", "dataset_id", "candidate_id"])
                    .loss.mean()
                    .groupby(["family_id", "candidate_id"])
                    .mean()
                    .groupby("candidate_id")
                    .mean()
                )
                preferred = min(
                    learner.candidate_ids, key=lambda name: (candidate_costs[name], name)
                )
                if preferred != learner.baseline_id:
                    raise ValueError("the source tie preference changed")
                frame = triple_rows(decisions, arrays["features"], info, eval_ids)
                episodes, preferences, pairs = learner.pair_scores(frame)
                if episodes != evaluation.episode_id.tolist():
                    raise ValueError("decision ordering changed")
                votes = np.zeros((len(eval_ids), 35))
                for column, (left, right) in enumerate(pairs):
                    left_vote = (preferences[:, column] > 0).astype(float) + 0.5 * (
                        preferences[:, column] == 0
                    )
                    votes[:, left] += left_vote
                    votes[:, right] += 1 - left_vote
                order = sorted(
                    range(35),
                    key=lambda i: (learner.candidate_ids[i] != preferred, learner.candidate_ids[i]),
                )
                winners = np.asarray(order)[votes[:, order].argmax(axis=1)]
                choices = np.array(
                    [info["option_names"].index(learner.candidate_ids[index]) for index in winners]
                )
                options = current_options(evaluation, bank, info["actions"])
                predictions = {
                    kind: options[np.arange(len(eval_ids)), choices],
                    "forecast_median_guarded": options[:, -1],
                }
                with np.load(directory / fold["prediction_path"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(eval_ids, saved["decision_indices"])
                    np.testing.assert_array_equal(choices, saved["choices"])
                    np.testing.assert_array_equal(
                        votes[np.arange(len(winners)), winners], saved["pairwise_wins"]
                    )
                    np.testing.assert_array_equal(predictions[kind], saved["point"])
                    np.testing.assert_array_equal(
                        predictions["forecast_median_guarded"], saved["reference"]
                    )
                future = decision_truth(evaluation, truth)
                reported = pd.read_parquet(directory / fold["scores_path"])
                if (
                    len(reported) != len(evaluation) * 2
                    or reported.duplicated(["episode_id", "method"]).any()
                ):
                    raise ValueError("matched decision scores have different coverage")
                for method, prediction in predictions.items():
                    residual = prediction - future
                    values = np.column_stack(
                        [np.abs(residual).mean(axis=1), np.square(residual).mean(axis=1)]
                    )
                    actual = (
                        reported[reported.method == method]
                        .set_index("episode_id")
                        .loc[evaluation.episode_id][["mae", "mse"]]
                        .to_numpy()
                    )
                    np.testing.assert_allclose(actual, values, rtol=0, atol=1e-10)
                    maximum_difference = max(
                        maximum_difference, float(np.abs(actual - values).max())
                    )
                    verified[kind].append(
                        evaluation.copy().assign(
                            model_id=model, method=method, mae=values[:, 0], mse=values[:, 1]
                        )
                    )
                model_count += 1
                decision_count += len(eval_ids)
    if (model_count, comparator_count, decision_count) != (60, 35700, 11232):
        raise ValueError("the matched audit did not cover all folds and comparisons")
    for model, family, kind in population_records:
        if (
            kind == "member_risk"
            and population_records[(model, family, kind)]
            != population_records[(model, family, "median_risk")]
        ):
            raise ValueError("the compared objectives used different inputs or populations")
    summaries = []
    for kind in studies:
        keys = [
            "model_id",
            "method",
            "source_episode_id",
            "origin_id",
            "family_id",
            "dataset_id",
            "item_id",
        ]
        windows = pd.concat(verified[kind]).groupby(keys)[["mae", "mse"]].mean().reset_index()
        stored = pd.read_parquet(args.study_root / kind / "episode_results.parquet")
        pd.testing.assert_frame_equal(
            windows.sort_values(keys).reset_index(drop=True),
            stored.sort_values(keys).reset_index(drop=True),
            check_dtype=False,
        )
        panels = [
            windows.assign(panel="full_development"),
            windows[windows.source_episode_id.isin(plan["decision_episode_ids"])].assign(
                panel="screening_90"
            ),
        ]
        keys = ["panel", "model_id", "method"]
        expected = (
            pd.concat(panels)
            .groupby([*keys, "family_id"])[["mae", "mse"]]
            .mean()
            .groupby(keys)
            .mean()
            .reset_index()
        )
        stored = pd.read_csv(args.study_root / kind / "summary.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(
            expected.sort_values(keys).reset_index(drop=True),
            stored.sort_values(keys).reset_index(drop=True),
            check_dtype=False,
            rtol=0,
            atol=1e-10,
        )
        expected["label_kind"] = kind
        summaries.append(expected)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifests": sources,
            "script_sha256": file_sha256(Path(__file__)),
            "verified_fold_models": model_count,
            "verified_comparators": comparator_count,
            "verified_decisions": decision_count,
            "maximum_metric_difference": maximum_difference,
            "matched_inputs_and_capacity": True,
            "limits": "source-development objective control; no new independent confirmation",
        },
    )
    print(summary[summary.panel == "full_development"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
