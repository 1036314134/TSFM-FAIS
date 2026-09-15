"""Independently aggregate votes, reconstruct all policies and recompute follow-up errors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import load_frozen_selector, result_panels  # noqa: E402

from tsfm_fais.routing.budgeted_portfolio import rank_pairwise_candidates  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def explicit_votes(learner, frame):
    ids, margins, pairs = learner.pair_scores(frame)
    votes = np.zeros((len(ids), len(learner.candidate_ids)))
    for column, (left, right) in enumerate(pairs):
        votes[:, left] += (margins[:, column] > 0) + 0.5 * (margins[:, column] == 0)
        votes[:, right] += (margins[:, column] < 0) + 0.5 * (margins[:, column] == 0)
    preference = sorted(
        range(len(learner.candidate_ids)),
        key=lambda i: (learner.candidate_ids[i] != learner.baseline_id, learner.candidate_ids[i]),
    )
    winner = np.asarray(preference)[votes[:, preference].argmax(axis=1)]
    return pd.DataFrame(
        {"episode_id": ids, "candidate_id": np.asarray(learner.candidate_ids)[winner]}
    )


def median_from_names(points, actions, selected):
    if len(selected) != 3 or len(set(selected)) != 3 or not set(selected) <= set(actions):
        raise ValueError("a portfolio must contain three distinct registered candidates")
    return np.median(points[[actions.index(name) for name in selected]], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "forecast-root",
        "policy-root",
        "source-bundle",
        "old-bundle",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed follow-up audits")
    output.mkdir(parents=True, exist_ok=True)
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((args.old_bundle / "manifest.json").read_text(encoding="utf-8"))
    controls = json.loads((args.old_bundle / old["controls_file"]).read_text(encoding="utf-8"))
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.prepared_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    maximum_prediction, maximum_metric, checked_decisions, checked_vectors = 0.0, 0.0, 0, 0
    model_records = []
    for model in ("chronos2", "timesfm2p5"):
        directory = args.policy_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        policy = json.loads((directory / "policy_predictions.json").read_text(encoding="utf-8"))
        forecast_dir = args.forecast_root / model
        forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed" or manifest["identity"][
            "prepared_sha256"
        ] != file_sha256(args.prepared_root / "manifest.json"):
            raise ValueError("the policy run uses a different preparation")
        for key, path in (
            ("source_bundle_sha256", args.source_bundle / "manifest.json"),
            ("old_bundle_sha256", args.old_bundle / "manifest.json"),
            ("forecast_sha256", forecast_dir / "manifest.json"),
        ):
            if manifest["identity"][key] != file_sha256(path):
                raise ValueError("a policy input manifest changed")
        for name, sha in policy["files_sha256"].items():
            if file_sha256(directory / name) != sha:
                raise ValueError("a policy decision or feature table changed")
        if file_sha256(directory / "policy_predictions.npz") != policy["prediction_sha256"]:
            raise ValueError("policy forecasts changed after scoring")
        features = pd.read_parquet(directory / "portfolio_features.parquet")
        old_features = pd.read_parquet(directory / "individual_features.parquet")
        choices = {}
        for kind in ("median_risk", "member_risk"):
            learner, _ = load_frozen_selector(args.source_bundle, bundle, model, kind, "label_kind")
            if len(learner.models) != 595:
                raise ValueError("the matched comparator count changed")
            regenerated = (
                explicit_votes(learner, features).set_index("episode_id").candidate_id.sort_index()
            )
            saved = (
                pd.read_parquet(directory / f"{kind}_choices.parquet")
                .set_index("episode_id")
                .candidate_id.sort_index()
            )
            pd.testing.assert_series_equal(regenerated, saved)
            choices[kind] = saved.to_dict()
            checked_decisions += len(saved)
            if kind == "median_risk" and policy["source_fixed_median_risk"] != learner.baseline_id:
                raise ValueError("the source-fixed baseline differs from the frozen model")
            del learner
        learner, _ = load_frozen_selector(
            args.old_bundle, old, model, "clean_forecast_mse", "objective"
        )
        ids, margins, pairs = learner.pair_scores(old_features)
        ranks = rank_pairwise_candidates(margins, pairs, learner.candidate_ids, learner.baseline_id)
        rankings = {
            identifier: [learner.candidate_ids[index] for index in row]
            for identifier, row in zip(ids, ranks, strict=True)
        }
        if rankings != json.loads(
            (directory / "old_teacher_rankings.json").read_text(encoding="utf-8")
        ):
            raise ValueError("the original teacher ranking did not replay")
        del learner
        with np.load(directory / "policy_predictions.npz", allow_pickle=False) as saved:
            bank, methods, episode_ids = (
                saved["point_z"],
                saved["methods"].tolist(),
                saved["episode_ids"].tolist(),
            )
        if (
            episode_ids != [row["episode_id"] for row in prep["episodes"]]
            or len(set(methods)) != 16
        ):
            raise ValueError("policy coverage or method identities changed")
        scores = pd.read_parquet(directory / "episode_results.parquet")
        if (
            len(scores) != len(episode_ids) * len(methods)
            or scores.duplicated(["episode_id", "method"]).any()
        ):
            raise ValueError("scoring silently lost or duplicated policy windows")
        score_index = scores.set_index(["episode_id", "method"])
        predicted = {row["episode_id"]: row for row in forecast["predictions"]}
        for index, record in enumerate(prep["episodes"]):
            row = predicted[record["episode_id"]]
            path = forecast_dir / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a base candidate prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                points, actions = saved["point_z"], saved["candidate_ids"].tolist()
            expected = dict(zip(actions, points, strict=True))
            for name in (
                "median_risk",
                "member_risk",
                "source_fixed_median_risk",
                "old_teacher_rank3",
                "old_source_fixed3",
            ):
                targets = []
                for target in range(2):
                    identifier = (
                        record["episode_id"]
                        if model == "chronos2"
                        else record["episode_id"] + f"|target={target}"
                    )
                    if name in choices:
                        label = choices[name][identifier]
                        if not label.startswith("median:"):
                            raise ValueError("invalid frozen portfolio identity")
                        selected = label[len("median:") :].split("+")
                    elif name == "source_fixed_median_risk":
                        selected = policy[name][len("median:") :].split("+")
                    elif name == "old_teacher_rank3":
                        selected = rankings[identifier][:3]
                    else:
                        selected = controls[model]["fixed3_clean_forecast_mse"][target]
                    targets.append(median_from_names(points[:, :, target], actions, selected))
                expected[name] = np.stack(targets, axis=1)
            expected.update(
                forecast_median_guarded=np.median(points, axis=0),
                forecast_mean_guarded=points.mean(axis=0),
                forecast_median_finite=np.median(points[:6], axis=0),
                forecast_mean_finite=points[:6].mean(axis=0),
            )
            rebuilt = np.stack([expected[name] for name in methods])
            np.testing.assert_array_equal(rebuilt, bank[index])
            maximum_prediction = max(maximum_prediction, float(np.abs(rebuilt - bank[index]).max()))
            checked_vectors += len(methods)
            with np.load(args.prepared_root / record["path"], allow_pickle=False) as saved:
                future, mask = saved["future"], saved["future_observed"]
            np.testing.assert_array_equal(mask, np.isfinite(future))
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            residual = bank[index] - (future - mean) / scale
            raw = residual * scale
            for method_index, method in enumerate(methods):
                saved_row = score_index.loc[(record["episode_id"], method)]
                if any(
                    saved_row[key] != record[key]
                    for key in (
                        "origin_id",
                        "dataset_id",
                        "family_id",
                        "item_id",
                        "panel",
                        "mechanism",
                        "missing_rate",
                        "mask_seed",
                    )
                ):
                    raise ValueError("a score was assigned to the wrong evaluation condition")
                if saved_row.native_missing_context != record["window"]["context_has_missing"]:
                    raise ValueError("natural and synthetic missingness labels were mixed")
                for metric, value in (
                    ("mae", np.abs(residual)),
                    ("mse", residual**2),
                    ("raw_mae", np.abs(raw)),
                    ("raw_mse", raw**2),
                ):
                    expected_value = np.mean(
                        [value[method_index, mask[:, target], target].mean() for target in range(2)]
                    )
                    maximum_metric = max(maximum_metric, abs(expected_value - saved_row[metric]))
                    np.testing.assert_allclose(
                        saved_row[metric], expected_value, rtol=1e-10, atol=1e-10
                    )
        actual_summary = pd.read_csv(directory / "summary.csv").set_index(["panel", "method"])
        for panel_name, panel in result_panels(scores):
            keys = ["method", "family_id", "dataset_id", "item_id"]
            metrics = ["mae", "mse", "raw_mae", "raw_mse"]
            item = panel.groupby(keys)[metrics].mean()
            dataset = item.groupby(keys[:-1])[metrics].mean()
            family = dataset.groupby(keys[:-2])[metrics].mean()
            summary = family.groupby("method")[metrics].mean()
            np.testing.assert_allclose(
                actual_summary.loc[panel_name].loc[summary.index, metrics],
                summary,
                rtol=1e-12,
                atol=1e-12,
            )
        model_records.append(
            {
                "model_id": model,
                "manifest_sha256": file_sha256(directory / "manifest.json"),
                "episodes": len(episode_ids),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "models": model_records,
            "verified_new_selector_decisions": checked_decisions,
            "verified_policy_vectors": checked_vectors,
            "maximum_prediction_difference": maximum_prediction,
            "maximum_metric_difference": maximum_metric,
            "limits": "forecast decisions and metrics audited; uncertainty and generalization claims require the separate readout",
        },
    )


if __name__ == "__main__":
    main()
