"""Recompute supplementary composition and scores against the frozen primary bank."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402
from evaluate_followup_motm import METHODS, matching_candidate  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "motm-root",
        "forecast-root",
        "policy-root",
        "study-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MoTM supplemental audits")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    motm = json.loads((args.motm_root / "manifest.json").read_text(encoding="utf-8"))
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.prepared_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    references = {row["episode_id"]: row for row in motm["episodes"]}
    summaries, family_tables, model_records, checked = [], [], [], 0
    maximum = 0.0
    for model in ("chronos2", "timesfm2p5"):
        directory = args.study_root / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        forecast_dir, policy_dir = args.forecast_root / model, args.policy_root / model
        base_manifest = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
        base_records = {row["episode_id"]: row for row in base_manifest["predictions"]}
        if manifest["status"] != "completed" or not manifest["parameters_unchanged"]:
            raise ValueError("complete frozen supplementary forecasts first")
        for key, path in (
            ("prepared_sha256", args.prepared_root / "manifest.json"),
            ("motm_sha256", args.motm_root / "manifest.json"),
            ("forecast_sha256", forecast_dir / "manifest.json"),
            ("primary_policy_sha256", policy_dir / "manifest.json"),
        ):
            if manifest["identity"][key] != file_sha256(path):
                raise ValueError("a supplementary source manifest changed")
        with np.load(policy_dir / "policy_predictions.npz", allow_pickle=False) as saved:
            primary, primary_methods = saved["point_z"], saved["methods"].tolist()
            if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                raise ValueError("primary policy order changed")
        predictions = {row["episode_id"]: row for row in manifest["predictions"]}
        expected_ids = {row["episode_id"] for row in prep["episodes"]}
        if set(predictions) != expected_ids or len(manifest["predictions"]) != len(expected_ids):
            raise ValueError("supplemental prediction coverage changed")
        scores = pd.read_parquet(directory / "episode_results.parquet")
        if (
            file_sha256(directory / "episode_results.parquet") != manifest["scores_sha256"]
            or len(scores) != len(expected_ids) * 4
            or scores.duplicated(["episode_id", "method"]).any()
        ):
            raise ValueError("supplemental score coverage changed")
        by_episode = scores.set_index(["episode_id", "method"])
        reused_count = 0
        for index, record in enumerate(prep["episodes"]):
            pred = predictions[record["episode_id"]]
            ref = references[record["episode_id"]]
            base = base_records[record["episode_id"]]
            for path, sha in (
                (directory / pred["path"], pred["sha256"]),
                (args.motm_root / ref["path"], ref["sha256"]),
                (forecast_dir / base["path"], base["sha256"]),
                (args.prepared_root / record["path"], record["sha256"]),
            ):
                if file_sha256(path) != sha:
                    raise ValueError("an input or forecast artifact changed")
            with np.load(args.prepared_root / record["path"], allow_pickle=False) as saved:
                context, candidates, actions = (
                    saved["context"],
                    saved["candidate_values"],
                    saved["candidate_ids"].tolist(),
                )
                future, mask = saved["future"], saved["future_observed"]
            with np.load(args.motm_root / ref["path"], allow_pickle=False) as saved:
                completed = saved["values"]
            if completed.shape != context.shape or not np.isfinite(completed).all():
                raise ValueError("invalid MoTM completed context")
            np.testing.assert_array_equal(
                completed[np.isfinite(context)], context[np.isfinite(context)]
            )
            with np.load(forecast_dir / base["path"], allow_pickle=False) as saved:
                base_points = saved["point_z"]
            with np.load(directory / pred["path"], allow_pickle=False) as saved:
                points, reused = saved["point_z"], int(saved["reused_candidate_index"])
                if saved["methods"].tolist() != list(METHODS):
                    raise ValueError("supplemental method order changed")
            expected_reuse = matching_candidate(
                context, candidates, actions, completed, joint=model == "chronos2"
            )
            if reused != (-1 if expected_reuse is None else expected_reuse):
                raise ValueError("a forecast was reused for a different effective input")
            if reused >= 0:
                np.testing.assert_array_equal(points[2], base_points[reused])
                reused_count += 1
            for position, name in enumerate(METHODS[:2]):
                np.testing.assert_array_equal(
                    points[position], primary[index, primary_methods.index(name)]
                )
            np.testing.assert_array_equal(
                points[3], np.median(np.concatenate([base_points, points[2][None]]), axis=0)
            )
            np.testing.assert_array_equal(mask, np.isfinite(future))
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            residual = points - (future - mean) / scale
            for position, method in enumerate(METHODS):
                saved_row = by_episode.loc[(record["episode_id"], method)]
                if any(
                    saved_row[key] != record[key]
                    for key in ("dataset_id", "family_id", "item_id", "origin_id", "panel")
                ):
                    raise ValueError("a supplementary score has the wrong metadata")
                for metric, value in (
                    ("mae", abs(residual)),
                    ("mse", residual**2),
                    ("raw_mae", abs(residual * scale)),
                    ("raw_mse", (residual * scale) ** 2),
                ):
                    expected = np.mean(
                        [value[position, mask[:, target], target].mean() for target in range(2)]
                    )
                    maximum = max(maximum, abs(expected - saved_row[metric]))
                    np.testing.assert_allclose(saved_row[metric], expected, rtol=1e-10, atol=1e-10)
                checked += 1
        saved_summary = pd.read_csv(directory / "summary.csv").set_index(["panel", "method"])
        for name, panel in result_panels(scores):
            keys, metrics = (
                ["method", "family_id", "dataset_id", "item_id"],
                ["mae", "mse", "raw_mae", "raw_mse"],
            )
            item = panel.groupby(keys)[metrics].mean()
            dataset = item.groupby(keys[:-1])[metrics].mean()
            family = dataset.groupby(keys[:-2])[metrics].mean()
            summary = family.groupby("method")[metrics].mean()
            np.testing.assert_allclose(
                saved_summary.loc[name].loc[summary.index, metrics], summary, rtol=1e-12, atol=1e-12
            )
            summaries.append(
                summary.reset_index().assign(
                    model_id=model, panel=name, families=panel.family_id.nunique()
                )
            )
            family_tables.append(family.reset_index().assign(model_id=model, panel=name))
        model_records.append(
            {
                "model_id": model,
                "manifest_sha256": file_sha256(directory / "manifest.json"),
                "reused_forecasts": reused_count,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    pd.concat(family_tables, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "models": model_records,
            "verified_policy_scores": checked,
            "maximum_metric_difference": maximum,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "limits": "late supplementary comparator; standalone MoTM and eight-member median budgets differ from the primary selector; Solar-trained reference component disclosed",
        },
    )


if __name__ == "__main__":
    main()
