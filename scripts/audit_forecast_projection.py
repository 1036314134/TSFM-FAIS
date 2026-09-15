"""Independently replay fitted projections, mixture weights and forecast scores."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_forecast_projection import row_positions, vectors_for_decisions  # noqa: E402

from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.preforecast import METADATA, decision_keys  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "input-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed projection audits")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or len(manifest["folds"]) != 90:
        raise ValueError("finish all projection target comparisons")
    table_path = args.accuracy_root / "candidate_accuracy.parquet"
    if (
        file_sha256(table_path) != manifest["identity"]["candidate_table_sha256"]
        or file_sha256(args.accuracy_root / "manifest.json")
        != manifest["identity"]["accuracy_manifest_sha256"]
    ):
        raise ValueError("the projection data source changed")
    schema = pq.read_schema(table_path).names
    frame = pd.read_parquet(
        table_path, columns=[name for name in (*METADATA, *FORECAST_FEATURES) if name in schema]
    )
    frame = forecast_response_inputs(
        frame[~frame.candidate_id.isin(["native_missing", "vendor_missing"])]
    )
    origins = frame[["origin_id", "family_id", "split"]].drop_duplicates().set_index("origin_id")
    banks = {}
    for model in ("chronos2", "timesfm2p5"):
        path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(path) != accuracy["prediction_arrays"][path.name]:
            raise ValueError("candidate forecasts changed")
        actions = sorted(
            set(accuracy["action_orders"][model]) - {"native_missing", "vendor_missing"}
        )
        banks[model] = np.load(path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in actions]
        ]
    truth_path = args.accuracy_root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("forecast outcomes changed")
    truth_bank = np.load(truth_path, mmap_mode="r")[:, None]
    audits, verified, max_metric, max_estimate, max_gap = [], 0, 0.0, 0.0, 0.0
    for record in manifest["folds"]:
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a fold record changed")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if (
            fold["identity_sha256"] != manifest["identity_sha256"]
            or fold["source_outcome_supervision"]
            or not fold["source_complete_history_supervision"]
        ):
            raise ValueError("invalid fold identity or supervision boundary")
        training = origins.loc[fold["training_origins"]]
        if set(training.split) != {"train"} or (training.family_id == fold["held_family"]).any():
            raise ValueError("training used an evaluation time or family")
        for kind in ("model", "predictions", "scores"):
            if file_sha256(root / fold[kind + "_path"]) != fold[kind + "_sha256"]:
                raise ValueError("a fitted projection artifact changed")
        learner = joblib.load(root / fold["model_path"])
        if set(learner.feature_names) != set(FORECAST_FEATURES) or set(
            fold["feature_names"]
        ) != set(FORECAST_FEATURES):
            raise ValueError("the regressor uses undeclared inputs")
        model, kind = fold["model_id"], fold["target_kind"]
        actions = sorted(
            set(accuracy["action_orders"][model]) - {"native_missing", "vendor_missing"}
        )
        if list(learner.candidate_ids) != actions or fold["candidate_ids"] != actions:
            raise ValueError("the fitted candidate pool changed")
        evaluation = decision_keys(
            frame[
                (frame.model_id == model)
                & (frame.family_id == fold["held_family"])
                & (frame.split == "validation")
                & frame.target_slot.isin([-1] if model == "chronos2" else [0, 1])
            ]
        )
        decisions = evaluation.drop_duplicates("episode_id")
        row_indices, action_indices = row_positions(evaluation, decisions, actions)
        estimates = np.empty((len(decisions), 7))
        estimates[row_indices, action_indices] = learner.predict(evaluation)
        with np.load(root / fold["predictions_path"], allow_pickle=False) as saved:
            if str(saved["identity_sha256"]) != manifest["identity_sha256"]:
                raise ValueError("saved predictions have a different identity")
            np.testing.assert_array_equal(
                saved["decision_ids"], decisions.episode_id.to_numpy(dtype=str)
            )
            np.testing.assert_array_equal(
                saved["episode_indices"], decisions.episode_index.to_numpy(int)
            )
            np.testing.assert_array_equal(
                saved["target_slots"], decisions.target_slot.to_numpy(int)
            )
            np.testing.assert_allclose(estimates, saved["estimates"], rtol=0, atol=1e-12)
            max_estimate = max(max_estimate, float(np.abs(estimates - saved["estimates"]).max()))
            weights, stored_point = saved["weights"], saved["point_z"]
            points = vectors_for_decisions(decisions, banks[model])
            median = np.sort(points, axis=1)[:, 3]
            changes = points - median[:, None]
            energy = (changes * changes).sum(axis=2) / points.shape[2]
            if kind == "direct_risk":
                alignment = (energy - estimates) / 2
            elif kind == "raw_projection":
                alignment = estimates.copy()
            elif kind == "unit_projection":
                alignment = np.sqrt(energy) * estimates
            else:
                raise ValueError("unknown projection target")
            alignment[energy <= 1e-24] = 0
            if weights.shape != (len(decisions), 8) or weights.min() < 0:
                raise ValueError("weights are outside the declared convex combination")
            np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-10)
            displacement = (changes * weights[:, 1:, None]).sum(axis=1)
            point = median + displacement
            np.testing.assert_allclose(point, stored_point, rtol=0, atol=1e-9)
            gradient = 2 * (
                (changes * displacement[:, None]).sum(axis=2) / points.shape[2] - alignment
            )
            gradient = np.column_stack([np.zeros(len(decisions)), gradient])
            normalized_gap = ((gradient * weights).sum(axis=1) - gradient.min(axis=1)) / saved[
                "objective_scale"
            ]
            if normalized_gap.max() > 1e-7 + 1e-10:
                raise ValueError("independent numerical optimality check failed")
            np.testing.assert_allclose(
                np.maximum(normalized_gap, 0), saved["normalized_duality_gap"], rtol=0, atol=1e-9
            )
            max_gap = max(max_gap, float(normalized_gap.max()))
        truth = vectors_for_decisions(decisions, truth_bank)[:, 0]
        scores = pd.read_parquet(root / fold["scores_path"])
        scores = scores[scores.method == kind].set_index("episode_id").loc[decisions.episode_id]
        expected = np.column_stack(
            [np.abs(point - truth).mean(axis=1), np.square(point - truth).mean(axis=1)]
        )
        np.testing.assert_allclose(expected, scores[["mae", "mse"]].to_numpy(), rtol=0, atol=1e-9)
        max_metric = max(
            max_metric, float(np.abs(expected - scores[["mae", "mse"]].to_numpy()).max())
        )
        verified += len(decisions)
        audits.append(
            {
                "model_id": model,
                "family_id": fold["held_family"],
                "target_kind": kind,
                "decisions": len(decisions),
                "maximum_normalized_duality_gap": float(normalized_gap.max()),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_folds": len(audits),
            "verified_decisions": verified,
            "maximum_estimate_difference": max_estimate,
            "maximum_metric_difference": max_metric,
            "maximum_normalized_duality_gap": max_gap,
            "audits": audits,
            "limits": "fitted-model, input-boundary, optimizer and score audit; does not establish real-risk optimality or independent confirmation",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "verified_folds": len(audits),
                "verified_decisions": verified,
                "maximum_metric_difference": max_metric,
                "maximum_normalized_duality_gap": max_gap,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
