"""Describe changed-gate transfer on already-used follow-up data, without refitting."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aligned_portfolio_io import decision_vectors  # noqa: E402
from apply_followup_policies import result_panels  # noqa: E402
from audit_shared_forecast_gate import replay_network  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "previous-policy-root",
        "motm-study-root",
        "source-bundle",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed transfer diagnostics")
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        bundle["status"] != "completed"
        or Path(bundle["identity"]["diagnostic_root"]).resolve() != output
        or len(prep["episodes"]) != 823
    ):
        raise ValueError(
            "complete the frozen source gates and retain the declared diagnostic destination"
        )
    if (
        file_sha256(ROOT / "src/tsfm_fais/routing/forecast_gate.py")
        != bundle["identity"]["module_sha256"]
    ):
        raise ValueError("the frozen gate changed")
    controls_path = args.source_bundle / "controls.json"
    if file_sha256(controls_path) != bundle["controls_sha256"]:
        raise ValueError("source-fixed gate controls changed")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("prefix scoring statistics changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "data_role": "previously used 823-task follow-up; diagnostic development reuse, not independent confirmation",
        "new_fits": 0,
        "new_forecaster_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial diagnostic identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    torch.set_num_threads(1)
    model_records, summaries = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        directory = output / model_id
        directory.mkdir(exist_ok=True)
        old_dir = args.previous_policy_root / model_id
        previous = json.loads((old_dir / "policy_predictions.json").read_text(encoding="utf-8"))
        feature_path = old_dir / "individual_features.parquet"
        if (
            file_sha256(feature_path) != previous["files_sha256"][feature_path.name]
            or file_sha256(old_dir / "policy_predictions.npz") != previous["prediction_sha256"]
        ):
            raise ValueError("the audited current-input features or previous predictions changed")
        frame = pd.read_parquet(feature_path)
        decisions = frame.drop_duplicates("episode_id")[
            ["episode_id", "source_episode_id", "episode_index", "target_slot"]
        ].reset_index(drop=True)
        expected = (
            {(index, -1) for index in range(823)}
            if model_id == "chronos2"
            else {(index, target) for index in range(823) for target in (0, 1)}
        )
        if set(zip(decisions.episode_index, decisions.target_slot, strict=True)) != expected or len(
            decisions
        ) != len(expected):
            raise ValueError("the diagnostic decisions changed target coverage")
        for row in decisions.itertuples(index=False):
            if row.source_episode_id != prep["episodes"][row.episode_index]["episode_id"]:
                raise ValueError("a current decision was mapped to another history")
        actions = controls[model_id]["actions"]
        order = pd.MultiIndex.from_product(
            [decisions.episode_id, actions], names=["episode_id", "candidate_id"]
        )
        features = (
            frame.set_index(["episode_id", "candidate_id"])
            .loc[order, list(FORECAST_FEATURES)]
            .to_numpy(np.float32)
            .reshape(len(decisions), 7, 33)
        )
        with np.load(old_dir / "policy_predictions.npz", allow_pickle=False) as saved:
            old_points, old_methods = saved["point_z"], saved["methods"].tolist()
            if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                raise ValueError("previous prediction order changed")
        bank = old_points[:, [old_methods.index(name) for name in actions]]
        vectors = decision_vectors(decisions, bank)
        predictions, weight_records = {}, {}
        for objective in ("ensemble", "member"):
            weights = []
            entries = [
                row
                for row in bundle["models"]
                if row["model_id"] == model_id and row["objective"] == objective
            ]
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("all three fixed seeds must be used in order")
            for entry in entries:
                path = args.source_bundle / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a frozen source gate changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["identity_sha256"] != bundle["identity_sha256"]
                    or len(saved["training_origins"]) != 165
                    or len(saved["training_families"]) != 15
                ):
                    raise ValueError("the source gate fitting population changed")
                model = SharedForecastGate()
                model.load_state_dict(saved["state_dict"])
                model.eval()
                current = predict_weights(model, features)
                np.testing.assert_array_equal(
                    current, replay_network(saved["state_dict"], features)
                )
                weights.append(current)
            averaged = np.mean(weights, axis=0)
            weight_records[objective] = averaged
            predictions[objective + "_gate"] = compose_forecasts(vectors, averaged)
        fixed = np.repeat(
            np.asarray(controls[model_id]["convex_weights"])[None], len(decisions), axis=0
        )
        predictions["gate_source_fixed_convex"] = compose_forecasts(vectors, fixed)
        predictions["gate_source_fixed_single"] = vectors[:, controls[model_id]["single_index"]]
        new_points = np.full((823, 4, 96, 2), np.nan)
        for position, values in enumerate(predictions.values()):
            for index, row in enumerate(decisions.itertuples(index=False)):
                if row.target_slot == -1:
                    new_points[row.episode_index, position] = values[index].reshape(96, 2)
                else:
                    new_points[row.episode_index, position, :, row.target_slot] = values[index]
        motm_dir = args.motm_study_root / model_id
        motm = json.loads((motm_dir / "manifest.json").read_text(encoding="utf-8"))
        motm_map = {row["episode_id"]: row for row in motm["predictions"]}
        extras = []
        for row in prep["episodes"]:
            record = motm_map[row["episode_id"]]
            path = motm_dir / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a supplementary comparator changed")
            with np.load(path, allow_pickle=False) as saved:
                names = saved["methods"].tolist()
                extras.append(
                    saved["point_z"][
                        [names.index("motm_reference"), names.index("forecast_median_with_motm")]
                    ]
                )
        methods = [*old_methods, "motm_reference", "forecast_median_with_motm", *predictions]
        all_points = np.concatenate([old_points, np.stack(extras), new_points], axis=1)
        if (
            len(set(methods)) != 22
            or all_points.shape != (823, 22, 96, 2)
            or not np.isfinite(all_points).all()
        ):
            raise ValueError("the complete diagnostic prediction bank is invalid")
        path = directory / "predictions.npz"
        _save_npz(
            path,
            point_z=all_points,
            methods=np.asarray(methods),
            identity_sha256=np.asarray(identity_sha),
            **weight_records,
        )
        _write_json(
            directory / "predictions_frozen.json",
            {
                "sha256": file_sha256(path),
                "future_arrays_read": False,
                "data_role": identity["data_role"],
            },
        )
        rows = []
        for index, record in enumerate(prep["episodes"]):
            source = args.prepared_root / record["path"]
            if file_sha256(source) != record["sha256"]:
                raise ValueError("the audited future changed")
            with np.load(source, allow_pickle=False) as saved:
                future, mask = saved["future"], saved["future_observed"]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            errors, _ = observed_future_errors(
                all_points[index] * scale + mean, future, mask, scale
            )
            for position, method in enumerate(methods):
                rows.append(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "episode_id",
                                "origin_id",
                                "dataset_id",
                                "family_id",
                                "item_id",
                                "panel",
                            )
                        },
                        "model_id": model_id,
                        "method": method,
                        "native_missing_context": record["window"]["context_has_missing"],
                        **{name: float(value[position].mean()) for name, value in errors.items()},
                    }
                )
        scores = pd.DataFrame(rows)
        previous_scores = pd.read_parquet(old_dir / "episode_results.parquet").set_index(
            ["episode_id", "method"]
        )
        replayed = (
            scores[scores.method.isin(old_methods)]
            .set_index(["episode_id", "method"])
            .loc[previous_scores.index]
        )
        np.testing.assert_array_equal(
            replayed[["mae", "mse", "raw_mae", "raw_mse"]],
            previous_scores[["mae", "mse", "raw_mae", "raw_mse"]],
        )
        scores.to_parquet(directory / "episode_results.parquet", index=False)
        for name, panel in result_panels(scores):
            _, families, summary = hierarchical_metrics(panel)
            families.to_csv(directory / f"{name}_family_metrics.csv", index=False)
            summaries.append(summary.assign(panel=name, families=panel.family_id.nunique()))
        model_records.append(
            {
                "model_id": model_id,
                "predictions_sha256": file_sha256(path),
                "scores_sha256": file_sha256(directory / "episode_results.parquet"),
            }
        )
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": model_records,
            "gate_weight_replay": "exact",
            "previous_scores_replay": "exact",
            "limits": "used-data transfer diagnostic only; a successful result still needs unused confirmation data",
        },
    )


if __name__ == "__main__":
    main()
