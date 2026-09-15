"""Replay all fixed R10 metric objectives on already-used target forecast banks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from apply_followup_policies import result_panels
from audit_shared_forecast_gate import replay_network
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs
from evaluate_r6_policies import restore
from latent_source_inputs import ROOT, read_json
from native_source_transfer_io import input_arguments
from run_native_confirmation import hierarchical_metrics
from train_calibrated_source_gates import probability_from_state

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors
from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.set_defaults(comparison_root=ROOT / "artifacts/iclr27-r9/source-expansion-transfer-v001")
    for name, path in {
        "study-root": "artifacts/iclr27-r10/metric-source-v002",
        "study-audit": "artifacts/iclr27-r10/metric-source-audit-v002",
        "reference-root": "artifacts/iclr27-r7/latent-source-gates-v001",
        "reference-audit": "artifacts/iclr27-r7/latent-source-audit-v001",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "protocol": "docs/iclr2027/R10_TRANSFER_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source transfer results")
    studies = {
        "metric": (args.study_root, read_json(args.study_root / "manifest.json")),
        "r7": (args.reference_root, read_json(args.reference_root / "manifest.json")),
    }
    for name, audit_root in (("metric", args.study_audit), ("r7", args.reference_audit)):
        audit = read_json(audit_root / "manifest.json")
        if (
            audit["status"] != "completed"
            or audit["verified_checkpoints"] != 384
            or audit["study_sha256"] != file_sha256(studies[name][0] / "manifest.json")
        ):
            raise ValueError("complete and audit both source studies first")
    cohorts = {
        "r6": (args.r6_prepared, read_json(args.r6_prepared / "manifest.json"), (96, 192)),
        "legacy_native": (
            args.legacy_input / "prepared",
            read_json(args.legacy_input / "prepared/manifest.json"),
            (96,),
        ),
    }
    scalers = {}
    for name, (root, prep, _) in cohorts.items():
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("target prefix statistics changed")
        scalers[name] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "source_manifests": {
            name: file_sha256(root / "manifest.json") for name, (root, _) in studies.items()
        },
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "target_loader_sha256": file_sha256(ROOT / "scripts/evaluate_r6_geometry_gates.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    actions = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    banks, maximum_delta, checked = [], 0.0, 0
    for model_id in ("chronos2", "timesfm2p5"):
        base_model = read_json(args.base_root / model_id / "manifest.json")
        with np.load(
            args.base_root / model_id / base_model["episodes"][0]["path"], allow_pickle=False
        ) as saved:
            if saved["actions"].tolist() != actions:
                raise ValueError("the source candidate order changed")
        configurations = [
            ("metric", f"metric_{condition}", condition, condition.split("_")[1])
            for condition in ("mae_teacher", "mae_future", "joint_teacher", "joint_future")
        ] + [("r7", f"r7_point_{label}", None, label) for label in ("teacher", "future")]
        loaded = {}
        for study_name, method, regime, label in configurations:
            root, study = studies[study_name]
            entries = [
                row
                for row in study["checkpoints"]
                if row["model_id"] == model_id
                and row["held_family"] is None
                and (
                    row.get("condition") == regime
                    if study_name == "metric"
                    else row.get("condition") == f"point_{label}"
                )
            ]
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("a transfer procedure has missing source seeds")
            loaded[method] = []
            for entry in entries:
                path = root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a fixed full-source checkpoint changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if len(set(saved["training_origins"])) != 165:
                    raise ValueError("source population changed")
                loaded[method].append(saved)
        for cohort, (_, prep, horizons) in cohorts.items():
            target_origins = {row["origin_id"] for row in prep["episodes"]}
            target_families = {row["family_id"] for row in prep["episodes"]}
            for models in loaded.values():
                for saved in models:
                    if (
                        set(saved["training_origins"]) & target_origins
                        or set(saved["training_families"]) & target_families
                    ):
                        raise ValueError("a target family or history entered source fitting")
            for horizon in horizons:
                decisions, base, vectors, _, median = (
                    r6_inputs(args, model_id, horizon, actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, actions, prep, scalers[cohort])
                )
                features = np.ascontiguousarray(np.pad(base, ((0, 0), (0, 0), (0, 64))))
                predictions = {}
                for method, models in loaded.items():
                    weights = []
                    for _seed, saved in zip((5101, 5102, 5103), models, strict=True):
                        probability = probability_from_state(saved["state_dict"], features)
                        np.testing.assert_array_equal(
                            probability, replay_network(saved["state_dict"], features)
                        )
                        weights.append(probability)
                    for name, probability in [
                        (method, np.mean(weights, axis=0)),
                        *[
                            (f"{method}_seed{seed}", weight)
                            for seed, weight in zip((5101, 5102, 5103), weights, strict=True)
                        ],
                    ]:
                        point = compose_forecasts(vectors, probability)
                        probability = probability / probability.sum(1, keepdims=True)
                        direct = (vectors * probability[:, :, None]).sum(1)
                        maximum_delta = max(maximum_delta, float(abs(point - direct).max()))
                        np.testing.assert_allclose(point, direct, rtol=1e-12, atol=1e-12)
                        predictions[name] = restore(
                            point, decisions, len(prep["episodes"]), horizon, model_id == "chronos2"
                        )
                        checked += len(decisions)
                for source_name, (root, _) in studies.items():
                    controls = (
                        {
                            condition: read_json(
                                root / model_id / "full_source" / f"fixed_{condition}.json"
                            )
                            for condition in (
                                "mae_teacher",
                                "mae_future",
                                "joint_teacher",
                                "joint_future",
                            )
                        }
                        if source_name == "metric"
                        else read_json(root / model_id / "full_source/controls.json")
                    )
                    for label in controls:
                        probability = np.broadcast_to(
                            controls[label]["weights"], (len(decisions), 7)
                        )
                        predictions[f"{source_name}_fixed_{label}"] = restore(
                            compose_forecasts(vectors, probability),
                            decisions,
                            len(prep["episodes"]),
                            horizon,
                            model_id == "chronos2",
                        )
                        predictions[f"{source_name}_single_{label}"] = restore(
                            vectors[:, controls[label]["single_index"]],
                            decisions,
                            len(prep["episodes"]),
                            horizon,
                            model_id == "chronos2",
                        )
                predictions["forecast_median_guarded"] = restore(
                    np.median(vectors, axis=1),
                    decisions,
                    len(prep["episodes"]),
                    horizon,
                    model_id == "chronos2",
                )
                np.testing.assert_array_equal(predictions["forecast_median_guarded"], median)
                path = output / cohort / model_id / f"h{horizon}_predictions.npz"
                _save_npz(
                    path,
                    point_z=np.stack(list(predictions.values()), axis=1),
                    methods=np.asarray(list(predictions)),
                    episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                )
                banks.append(
                    {
                        "cohort": cohort,
                        "model_id": model_id,
                        "horizon": horizon,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
        print(f"{model_id}: all fixed transfer predictions saved", flush=True)
    _write_json(
        output / "prediction_freeze.json",
        {"banks": banks, "identity": identity, "target_future_arrays_read": False},
    )
    records, metric_delta = [], 0.0
    for bank_entry in banks:
        cohort, horizon = bank_entry["cohort"], bank_entry["horizon"]
        root, prep, _ = cohorts[cohort]
        with np.load(output / bank_entry["path"], allow_pickle=False) as saved:
            bank, names = saved["point_z"], saved["methods"].tolist()
        if len(names) != 37:
            raise ValueError("the prespecified transfer method set changed")
        for index, row in enumerate(prep["episodes"]):
            path = root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a target future window changed")
            with np.load(path, allow_pickle=False) as saved:
                truth, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            scaler = scalers[cohort][(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            raw = bank[index] * scale + mean
            errors, _ = observed_future_errors(
                raw, truth, observed, scale, minimum_observed=horizon // 2
            )
            direct = [
                (raw[:, :, slot][:, observed[:, slot]] - truth[observed[:, slot], slot][None])
                / scale[slot]
                for slot in (0, 1)
            ]
            for metric, values in (
                ("mae", np.mean([abs(delta).mean(1) for delta in direct], axis=0)),
                ("mse", np.mean([(delta**2).mean(1) for delta in direct], axis=0)),
            ):
                metric_delta = max(metric_delta, float(abs(values - errors[metric].mean(1)).max()))
                np.testing.assert_allclose(values, errors[metric].mean(1), rtol=1e-12, atol=1e-12)
            records.extend(
                {
                    **{
                        key: row[key]
                        for key in ("episode_id", "origin_id", "family_id", "dataset_id", "item_id")
                    },
                    "cohort": cohort,
                    "model_id": bank_entry["model_id"],
                    "horizon": horizon,
                    "method": method,
                    "panel": row.get("panel", "legacy_native"),
                    "native_missing_context": row["window"]["context_has_missing"],
                    **{name: float(value[position].mean()) for name, value in errors.items()},
                }
                for position, method in enumerate(names)
            )
    frame = pd.DataFrame(records)
    families, summaries = [], []
    for (cohort, _, horizon), group in frame.groupby(["cohort", "model_id", "horizon"]):
        panels = (
            result_panels(group)
            if cohort == "r6"
            else (
                ("all_registered", group),
                ("naturally_missing", group[group.native_missing_context]),
                ("complete_context", group[~group.native_missing_context]),
            )
        )
        for panel_name, panel in panels:
            _, family, summary = hierarchical_metrics(panel)
            families.append(family.assign(cohort=cohort, horizon=horizon, panel=panel_name))
            summaries.append(
                summary.assign(
                    cohort=cohort,
                    horizon=horizon,
                    panel=panel_name,
                    families=panel.family_id.nunique(),
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    original = pd.read_csv(
        args.comparison_root / "comparison_summary.csv", float_precision="round_trip"
    )
    keys = ["cohort", "model_id", "horizon", "panel", "method"]
    actual = summary[summary.method == "forecast_median_guarded"].set_index(keys)
    expected = original.set_index(keys).loc[actual.index]
    np.testing.assert_allclose(
        actual[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    frame.to_parquet(output / "episode_results.parquet", index=False)
    shared = summary[~summary.method.str.startswith("metric_")].set_index(keys)
    reference = original.set_index(keys).loc[shared.index]
    np.testing.assert_allclose(
        shared[["mae", "mse"]], reference[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    pd.concat(
        [original, summary[summary.method.str.startswith("metric_")]], ignore_index=True
    ).to_csv(output / "comparison_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "prediction_banks": banks,
            "replayed_decision_predictions": checked,
            "maximum_prediction_difference": maximum_delta,
            "maximum_metric_difference": metric_delta,
            "score_rows": len(frame),
            "new_fits": 0,
            "new_forecaster_calls": 0,
            "limits": "short transfer evaluation on previously used target cohorts; retains all source conditions and original comparisons",
        },
    )


if __name__ == "__main__":
    main()
