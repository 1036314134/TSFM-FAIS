"""Evaluate the frozen positional source models on both previously used target cohorts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs  # noqa: E402
from positional_forecast_portfolio import (  # noqa: E402
    PositionalPortfolio,
    position_inputs,
    predict_position,
)
from positional_portfolio_io import restore_positions, target_nodes  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "study-root",
        "study-audit",
        "r6-prepared",
        "r6-policy",
        "r6-audit",
        "legacy-input",
        "legacy-results",
        "comparison-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed positional target evaluations")
    study, audit = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.study_audit / "manifest.json"),
    )
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.study_root / "manifest.json")
        or audit["verified_checkpoints"] != 192
    ):
        raise ValueError("independently audit the complete source study before transfer")
    for key, name in (
        ("model_module_sha256", "positional_forecast_portfolio.py"),
        ("io_module_sha256", "positional_portfolio_io.py"),
    ):
        if file_sha256(ROOT / "scripts" / name) != study["identity"][key]:
            raise ValueError("positional inference changed after fitting")
    cohorts = {
        "r6": (args.r6_prepared, read_json(args.r6_prepared / "manifest.json"), (96, 192)),
        "legacy_native": (
            args.legacy_input / "prepared",
            read_json(args.legacy_input / "prepared/manifest.json"),
            (96,),
        ),
    }
    scalers = {}
    for cohort, (root, prep, _) in cohorts.items():
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("original prefix normalization changed")
        scalers[cohort] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "study_audit_sha256": file_sha256(args.study_audit / "manifest.json"),
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "target_input_loader_sha256": file_sha256(ROOT / "scripts/evaluate_r6_geometry_gates.py"),
        "seed_aggregation": "average the three bounded predictions, with the median as numerical anchor",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    banks, checked_predictions, range_violations = [], 0, 0
    names = ["position_local", "position_pooled", "forecast_median_guarded"]
    for model_id in ("chronos2", "timesfm2p5"):
        models = [row for row in study["source_models"] if row["model_id"] == model_id]
        actions = models[0]["actions"]
        if any(row["actions"] != actions for row in models):
            raise ValueError("source candidate identities changed")
        for cohort, (_root, prep, horizons) in cohorts.items():
            for horizon in horizons:
                loaded = (
                    r6_inputs(args, model_id, horizon, actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, actions, prep, scalers[cohort])
                )
                original_decisions, base, vectors, _candidate_bank, median = loaded
                decisions, base, points = target_nodes(
                    original_decisions, base, vectors, joint=model_id == "chronos2"
                )
                inputs = position_inputs(base, points)
                np.testing.assert_array_equal(
                    restore_positions(inputs["median"], decisions, len(prep["episodes"]), horizon),
                    median,
                )
                predictions = []
                saved_points = {}
                for mode in ("local", "pooled"):
                    selected = [row for row in models if row["mode"] == mode]
                    if [row["seed"] for row in selected] != [5101, 5102, 5103]:
                        raise ValueError("a target condition lost a source seed")
                    outputs = []
                    for entry in selected:
                        path = args.study_root / entry["path"]
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError("a full source checkpoint changed")
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (
                            saved["identity_sha256"] != study["identity_sha256"]
                            or len(set(saved["training_origins"])) != 165
                        ):
                            raise ValueError("a model belongs to another source population")
                        if set(saved["training_origins"]) & {
                            row["origin_id"] for row in prep["episodes"]
                        }:
                            raise ValueError("a target origin entered source fitting")
                        model = PositionalPortfolio(mode).eval().requires_grad_(False)
                        model.load_state_dict(saved["state_dict"])
                        point = predict_position(model, inputs)
                        range_violations += int(
                            np.count_nonzero((point < inputs["lower"]) | (point > inputs["upper"]))
                        )
                        outputs.append(point)
                        checked_predictions += len(point)
                    reference = inputs["median"]
                    average = np.clip(
                        reference + np.mean(np.stack(outputs) - reference[None], axis=0),
                        inputs["lower"],
                        inputs["upper"],
                    )
                    predictions.append(
                        restore_positions(average, decisions, len(prep["episodes"]), horizon)
                    )
                    saved_points[mode + "_seed_points"] = np.stack(outputs)
                predictions.append(median)
                bank = np.stack(predictions, axis=1)
                if (
                    bank.shape != (len(prep["episodes"]), 3, horizon, 2)
                    or not np.isfinite(bank).all()
                ):
                    raise ValueError("the positional target forecast coverage changed")
                path = output / cohort / model_id / f"h{horizon}_predictions.npz"
                _save_npz(
                    path,
                    point_z=bank,
                    methods=np.asarray(names),
                    episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                    **saved_points,
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
    if range_violations:
        raise ValueError("a positional seed prediction left its candidate range")
    _write_json(
        output / "prediction_freeze.json",
        {"banks": banks, "identity": identity, "evaluation_future_arrays_read": False},
    )
    records, metric_difference = [], 0.0
    for entry in banks:
        cohort, horizon = entry["cohort"], entry["horizon"]
        root, prep, _ = cohorts[cohort]
        path = output / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a fixed target prediction bank changed")
        with np.load(path, allow_pickle=False) as saved:
            bank = saved["point_z"]
        for index, row in enumerate(prep["episodes"]):
            path = root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an original target scoring window changed")
            with np.load(path, allow_pickle=False) as saved:
                truth, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            scaler = scalers[cohort][(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            raw = bank[index] * scale + mean
            errors, _ = observed_future_errors(
                raw, truth, observed, scale, minimum_observed=horizon // 2
            )
            direct_mae, direct_mse = [], []
            for slot in (0, 1):
                delta = (
                    raw[:, :, slot][:, observed[:, slot]] - truth[observed[:, slot], slot][None]
                ) / scale[slot]
                direct_mae.append(abs(delta).mean(1))
                direct_mse.append((delta**2).mean(1))
            for metric, value in (
                ("mae", np.mean(direct_mae, axis=0)),
                ("mse", np.mean(direct_mse, axis=0)),
            ):
                metric_difference = max(
                    metric_difference, float(abs(value - errors[metric].mean(1)).max())
                )
                np.testing.assert_allclose(value, errors[metric].mean(1), rtol=1e-12, atol=1e-12)
            records.extend(
                {
                    **{
                        key: row[key]
                        for key in ("episode_id", "origin_id", "family_id", "dataset_id", "item_id")
                    },
                    "cohort": cohort,
                    "model_id": entry["model_id"],
                    "horizon": horizon,
                    "method": method,
                    "panel": row.get("panel", "legacy_native"),
                    "native_missing_context": row["window"]["context_has_missing"],
                    **{name: float(values[position].mean()) for name, values in errors.items()},
                }
                for position, method in enumerate(names)
            )
    frame = pd.DataFrame(records)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    families, summaries = [], []
    for (cohort, _model, horizon), group in frame.groupby(["cohort", "model_id", "horizon"]):
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
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(
        args.comparison_root / "comparison_summary.csv", float_precision="round_trip"
    )
    keys = ["cohort", "model_id", "horizon", "panel", "method"]
    reference = summary[summary.method == "forecast_median_guarded"].set_index(keys)
    expected = original.set_index(keys).loc[reference.index]
    np.testing.assert_allclose(
        reference[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    pd.concat(
        [original, summary[summary.method != "forecast_median_guarded"]], ignore_index=True
    ).to_csv(output / "comparison_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "prediction_banks": banks,
            "seed_target_trajectories_replayed": checked_predictions,
            "range_violations": range_violations,
            "maximum_normalized_metric_difference": metric_difference,
            "score_rows": len(frame),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_fits": 0,
            "limits": "local and pooled modes fixed before readout; both target cohorts already used; preserves all prior baselines and failures",
        },
    )


if __name__ == "__main__":
    main()
