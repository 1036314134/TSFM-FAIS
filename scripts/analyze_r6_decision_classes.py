"""Compare unavailable future oracles under fixed versus coordinate-wise forecast weights."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.forecast_gate import compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "r6-prepared",
        "r6-policy",
        "r6-audit",
        "legacy-input",
        "legacy-results",
        "comparison-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed decision-class diagnostics")
    cohorts = {"r6": args.r6_prepared, "legacy_native": args.legacy_input / "prepared"}
    actions = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    output.mkdir(parents=True, exist_ok=True)
    records, bank_records, max_gap, verified_cases = [], [], 0.0, 0
    for cohort, prepared_root in cohorts.items():
        prep = read_json(prepared_root / "manifest.json")
        scaler_path = prepared_root / "standardizers.json"
        if (
            prep["status"] != "completed"
            or file_sha256(scaler_path) != prep["standardizers_sha256"]
        ):
            raise ValueError("original prepared inputs or scalers changed")
        scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(scaler_path)}
        for model_id in ("chronos2", "timesfm2p5"):
            if cohort == "r6":
                root = args.r6_policy / model_id
                audit = read_json(args.r6_audit / "manifest.json")
                expected = next(row for row in audit["models"] if row["model_id"] == model_id)
                if file_sha256(root / "manifest.json") != expected["manifest_sha256"]:
                    raise ValueError("the audited R6 policy changed")
                entry = next(
                    row
                    for row in read_json(root / "manifest.json")["horizons"]
                    if row["horizon"] == 96
                )
                directory = root / entry["directory"]
                if file_sha256(directory / "predictions_frozen.json") != entry["marker_sha256"]:
                    raise ValueError("the R6 prediction binding changed")
                path = directory / "policy_predictions.npz"
                digest = read_json(directory / "predictions_frozen.json")["prediction_sha256"]
            else:
                audit = read_json(args.legacy_results / "manifest.json")
                if audit["status"] != "completed" or audit["identity"][
                    "input_manifest_sha256"
                ] != file_sha256(prepared_root / "manifest.json"):
                    raise ValueError("legacy source inputs changed")
                entry = next(
                    row
                    for row in read_json(args.legacy_results / "prediction_freeze.json")["banks"]
                    if row["model_id"] == model_id
                )
                path, digest = args.legacy_results / entry["path"], entry["sha256"]
            if file_sha256(path) != digest:
                raise ValueError("a complete forecast bank changed")
            with np.load(path, allow_pickle=False) as saved:
                if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                    raise ValueError("forecast identities or ordering changed")
                order = saved["methods"].tolist()
                bank = saved["point_z"]
            points = bank[:, [order.index(name) for name in actions]]
            median = bank[:, order.index("forecast_median_guarded")]
            median8 = bank[:, order.index("forecast_median_with_motm")]
            truth, masks, means, scales = [], [], [], []
            for row in prep["episodes"]:
                path = prepared_root / row["path"]
                if file_sha256(path) != row["sha256"]:
                    raise ValueError("an original scoring window changed")
                with np.load(path, allow_pickle=False) as saved:
                    raw, observed = saved["future"][:96], saved["future_observed"][:96]
                scaler = scalers[(row["dataset_id"], row["item_id"])]
                mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
                if not np.array_equal(np.isfinite(raw), observed) or observed.sum(0).min() < 48:
                    raise ValueError("future observation support changed")
                truth.append((raw - mean) / scale)
                masks.append(observed)
                means.append(mean)
                scales.append(scale)
            truth, masks, means, scales = map(np.stack, (truth, masks, means, scales))
            count = len(truth)
            multiplier = np.sqrt(96 * masks / masks.sum(1)[:, None, :])
            weighted_points = points * multiplier[:, None]
            weighted_truth = np.where(masks, truth, 0.0) * multiplier
            fixed, single = np.empty_like(truth), np.empty_like(truth)
            weight_outputs = {}
            for slot in [-1] if model_id == "chronos2" else [0, 1]:
                vectors = (
                    weighted_points.reshape(count, 7, -1)
                    if slot == -1
                    else weighted_points[:, :, :, slot]
                )
                target = (
                    weighted_truth.reshape(count, -1) if slot == -1 else weighted_truth[:, :, slot]
                )
                _, _, _, gram = forecast_geometry(vectors)
                alignment = projection_targets(vectors, target)["raw_projection"]
                weights, _, _ = simplex_quadratic_weights(gram, alignment)
                prediction = compose_forecasts(vectors, weights)
                gradient = 2 * np.mean(vectors * (prediction - target)[:, None], axis=2)
                gap = ((gradient * weights).sum(1) - gradient.min(1)) / np.maximum(
                    1.0, abs(gradient).max(1)
                )
                if gap.max() > 1e-7:
                    raise ValueError("fixed-weight oracle failed direct optimality")
                max_gap = max(max_gap, float(gap.max()))
                individual_cost = ((vectors - target[:, None]) ** 2).mean(2)
                choices = individual_cost.argmin(1)
                if (((prediction - target) ** 2).mean(1) - individual_cost.min(1)).max() > 1e-8:
                    raise ValueError("a feasible single candidate beats the convex optimum")
                raw_vectors = points.reshape(count, 7, -1) if slot == -1 else points[:, :, :, slot]
                estimated = compose_forecasts(raw_vectors, weights)
                chosen = raw_vectors[np.arange(count), choices]
                if slot == -1:
                    fixed[:] = estimated.reshape(count, 96, 2)
                    single[:] = chosen.reshape(count, 96, 2)
                else:
                    fixed[:, :, slot], single[:, :, slot] = estimated, chosen
                weight_outputs[f"slot_{slot}"] = weights
            coordinate = np.where(masks, np.clip(truth, points.min(1), points.max(1)), median)
            methods = {
                "future_oracle_fixed_convex": fixed,
                "future_oracle_coordinate_convex": coordinate,
                "future_oracle_single": single,
                "forecast_median_guarded": median,
                "forecast_median_with_motm": median8,
            }
            path = output / cohort / model_id / "oracle_predictions.npz"
            _save_npz(
                path,
                point_z=np.stack(list(methods.values()), axis=1),
                methods=np.asarray(list(methods)),
                episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                **weight_outputs,
            )
            bank_records.append(
                {
                    "cohort": cohort,
                    "model_id": model_id,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
            scores_by_method = {}
            for method, predictions in methods.items():
                local_scores = []
                for index, row in enumerate(prep["episodes"]):
                    with np.load(prepared_root / row["path"], allow_pickle=False) as saved:
                        raw_truth = saved["future"][:96]
                    errors, _ = observed_future_errors(
                        (predictions[index] * scales[index] + means[index])[None],
                        raw_truth,
                        masks[index],
                        scales[index],
                        minimum_observed=48,
                    )
                    scores = {name: float(value.mean()) for name, value in errors.items()}
                    local_scores.append([scores["mae"], scores["mse"]])
                    records.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "episode_id",
                                    "origin_id",
                                    "family_id",
                                    "dataset_id",
                                    "item_id",
                                )
                            },
                            "cohort": cohort,
                            "model_id": model_id,
                            "horizon": 96,
                            "method": method,
                            "panel": row.get("panel", "legacy_native"),
                            "native_missing_context": row["window"]["context_has_missing"],
                            **scores,
                        }
                    )
                scores_by_method[method] = np.asarray(local_scores)
            for comparator in (
                "future_oracle_fixed_convex",
                "future_oracle_single",
                "forecast_median_guarded",
            ):
                if (
                    scores_by_method["future_oracle_coordinate_convex"]
                    - scores_by_method[comparator]
                ).max() > 1e-8:
                    raise ValueError("coordinate-wise feasible-set inclusion failed")
            verified_cases += count
    frame = pd.DataFrame(records)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summaries, families = [], []
    for (cohort, _model), group in frame.groupby(["cohort", "model_id"]):
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
            families.append(family.assign(cohort=cohort, horizon=96, panel=panel_name))
            summaries.append(
                summary.assign(
                    cohort=cohort, horizon=96, panel=panel_name, families=panel.family_id.nunique()
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(
        args.comparison_root / "comparison_summary.csv", float_precision="round_trip"
    )
    keys = ["cohort", "model_id", "horizon", "panel", "method"]
    controls = summary[
        summary.method.isin(["forecast_median_guarded", "forecast_median_with_motm"])
    ].set_index(keys)
    expected = original.set_index(keys).loc[controls.index]
    np.testing.assert_allclose(
        controls[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": file_sha256(args.protocol),
            "verified_model_windows": verified_cases,
            "maximum_fixed_oracle_optimality_gap": max_gap,
            "prediction_banks": bank_records,
            "score_rows": len(frame),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_models_fitted": 0,
            "limits": "all three oracles use evaluation futures and are unavailable; coordinate-wise includes extra target freedom for Chronos; no claim of learnable or deployable gains",
        },
    )


if __name__ == "__main__":
    main()
