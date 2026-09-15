"""Score fixed forecast banks with observed-target metrics and preserved references."""

import numpy as np
import pandas as pd
from apply_followup_policies import result_panels
from run_native_confirmation import hierarchical_metrics

from tsfm_fais.forecasting.observed_accuracy import observed_future_errors
from tsfm_fais.utility_experiment import _write_json, file_sha256


def score_transfer_banks(
    output,
    banks,
    cohorts,
    scalers,
    comparison_root,
    identity,
    checked,
    maximum_delta,
    prefix="scope_",
    aliases=(),
):
    records, metric_delta = [], 0.0
    for bank_entry in banks:
        cohort, horizon = bank_entry["cohort"], bank_entry["horizon"]
        root, prep, _ = cohorts[cohort]
        with np.load(output / bank_entry["path"], allow_pickle=False) as saved:
            bank, names = saved["point_z"], saved["methods"].tolist()
        if len(names) != bank_entry["method_count"]:
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
    original = pd.read_csv(comparison_root / "comparison_summary.csv", float_precision="round_trip")
    keys = ["cohort", "model_id", "horizon", "panel", "method"]
    actual = summary[summary.method == "forecast_median_guarded"].set_index(keys)
    expected = original.set_index(keys).loc[actual.index]
    np.testing.assert_allclose(
        actual[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    frame.to_parquet(output / "episode_results.parquet", index=False)
    shared = summary[~summary.method.str.startswith(prefix)].set_index(keys)
    reference = original.set_index(keys).loc[shared.index]
    np.testing.assert_allclose(
        shared[["mae", "mse"]], reference[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    for model_id, method, reference_method in aliases:
        current = summary[(summary.model_id == model_id) & (summary.method == method)].copy()
        current["method"] = reference_method
        current = current.set_index(keys)
        previous = original.set_index(keys).loc[current.index]
        np.testing.assert_allclose(
            current[["mae", "mse"]], previous[["mae", "mse"]], rtol=1e-12, atol=1e-12
        )
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    pd.concat([original, summary[summary.method.str.startswith(prefix)]], ignore_index=True).to_csv(
        output / "comparison_summary.csv", index=False
    )
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
