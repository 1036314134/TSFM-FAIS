"""Compare teacher and observed historical-future labels on matched prefix windows."""

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
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def observed_vectors(points, target, observed, slot):
    """Embed equal-target observed-cell MSE into an ordinary vector mean."""
    counts = observed.sum(axis=1)
    if points.shape[2:] != (96, 2) or (counts < 48).any():
        raise ValueError("both historical targets need the declared observation support")
    multiplier = np.sqrt(96 * observed / counts[:, None, :])
    candidates = points * multiplier[:, None]
    targets = np.where(observed, target, 0.0) * multiplier
    if slot == -1:
        return candidates.reshape(len(points), 7, -1), targets.reshape(len(points), -1)
    return candidates[:, :, :, slot], targets[:, :, slot]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort-root",
        "prepared-root",
        "calibration-input-root",
        "calibration-root",
        "forecast-root",
        "method-freeze",
        "original-prefix-result",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed matched-label controls")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        name: file_sha256(path)
        for name, path in {
            "script": Path(__file__),
            "protocol": args.protocol,
            "cohort": args.cohort_root / "manifest.json",
            "prepared": args.prepared_root / "manifest.json",
            "calibration_input": args.calibration_input_root / "manifest.json",
            "previous_audit": args.original_prefix_result / "manifest.json",
            "method": args.method_freeze,
        }.items()
    }
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cohort = read_json(args.cohort_root / "manifest.json")
    prep = read_json(args.prepared_root / "manifest.json")
    inputs = read_json(args.calibration_input_root / "manifest.json")
    prior = read_json(args.original_prefix_result / "manifest.json")
    binding = read_json(args.method_freeze)
    if any(row["status"] != "completed" for row in (cohort, prep, inputs, prior)):
        raise ValueError("finish the preceding source and prediction audits")
    if file_sha256(Path(binding["controls_path"])) != binding["controls_sha256"]:
        raise ValueError("source weights changed")
    controls = read_json(Path(binding["controls_path"]))
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != inputs["identity"]["standardizers_sha256"]:
        raise ValueError("prefix standardization changed")
    scalers = {(row["dataset_id"], row["item_id"]): row for row in read_json(scaler_path)}
    sources = {(row["dataset_id"], row["item_id"]): row for row in cohort["sources"]}
    prefixes = {}
    for key, row in sources.items():
        if file_sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("an original source changed")
        prefixes[key] = np.load(row["path"], mmap_mode="r")[: row["prefix_end"], :2]
    historical_targets, eligibility = {}, []
    for history in inputs["histories"]:
        key = (history["dataset_id"], history["item_id"])
        prefix, origin = prefixes[key], history["origin"]
        future = prefix[origin : origin + 96]
        eligible = len(future) == 96 and np.isfinite(future).sum(0).min() >= 48
        eligibility.append(
            {
                **{
                    name: history[name]
                    for name in ("history_id", "dataset_id", "item_id", "origin")
                },
                "prefix_end": len(prefix),
                "inside_prefix": origin + 96 <= len(prefix),
                "eligible": bool(eligible),
                "future_observed": np.isfinite(future).sum(0).tolist(),
            }
        )
        if eligible:
            mean, scale = (
                np.asarray(scalers[key]["mean"])[:2],
                np.asarray(scalers[key]["scale"])[:2],
            )
            historical_targets[history["history_id"]] = (
                (future - mean) / scale,
                np.isfinite(future),
            )
    _write_json(output / "eligibility.json", eligibility)
    weights, fitting, maximum_gap, checked_labels = {}, [], 0.0, 0
    for model in ("chronos2", "timesfm2p5"):
        root = args.calibration_root / model
        original_model = next(
            row for row in prior["calibration_audit"]["models"] if row["model_id"] == model
        )
        if file_sha256(root / "manifest.json") != original_model["manifest_sha256"]:
            raise ValueError("the audited historical forecast bank changed")
        manifest = read_json(root / "manifest.json")
        if manifest["identity"]["input_manifest_sha256"] != identity["calibration_input"]:
            raise ValueError("calibration inputs differ from the audited bank")
        teachers, predictions = {}, {}
        for entries, destination, name in (
            (manifest["teachers"], teachers, "history_id"),
            (manifest["predictions"], predictions, "episode_id"),
        ):
            for entry in entries:
                path = root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a cached historical prediction changed")
                with np.load(path, allow_pickle=False) as saved:
                    destination[entry[name]] = saved["point_z"]
        source_weights = np.asarray(controls[model]["convex_weights"])
        for key in sources:
            selected = [
                row
                for row in inputs["episodes"]
                if (row["dataset_id"], row["item_id"]) == key
                and row["history_id"] in historical_targets
            ]
            history_ids = sorted({row["history_id"] for row in selected})
            if len(selected) != 18 * len(history_ids):
                raise ValueError("matched history masks lost coverage")
            eligible = len(history_ids) >= 2
            if eligible:
                points = np.stack([predictions[row["episode_id"]] for row in selected])
                teacher = np.stack([teachers[row["history_id"]] for row in selected])
                future = np.stack([historical_targets[row["history_id"]][0] for row in selected])
                observed = np.stack([historical_targets[row["history_id"]][1] for row in selected])
            for slot in [-1] if model == "chronos2" else [0, 1]:
                record = {
                    "model_id": model,
                    "dataset_id": key[0],
                    "item_id": key[1],
                    "target_slot": slot,
                    "eligible_history_ids": history_ids,
                    "uses_source_fallback": not eligible,
                    "source_weights": source_weights.tolist(),
                    "objectives": {},
                }
                for kind in ("teacher", "future"):
                    local, gap, metrics = source_weights.copy(), 0.0, {}
                    if eligible:
                        vectors, target = observed_vectors(
                            points, teacher if kind == "teacher" else future, observed, slot
                        )
                        _, _, _, gram = forecast_geometry(vectors)
                        alignment = projection_targets(vectors, target)["raw_projection"]
                        optimum, _, _ = simplex_quadratic_weights(
                            gram.mean(0)[None], alignment.mean(0)[None]
                        )
                        local = optimum[0]
                        estimate = np.sum(vectors * local[None, :, None], axis=1)
                        gradient = 2 * np.mean(vectors * (estimate - target)[:, None], axis=(0, 2))
                        gap = float(
                            (gradient @ local - gradient.min()) / max(1.0, abs(gradient).max())
                        )
                        if gap > 1e-7:
                            raise ValueError("direct historical-residual optimality check failed")
                        for name, weight in (
                            ("local", local),
                            ("half", (local + source_weights) / 2),
                            ("source", source_weights),
                        ):
                            prediction = np.sum(vectors * weight[None, :, None], axis=1)
                            metrics[name] = float(((prediction - target) ** 2).mean())
                        if metrics["local"] > metrics["source"] + 1e-10:
                            raise ValueError(
                                "the local risk is worse than a feasible source mixture"
                            )
                        # Verify the weighted-vector loss against per-target observed-cell MSE.
                        ordinary = np.sum(points * local[None, :, None, None], axis=1)
                        reference = teacher if kind == "teacher" else future
                        slots = (0, 1) if slot == -1 else (slot,)
                        direct = np.mean(
                            [
                                np.mean(
                                    [
                                        (
                                            (
                                                ordinary[index, observed[index, :, k], k]
                                                - reference[index, observed[index, :, k], k]
                                            )
                                            ** 2
                                        ).mean()
                                        for k in slots
                                    ]
                                )
                                for index in range(len(points))
                            ]
                        )
                        np.testing.assert_allclose(direct, metrics["local"], rtol=1e-12, atol=1e-12)
                        checked_labels += len(selected)
                    maximum_gap = max(maximum_gap, gap)
                    record["objectives"][kind] = {
                        "local_weights": local.tolist(),
                        "weights": ((local + source_weights) / 2).tolist(),
                        "direct_optimality_gap": gap,
                        "matched_calibration_mse": metrics,
                    }
                weights[(model, *key, slot)] = record
                fitting.append(record)
    _write_json(output / "weights.json", fitting)
    names = [
        "prefix_teacher_half_matched",
        "prefix_future_half_matched",
        "gate_source_fixed_convex",
    ]
    banks = []
    for model in ("chronos2", "timesfm2p5"):
        root = args.forecast_root / model
        for entry in read_json(root / "manifest.json")["horizons"]:
            horizon, path = entry["horizon"], root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("the evaluated forecasting bank changed")
            predictions = {row["episode_id"]: row for row in read_json(path)["predictions"]}
            bank = np.empty((len(prep["episodes"]), 3, horizon, 2))
            for index, row in enumerate(prep["episodes"]):
                entry = predictions[row["episode_id"]]
                prediction_path = path.parent / entry["path"]
                if file_sha256(prediction_path) != entry["sha256"]:
                    raise ValueError("an evaluated candidate forecast changed")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    order = saved["candidate_ids"].tolist()
                    points = saved["point_z"][
                        [order.index(name) for name in controls[model]["actions"]]
                    ]
                for slot in (0, 1):
                    fitted = weights[
                        (
                            model,
                            row["dataset_id"],
                            row["item_id"],
                            -1 if model == "chronos2" else slot,
                        )
                    ]
                    for method_index, value in enumerate(
                        (
                            fitted["objectives"]["teacher"]["weights"],
                            fitted["objectives"]["future"]["weights"],
                            fitted["source_weights"],
                        )
                    ):
                        vector = np.asarray(value)
                        vector /= vector.sum()
                        bank[index, method_index, :, slot] = points[0, :, slot] + np.sum(
                            vector[:, None] * (points[:, :, slot] - points[:1, :, slot]), axis=0
                        )
            if bank.shape != (1522, 3, horizon, 2) or not np.isfinite(bank).all():
                raise ValueError("complete policy coverage is required")
            target_path = output / model / f"h{horizon}_predictions.npz"
            _save_npz(
                target_path,
                point_z=bank,
                methods=np.asarray(names),
                episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
            )
            banks.append(
                {
                    "model_id": model,
                    "horizon": horizon,
                    "path": str(target_path.relative_to(output)),
                    "sha256": file_sha256(target_path),
                }
            )
    _write_json(
        output / "prediction_freeze.json",
        {
            "banks": banks,
            "weights_sha256": file_sha256(output / "weights.json"),
            "evaluation_futures_read": False,
        },
    )
    rows, max_metric_delta = [], 0.0
    for entry in banks:
        model, horizon = entry["model_id"], entry["horizon"]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            bank = saved["point_z"]
        for index, record in enumerate(prep["episodes"]):
            key = (record["dataset_id"], record["item_id"])
            path = args.prepared_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("an original evaluation window changed")
            with np.load(path, allow_pickle=False) as saved:
                future, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            origin = record["window"]["origin"]
            np.testing.assert_array_equal(
                future, np.load(sources[key]["path"], mmap_mode="r")[origin : origin + horizon, :2]
            )
            mean, scale = (
                np.asarray(scalers[key]["mean"])[:2],
                np.asarray(scalers[key]["scale"])[:2],
            )
            errors, _ = observed_future_errors(
                bank[index] * scale + mean, future, observed, scale, minimum_observed=horizon // 2
            )
            normalized = (future - mean) / scale
            direct = {"mae": [], "mse": []}
            for slot in (0, 1):
                delta = (
                    bank[index, :, :, slot][:, observed[:, slot]]
                    - normalized[observed[:, slot], slot][None]
                )
                direct["mae"].append(abs(delta).mean(1))
                direct["mse"].append((delta**2).mean(1))
            for metric in direct:
                reference = np.mean(direct[metric], axis=0)
                max_metric_delta = max(
                    max_metric_delta, float(abs(reference - errors[metric].mean(1)).max())
                )
                np.testing.assert_allclose(
                    reference, errors[metric].mean(1), rtol=1e-12, atol=1e-12
                )
            for position, method in enumerate(names):
                rows.append(
                    {
                        **{
                            name: record[name]
                            for name in (
                                "episode_id",
                                "origin_id",
                                "family_id",
                                "dataset_id",
                                "item_id",
                                "panel",
                                "mechanism",
                                "missing_rate",
                                "mask_seed",
                            )
                        },
                        "model_id": model,
                        "horizon": horizon,
                        "method": method,
                        "native_missing_context": record["window"]["context_has_missing"],
                        **{name: float(value[position].mean()) for name, value in errors.items()},
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_parquet(output / "episode_results.parquet", index=False)
    summaries, family_rows = [], []
    for (_model, horizon), group in scores.groupby(["model_id", "horizon"]):
        for panel_name, panel in result_panels(group):
            _, family, summary = hierarchical_metrics(panel)
            family_rows.append(family.assign(horizon=horizon, panel=panel_name))
            summaries.append(
                summary.assign(
                    horizon=horizon,
                    panel=panel_name,
                    families=panel.family_id.nunique(),
                    origins=panel.origin_id.nunique(),
                )
            )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(family_rows, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(
        args.original_prefix_result / "comparison_summary.csv", float_precision="round_trip"
    )
    keys = ["model_id", "horizon", "panel", "method"]
    control = summary[summary.method == names[2]].set_index(keys).sort_index()
    reference = original.set_index(keys).loc[control.index]
    np.testing.assert_allclose(
        control[["mae", "mse"]], reference[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    pd.concat([original, summary[summary.method.isin(names[:2])]], ignore_index=True).to_csv(
        output / "comparison_summary.csv", index=False
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "matched_history_count": len(
                {
                    identifier
                    for row in fitting
                    if not row["uses_source_fallback"]
                    for identifier in row["eligible_history_ids"]
                }
            ),
            "matched_item_count": len(
                {
                    (row["dataset_id"], row["item_id"])
                    for row in fitting
                    if not row["uses_source_fallback"]
                }
            ),
            "checked_weighted_label_rows": checked_labels,
            "score_rows": len(scores),
            "maximum_optimality_gap": maximum_gap,
            "maximum_normalized_metric_difference": max_metric_delta,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_imputer_fits": 0,
            "limits": "matched past-label diagnostic on used R6 data; fixed half shrinkage; masks and horizons do not create independent samples",
        },
    )


if __name__ == "__main__":
    main()
