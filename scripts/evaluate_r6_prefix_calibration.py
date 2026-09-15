"""Audit prefix-only weights and evaluate their frozen outputs on the used R6 cohort."""

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

from tsfm_fais.forecasting.accuracy import PrefixStandardizer  # noqa: E402
from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_calibration(args, sources):
    inputs = read_json(args.input_root / "manifest.json")
    if (
        inputs["status"] != "completed"
        or len(inputs["histories"]) != 43
        or len(inputs["episodes"]) != 774
    ):
        raise ValueError("incomplete prefix input coverage")
    history_map = {row["history_id"]: row for row in inputs["histories"]}
    prefixes = {}
    for source in sources.values():
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError("an original source changed")
        prefixes[(source["dataset_id"], source["item_id"])] = np.load(path, mmap_mode="r")[
            : source["prefix_end"]
        ]
    for fit in inputs["fitting"]:
        path = args.input_root / "imputers" / fit["dataset_id"] / "training_batch.npz"
        if file_sha256(path) != fit["training_sha256"]:
            raise ValueError("a neural fitting batch changed")
        with np.load(path, allow_pickle=False) as saved:
            if len(saved["values"]) != 64:
                raise ValueError("neural fitting budget changed")
            for index, identifier in enumerate(saved["window_ids"].tolist()):
                item, offset = identifier.rsplit("@", 1)
                start = int(offset.split("|", 1)[0])
                prefix = prefixes[(fit["dataset_id"], item)]
                if start + 96 > int(0.6 * len(prefix)):
                    raise ValueError("a neural imputer used the calibration interval")
                mask = saved["observed"][index]
                np.testing.assert_array_equal(
                    saved["values"][index][mask], prefix[start : start + 96][mask]
                )
        for entry in fit["imputers"]:
            marker = Path(entry["path"])
            if file_sha256(marker) != entry["sha256"] or entry["status"] != "fitted":
                raise ValueError("a neural imputer fitting record changed")
            for artifact in read_json(marker)["files"]:
                if (
                    file_sha256(marker.parent / entry["candidate_id"] / artifact["path"])
                    != artifact["sha256"]
                ):
                    raise ValueError("a fitted imputer checkpoint changed")
    clean_histories = {}
    for history in inputs["histories"]:
        path = args.input_root / history["path"]
        if file_sha256(path) != history["sha256"]:
            raise ValueError("a calibration history changed")
        prefix = prefixes[(history["dataset_id"], history["item_id"])]
        origin = history["origin"]
        if not int(0.6 * len(prefix)) <= origin - 96 < origin <= len(prefix):
            raise ValueError("a teacher context violates the prefix boundary")
        with np.load(path, allow_pickle=False) as saved:
            clean = saved["clean"]
        np.testing.assert_array_equal(clean, prefix[origin - 96 : origin])
        if not np.isfinite(clean).all():
            raise ValueError("a teacher uses an originally missing observation")
        clean_histories[history["history_id"]] = clean
    for record in inputs["episodes"]:
        path = args.input_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a calibration episode changed")
        with np.load(path, allow_pickle=False) as saved:
            context = saved["context"]
            observed = np.isfinite(context)
            clean = clean_histories[record["history_id"]]
            np.testing.assert_array_equal(context[observed], clean[observed])
            for point in saved["candidate_values"]:
                np.testing.assert_array_equal(point[observed], clean[observed])
    binding = read_json(args.method_freeze)
    if file_sha256(Path(binding["controls_path"])) != binding["controls_sha256"]:
        raise ValueError("the original source control changed")
    controls = read_json(Path(binding["controls_path"]))
    weights, model_records, max_gap, max_metric_delta = {}, [], 0.0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.calibration_root / model_id
        manifest = read_json(root / "manifest.json")
        if manifest["status"] != "completed" or manifest["identity"][
            "input_manifest_sha256"
        ] != file_sha256(args.input_root / "manifest.json"):
            raise ValueError("a fitted calibration has different inputs")
        if file_sha256(root / "weights.json") != manifest["weights_sha256"]:
            raise ValueError("fitted weights changed")
        if len(manifest["teachers"]) != 43 or len(manifest["predictions"]) != 774:
            raise ValueError("calibration forecast coverage changed")
        teachers, predictions = {}, {}
        for entries, destination, id_key in (
            (manifest["teachers"], teachers, "history_id"),
            (manifest["predictions"], predictions, "episode_id"),
        ):
            for entry in entries:
                path = root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a historical forecast changed")
                with np.load(path, allow_pickle=False) as saved:
                    destination[entry[id_key]] = saved["point_z"]
        fitted = read_json(root / "weights.json")
        if len(fitted) != (15 if model_id == "chronos2" else 30):
            raise ValueError("fitted series/target coverage changed")
        for entry in fitted:
            source_weights = np.asarray(entry["source_weights"])
            local = np.asarray(entry["local_weights"])
            primary = np.asarray(entry["primary_weights"])
            np.testing.assert_array_equal(source_weights, controls[model_id]["convex_weights"])
            np.testing.assert_array_equal(primary, 0.5 * (source_weights + local))
            if min(local) < 0 or abs(local.sum() - 1) > 1e-10:
                raise ValueError("local weights are outside the probability simplex")
            episodes = [
                row
                for row in inputs["episodes"]
                if (row["dataset_id"], row["item_id"]) == (entry["dataset_id"], entry["item_id"])
            ]
            if not episodes:
                np.testing.assert_array_equal(local, source_weights)
                if not entry["uses_source_fallback"]:
                    raise ValueError("an unsupported item is missing its source fallback")
            else:
                unique_histories = {row["history_id"] for row in episodes}
                if not 2 <= len(unique_histories) <= 4 or len(episodes) != 18 * len(
                    unique_histories
                ):
                    raise ValueError("unequal historical condition support")
                for identifier in unique_histories:
                    if history_map[identifier]["origin"] not in entry["origins"]:
                        raise ValueError("a calibration origin is unregistered")
                points = np.stack([predictions[row["episode_id"]] for row in episodes])
                teacher = np.stack([teachers[row["history_id"]] for row in episodes])
                slot = entry["target_slot"]
                vectors = (
                    points.reshape(len(points), 7, -1) if slot == -1 else points[:, :, :, slot]
                )
                target = teacher.reshape(len(teacher), -1) if slot == -1 else teacher[:, :, slot]
                # Direct residual derivatives check optimality without the fitted centered Gram construction.
                estimate = np.sum(vectors * local[None, :, None], axis=1)
                gradient = 2 * np.mean(vectors * (estimate - target)[:, None], axis=(0, 2))
                gap = float(
                    (np.dot(gradient, local) - gradient.min()) / max(1.0, np.abs(gradient).max())
                )
                max_gap = max(max_gap, gap)
                if gap > 1e-7:
                    raise ValueError("direct teacher-residual optimality check failed")
                for name, value in (
                    ("local", local),
                    ("source", source_weights),
                    ("half_local", primary),
                ):
                    point = np.sum(vectors * value[None, :, None], axis=1)
                    metrics = {
                        "mae": np.abs(point - target).mean(),
                        "mse": ((point - target) ** 2).mean(),
                    }
                    for metric, score in metrics.items():
                        delta = abs(
                            float(score) - entry["calibration_teacher_metrics"][name][metric]
                        )
                        max_metric_delta = max(max_metric_delta, delta)
                        if delta > 1e-10:
                            raise ValueError("a calibration teacher score did not replay")
            weights[(model_id, entry["dataset_id"], entry["item_id"], entry["target_slot"])] = entry
        model_records.append(
            {"model_id": model_id, "manifest_sha256": file_sha256(root / "manifest.json")}
        )
    return (
        weights,
        controls,
        {
            "models": model_records,
            "checked_histories": 43,
            "checked_inputs": 774,
            "checked_fitted_weights": len(weights),
            "maximum_direct_optimality_gap": max_gap,
            "maximum_teacher_metric_difference": max_metric_delta,
            "prefix_boundaries_verified": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root",
        "calibration-root",
        "cohort-root",
        "prepared-root",
        "forecast-root",
        "method-freeze",
        "readout-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed prefix-calibration results")
    output.mkdir(parents=True, exist_ok=True)
    cohort = read_json(args.cohort_root / "manifest.json")
    sources = {(row["dataset_id"], row["item_id"]): row for row in cohort["sources"]}
    weights, controls, audit = audit_calibration(args, sources)
    _write_json(output / "calibration_audit.json", audit)
    prep = read_json(args.prepared_root / "manifest.json")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.prepared_root / "standardizers.json")
    }
    for key, source in sources.items():
        scaler = PrefixStandardizer.fit(
            np.load(source["path"], mmap_mode="r")[: source["prefix_end"]]
        )
        np.testing.assert_array_equal(scaler.mean, scalers[key]["mean"])
        np.testing.assert_array_equal(scaler.scale, scalers[key]["scale"])
    names = [
        "prefix_teacher_half",
        "prefix_teacher_local",
        "gate_source_fixed_convex",
        "forecast_median_guarded",
        "forecast_median_with_motm",
    ]
    banks, maximum_prediction_difference = [], 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.forecast_root / model_id
        forecast = read_json(root / "manifest.json")
        for horizon_entry in forecast["horizons"]:
            horizon = horizon_entry["horizon"]
            marker = root / horizon_entry["path"]
            if file_sha256(marker) != horizon_entry["sha256"]:
                raise ValueError("frozen evaluation forecast metadata changed")
            predictions = {row["episode_id"]: row for row in read_json(marker)["predictions"]}
            bank = []
            for record in prep["episodes"]:
                forecast_entry = predictions[record["episode_id"]]
                path = marker.parent / forecast_entry["path"]
                if file_sha256(path) != forecast_entry["sha256"]:
                    raise ValueError("an original candidate prediction changed")
                with np.load(path, allow_pickle=False) as saved:
                    all_points = saved["point_z"]
                    order = saved["candidate_ids"].tolist()
                points = all_points[[order.index(name) for name in controls[model_id]["actions"]]]
                mixtures = np.empty((3, horizon, 2))
                for slot in (0, 1):
                    entry = weights[
                        (
                            model_id,
                            record["dataset_id"],
                            record["item_id"],
                            -1 if model_id == "chronos2" else slot,
                        )
                    ]
                    for method_index, field in enumerate(
                        ("primary_weights", "local_weights", "source_weights")
                    ):
                        vector = np.asarray(entry[field])
                        mixed = points[0, :, slot] + np.sum(
                            vector[:, None] * (points[:, :, slot] - points[:1, :, slot]), axis=0
                        )
                        direct = np.sum(vector[:, None] * points[:, :, slot], axis=0)
                        maximum_prediction_difference = max(
                            maximum_prediction_difference, float(np.max(np.abs(mixed - direct)))
                        )
                        np.testing.assert_allclose(mixed, direct, rtol=1e-12, atol=1e-12)
                        mixtures[method_index, :, slot] = mixed
                bank.append(
                    np.concatenate(
                        [
                            mixtures,
                            np.median(points, axis=0)[None],
                            np.median(all_points, axis=0)[None],
                        ],
                        axis=0,
                    )
                )
            bank = np.stack(bank)
            if bank.shape != (1522, 5, horizon, 2) or not np.isfinite(bank).all():
                raise ValueError("new policy output coverage changed")
            path = output / model_id / f"h{horizon}_predictions.npz"
            _save_npz(
                path,
                point_z=bank,
                methods=np.asarray(names),
                episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
            )
            banks.append(
                {
                    "model_id": model_id,
                    "horizon": horizon,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                }
            )
    _write_json(
        output / "prediction_freeze.json",
        {
            "banks": banks,
            "calibration_audit_sha256": file_sha256(output / "calibration_audit.json"),
            "evaluation_future_arrays_read": False,
        },
    )
    rows, max_normalized_delta = [], 0.0
    for entry in banks:
        horizon, model_id = entry["horizon"], entry["model_id"]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            bank = saved["point_z"]
        for index, record in enumerate(prep["episodes"]):
            path = args.prepared_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("an original prepared evaluation window changed")
            with np.load(path, allow_pickle=False) as saved:
                future, observed = saved["future"][:horizon], saved["future_observed"][:horizon]
            source = sources[(record["dataset_id"], record["item_id"])]
            origin = record["window"]["origin"]
            np.testing.assert_array_equal(
                future, np.load(source["path"], mmap_mode="r")[origin : origin + horizon, :2]
            )
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            errors, counts = observed_future_errors(
                bank[index] * scale + mean, future, observed, scale, minimum_observed=horizon // 2
            )
            normalized = (future - mean) / scale
            direct_mae, direct_mse = [], []
            for slot in (0, 1):
                difference = (
                    bank[index, :, :, slot][:, observed[:, slot]]
                    - normalized[observed[:, slot], slot][None]
                )
                direct_mae.append(np.abs(difference).mean(axis=1))
                direct_mse.append((difference**2).mean(axis=1))
            for metric, direct in (
                ("mae", np.mean(direct_mae, axis=0)),
                ("mse", np.mean(direct_mse, axis=0)),
            ):
                max_normalized_delta = max(
                    max_normalized_delta, float(np.max(abs(errors[metric].mean(1) - direct)))
                )
                np.testing.assert_allclose(errors[metric].mean(1), direct, rtol=1e-12, atol=1e-12)
            for position, name in enumerate(names):
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
                                "mechanism",
                                "missing_rate",
                                "mask_seed",
                            )
                        },
                        "model_id": model_id,
                        "horizon": horizon,
                        "method": name,
                        "native_missing_context": record["window"]["context_has_missing"],
                        "future_observed_target0": int(counts[0]),
                        "future_observed_target1": int(counts[1]),
                        **{
                            metric: float(value[position].mean())
                            for metric, value in errors.items()
                        },
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_parquet(output / "episode_results.parquet", index=False)
    summaries, families = [], []
    for (_model_id, horizon), group in scores.groupby(["model_id", "horizon"]):
        for panel_name, panel in result_panels(group):
            _, family, summary = hierarchical_metrics(panel)
            families.append(family.assign(horizon=horizon, panel=panel_name))
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
    pd.concat(families, ignore_index=True).to_csv(output / "family_metrics.csv", index=False)
    original = pd.read_csv(args.readout_root / "summary.csv", float_precision="round_trip")
    keys = ["model_id", "horizon", "panel", "method"]
    overlap = summary[summary.method.isin(names[2:])].set_index(keys).sort_index()
    expected = original.set_index(keys).loc[overlap.index]
    np.testing.assert_allclose(
        overlap[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    combined = pd.concat([original, summary[summary.method.isin(names[:2])]], ignore_index=True)
    combined.to_csv(output / "comparison_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "calibration_audit": audit,
            "prediction_freeze_sha256": file_sha256(output / "prediction_freeze.json"),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "score_rows": len(scores),
            "maximum_prediction_replay_difference": maximum_prediction_difference,
            "maximum_normalized_metric_difference": max_normalized_delta,
            "source_control_replay_max_difference": float(
                np.max(
                    abs(overlap[["mae", "mse"]].to_numpy() - expected[["mae", "mse"]].to_numpy())
                )
            ),
            "limits": "post-confirmation development on used R6 data; two fixed new policies; all original baselines retained; no independent improvement claim",
        },
    )


if __name__ == "__main__":
    main()
