"""Re-score cached forecasts and export objective-aligned deployment features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.data import load_dataset, load_manifest  # noqa: E402
from tsfm_fais.forecasting.accuracy import (  # noqa: E402
    PrefixStandardizer,
    forecast_errors,
    guarded_direct_forecast,
    recover_legacy_chronos_median,
)
from tsfm_fais.routing.utility import response_features, sequence_features  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


class RowWriter:
    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_suffix(".partial.parquet")
        self.rows = []
        self.writer = None
        self.count = 0

    def add(self, row: dict):
        self.rows.append(row)
        if len(self.rows) >= 3000:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temporary, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.temporary.replace(self.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pause-ms", type=float, default=5.0)
    parser.add_argument("--chronos-repair-root", type=Path)
    args = parser.parse_args()
    import psutil

    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[-1]])
    root, output = args.source_root.resolve(), args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("accuracy export already completed; preserve its evidence")
    manifest_path = root / "episodes_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["identity"]["config"]
    data_manifest = load_manifest(config["data_manifest"])
    targets = config["target_indices"]
    models = ("chronos2", "timesfm2p5")
    chronos_levels = json.loads(
        (root / "chronos2/forecast_identity.json").read_text(encoding="utf-8")
    )["forecast_spec"]["quantile_levels"]
    recovered_count = 0
    repair_records = {}
    if args.chronos_repair_root is not None:
        repair_manifest = json.loads(
            (args.chronos_repair_root / "manifest.json").read_text(encoding="utf-8")
        )
        if repair_manifest.get("status") != "completed" or repair_manifest["identity"][
            "source_episode_manifest_sha256"
        ] != file_sha256(manifest_path):
            raise ValueError(
                "corrected Chronos forecasts are incomplete or bound to another source"
            )
        repair_records = {row["episode_id"]: row for row in repair_manifest["episodes"]}
    prediction_hashes = {
        model: {
            row["episode_id"]: row["sha256"]
            for row in json.loads(
                (root / model / "forecast_manifest.json").read_text(encoding="utf-8")
            )["episodes"]
        }
        for model in models
    }
    vendor_root = root / "timesfm-vendor-missing-v001"
    vendor_manifest = json.loads((vendor_root / "manifest.json").read_text(encoding="utf-8"))
    vendor_hashes = {row["episode_id"]: row["sha256"] for row in vendor_manifest["episodes"]}
    standards, standard_records = {}, []
    for dataset in manifest["datasets"]:
        valid_items = [item for item in dataset["items"] if "prefix_end" in item]
        if not valid_items:
            continue
        for path, digest in dataset["sources"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError("raw data changed since episode preparation")
        items = {
            item.item_id: item for item in load_dataset(data_manifest.get(dataset["dataset_id"]))
        }
        for item in valid_items:
            scaler = PrefixStandardizer.fit(items[item["item_id"]].values[: item["prefix_end"]])
            standards[(dataset["dataset_id"], item["item_id"])] = scaler
            standard_records.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "item_id": item["item_id"],
                    "prefix_end": item["prefix_end"],
                    "mean": scaler.mean.tolist(),
                    "scale": scaler.scale.tolist(),
                    "constant_channels": np.flatnonzero(scaler.constant).tolist(),
                    "observed_count": scaler.observed_count.tolist(),
                }
            )
        del items
    _write_json(output / "standardizers.json", standard_records)
    candidates = RowWriter(output / "candidate_accuracy.parquet")
    controls = RowWriter(output / "control_accuracy.parquet")
    n, h, k = len(manifest["episodes"]), config["horizon"], len(targets)
    point_arrays = {
        model: np.lib.format.open_memmap(
            output / f"{model}_point_z.partial.npy",
            mode="w+",
            dtype="float64",
            shape=(n, len(config["candidate_ids"]) + 2, h, k),
        )
        for model in models
    }
    truth_array = np.lib.format.open_memmap(
        output / "truth_z.partial.npy", mode="w+", dtype="float64", shape=(n, h, k)
    )
    action_orders = {}
    started = time.monotonic()
    for index, record in enumerate(manifest["episodes"]):
        if psutil.virtual_memory().available < 8 * 1024**3:
            while psutil.virtual_memory().available < 8 * 1024**3:
                time.sleep(10)
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("candidate cache changed")
        with np.load(path, allow_pickle=False) as episode:
            scaler = standards[(record["dataset_id"], record["item_id"])]
            ids = episode["candidate_ids"].tolist()
            completions = episode["candidate_values"]
            context = episode["context"]
            truth = episode["future"][:, targets]
            locf = completions[ids.index("locf")]
            truth_array[index] = (truth - scaler.mean[targets]) / scaler.scale[targets]
            slot_groups = [(-1, targets), *[(j, [target]) for j, target in enumerate(targets)]]
            static_cache = {}
            for slot, selected in slot_groups:
                for action_index in range(len(ids) + 1):
                    direct = action_index == len(ids)
                    static_cache[(slot, action_index)] = sequence_features(
                        context - scaler.mean,
                        (locf if direct else completions[action_index]) - scaler.mean,
                        locf - scaler.mean,
                        scaler.scale,
                        selected,
                        period=record["period"],
                        native_coverage=1.0
                        if direct
                        else float(episode["native_coverage"][action_index]),
                    )
            base = {
                key: record[key]
                for key in (
                    "episode_id",
                    "origin_id",
                    "dataset_id",
                    "family_id",
                    "item_id",
                    "origin",
                    "split",
                    "mechanism",
                    "missing_rate",
                    "mask_seed",
                )
            }
            base["episode_index"] = index
            for model in models:
                forecast_path = root / model / "predictions" / path.name
                corrected = model == "chronos2" and record["episode_id"] in repair_records
                if (
                    model == "chronos2"
                    and args.chronos_repair_root
                    and context.shape[1] in (len(chronos_levels), config["horizon"])
                    and not corrected
                ):
                    raise ValueError("corrected forecasts omit an affected episode")
                expected_hash = prediction_hashes[model][record["episode_id"]]
                if corrected:
                    forecast_path = (
                        args.chronos_repair_root / repair_records[record["episode_id"]]["path"]
                    )
                    expected_hash = repair_records[record["episode_id"]]["sha256"]
                if file_sha256(forecast_path) != expected_hash:
                    raise ValueError("forecast cache changed")
                with np.load(forecast_path, allow_pickle=False) as prediction:
                    point, quantiles, clean = (
                        prediction["point"],
                        prediction["quantiles"],
                        prediction["clean_point"],
                    )
                    if (
                        model == "chronos2"
                        and not corrected
                        and context.shape[1] == config["horizon"]
                    ):
                        raise ValueError("legacy D==H collision needs a fresh forecast")
                    if (
                        model == "chronos2"
                        and not corrected
                        and context.shape[1] == len(chronos_levels)
                    ):
                        point = recover_legacy_chronos_median(
                            quantiles, targets, chronos_levels, context.shape[1]
                        )
                        recovered_count += 1
                actions = ids + (["native_missing"] if len(point) > len(ids) else [])
                legacy_point = point
                if model == "timesfm2p5":
                    vendor_path = vendor_root / "predictions" / path.name
                    if file_sha256(vendor_path) != vendor_hashes[record["episode_id"]]:
                        raise ValueError("vendor forecast changed")
                    with np.load(vendor_path, allow_pickle=False) as vendor:
                        point = np.concatenate([point, vendor["point"][None]])
                        quantiles = np.concatenate([quantiles, vendor["quantiles"][None]])
                    actions = actions + ["vendor_missing"]
                guarded, fallback_targets = guarded_direct_forecast(
                    context, targets, point[-1], point[ids.index("locf")], joint=model == "chronos2"
                )
                point = np.concatenate([point, guarded[None]])
                actions.append("guarded_direct")
                action_orders[model] = actions
                point_arrays[model][index] = (point - scaler.mean[targets]) / scaler.scale[targets]
                error = forecast_errors(point, truth, scaler.scale[targets])
                pool = np.median(point[: len(ids)], axis=0)
                reference = point[ids.index("locf")]
                supported_point = point[[*range(len(ids)), len(point) - 1]]
                control_errors = {
                    method: forecast_errors(values[None], truth, scaler.scale[targets])
                    for method, values in [
                        *(
                            [("clean", clean)]
                            if model != "chronos2" or args.chronos_repair_root
                            else []
                        ),
                        ("forecast_mean_finite", np.mean(point[: len(ids)], axis=0)),
                        ("forecast_median_finite", pool),
                        ("forecast_mean_all", np.mean(point[:-1], axis=0)),
                        ("forecast_median_all", np.median(point[:-1], axis=0)),
                        ("forecast_mean_guarded", np.mean(supported_point, axis=0)),
                        ("forecast_median_guarded", np.median(supported_point, axis=0)),
                        ("forecast_mean_legacy", np.mean(legacy_point, axis=0)),
                        ("forecast_median_legacy", np.median(legacy_point, axis=0)),
                    ]
                }
                for slot, selected in slot_groups:
                    positions = list(range(len(targets))) if slot == -1 else [slot]
                    for action_index, action in enumerate(actions):
                        direct = action_index >= len(ids)
                        features = dict(static_cache[(slot, min(action_index, len(ids)))])
                        features.update(
                            {
                                "static.direct_missing": float(direct),
                                "static.empty_channel_fraction": float(
                                    np.isnan(context).all(0).mean()
                                ),
                                "static.empty_target_fraction": float(
                                    np.isnan(context[:, selected]).all(0).mean()
                                ),
                                "static.fallback_target_fraction": float(
                                    fallback_targets[positions].mean()
                                )
                                if action == "guarded_direct"
                                else 0.0,
                            }
                        )
                        features.update(
                            response_features(
                                point[action_index][:, positions],
                                reference[:, positions],
                                pool[:, positions],
                                locf[-1, selected],
                                scaler.scale[selected],
                                None,
                            )
                        )
                        candidates.add(
                            base
                            | {
                                "model_id": model,
                                "target_slot": slot,
                                "candidate_id": action,
                                **{
                                    metric: float(values[action_index, positions].mean())
                                    for metric, values in error.items()
                                },
                                **features,
                            }
                        )
                    for method, control_error in control_errors.items():
                        controls.add(
                            base
                            | {
                                "model_id": model,
                                "target_slot": slot,
                                "method": method,
                                **{
                                    metric: float(values[0, positions].mean())
                                    for metric, values in control_error.items()
                                },
                            }
                        )
        if (index + 1) % 100 == 0 or index + 1 == len(manifest["episodes"]):
            state = {
                "completed": index + 1,
                "total": len(manifest["episodes"]),
                "elapsed_seconds": time.monotonic() - started,
            }
            _write_json(output / "progress.json", state)
            print(json.dumps(state), flush=True)
        time.sleep(max(0, args.pause_ms / 1000))
    candidates.close()
    controls.close()
    for model in models:
        point_arrays[model].flush()
        point_arrays[model]._mmap.close()
        (output / f"{model}_point_z.partial.npy").replace(output / f"{model}_point_z.npy")
    truth_array.flush()
    truth_array._mmap.close()
    (output / "truth_z.partial.npy").replace(output / "truth_z.npy")
    _write_json(
        output / "manifest.json",
        {
            "evidence_role": "development",
            "source_root": str(root),
            "source_episode_manifest_sha256": file_sha256(manifest_path),
            "script_sha256": file_sha256(Path(__file__)),
            "metric_module_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/accuracy.py"),
            "candidate_rows": candidates.count,
            "control_rows": controls.count,
            "action_orders": action_orders,
            "prediction_arrays": {path.name: file_sha256(path) for path in output.glob("*.npy")},
            "guarded_direct": "joint model falls back to LOCF for all targets when any input channel is wholly unobserved; independent model falls back only for a wholly unobserved target; raw direct baseline is retained separately",
            "standardization": "population mean/std from the original fit prefix, shared by all candidates; constant channels use scale 1; cached predictions and truth are transformed for scoring",
            "primary_metrics": ["mae", "mse"],
            "reconstruction_error_role": "auxiliary only",
            "recovered_chronos_episodes": recovered_count,
            "chronos_repair_manifest_sha256": file_sha256(
                args.chronos_repair_root / "manifest.json"
            )
            if args.chronos_repair_root
            else None,
            "legacy_axis_correction": "fresh corrected forecasts replace affected episodes when a repair root is supplied; otherwise retained Chronos-2 medians are recovered algebraically and Chronos clean output is excluded; response features use point forecasts only for both models",
            "forecaster_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(json.dumps({"status": "completed", "candidate_rows": candidates.count}), flush=True)


if __name__ == "__main__":
    main()
