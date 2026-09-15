"""Validate source quantiles against frozen point forecasts without reading outcomes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guarded_outputs(context, native_point, native_quantiles, locf_point, locf_quantiles, *, joint):
    context = np.asarray(context, float)
    empty = ~np.isfinite(context).any(axis=0)
    fallback = np.repeat(empty.any(), 2) if joint else empty[:2]
    point = np.where(fallback[None], locf_point, native_point)
    quantiles = np.where(fallback[None, :, None], locf_quantiles, native_quantiles)
    return point, quantiles, fallback


def interval_quality(quantiles, scales):
    quantiles, scales = np.asarray(quantiles, float), np.asarray(scales, float)
    if quantiles.ndim != 4 or quantiles.shape[-2:] != (2, 3) or scales.shape != (2,):
        raise ValueError("expected [candidate,horizon,target,{0.1,0.5,0.9}] quantiles")
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("original target scales must be finite and positive")
    finite = np.isfinite(quantiles).all(axis=-1)
    crossed = finite & (
        (quantiles[..., 0] > quantiles[..., 1]) | (quantiles[..., 1] > quantiles[..., 2])
    )
    reversed_endpoints = finite & (quantiles[..., 0] > quantiles[..., 2])
    width = (quantiles[..., 2] - quantiles[..., 0]) / scales
    # Preserve signed widths and quality flags; do not reorder predicted quantiles.
    return {
        "finite": finite,
        "crossed": crossed,
        "reversed_endpoints": reversed_endpoints,
        "width_by_target": np.clip(width.mean(axis=1), -1e8, 1e8),
        "width_joint": np.clip(width.mean(axis=(1, 2)), -1e8, 1e8),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "artifacts/iclr27-r3/development-expanded-v001"
    )
    parser.add_argument(
        "--accuracy-root", type=Path, default=ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    )
    parser.add_argument(
        "--chronos-repair-root",
        type=Path,
        default=ROOT / "artifacts/iclr27-r4/chronos-layout-fixed-v001",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R6_SOURCE_QUANTILE_AUDIT_PLAN.md"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed quantile audits")
    source_path, accuracy_path = (
        root / "episodes_manifest.json",
        args.accuracy_root / "manifest.json",
    )
    source, accuracy = read_json(source_path), read_json(accuracy_path)
    repair_path = args.chronos_repair_root / "manifest.json"
    repair = read_json(repair_path)
    if (
        accuracy["source_episode_manifest_sha256"] != file_sha256(source_path)
        or repair["status"] != "completed"
        or repair["identity"]["source_episode_manifest_sha256"] != file_sha256(source_path)
        or accuracy["chronos_repair_manifest_sha256"] != file_sha256(repair_path)
    ):
        raise ValueError("the corrected source prediction provenance changed")
    corrected = {row["episode_id"]: row for row in repair["episodes"]}
    if len(corrected) != 1512 or len(source["episodes"]) != 7812:
        raise ValueError("source or corrected-case coverage changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    vendor_root = root / "timesfm-vendor-missing-v001"
    vendor = read_json(vendor_root / "manifest.json")
    if vendor["identity"]["episode_manifest_sha256"] != file_sha256(source_path):
        raise ValueError("vendor missing-input predictions belong to another source")
    vendor_hashes = {row["episode_id"]: row["sha256"] for row in vendor["episodes"]}
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "source_manifest_sha256": file_sha256(source_path),
        "accuracy_manifest_sha256": file_sha256(accuracy_path),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "repair_manifest_sha256": file_sha256(repair_path),
        "vendor_manifest_sha256": file_sha256(vendor_root / "manifest.json"),
        "quantile_levels": [0.1, 0.5, 0.9],
        "horizon": 96,
        "targets": [0, 1],
        "future_arrays_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial source quantile audit identity changed")
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    models, quality_rows, input_rows = [], [], []
    for model in ("chronos2", "timesfm2p5"):
        forecast_manifest = read_json(root / model / "forecast_manifest.json")
        forecast_hashes = {
            row["episode_id"]: row["sha256"] for row in forecast_manifest["episodes"]
        }
        spec = read_json(root / model / "forecast_identity.json")["forecast_spec"]
        if (
            spec["quantile_levels"] != [0.1, 0.5, 0.9]
            or spec["target_indices"] != [0, 1]
            or spec["horizon"] != 96
        ):
            raise ValueError("source quantile semantics differ from the registered levels")
        bank_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(bank_path) != accuracy["prediction_arrays"][bank_path.name]:
            raise ValueError("the corrected point prediction bank changed")
        bank = np.load(bank_path, mmap_mode="r")
        actions = sorted([*source["identity"]["config"]["candidate_ids"], "guarded_direct"])
        order = [accuracy["action_orders"][model].index(name) for name in actions]
        qpath = output / f"{model}_quantiles_z.npy"
        qbank = np.lib.format.open_memmap(
            qpath, mode="w+", dtype=np.float64, shape=(7812, 7, 96, 2, 3)
        )
        widths, joint_widths = np.empty((7812, 7, 2)), np.empty((7812, 7))
        corrected_count, fallback_count, point_delta, median_delta = 0, 0, 0.0, 0.0
        local_quality = []
        for index, row in enumerate(source["episodes"]):
            episode_path = root / row["path"]
            if file_sha256(episode_path) != row["sha256"]:
                raise ValueError("an original source input changed")
            with np.load(episode_path, allow_pickle=False) as saved:
                context = saved["context"]
                ids = saved["candidate_ids"].tolist()
            path = root / model / "predictions" / episode_path.name
            expected_hash = forecast_hashes[row["episode_id"]]
            repaired = model == "chronos2" and row["episode_id"] in corrected
            if model == "chronos2" and context.shape[1] in (3, 96) and not repaired:
                raise ValueError("an ambiguous Chronos case lacks a corrected forecast")
            if repaired:
                item = corrected[row["episode_id"]]
                path, expected_hash = args.chronos_repair_root / item["path"], item["sha256"]
                corrected_count += 1
            if file_sha256(path) != expected_hash:
                raise ValueError("a source quantile file changed")
            input_rows.append(
                {
                    "model_id": model,
                    "episode_id": row["episode_id"],
                    "path": str(path),
                    "sha256": expected_hash,
                    "corrected": repaired,
                }
            )
            with np.load(path, allow_pickle=False) as saved:
                point, quantiles = saved["point"], saved["quantiles"]
            expected_count = 7 if model == "chronos2" else 6
            if point.shape != (expected_count, 96, 2) or quantiles.shape != (
                expected_count,
                96,
                2,
                3,
            ):
                raise ValueError("source point and quantile coordinates do not align")
            if model == "timesfm2p5":
                path = vendor_root / "predictions" / episode_path.name
                expected_hash = vendor_hashes[row["episode_id"]]
                if file_sha256(path) != expected_hash:
                    raise ValueError("a vendor quantile file changed")
                with np.load(path, allow_pickle=False) as saved:
                    native_point, native_quantiles = saved["point"], saved["quantiles"]
                if native_point.shape != (96, 2) or native_quantiles.shape != (96, 2, 3):
                    raise ValueError("vendor point and quantile coordinates differ")
                point = np.concatenate([point, native_point[None]])
                quantiles = np.concatenate([quantiles, native_quantiles[None]])
                input_rows.append(
                    {
                        "model_id": model,
                        "episode_id": row["episode_id"],
                        "path": str(path),
                        "sha256": expected_hash,
                        "corrected": False,
                    }
                )
            locf = ids.index("locf")
            guarded_point, guarded_quantiles, fallback = guarded_outputs(
                context,
                point[-1],
                quantiles[-1],
                point[locf],
                quantiles[locf],
                joint=model == "chronos2",
            )
            fallback_count += int(fallback.sum())
            selected = [ids.index(name) if name != "guarded_direct" else 6 for name in actions]
            point = np.concatenate([point[:6], guarded_point[None]])[selected]
            quantiles = np.concatenate([quantiles[:6], guarded_quantiles[None]])[selected]
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
            point_z = (point - mean) / scale
            point_delta = max(point_delta, float(abs(point_z - bank[index, order]).max()))
            np.testing.assert_array_equal(point_z, bank[index, order])
            if model == "chronos2":
                np.testing.assert_array_equal(quantiles[..., 1], point)
            median_delta = max(median_delta, float(abs((quantiles[..., 1] - point) / scale).max()))
            qbank[index] = (quantiles - mean[None, None, :, None]) / scale[None, None, :, None]
            quality = interval_quality(quantiles, scale)
            widths[index], joint_widths[index] = quality["width_by_target"], quality["width_joint"]
            for candidate, action in enumerate(actions):
                current = {
                    "model_id": model,
                    "episode_id": row["episode_id"],
                    "origin_id": row["origin_id"],
                    "family_id": row["family_id"],
                    "split": row["split"],
                    "action": action,
                    "nonfinite_coordinates": int((~quality["finite"][candidate]).sum()),
                    "crossed_coordinates": int(quality["crossed"][candidate].sum()),
                    "reversed_endpoints": int(quality["reversed_endpoints"][candidate].sum()),
                    "mean_interval_width": float(joint_widths[index, candidate]),
                }
                local_quality.append(current)
            if (index + 1) % 1000 == 0:
                print(f"{model}: {index + 1}/7812 source inputs checked", flush=True)
        qbank.flush()
        del qbank
        width_path, joint_path = (
            output / f"{model}_width_by_target.npy",
            output / f"{model}_width_joint.npy",
        )
        np.save(width_path, widths, allow_pickle=False)
        np.save(joint_path, joint_widths, allow_pickle=False)
        quality = pd.DataFrame(local_quality)
        quality_rows.append(quality)
        models.append(
            {
                "model_id": model,
                "actions": actions,
                "episodes": 7812,
                "corrected_episodes": corrected_count,
                "fallback_targets": fallback_count,
                "maximum_point_difference": point_delta,
                "maximum_point_median_difference_z": median_delta,
                "nonfinite_coordinates": int(quality.nonfinite_coordinates.sum()),
                "crossed_coordinates": int(quality.crossed_coordinates.sum()),
                "reversed_endpoints": int(quality.reversed_endpoints.sum()),
                "files": {path.name: file_sha256(path) for path in (qpath, width_path, joint_path)},
            }
        )
    pd.concat(quality_rows, ignore_index=True).to_parquet(
        output / "interval_quality.parquet", index=False
    )
    pd.DataFrame(input_rows).to_parquet(output / "source_files.parquet", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": models,
            "quality_sha256": file_sha256(output / "interval_quality.parquet"),
            "source_files_sha256": file_sha256(output / "source_files.parquet"),
            "new_forecaster_calls": 0,
            "new_fits": 0,
            "limits": "source availability and numeric integrity only; widths are not accuracy or calibration guarantees; native evaluation quantiles are not collected",
        },
    )


if __name__ == "__main__":
    main()
