"""Inspect prefix calibration support and an explicitly unavailable fixed-mixture oracle."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeze_followup_cohort import evenly_spaced  # noqa: E402

from tsfm_fais.routing.forecast_gate import compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort-root",
        "prepared-root",
        "forecast-root",
        "method-freeze",
        "audit-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (args.output_root / "manifest.json").exists():
        raise ValueError("preserve completed post-confirmation diagnostics")
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    method = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed":
        raise ValueError("complete independent confirmation checks before explanatory analysis")
    controls = json.loads(Path(method["controls_path"]).read_text(encoding="utf-8"))
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.prepared_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    availability = []
    for source in cohort["sources"]:
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError("an original trajectory changed")
        values = np.load(path, mmap_mode="r")
        prefix_end = source["prefix_end"]
        inner_fit_end = max(96, int(0.6 * prefix_end))
        candidates = [
            origin
            for origin in range(inner_fit_end + 96, prefix_end + 1, 96)
            if np.isfinite(values[origin - 96 : origin]).all()
        ]
        selected = evenly_spaced(candidates, 4)
        availability.append(
            {
                "dataset_id": source["dataset_id"],
                "item_id": source["item_id"],
                "prefix_end": prefix_end,
                "candidate_inner_fit_end": inner_fit_end,
                "eligible_complete_prefix_histories": len(candidates),
                "candidate_calibration_origins": selected,
                "at_least_two_histories": len(selected) >= 2,
            }
        )
    records = [row for row in prep["episodes"] if row["panel"] == "new_synthetic"]
    rows, oracle_records = [], []
    for model_id in ("chronos2", "timesfm2p5"):
        root = args.forecast_root / model_id
        forecast = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        actions = controls[model_id]["actions"]
        source_weights = np.asarray(controls[model_id]["convex_weights"])
        for horizon_entry in forecast["horizons"]:
            horizon = horizon_entry["horizon"]
            path = root / horizon_entry["path"]
            if file_sha256(path) != horizon_entry["sha256"]:
                raise ValueError("candidate horizon metadata changed")
            predictions = {
                row["episode_id"]: row
                for row in json.loads(path.read_text(encoding="utf-8"))["predictions"]
            }
            for dataset, item in sorted({(row["dataset_id"], row["item_id"]) for row in records}):
                selected = [
                    row
                    for row in records
                    if row["dataset_id"] == dataset and row["item_id"] == item
                ]
                points, truth = [], []
                for record in selected:
                    entry = predictions[record["episode_id"]]
                    point_path = path.parent / entry["path"]
                    source_path = args.prepared_root / record["path"]
                    if (
                        file_sha256(point_path) != entry["sha256"]
                        or file_sha256(source_path) != record["sha256"]
                    ):
                        raise ValueError("a checked forecast or future changed")
                    with np.load(point_path, allow_pickle=False) as saved:
                        order = saved["candidate_ids"].tolist()
                        points.append(saved["point_z"][[order.index(name) for name in actions]])
                    with np.load(source_path, allow_pickle=False) as saved:
                        future = saved["future"][:horizon]
                    scaler = scalers[(dataset, item)]
                    truth.append(
                        (future - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
                    )
                points, truth = np.stack(points), np.stack(truth)
                if not np.isfinite(truth).all():
                    raise ValueError("this oracle diagnostic requires complete synthetic futures")
                for slot in [-1] if model_id == "chronos2" else [0, 1]:
                    vectors = (
                        points.reshape(len(points), 7, -1) if slot == -1 else points[:, :, :, slot]
                    )
                    target = truth.reshape(len(truth), -1) if slot == -1 else truth[:, :, slot]
                    _, _, _, gram = forecast_geometry(vectors)
                    alignment = projection_targets(vectors, target)["raw_projection"]
                    optimum, certificate, _ = simplex_quadratic_weights(
                        gram.mean(0)[None], alignment.mean(0)[None]
                    )
                    local = compose_forecasts(vectors, np.repeat(optimum, len(points), axis=0))
                    global_fixed = compose_forecasts(
                        vectors, np.repeat(source_weights[None], len(points), axis=0)
                    )
                    if ((local - target) ** 2).mean() > (
                        (global_fixed - target) ** 2
                    ).mean() + 1e-9:
                        raise ValueError(
                            "the fixed-mixture oracle is worse than a feasible source mixture"
                        )
                    for label, prediction in (
                        ("unavailable_item_fixed_oracle", local),
                        ("source_fixed_convex", global_fixed),
                    ):
                        rows.extend(
                            {
                                "model_id": model_id,
                                "horizon": horizon,
                                "dataset_id": dataset,
                                "family_id": record["family_id"],
                                "item_id": item,
                                "episode_id": record["episode_id"],
                                "origin_id": record["origin_id"],
                                "target_slot": slot,
                                "method": label,
                                "mae": float(abs(prediction[index] - target[index]).mean()),
                                "mse": float(((prediction[index] - target[index]) ** 2).mean()),
                            }
                            for index, record in enumerate(selected)
                        )
                    oracle_records.append(
                        {
                            "model_id": model_id,
                            "horizon": horizon,
                            "dataset_id": dataset,
                            "item_id": item,
                            "target_slot": slot,
                            "weights": optimum[0].tolist(),
                            "optimality_gap": float(certificate[0]),
                            "uses_evaluation_future": True,
                        }
                    )
    frame = pd.DataFrame(rows)
    keys = [
        "model_id",
        "horizon",
        "method",
        "family_id",
        "dataset_id",
        "item_id",
        "origin_id",
        "episode_id",
    ]
    episodes = frame.groupby(keys)[["mae", "mse"]].mean().reset_index()
    items = episodes.groupby(keys[:-2])[["mae", "mse"]].mean()
    datasets = items.groupby(keys[:-3])[["mae", "mse"]].mean()
    families = datasets.groupby(keys[:-4])[["mae", "mse"]].mean()
    summary = families.groupby(["model_id", "horizon", "method"])[["mae", "mse"]].mean()
    args.output_root.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(args.output_root / "episode_metrics.parquet", index=False)
    families.to_csv(args.output_root / "family_metrics.csv")
    summary.to_csv(args.output_root / "summary.csv")
    _write_json(args.output_root / "prefix_availability.json", availability)
    _write_json(
        args.output_root / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
            "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
            "method_freeze_sha256": file_sha256(args.method_freeze),
            "confirmation_audit_sha256": file_sha256(args.audit_root / "manifest.json"),
            "oracles": oracle_records,
            "new_forecaster_calls": 0,
            "new_deployable_models": 0,
            "limits": "post-confirmation diagnostic; oracle weights use evaluation futures and are not deployable; prefix counts alone do not establish calibration effectiveness; all original R6 results remain unchanged",
        },
    )


if __name__ == "__main__":
    main()
