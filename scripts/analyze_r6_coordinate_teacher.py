"""Check coordinate-wise teacher projections without fitting to evaluation futures."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-diagnostic", "prepared-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed coordinate teacher diagnostics")
    source = read_json(args.teacher_diagnostic / "manifest.json")
    prep = read_json(args.prepared_root / "manifest.json")
    if source["status"] != "completed" or source["identity"]["prepared_sha256"] != file_sha256(
        args.prepared_root / "manifest.json"
    ):
        raise ValueError("finish the matching teacher and input audit")
    episodes = [row for row in prep["episodes"] if row["panel"] == "new_synthetic"]
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.prepared_root / "standardizers.json")
    }
    if file_sha256(args.prepared_root / "standardizers.json") != prep["standardizers_sha256"]:
        raise ValueError("prefix statistics changed")
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
    banks = []
    for entry in source["prediction_banks"]:
        path = args.teacher_diagnostic / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("an audited teacher/policy bank changed")
        with np.load(path, allow_pickle=False) as saved:
            if saved["episode_ids"].tolist() != [row["episode_id"] for row in episodes]:
                raise ValueError("teacher and input ordering differ")
            names, bank = saved["methods"].tolist(), saved["point_z"]
        points = bank[:, [names.index(name) for name in actions]]
        teacher = bank[:, names.index("complete_history_teacher")]
        result = np.clip(teacher, points.min(1), points.max(1))
        if not np.isfinite(result).all():
            raise ValueError("coordinate teacher projection is nonfinite")
        for method in ("forecast_median_guarded", "unavailable_teacher_window_convex"):
            reference = bank[:, names.index(method)]
            for exponent in (1, 2):
                actual_cost = (abs(result - teacher) ** exponent).mean((1, 2))
                reference_cost = (abs(reference - teacher) ** exponent).mean((1, 2))
                if (actual_cost - reference_cost).max() > 1e-10:
                    raise ValueError("the coordinate teacher optimum lost a feasible reference")
        path = output / entry["model_id"] / f"h{entry['horizon']}_predictions.npz"
        _save_npz(
            path,
            point_z=result,
            teacher_z=teacher,
            episode_ids=np.asarray([row["episode_id"] for row in episodes]),
        )
        banks.append(
            {
                "model_id": entry["model_id"],
                "horizon": entry["horizon"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    _write_json(
        output / "prediction_freeze.json", {"banks": banks, "evaluation_future_arrays_read": False}
    )
    rows = []
    for entry in banks:
        horizon = entry["horizon"]
        with np.load(output / entry["path"], allow_pickle=False) as saved:
            points, teachers = saved["point_z"], saved["teacher_z"]
        for index, row in enumerate(episodes):
            path = args.prepared_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an original future window changed")
            with np.load(path, allow_pickle=False) as saved:
                raw = saved["future"][:horizon]
            if not np.isfinite(raw).all():
                raise ValueError("this diagnostic is restricted to complete synthetic futures")
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            truth = (raw - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
            rows.append(
                {
                    **{
                        key: row[key]
                        for key in ("episode_id", "origin_id", "dataset_id", "family_id", "item_id")
                    },
                    "model_id": entry["model_id"],
                    "horizon": horizon,
                    "method": "unavailable_teacher_coordinate_convex",
                    "mae": float(abs(points[index] - truth).mean()),
                    "mse": float(((points[index] - truth) ** 2).mean()),
                    "teacher_mae": float(abs(points[index] - teachers[index]).mean()),
                    "teacher_mse": float(((points[index] - teachers[index]) ** 2).mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_metrics.parquet", index=False)
    keys = ["model_id", "horizon", "method", "family_id", "dataset_id", "item_id", "origin_id"]
    metrics = ["mae", "mse", "teacher_mae", "teacher_mse"]
    origin = frame.groupby(keys)[metrics].mean()
    item = origin.groupby(keys[:-1])[metrics].mean()
    dataset = item.groupby(keys[:-2])[metrics].mean()
    family = dataset.groupby(keys[:-3])[metrics].mean()
    summary = family.groupby(keys[:3])[metrics].mean().reset_index()
    family.to_csv(output / "family_metrics.csv")
    summary.to_csv(output / "summary.csv", index=False)
    old = pd.read_csv(args.teacher_diagnostic / "summary.csv", float_precision="round_trip")
    pd.concat([old, summary], ignore_index=True).to_csv(
        output / "comparison_summary.csv", index=False
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "parent_sha256": file_sha256(args.teacher_diagnostic / "manifest.json"),
            "protocol_sha256": file_sha256(args.protocol),
            "score_rows": len(frame),
            "prediction_banks": banks,
            "summary_sha256": file_sha256(output / "summary.csv"),
            "new_forecaster_calls": 0,
            "new_models_fitted": 0,
            "limits": "coordinate teacher projections use hidden complete histories; actual future improvements are descriptive and not guaranteed or deployable",
        },
    )


if __name__ == "__main__":
    main()
