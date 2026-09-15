"""Recompute projection diagnostics from saved arrays without a forecasting model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(prediction, target):
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    if not np.isfinite(residual).all():
        raise ValueError("saved diagnostic arrays contain nonfinite values")
    return {"mae": float(np.abs(residual).mean()), "mse": float((residual**2).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / "verification.json"
    if output.exists():
        raise ValueError("preserve completed array verification")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "completed"
        or not manifest["parameter_digest_unchanged"]
        or manifest["identity"]["optimization_target"] != "forecast_median"
    ):
        raise ValueError("a completed teacher projection with frozen parameters is required")
    records = []
    for record in manifest["records"]:
        name = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:20] + ".npz"
        path = root / name
        if digest(path) != record["arrays_sha256"]:
            raise ValueError("saved projection arrays changed")
        with np.load(path, allow_pickle=False) as saved:
            teacher, truth = saved["teacher"], saved["source_future"]
            np.testing.assert_array_equal(
                teacher, np.median(saved["teacher_candidate_forecasts"], axis=0)
            )
            actual_teacher = errors(teacher, truth)
            for metric in ("mae", "mse"):
                np.testing.assert_allclose(
                    actual_teacher[metric],
                    record["teacher_forecast_errors"][metric],
                    rtol=2e-5,
                    atol=1e-7,
                )
            initial = errors(saved["initial_forecast"], teacher)["mse"]
            for variant in record["variants"]:
                prediction = saved[variant["granularity"] + "_best_forecast"]
                distance = errors(prediction, teacher)["mse"]
                actual = errors(prediction, truth)
                np.testing.assert_allclose(
                    initial, variant["initial_objective_mse"], rtol=2e-5, atol=1e-7
                )
                np.testing.assert_allclose(
                    distance, variant["best_objective_mse"], rtol=2e-5, atol=1e-7
                )
                for metric in ("mae", "mse"):
                    np.testing.assert_allclose(
                        actual[metric],
                        variant["source_forecast_errors_at_best_objective"][metric],
                        rtol=2e-5,
                        atol=1e-7,
                    )
                records.append(
                    {
                        "episode_id": record["episode_id"],
                        "granularity": variant["granularity"],
                        "initial_teacher_mse": initial,
                        "best_teacher_mse": distance,
                        "source_forecast_mae": actual["mae"],
                        "source_forecast_mse": actual["mse"],
                    }
                )
    report = {
        "status": "passed",
        "manifest_sha256": digest(root / "manifest.json"),
        "verification_script_sha256": digest(Path(__file__)),
        "source_windows": len(manifest["records"]),
        "variants_checked": len(records),
        "teacher_arrays_exact": True,
        "records": records,
        "interpretation": "independent NumPy recomputation of saved teacher distances and source forecast errors; no model calls, generalization claims or global optimum claims",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "source_windows", "variants_checked", "teacher_arrays_exact")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
