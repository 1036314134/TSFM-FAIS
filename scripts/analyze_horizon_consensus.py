"""Check the decision-granularity gap using fixed cached forecast-consensus rules."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.forecasting.horizon_consensus import consensus_medoid_segments  # noqa: E402
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed horizon-consensus diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source_path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("forecast source episodes changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    selected = [
        (index, record)
        for index, record in enumerate(source["episodes"])
        if record["split"] == "validation"
    ]
    if len(selected) != 1872:
        raise ValueError("this diagnostic requires the full original development panel")
    indices = [index for index, _ in selected]
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/horizon_consensus.py"),
        "blocks": [1, 8, 96],
        "target_modes": ["joint", "separate"],
        "choice_objective": "distance to current seven-candidate coordinate median; no outcomes",
        "role": "forecast-output comparison; not one imputed context or a new-method claim",
    }
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    results, files, parity = [], [], []
    reference = pd.read_csv(root / "analysis-joint-mae-mse-v001/summary.csv")
    for model in ("chronos2", "timesfm2p5"):
        array_path = root / f"{model}_point_z.npy"
        if file_sha256(array_path) != accuracy["prediction_arrays"][array_path.name]:
            raise ValueError("a cached forecast array changed")
        positions = [
            index
            for index, name in enumerate(accuracy["action_orders"][model])
            if name not in {"native_missing", "vendor_missing"}
        ]
        values = np.load(array_path, mmap_mode="r")[indices][:, positions]
        if values.shape != (1872, 7, 96, 2):
            raise ValueError("expected the supported seven-action forecast pool")
        median = np.median(values, axis=1)
        forecasts = {"forecast_median_guarded": median}
        choices, diagnostics = {}, {}
        for joint in (True, False):
            previous = np.full(len(values), np.inf)
            for blocks in identity["blocks"]:
                name = f"consensus_{'joint' if joint else 'target'}_b{blocks}"
                prediction, choice = consensus_medoid_segments(
                    values, blocks=blocks, joint_targets=joint
                )
                residual = ((prediction - median) ** 2).mean(axis=(1, 2))
                if np.any(residual > previous + 1e-10):
                    raise ValueError(
                        "refining a block partition increased its minimum consensus distance"
                    )
                if blocks == 96 and not joint:
                    np.testing.assert_array_equal(prediction, median)
                previous = residual
                forecasts[name], choices[name] = prediction, choice
                diagnostics[name] = {
                    "residual_to_median_mse": float(residual.mean()),
                    "mean_changes": float(np.mean((np.diff(choice, axis=1) != 0).sum(axis=1)))
                    if blocks > 1
                    else 0.0,
                }
        cache = output / f"{model}_choices.npz"
        _save_npz(cache, **choices)
        files.append({"path": cache.name, "sha256": file_sha256(cache)})
        # Actual futures are read only after all observable-consensus choices are fixed.
        truth_path = root / "truth_z.npy"
        if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
            raise ValueError("the cached standardized outcomes changed")
        truth = np.load(truth_path, mmap_mode="r")[indices]
        for method, point in forecasts.items():
            error = point - truth
            mae, mse = np.abs(error).mean(axis=(1, 2)), (error**2).mean(axis=(1, 2))
            for number, (_, record) in enumerate(selected):
                results.append(
                    {
                        "model_id": model,
                        "method": method,
                        "episode_id": record["episode_id"],
                        "origin_id": record["origin_id"],
                        "family_id": record["family_id"],
                        "dataset_id": record["dataset_id"],
                        "mae": mae[number],
                        "mse": mse[number],
                    }
                )
        local = pd.DataFrame(results)
        local = local[(local.model_id == model) & (local.method == "forecast_median_guarded")]
        measured = np.array([family_macro(local, key) for key in ("mae", "mse")])
        prior = reference[
            (reference.model_id == model)
            & (reference.scope == "unseen_family")
            & (reference.method == "forecast_median_guarded")
        ][["mae", "mse"]].to_numpy()
        if prior.shape != (1, 2):
            raise ValueError("the median reference must be unique")
        np.testing.assert_allclose(measured, prior[0], rtol=0, atol=1e-10)
        parity.append(
            {
                "model_id": model,
                "maximum_reference_difference": float(np.abs(measured - prior[0]).max()),
                "diagnostics": diagnostics,
            }
        )
    frame = pd.DataFrame(results)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family = frame.groupby(["model_id", "method", "family_id"])[["mae", "mse"]].mean().reset_index()
    family.to_csv(output / "family_metrics.csv", index=False)
    summary = family.groupby(["model_id", "method"])[["mae", "mse"]].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "summary": summary.to_dict("records"),
            "choices": files,
            "parity": parity,
            "logical_forecaster_queries_added": 0,
            "limitations": [
                "finer forecast-output selection need not correspond to one completed input",
                "consensus distance is not a lower bound on actual forecasting risk",
                "original R4 raw-input recipe; not the R5 before-model standardization table",
            ],
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
