"""Check duplicate-row rounding and exact reuse for the shared-weight control."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pyarrow.dataset  # noqa: F401 - load the Arrow extension before Torch on Windows.
import torch
from evaluate_r6_geometry_gates import r6_inputs
from latent_source_inputs import ROOT, read_json
from shared_context_inference import broadcast_context_weights
from train_calibrated_source_gates import probability_from_state

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    output = ROOT / "artifacts/iclr27-r11/shared-inference-diagnostic-v001"
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed shared-inference diagnostic")
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[-1:])
    torch.set_num_threads(1)
    args = SimpleNamespace(
        r6_policy=ROOT / "artifacts/iclr27-r6/policy-results-v002",
        r6_audit=ROOT / "artifacts/iclr27-r6/policy-audit-v002",
    )
    prep = read_json(ROOT / "artifacts/iclr27-r6/confirmation-v001/prepared/manifest.json")
    actions = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    records = []
    for horizon in (96, 192):
        decisions, base, _, bank, _ = r6_inputs(args, "chronos2", horizon, actions, prep)
        global_features = np.empty_like(base)
        global_features[decisions.episode_index.to_numpy(int)] = base
        features = np.ascontiguousarray(
            np.pad(np.repeat(global_features, 2, axis=0), ((0, 0), (0, 0), (0, 64)))
        )
        vectors = bank.transpose(0, 3, 1, 2).reshape(2 * len(prep["episodes"]), 7, horizon)
        np.testing.assert_array_equal(features[::2], features[1::2])
        for seed in (5101, 5102, 5103):
            path = (
                ROOT
                / f"artifacts/iclr27-r11/target-local-source-v001/full_source/broadcast_{seed}.pt"
            )
            saved = torch.load(path, map_location="cpu", weights_only=True)
            duplicate = probability_from_state(saved["state_dict"], features)
            shared = broadcast_context_weights(saved["state_dict"], features)
            np.testing.assert_array_equal(shared[::2], shared[1::2])
            records.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "model_sha256": file_sha256(path),
                    "duplicate_row_maximum_weight_difference": float(
                        abs(duplicate[::2] - duplicate[1::2]).max()
                    ),
                    "single_inference_weight_difference": float(abs(duplicate - shared).max()),
                    "maximum_forecast_difference": float(
                        abs(
                            compose_forecasts(vectors, duplicate)
                            - compose_forecasts(vectors, shared)
                        ).max()
                    ),
                    "shared_weight_difference": 0.0,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "checks": records,
            "future_arrays_read": False,
            "new_fits": 0,
            "new_forecaster_calls": 0,
        },
    )
    print(records, flush=True)


if __name__ == "__main__":
    main()
