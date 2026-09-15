"""Inspect the stopped gate replay without reading future outcomes or changing models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import evaluate_r6_policies  # noqa: E402,F401
from audit_shared_forecast_gate import replay_network  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _write_json  # noqa: E402


def main():
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity([15])
    torch.set_num_threads(1)
    binding = json.loads((ROOT / "artifacts/iclr27-r6/method-freeze-v001/manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in binding["source_models"] if row["model_id"] == "chronos2" and row["objective"] == "ensemble" and row["seed"] == 5101)
    state = torch.load(entry["path"], map_location="cpu", weights_only=True)["state_dict"]
    directory = ROOT / "artifacts/iclr27-r6/policy-results-v001/chronos2/h96"
    frame = pd.read_parquet(directory / "individual_features.parquet")
    decisions = pd.read_parquet(directory / "decisions.parquet")
    actions = sorted(frame.candidate_id.unique())
    matrix = frame.set_index(["episode_id", "candidate_id"])
    order = pd.MultiIndex.from_product([decisions.episode_id, actions], names=["episode_id", "candidate_id"])
    bulk = matrix.loc[order, list(FORECAST_FEATURES)].to_numpy(np.float32).reshape(len(decisions), 7, 33)
    pieces = []
    for identifier in decisions.episode_id:
        pieces.append(matrix.loc[[(identifier, action) for action in actions], list(FORECAST_FEATURES)].to_numpy(np.float32).reshape(1, 7, 33))
    concatenated = np.concatenate(pieces)
    np.testing.assert_array_equal(bulk, concatenated)
    rows = []
    for name, features in (("bulk", bulk), ("concatenated", concatenated), ("contiguous", np.ascontiguousarray(concatenated))):
        for requires_grad in (True, False):
            model = SharedForecastGate().eval()
            model.load_state_dict(state)
            model.requires_grad_(requires_grad)
            normal = predict_weights(model, features)
            replay = replay_network(state, features)
            rows.append({"input": name, "strides": features.strides, "requires_grad": requires_grad,
                         "maximum_difference": float(abs(normal - replay).max()), "different_entries": int((normal != replay).sum())})
    _write_json(ROOT / "artifacts/iclr27-r6/gate-replay-diagnostic-v001/diagnosis.json", {"status": "completed", "default_dtype": str(torch.get_default_dtype()), "comparisons": rows, "future_arrays_read": False})
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
