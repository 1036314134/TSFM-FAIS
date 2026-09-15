"""Verify canonical feature packing on every frozen gate without reading future labels."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from audit_shared_forecast_gate import replay_network  # noqa: E402
from r6_policy_inputs import pack_gate_features  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    process = psutil.Process()
    process.nice(psutil.IDLE_PRIORITY_CLASS)
    process.cpu_affinity([15])
    torch.set_num_threads(1)
    directory = ROOT / "artifacts/iclr27-r6/policy-results-v001/chronos2/h96"
    frame = pd.read_parquet(directory / "individual_features.parquet")
    decisions = pd.read_parquet(directory / "decisions.parquet")
    actions = sorted(frame.candidate_id.unique())
    order = pd.MultiIndex.from_product([decisions.episode_id, actions], names=["episode_id", "candidate_id"])
    original = frame.set_index(["episode_id", "candidate_id"]).loc[order, list(FORECAST_FEATURES)].to_numpy(np.float32).reshape(len(decisions), 7, 33)
    packed = pack_gate_features(original)
    np.testing.assert_array_equal(original, packed)
    binding = json.loads((ROOT / "artifacts/iclr27-r6/method-freeze-v001/manifest.json").read_text(encoding="utf-8"))
    future_root = ROOT / "artifacts/iclr27-r6/source-future-control-v001"
    future = json.loads((future_root / "manifest.json").read_text(encoding="utf-8"))
    entries = [(Path(row["path"]), row["sha256"]) for row in binding["source_models"]]
    entries.extend((future_root / row["path"], row["sha256"]) for row in future["models"])
    checks = []
    for path, digest in entries:
        if file_sha256(path) != digest:
            raise ValueError("a frozen checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        model = SharedForecastGate().eval()
        model.load_state_dict(state)
        normal, replay = predict_weights(model, packed), replay_network(state, packed)
        np.testing.assert_array_equal(normal, replay)
        if file_sha256(path) != digest:
            raise ValueError("a checkpoint changed during the layout test")
        checks.append({"path": str(path), "sha256": digest, "maximum_difference": float(abs(normal - replay).max())})
    output = ROOT / "artifacts/iclr27-r6/gate-replay-diagnostic-v001/canonical_verification.json"
    if output.exists():
        raise ValueError("preserve the completed canonical layout verification")
    _write_json(output, {"status": "completed", "input_values_unchanged": True, "future_arrays_read": False,
                         "checkpoints": checks, "feature_table_sha256": file_sha256(directory / "individual_features.parquet"),
                         "scope": "same real feature values from the stopped unscored run; all 18 frozen checkpoints; no parameter or metric-dependent change"})
    print(json.dumps({"status": "completed", "models": len(checks), "maximum_difference": max(row["maximum_difference"] for row in checks)}), flush=True)


if __name__ == "__main__":
    main()
