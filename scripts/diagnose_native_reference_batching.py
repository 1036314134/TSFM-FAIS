"""Compare the same frozen source gate under original and concatenated batching."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_r6_geometry_gates import r6_inputs
from native_source_transfer_io import input_arguments, native_bank_path, read_json
from train_shared_forecast_gate import predict_weights

from tsfm_fais.routing.forecast_gate import SharedForecastGate, compose_forecasts
from tsfm_fais.utility_experiment import _write_json, file_sha256


def predict(args, model_id, features, points):
    probabilities = []
    control = read_json(args.future_control / "manifest.json")
    for entry in (row for row in control["models"] if row["model_id"] == model_id):
        path = args.future_control / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a source control checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        model = SharedForecastGate().eval().requires_grad_(False)
        model.load_state_dict(state)
        probabilities.append(predict_weights(model, features))
    return compose_forecasts(points, np.mean(probabilities, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for model_id in ("chronos2", "timesfm2p5"):
        current = pd.read_parquet(args.study_root / model_id / "decisions.parquet")
        with np.load(args.study_root / model_id / "inputs.npz", allow_pickle=False) as saved:
            features, points = saved["features"], saved["points"]
        actions = next(
            row["actions"]
            for row in read_json(args.future_control / "manifest.json")["models"]
            if row["model_id"] == model_id
        )
        original, original_features, original_points, _, _ = r6_inputs(
            args, model_id, 96, actions, read_json(args.r6_prepared / "manifest.json")
        )
        selected = np.flatnonzero(current.cohort.to_numpy() == "r6")
        indices = pd.Index(original.episode_id).get_indexer(current.iloc[selected].episode_id)
        if (indices < 0).any():
            raise ValueError("current native queries lost their original identifiers")
        np.testing.assert_array_equal(features[selected], original_features[indices])
        np.testing.assert_array_equal(points[selected], original_points[indices])
        with np.load(native_bank_path(args, model_id, "r6"), allow_pickle=False) as saved:
            bank = saved["point_z"][:, saved["methods"].tolist().index("source_future_gate")]
        from aligned_portfolio_io import decision_truth

        expected = decision_truth(original, bank)
        whole = predict(args, model_id, original_features, original_points)
        concatenated = predict(args, model_id, features, points)[selected]
        original_delta = float(abs(whole - expected).max())
        changed_delta = float(abs(concatenated - expected[indices]).max())
        np.testing.assert_allclose(whole, expected, rtol=0, atol=1e-12)
        records.append(
            {
                "model_id": model_id,
                "original_batch_maximum_prediction_difference": original_delta,
                "concatenated_batch_maximum_prediction_difference": changed_delta,
                "native_decisions": len(selected),
                "features_and_candidates_exactly_equal": True,
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "records": records,
            "new_fits": 0,
            "new_forecaster_calls": 0,
            "conclusion": "Use the previously audited source-control predictions for comparisons; input values and fitted parameters did not change.",
        },
    )
    print(records, flush=True)


if __name__ == "__main__":
    main()
