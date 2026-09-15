import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "predictive_consensus",
    Path(__file__).resolve().parents[2] / "scripts/analyze_predictive_consensus.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_simplex_projection_preserves_feasibility_and_beats_vertex_teacher_distance():
    predictions = np.array(
        [
            [[0.0, 3.0], [2.0, 0.0], [1.0, 1.0]],
            [[2.0, 1.0], [0.0, 2.0], [3.0, 0.0]],
            [[4.0, 0.0], [1.0, 4.0], [0.0, 2.0]],
        ]
    )
    result = MODULE.consensus_controls(predictions)
    teacher = result["median"][0]
    for method in ("simplex_sequence", "simplex_target"):
        forecast, weights = result[method]
        assert np.min(weights) >= -1e-10
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-8)
        distance = ((forecast - teacher) ** 2).mean()
        assert (
            distance <= min(((candidate - teacher) ** 2).mean() for candidate in predictions) + 1e-8
        )


def test_identical_predictions_remain_identical_for_every_control():
    prediction = np.arange(8.0).reshape(4, 2)
    outputs = MODULE.consensus_controls(np.stack([prediction] * 5))
    for forecast, _ in outputs.values():
        np.testing.assert_allclose(forecast, prediction, rtol=0, atol=1e-12)
