import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audit_metric_source_gates import compare_previous_models, direct_control  # noqa: E402
from metric_source_gate import (  # noqa: E402
    EPSILON,
    fit_fixed_metric,
    fixed_objective,
    metric_objective,
    polish_simplex,
)
from train_calibrated_source_gates import probability_from_state  # noqa: E402
from train_metric_source_gates import fit_metric_gate  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)


def test_metric_gradient_matches_direct_errors_on_the_simplex():
    rng = np.random.default_rng(52)
    points, target = rng.normal(size=(4, 7, 9)), rng.normal(size=(4, 9))
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, target)["raw_projection"]
    logits = torch.tensor(rng.normal(size=(4, 7)), dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(1)
    p, y = torch.tensor(points), torch.tensor(target)
    error = torch.einsum("na,naq->nq", probability, p) - y
    smoothed = torch.sqrt(error.square() + EPSILON**2)
    assert float((smoothed - error.abs()).min()) >= 0
    assert float((smoothed - error.abs()).max()) <= EPSILON
    direct = (smoothed + error.square()).mean(1) / 2
    relative = metric_objective(
        probability, p, y, torch.tensor(gram), torch.tensor(alignment), "joint"
    )
    first = torch.autograd.grad(direct.sum(), logits, retain_graph=True)[0]
    second = torch.autograd.grad(relative.sum(), logits)[0]
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)


def test_convex_control_recovers_median_and_joint_tradeoff_for_outlier():
    points = np.array([[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]])
    target = np.array([[0.0, 0.0, 10.0]])
    mae = fit_fixed_metric(points, target, np.ones(1), "mae")
    joint = fit_fixed_metric(points, target, np.ones(1), "joint")
    assert 0 <= 10 * mae["weights"][1] < 0.001
    assert abs(10 * joint["weights"][1] - (10 / 3 - 0.5 / 3)) < 0.001
    assert mae["single_index"] == 0 and joint["single_index"] == 0
    assert mae["optimality_gap"] <= 1e-7 and joint["optimality_gap"] <= 1e-7


def test_convex_gradient_matches_numerical_directional_derivative():
    rng = np.random.default_rng(72)
    points, target = rng.normal(size=(5, 7, 6)), rng.normal(size=(5, 6))
    objective = fixed_objective(points, target, np.arange(1, 6), "joint")
    probability = np.ones(7) / 7
    direction = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, gradient = objective(probability)
    step = 1e-6
    numerical = (
        objective(probability + step * direction)[0] - objective(probability - step * direction)[0]
    ) / (2 * step)
    np.testing.assert_allclose(numerical, gradient @ direction, rtol=1e-6, atol=1e-8)


def test_independent_fixed_control_reconstructs_loss_and_gradient():
    rng = np.random.default_rng(88)
    points, target = rng.normal(size=(5, 7, 9)), rng.normal(size=(5, 9))
    weights = np.arange(1, 6, dtype=float)
    initial = weights.copy()
    probability = np.arange(1, 8, dtype=float) / 28
    for kind in ("mae", "joint"):
        expected = fixed_objective(points, target, weights, kind)(probability)
        actual = direct_control(points, target, weights, probability, kind)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(weights, initial)


def test_metric_fit_excludes_unknown_future_and_replays_prediction():
    torch.set_num_threads(1)
    rng = np.random.default_rng(62)
    frame = pd.DataFrame(
        {
            "family_id": ["f"] * 8,
            "dataset_id": ["d"] * 8,
            "origin_id": list(map(str, range(8))),
            "episode_id": list(map(str, range(8))),
        }
    )
    points, target = rng.normal(size=(8, 7, 9)), rng.normal(size=(8, 9))
    features = rng.normal(size=(8, 7, 97)).astype(np.float32)
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, target)["raw_projection"]
    target[6:] = np.nan
    alignment[6:] = np.nan
    saved = fit_metric_gate(
        frame, features, points, target, gram, alignment, np.arange(6), 5101, "joint"
    )
    assert len(saved["history"]) == 25
    assert set(saved["training_origins"]) == set(map(str, range(6)))
    probability = probability_from_state(saved["state_dict"], features[6:])
    assert probability.shape == (2, 7)
    np.testing.assert_allclose(probability.sum(1), 1)


def test_precision_refinement_reaches_boundary_without_changing_accepted_solutions():
    target = np.array([1.0, 0.0, 0.0])

    def objective(value):
        delta = value - target
        return float(delta @ delta), 2 * delta

    initial = np.array([0.4, 0.3, 0.3])
    refined, steps = polish_simplex(objective, initial)
    np.testing.assert_allclose(refined, target, rtol=0, atol=1e-12)
    assert 0 < steps <= 200
    np.testing.assert_array_equal(initial, [0.4, 0.3, 0.3])
    accepted = np.array([1 - 1e-10, 1e-10, 0.0])
    repeated, steps = polish_simplex(objective, accepted)
    np.testing.assert_array_equal(repeated, accepted)
    assert steps == 0


def test_previous_attempt_comparison_requires_exact_model_states(tmp_path):
    import json

    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old_identity = {"module_sha256": "old", "protocol_sha256": "old", "script_sha256": "unchanged"}
    identity = {**old_identity, "module_sha256": "new", "protocol_sha256": "new"}
    (old_root / "identity.json").write_text(json.dumps(old_identity), encoding="utf-8")
    saved = {
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "initial_parameter_sha256": "same",
        "train_indices_sha256": "same",
        "training_origins": ["o"],
        "training_families": ["f"],
        "history": [{"epoch": 1, "loss": 0.5}],
        "metadata": {"condition": "joint_future", "identity_sha256": "old"},
    }
    torch.save(saved, old_root / "model.pt")
    torch.save(
        {**saved, "metadata": {**saved["metadata"], "identity_sha256": "new"}},
        new_root / "model.pt",
    )
    assert compare_previous_models(old_root, new_root, identity) == 1
    saved["state_dict"]["weight"][0] += 0.01
    torch.save(saved, new_root / "model.pt")
    with pytest.raises(AssertionError):
        compare_previous_models(old_root, new_root, identity)
