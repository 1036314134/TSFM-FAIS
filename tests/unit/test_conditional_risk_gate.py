import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from audit_conditional_risk_gate import direct_fixed  # noqa: E402
from conditional_risk_gate import (  # noqa: E402
    conditional_objective,
    fit_conditional_gate,
    fixed_terms,
    fixed_value_gradient,
)
from pool_gate_model import fit_pool_gate  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)


def test_expected_loss_and_gradient_match_sampled_gaussian_futures():
    rng = np.random.default_rng(16601)
    points = rng.normal(size=(1, 8, 4))
    target = rng.normal(size=(1, 4))
    variance = np.full((1, 4), 0.7)
    weight = np.full(8, 1 / 8)
    probability = torch.tensor(weight[None], dtype=torch.float64, requires_grad=True)
    gram = forecast_geometry(points)[3]
    alignment = projection_targets(points, target)["raw_projection"]
    value = conditional_objective(
        probability,
        *[torch.tensor(x) for x in (points, target, variance, gram, alignment)],
        torch.tensor([True]),
        True,
    )
    gradient = torch.autograd.grad(value.sum(), probability)[0].numpy()[0]
    terms = fixed_terms(points, target, variance, np.array([True]), np.ones(1), True)
    risk, fixed_gradient, _ = fixed_value_gradient(terms, weight)
    constant = 0.5 * (((np.median(points, axis=1) - target) ** 2 + variance).mean())
    np.testing.assert_allclose(float(value.detach()) + constant, risk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradient - gradient[0], fixed_gradient - fixed_gradient[0], rtol=1e-12, atol=1e-12
    )
    direct_value, direct_gradient = direct_fixed(
        points, target, variance, np.array([True]), np.ones(1), weight, True, None
    )
    np.testing.assert_allclose(direct_value, risk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct_gradient, fixed_gradient, rtol=1e-12, atol=1e-12)
    draws = target + rng.normal(size=(50000, 4)) * np.sqrt(variance)
    prediction = weight @ points[0]
    error = prediction - draws
    samples = (abs(error) + error**2).mean(1) / 2
    assert abs(samples.mean() - risk) < 5 * samples.std(ddof=1) / np.sqrt(len(samples))


def test_nonsmooth_fixed_certificate_selects_a_valid_zero_residual_subgradient():
    points = np.repeat(np.linspace(-1, 1, 8)[None, :, None], 2, axis=0)
    target = np.array([[0.1], [0.0]])
    terms = fixed_terms(
        points, target, np.zeros_like(target), np.array([False, True]), np.array([1.0, 3.0]), False
    )
    probability = np.full(8, 1 / 8)
    value, gradient, certificate = fixed_value_gradient(terms, probability, certificate=True)
    assert certificate["tie_indices"] == [1]
    assert abs(certificate["tie_subgradient"][0]) <= 1
    assert gradient @ probability - gradient.min() < 1e-10
    direct_value, direct_gradient = direct_fixed(
        points,
        target,
        np.zeros_like(target),
        np.array([False, True]),
        np.array([1.0, 3.0]),
        probability,
        False,
        certificate,
    )
    np.testing.assert_allclose(direct_value, value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(direct_gradient, gradient, rtol=1e-12, atol=1e-12)
    direction = np.r_[1.0, np.zeros(6), -1.0]
    for step in (-1e-4, 1e-4):
        perturbed = fixed_value_gradient(terms, probability + step * direction)[0]
        assert perturbed >= value


def test_original_training_replays_and_ignores_validation_labels():
    torch.set_num_threads(1)
    rng = np.random.default_rng(16602)
    frame = pd.DataFrame(
        {
            "family_id": ["a"] * 20 + ["held"] * 2,
            "dataset_id": ["dataset"] * 22,
            "episode_id": [str(i) for i in range(22)],
            "origin_id": [f"origin{i}" for i in range(22)],
        }
    )
    points = rng.normal(size=(22, 8, 4))
    target = rng.normal(size=(22, 4))
    features = rng.normal(size=(22, 8, 97)).astype(np.float32)
    features[:, :, 33:] = 0
    gram = forecast_geometry(points)[3]
    alignment = projection_targets(points, target)["raw_projection"]
    indices = np.arange(20)
    original = fit_pool_gate(
        frame, features, points, target, gram, alignment, indices, 5101, "joint"
    )
    data = {
        "features": features,
        "points": points,
        "truth": target.copy(),
        "mean": target.copy(),
        "variance": np.zeros_like(target),
        "simulated": np.zeros(22, bool),
        "gram": gram,
    }
    replay = fit_conditional_gate(frame, data, indices, 5101, 25, False)
    assert replay["history"] == original["history"]
    for name, value in original["state_dict"].items():
        torch.testing.assert_close(value, replay["state_dict"][name], rtol=0, atol=0)
    data["truth"][20:] = 1e12
    data["mean"][20:] = -1e12
    changed = fit_conditional_gate(frame, data, indices, 5101, 25, False)
    for name, value in replay["state_dict"].items():
        torch.testing.assert_close(value, changed["state_dict"][name], rtol=0, atol=0)
