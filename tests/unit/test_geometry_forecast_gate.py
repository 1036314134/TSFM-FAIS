import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_shared_forecast_gate import replay_network  # noqa: E402
from geometry_forecast_gate import geometry_features  # noqa: E402
from train_shared_forecast_gate import predict_weights  # noqa: E402

from tsfm_fais.routing.forecast_gate import SharedForecastGate  # noqa: E402


def test_geometry_features_match_direct_centered_products_and_matched_control():
    rng = np.random.default_rng(9201)
    points, base = rng.normal(size=(3, 7, 19)), rng.normal(size=(3, 7, 33))
    full = geometry_features(base, points, mode="full")
    diagonal = geometry_features(base, points, mode="diagonal")
    centered = points - np.median(points, axis=1)[:, None]
    expected = np.stack([np.stack([np.mean(x[a] * x, axis=1) for a in range(7)]) for x in centered])
    expected = (np.sign(expected) * np.log1p(abs(expected))).astype(np.float32)
    np.testing.assert_allclose(full[:, :, 40:], expected, atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(full[:, :, :40], diagonal[:, :, :40])
    np.testing.assert_array_equal(full[:, :, 33:40], np.broadcast_to(np.eye(7), (3, 7, 7)))
    np.testing.assert_array_equal(diagonal[:, :, 40:], full[:, :, 40:] * np.eye(7)[None])


def test_geometry_block_is_translation_and_coordinate_permutation_invariant():
    rng = np.random.default_rng(9202)
    points, base = rng.normal(size=(2, 7, 24)), np.zeros((2, 7, 33))
    transformed = points[:, :, ::-1] + rng.normal(size=(2, 1, 24))
    original = geometry_features(base, points, mode="full")
    shifted = geometry_features(base, transformed, mode="full")
    np.testing.assert_allclose(original, shifted, atol=1e-7, rtol=1e-7)


def test_identical_forecasts_have_zero_relationship_features():
    points = np.broadcast_to(np.arange(12), (2, 7, 12))
    for mode in ("full", "diagonal"):
        values = geometry_features(np.zeros((2, 7, 33)), points, mode=mode)
        np.testing.assert_array_equal(values[:, :, 40:], np.zeros((2, 7, 7)))


def test_nonuniform_geometry_gate_has_exact_matrix_inference_replay():
    torch.manual_seed(9203)
    model = SharedForecastGate(features=47).eval().requires_grad_(False)
    model.score[-1].weight.copy_(torch.linspace(-0.3, 0.5, 16)[None])
    model.action_bias.copy_(torch.linspace(-0.1, 0.1, 7))
    rng = np.random.default_rng(9203)
    features = geometry_features(
        rng.normal(size=(35, 7, 33)), rng.normal(size=(35, 7, 96)), mode="full"
    )
    actual = predict_weights(model, features)
    assert not np.allclose(actual, 1 / 7)
    np.testing.assert_array_equal(actual, replay_network(model.state_dict(), features))
