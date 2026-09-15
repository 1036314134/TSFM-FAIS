import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from disagreement_scale import ScaledPortfolio, decode, forecast_scale, predict_scaled  # noqa: E402
from positional_forecast_portfolio import position_inputs  # noqa: E402


def test_mad_does_not_amplify_a_single_extreme_candidate():
    outputs = []
    for extreme in (100.0, 1e9):
        points = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, extreme])[None, :, None]
        inputs = position_inputs(np.zeros((1, 7, 33)), points)
        scale = forecast_scale(points, "mad")
        np.testing.assert_array_equal(scale, [[2.0]])
        outputs.append(decode(inputs, scale, np.array([[2.0]])))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_zero_mad_preserves_consensus_and_has_zero_head_gradient():
    points = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 100.0])[None, :, None]
    inputs = position_inputs(np.zeros((1, 7, 33)), points)
    model = ScaledPortfolio()
    tensors = {
        name: torch.as_tensor(
            value, dtype=torch.float32 if name in ("context", "local") else torch.float64
        )
        for name, value in inputs.items()
    }
    prediction = model(**tensors, scale=torch.as_tensor(forecast_scale(points, "mad")))
    ((prediction - 1) ** 2).mean().backward()
    np.testing.assert_array_equal(prediction.detach(), [[0.0]])
    assert torch.equal(model.output.bias.grad, torch.zeros_like(model.output.bias.grad))
    assert torch.equal(model.output.weight.grad, torch.zeros_like(model.output.weight.grad))


def test_fixed_offset_decoder_respects_positive_affine_rescaling():
    rng = np.random.default_rng(18101)
    points = rng.normal(size=(4, 7, 9))
    offset = rng.normal(size=(4, 9))
    transformed = 3.2 * points - 4.7
    for kind in ("range", "mad"):
        inputs = position_inputs(np.zeros((4, 7, 33)), points)
        changed = position_inputs(np.zeros((4, 7, 33)), transformed)
        first = decode(inputs, forecast_scale(points, kind), offset)
        second = decode(changed, forecast_scale(transformed, kind), offset)
        np.testing.assert_allclose(second, 3.2 * first - 4.7, rtol=1e-12, atol=1e-12)


def test_nonuniform_scaled_prediction_replays_inside_bounds_at_both_horizons():
    rng = np.random.default_rng(18102)
    for mode in ("local", "pooled"):
        model = ScaledPortfolio(mode).eval().requires_grad_(False)
        model.output.weight.copy_(torch.linspace(-1, 1, 8)[None])
        model.output.bias.fill_(0.3)
        for horizon in (96, 192):
            points = rng.normal(size=(3, 7, horizon))
            inputs = position_inputs(rng.normal(size=(3, 7, 33)), points)
            for kind in ("unit", "range", "mad"):
                scale = forecast_scale(points, kind)
                actual = predict_scaled(model, inputs, scale, np.arange(3))
                tensors = {name: torch.as_tensor(value) for name, value in inputs.items()}
                direct = model(**tensors, scale=torch.as_tensor(scale)).numpy()
                np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-12)
                assert np.all(actual >= inputs["lower"]) and np.all(actual <= inputs["upper"])
