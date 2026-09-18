import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from covariance_adapter_core import (  # noqa: E402
    CovarianceAdapter,
    fit_static,
    forecast_mix,
    geometry,
    masked_smooth_mae,
    repaired_context,
    static_values,
)
from dynamic_posterior_core import condition_state  # noqa: E402


def fixture():
    rng = np.random.default_rng(3601)
    model = fit_static(rng.normal(size=(400, 4)))
    context = rng.normal(size=(12, 4))
    context[-4:, :2] = np.nan
    context[1, 2:] = np.nan
    context[3] = np.nan
    data = {
        "context": context,
        "mean": model["mean"],
        "scale": model["scale"],
        "keep": np.ones(4, bool),
        "base_values": static_values(context, model),
    }
    return data, model, geometry(data, model, "cpu")


def test_zero_adapter_exactly_replays_anchor_and_preserves_observations():
    data, _, prepared = fixture()
    net = CovarianceAdapter(4)
    value = repaired_context(net, prepared)
    torch.testing.assert_close(value, prepared["anchor"], rtol=0, atol=0)
    with torch.no_grad():
        net.correction.copy_(torch.randn_like(net.correction))
    value = repaired_context(net, prepared)
    mask = torch.tensor(np.isfinite(data["context"]))
    torch.testing.assert_close(value[mask], prepared["original"][mask], rtol=0, atol=0)
    torch.testing.assert_close(value[3], prepared["anchor"][3], rtol=0, atol=0)


def test_covariance_correction_stays_positive_definite_at_saturation():
    net = CovarianceAdapter()
    with torch.no_grad():
        net.correction.copy_(torch.randn_like(net.correction) * 1e5)
    transform = net.transform().detach().numpy()
    assert np.linalg.norm(transform - np.eye(17), 2) <= 0.5 + 1e-12
    assert np.linalg.svd(transform, compute_uv=False).min() >= 0.5 - 1e-12
    matrix = np.random.default_rng(35).normal(size=(17, 17))
    covariance = matrix @ matrix.T + 0.01 * np.eye(17)
    corrected = net(torch.tensor(covariance)).detach().numpy()
    assert (
        np.linalg.eigvalsh(corrected).min() >= 0.25 * np.linalg.eigvalsh(covariance).min() - 1e-10
    )


def test_differentiable_conditioning_matches_independent_conditional_means():
    data, model, prepared = fixture()
    net = CovarianceAdapter(4)
    with torch.no_grad():
        net.correction.fill_(0.7)
    covariance = net(prepared["covariance"]).detach().numpy()
    z = (data["context"] - data["mean"]) / data["scale"]
    old = np.stack([condition_state(model["center"], model["covariance"], row)[0] for row in z])
    new = np.stack([condition_state(model["center"], covariance, row)[0] for row in z])
    expected = prepared["anchor"].numpy() + (new - old).astype(np.float32)
    expected[np.isfinite(z)] = z.astype(np.float32)[np.isfinite(z)]
    actual = repaired_context(net, prepared)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)
    actual.square().mean().backward()
    assert net.correction.grad is not None and torch.isfinite(net.correction.grad).all()
    assert net.correction.grad.norm() > 0


def test_source_loss_ignores_unlabelled_channels_and_has_finite_gradients():
    value = torch.tensor([[2.0, 10.0], [4.0, 50.0]], requires_grad=True)
    truth = torch.tensor([[0.0, float("nan")], [float("nan"), float("nan")]])
    loss = masked_smooth_mae(value, truth)
    loss.backward()
    assert abs(loss.item() - (4.0 + 1e-6) ** 0.5) < 1e-6
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad[:, 1], torch.zeros(2), rtol=0, atol=0)
    assert value.grad[1, 0] == 0


def test_forecast_portfolio_has_target_specific_weights_and_preserves_constants():
    bank = np.array([[[1.0, 10.0], [1.0, 20.0]], [[3.0, 30.0], [3.0, 40.0]]])
    weight = np.array([[0.25, 0.75], [0.75, 0.25]])
    np.testing.assert_array_equal(forecast_mix(bank, weight), [[2.5, 15.0], [2.5, 25.0]])
