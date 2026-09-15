import numpy as np
import torch

from tsfm_fais.routing.forecast_gate import (
    SharedForecastGate,
    compose_forecasts,
    gate_objective,
    teacher_quadratics,
)
from tsfm_fais.routing.forecast_projection import projection_targets


def test_quadratic_gate_loss_equals_actual_returned_forecast_loss():
    rng = np.random.default_rng(91)
    points, teacher = rng.normal(size=(5, 7, 12)), rng.normal(size=(5, 12))
    risks = projection_targets(points, teacher)["direct_risk"]
    gram, alignment = teacher_quadratics(points, risks)
    weights = rng.dirichlet(np.ones(7), 5)
    tensors = [torch.tensor(value, dtype=torch.float64) for value in (weights, gram, alignment)]
    actual = gate_objective(*tensors, "ensemble").numpy()
    baseline = ((np.median(points, axis=1) - teacher) ** 2).mean(1)
    expected = ((compose_forecasts(points, weights) - teacher) ** 2).mean(1) - baseline
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    member = gate_objective(*tensors, "member").numpy()
    dispersion = (
        weights * ((points - compose_forecasts(points, weights)[:, None]) ** 2).mean(2)
    ).sum(1)
    np.testing.assert_allclose(member - actual, dispersion, atol=1e-12)


def test_opposing_errors_are_rewarded_only_by_the_returned_mixture_objective():
    points, teacher = np.array([[[-1.0], [1.0]]]), np.zeros((1, 1))
    gram, alignment = teacher_quadratics(points, projection_targets(points, teacher)["direct_risk"])
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    args = (weights, torch.tensor(gram), torch.tensor(alignment))
    assert gate_objective(*args, "ensemble").item() == 0
    assert gate_objective(*args, "member").item() == 1
    gate_objective(*args, "ensemble").sum().backward()
    assert torch.isfinite(weights.grad).all()


def test_shared_gate_starts_uniform_and_keeps_identical_forecasts():
    torch.manual_seed(5101)
    model = SharedForecastGate()
    training = np.arange(4 * 7 * 33, dtype=float).reshape(4, 7, 33)
    model.fit_normalization(training, np.ones(4))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    weights = model(torch.tensor(training, dtype=torch.float32)).detach().numpy()
    np.testing.assert_allclose(weights, 1 / 7, atol=1e-7)
    _ = model(torch.full((2, 7, 33), 1e12))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert sum(value.numel() for value in model.parameters()) == 1096
    values = np.repeat(np.array([[[1.3, -2.7]]]), 7, axis=1)
    np.testing.assert_array_equal(compose_forecasts(values, weights[:1]), values[:, 0])


def test_fitted_gate_checkpoint_round_trip(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train_shared_forecast_gate import fit_gate, predict_weights

    torch.set_num_threads(1)
    rng = np.random.default_rng(1001)
    features = rng.normal(size=(8, 7, 33))
    points, teacher = rng.normal(size=(8, 7, 4)), rng.normal(size=(8, 4))
    gram, alignment = teacher_quadratics(points, projection_targets(points, teacher)["direct_risk"])
    model, history = fit_gate(features, gram, alignment, np.ones(8), kind="ensemble", seed=5101)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "training_history": history, "seed": 5101}, path)
    restored = SharedForecastGate()
    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    np.testing.assert_array_equal(
        predict_weights(model, features), predict_weights(restored, features)
    )
