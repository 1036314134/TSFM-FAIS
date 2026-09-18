import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import forecast_calibration_core as core  # noqa: E402


def test_nan_labels_have_finite_masked_gradients_and_equal_target_weights():
    prediction = torch.tensor([[2.0, 1.0], [5.0, 3.0]], requires_grad=True)
    truth = torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])
    loss = core.smooth_mae(prediction, truth)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1, 0] == 0
    expected = ((4.0 + 1e-6) ** 0.5 + ((1.0 + 1e-6) ** 0.5 + (9.0 + 1e-6) ** 0.5) / 2) / 2
    np.testing.assert_allclose(loss.item(), expected, atol=1e-6)


def test_lp_case_balancing_and_dual_certificate():
    truth = np.array([[0.0, np.nan, np.nan], [2.0, 2.0, 2.0]])
    predictions = np.tile(np.array([0.0, 2.0])[None, :, None], (2, 1, 3))
    fitted = core.fixed_mae_fit(predictions, truth)
    np.testing.assert_allclose(fitted["objective"], 1.0, atol=1e-10)
    assert fitted["duality_gap"] < 1e-10
    assert fitted["feasibility_error"] < 1e-10


def test_source_peer_selection_cannot_see_calibration_labels(monkeypatch):
    random = np.random.default_rng(71)
    values = {str(i): random.normal(size=(1280, 11)) for i in range(4)}
    records = {s: {"values": v, "prefix_end": 640} for s, v in values.items()}
    monkeypatch.setattr(core, "sources", lambda: (records, {}))
    first = core.calibration_sources()[1:]
    for v in values.values():
        v[320:] = random.normal(size=v[320:].shape) * 1e5
    assert core.calibration_sources()[1:] == first


def test_future_and_hidden_truth_cannot_enter_gate_features():
    random = np.random.default_rng(91)
    full = random.normal(size=(700, 17))
    row = {"origin": 500, "outage_age": 24}
    first = core.calibration_context(full, row)
    changed = full.copy()
    changed[476:500, :2] = 1e10
    changed[500:] = -1e10
    second = core.calibration_context(changed, row)
    np.testing.assert_array_equal(first, second)
    requested = []

    class Regression:
        def fit(self, target, features):
            requested.append((target, features))
            return {"features": [], "beta": []}

    def data(x):
        return {
            "context": x,
            "mean": np.zeros(17),
            "scale": np.ones(17),
            "stat_names": np.array(["local_ridge", "peer_ridge"]),
            "stat_targets": np.stack([np.nan_to_num(x[:, :2]), np.nan_to_num(x[:, :2])]),
        }

    np.testing.assert_array_equal(
        core.observable_features(data(first), Regression()),
        core.observable_features(data(second), Regression()),
    )
    assert requested and all(0 not in f and 1 not in f for _, f in requested)


def test_input_mixing_preserves_observations_and_has_exact_endpoints():
    original = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    original[-3:, :2] = torch.nan
    original[1, 2] = torch.nan
    local = torch.nan_to_num(original[:, :2], nan=2.0)
    peer = torch.nan_to_num(original[:, :2], nan=-3.0)
    data = {
        "original": original,
        "local": local,
        "peer": peer,
        "keep": torch.ones(4, dtype=torch.bool),
    }
    for value, expected in ((0.0, local), (1.0, peer)):
        mixed = core.mixed_context(data, torch.full((2,), value))
        torch.testing.assert_close(mixed[:, :2], expected, rtol=0, atol=0)
    alpha = torch.tensor([0.1, 0.9], requires_grad=True)
    mixed = core.mixed_context(data, alpha)
    observed = torch.isfinite(original)
    torch.testing.assert_close(mixed[observed], original[observed], rtol=0, atol=0)
    mixed[:, :2].sum().backward()
    assert torch.isfinite(alpha.grad).all() and (alpha.grad != 0).all()


def test_objective_controls_have_same_capacity_and_initialization():
    a = core.new_gate("forecast_gate", "cpu")
    b = core.new_gate("imputation_gate", "cpu")
    assert sum(p.numel() for p in a.parameters()) == 97
    assert sum(p.numel() for p in core.new_gate("fixed_input_mix", "cpu").parameters()) == 2
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a(torch.ones(2, 10)), torch.full((2,), 0.5), rtol=0, atol=0)
