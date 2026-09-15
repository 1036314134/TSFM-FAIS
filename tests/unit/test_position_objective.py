import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from position_objective import fit_position_loss, position_loss  # noqa: E402
from positional_forecast_portfolio import fit_position, position_inputs  # noqa: E402


def test_mse_adapter_exactly_replays_original_fit_for_both_scopes():
    torch.set_num_threads(1)
    rng = np.random.default_rng(17101)
    inputs = position_inputs(rng.normal(size=(15, 7, 33)), rng.normal(size=(15, 7, 12)))
    teacher, weights = rng.normal(size=(15, 12)), rng.uniform(0.5, 1.5, size=15)
    settings = {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "gradient_norm": 1.0,
    }
    for mode in ("local", "pooled"):
        old, old_history, old_initial = fit_position(
            inputs, teacher, weights, mode=mode, seed=5101, settings=settings
        )
        current, history, initial = fit_position_loss(
            inputs, teacher, weights, mode=mode, seed=5101, settings=settings, objective="mse"
        )
        assert initial == old_initial
        assert history == [
            {"epoch": row["epoch"], "mean_training_loss": row["mean_training_teacher_mse"]}
            for row in old_history
        ]
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)


def test_joint_loss_has_the_expected_robust_location_gradient():
    value = torch.tensor(24.75, dtype=torch.float64, requires_grad=True)
    point = value.expand(1, 4)
    target = torch.tensor([[0.0, 0.0, 0.0, 100.0]], dtype=torch.float64)
    joint = torch.autograd.grad(
        position_loss(point, target, "joint").sum(), value, retain_graph=True
    )[0]
    mse = torch.autograd.grad(position_loss(point, target, "mse").sum(), value)[0]
    assert abs(float(joint)) < 1e-7
    np.testing.assert_allclose(float(mse), -0.5, rtol=0, atol=1e-12)
