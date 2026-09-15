import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def objective(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "composer_training", scripts / "train_differentiable_block_composer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.training_objective


def test_blended_loss_preserves_supervised_control_and_detaches_targets(objective):
    prediction = torch.tensor([[0.0]], requires_grad=True)
    truth = torch.tensor([[2.0]], requires_grad=True)
    teacher = torch.tensor([[1.0]], requires_grad=True)
    scales = {"mae": 1.0, "mse": 1.0}
    assert objective(prediction, truth, scales).item() == 3.0
    loss = objective(prediction, truth, scales, teacher=teacher, teacher_weight=0.5)
    assert loss.item() == 2.0
    loss.backward()
    assert prediction.grad.item() == -2.0
    assert truth.grad is None and teacher.grad is None


def test_pure_teacher_loss_does_not_read_source_future(objective):
    prediction, teacher = torch.tensor([[0.0]]), torch.tensor([[1.0]])
    scales = {"mae": 1.0, "mse": 1.0}
    first = objective(prediction, torch.zeros(1, 1), scales, teacher=teacher, teacher_weight=1.0)
    changed = objective(
        prediction, torch.full((1, 1), float("nan")), scales, teacher=teacher, teacher_weight=1.0
    )
    assert torch.equal(first, changed)


def test_teacher_configuration_requires_valid_weight_and_teacher(objective):
    values, scales = torch.zeros(1, 1), {"mae": 1.0, "mse": 1.0}
    with pytest.raises(ValueError, match="requires"):
        objective(values, values, scales, teacher_weight=0.5)
    with pytest.raises(ValueError, match="weight"):
        objective(values, values, scales, teacher_weight=1.5)
