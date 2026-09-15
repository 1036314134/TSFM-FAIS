import torch

from tsfm_fais.routing.consensus_projection import project_consensus


def test_projection_fits_consensus_with_observation_preserving_simplex_weights():
    context = torch.tensor([[2.0], [float("nan")]])
    candidates = torch.tensor([[[2.0], [0.0]], [[2.0], [1.0]], [[2.0], [2.0]]])
    predicted = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    native = torch.tensor([[0.3]])

    def forecast(values):
        assert values[0, 0].item() == 2.0
        return values[-1:]

    result = project_consensus(candidates, context, predicted, native, forecast, steps=20)
    assert result["checkpoints"][20]["teacher_mse"] < result["checkpoints"][0]["teacher_mse"]
    if result["weights"] is not None:
        assert (result["weights"] >= 0).all()
        torch.testing.assert_close(result["weights"].sum(0), torch.ones(1))


def test_no_missing_values_avoid_projection_calls():
    context = torch.ones(3, 1)

    def unexpected(_):
        raise AssertionError("complete context should not require optimization")

    result = project_consensus(
        torch.stack([context, context]), context, torch.ones(2, 2, 1), torch.ones(2, 1), unexpected
    )
    assert result["forward_calls"] == 0
    assert result["stop_reason"] == "complete_context"


def test_nonfinite_trial_keeps_the_observable_forecast_anchor():
    context = torch.tensor([[2.0], [float("nan")]])
    candidates = torch.tensor([[[2.0], [0.0]], [[2.0], [1.0]], [[2.0], [2.0]]])
    predicted = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    native = torch.tensor([[0.3]])
    result = project_consensus(
        candidates, context, predicted, native, lambda values: values[-1:] * float("nan")
    )
    assert result["stop_reason"] == "nonfinite_objective"
    torch.testing.assert_close(result["checkpoints"][20]["prediction"], native)
