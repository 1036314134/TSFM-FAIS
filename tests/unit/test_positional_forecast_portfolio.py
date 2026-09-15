import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from positional_forecast_portfolio import (  # noqa: E402
    PositionalPortfolio,
    position_inputs,
    predict_position,
)
from positional_portfolio_io import restore_positions, target_nodes  # noqa: E402


@pytest.mark.parametrize("mode", ["local", "pooled"])
def test_initial_prediction_is_exact_median_at_both_horizons(mode):
    rng = np.random.default_rng(9301)
    for horizon in (96, 192):
        points = rng.normal(size=(3, 7, horizon))
        inputs = position_inputs(rng.normal(size=(3, 7, 33)), points)
        model = PositionalPortfolio(mode).eval().requires_grad_(False)
        np.testing.assert_array_equal(predict_position(model, inputs), np.median(points, axis=1))


@pytest.mark.parametrize(
    "values,target,sign", [([0, 0, 0, 0, 1, 2, 3], 1.0, -1), ([0, 1, 2, 3, 3, 3, 3], 2.0, 1)]
)
def test_boundary_median_has_a_nonzero_gradient_toward_the_interior(values, target, sign):
    inputs = position_inputs(np.zeros((1, 7, 33)), np.asarray(values, float)[None, :, None])
    model = PositionalPortfolio()
    point = model(
        **{
            name: torch.as_tensor(
                value, dtype=torch.float32 if name in ("context", "local") else torch.float64
            )
            for name, value in inputs.items()
        }
    )
    ((point - target) ** 2).mean().backward()
    assert sign * float(model.output.bias.grad) > 0


def test_nonuniform_prediction_replays_and_stays_in_the_forecast_envelope():
    rng = np.random.default_rng(9302)
    inputs = position_inputs(rng.normal(size=(7, 7, 33)), rng.normal(size=(7, 7, 96)))
    for mode in ("local", "pooled"):
        model = PositionalPortfolio(mode).eval().requires_grad_(False)
        model.output.weight.copy_(torch.linspace(-1, 1, 8)[None])
        for bias in (-20, 0.3, 20):
            model.output.bias.fill_(bias)
            predicted = predict_position(model, inputs)
            assert np.all(predicted >= inputs["lower"]) and np.all(predicted <= inputs["upper"])
        if mode == "pooled":
            with torch.no_grad():
                offset = model.offsets(
                    torch.as_tensor(inputs["context"]), torch.as_tensor(inputs["local"])
                )
            assert torch.equal(offset, offset[:, :1].expand_as(offset))


def test_identical_candidates_are_preserved_after_nonzero_bias():
    points = np.broadcast_to(np.linspace(-1, 1, 20), (2, 7, 20))
    inputs = position_inputs(np.zeros((2, 7, 33)), points)
    model = PositionalPortfolio().eval().requires_grad_(False)
    model.output.bias.fill_(10)
    np.testing.assert_array_equal(predict_position(model, inputs), points[:, 0])


def test_a_wide_candidate_range_does_not_amplify_a_small_learned_correction():
    points = np.broadcast_to(np.asarray([0, 0, 0, 0, 1, 2, 1e6])[None, :, None], (1, 7, 3))
    inputs = position_inputs(np.zeros((1, 7, 33)), points)
    model = PositionalPortfolio().eval().requires_grad_(False)
    model.output.bias.fill_(0.01)
    np.testing.assert_allclose(
        predict_position(model, inputs), np.full((1, 3), 0.01), rtol=0, atol=1e-8
    )


def test_joint_targets_preserve_forecast_coordinate_order():
    points = np.arange(2 * 7 * 5 * 2).reshape(2, 7, 5, 2)
    decisions = pd.DataFrame(
        {"episode_id": ["a", "b"], "episode_index": [0, 1], "target_slot": [-1, -1]}
    )
    nodes, base, vectors = target_nodes(
        decisions, np.zeros((2, 7, 33)), points.reshape(2, 7, -1), joint=True
    )
    assert base.shape == (4, 7, 33)
    np.testing.assert_array_equal(restore_positions(vectors[:, 0], nodes, 2, 5), points[:, 0])


def test_interleaved_independent_targets_restore_in_episode_order():
    decisions = pd.DataFrame(
        {
            "episode_id": ["a0", "a1", "b0", "b1"],
            "episode_index": [0, 0, 1, 1],
            "target_slot": [0, 1, 0, 1],
        }
    )
    points = np.arange(2 * 5 * 2).reshape(2, 5, 2)
    values = np.stack(
        [points[row.episode_index, :, row.target_slot] for row in decisions.itertuples(index=False)]
    )
    np.testing.assert_array_equal(restore_positions(values, decisions, 2, 5), points)
