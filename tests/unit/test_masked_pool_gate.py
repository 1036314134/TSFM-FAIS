import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_real_pool import direct_observed_control  # noqa: E402
from masked_pool_gate import (
    fit_observed_gate,
    observed_fixed_objective,
    observed_geometry,
    observed_objective,
)  # noqa: E402
from metric_source_gate import EPSILON  # noqa: E402
from pool_gate_model import fit_pool_gate  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)


def test_observed_objective_has_correct_gradient_and_ignores_unknown_labels():
    rng = np.random.default_rng(610)
    points = rng.normal(size=(3, 8, 192))
    target = rng.normal(size=(3, 192))
    mask = np.ones_like(target, bool)
    mask[:, ::5] = False
    target[~mask] = np.nan
    g, b, u = observed_geometry(points, target, mask, True)
    logits = torch.tensor(rng.normal(size=(3, 8)), dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(1)
    p, y, observed, coord = (
        torch.tensor(points),
        torch.tensor(target),
        torch.tensor(mask),
        torch.tensor(u),
    )
    value = observed_objective(probability, p, y, torch.tensor(g), torch.tensor(b), observed, coord)
    changed = y.clone()
    changed[~observed] = 1e9
    torch.testing.assert_close(
        value,
        observed_objective(
            probability, p, changed, torch.tensor(g), torch.tensor(b), observed, coord
        ),
        rtol=0,
        atol=0,
    )
    prediction = torch.einsum("na,naq->nq", probability, p)
    error = torch.where(observed, prediction - torch.nan_to_num(y), 0.0)
    direct = ((error.square() + torch.sqrt(error.square() + EPSILON**2)) * coord).sum(1) / 2
    first = torch.autograd.grad(value.sum(), logits, retain_graph=True)[0]
    second = torch.autograd.grad(direct.sum(), logits)[0]
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)


def test_observed_fixed_loss_matches_independent_residual_reconstruction():
    rng = np.random.default_rng(620)
    points = rng.normal(size=(4, 8, 96))
    truth = rng.normal(size=(4, 96))
    observed = np.ones_like(truth, bool)
    observed[:, ::4] = False
    truth[~observed] = np.nan
    _, _, coordinates = observed_geometry(points, truth, observed, False)
    weights = np.arange(1, 5, dtype=float)
    probability = np.arange(1, 9, dtype=float) / 36
    expected = observed_fixed_objective(points, truth, coordinates, weights)(probability)
    actual = direct_observed_control(points, truth, observed, weights, probability, False)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)


def test_complete_observation_path_reproduces_original_pool_training():
    torch.set_num_threads(1)
    rng = np.random.default_rng(630)
    n = 7
    frame = pd.DataFrame(
        {
            "family_id": ["f"] * n,
            "dataset_id": ["d"] * n,
            "origin_id": list(map(str, range(n))),
            "episode_id": list(map(str, range(n))),
        }
    )
    points = rng.normal(size=(n, 8, 96))
    truth = rng.normal(size=(n, 96))
    features = rng.normal(size=(n, 8, 97)).astype(np.float32)
    gram = forecast_geometry(points)[3]
    alignment = projection_targets(points, truth)["raw_projection"]
    data = {
        "points": points,
        "truth": truth,
        "features": features,
        "observed": np.ones_like(truth, bool),
        "coordinates": np.full_like(truth, 1 / 96),
        "gram": gram,
        "alignment": alignment,
    }
    indices = np.arange(n)
    original = fit_pool_gate(
        frame, features, points, truth, gram, alignment, indices, 5101, "joint"
    )
    current = fit_observed_gate(frame, data, indices, 5101, 25)
    assert original["history"] == current["history"]
    for name in original["state_dict"]:
        torch.testing.assert_close(
            original["state_dict"][name], current["state_dict"][name], rtol=0, atol=0
        )
