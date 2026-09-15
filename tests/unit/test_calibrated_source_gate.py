import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from calibrated_source_gate import (  # noqa: E402
    choose_configuration,
    fit_snapshots,
    regularized_objective,
    split_origins,
)
from train_calibrated_source_gates import probability_from_state  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)


def test_origin_split_keeps_masks_targets_together_and_purges_future_overlap():
    positions = {f"o{i}": value for i, value in enumerate([100, 140, 190, 250, 300])}
    frame = pd.DataFrame(
        [
            {
                "origin_id": origin,
                "family_id": "f",
                "dataset_id": "d",
                "item_id": "s",
                "target_slot": target,
                "mask": mask,
            }
            for origin in positions
            for target in (0, 1)
            for mask in (1, 2)
        ]
    )
    fit, calibration, purged = split_origins(frame, np.arange(len(frame)), positions)
    assert set(frame.iloc[fit].origin_id) == {"o0", "o1"}
    assert set(frame.iloc[calibration].origin_id) == {"o3", "o4"}
    assert set(frame.iloc[purged].origin_id) == {"o2"}
    shuffled = frame.sample(frac=1, random_state=4).reset_index(drop=True)
    repeated = split_origins(shuffled, np.arange(len(shuffled)), positions)
    assert [set(shuffled.iloc[ids].origin_id) for ids in repeated] == [
        set(frame.iloc[ids].origin_id) for ids in (fit, calibration, purged)
    ]


def test_regularization_matches_direct_forecast_loss_and_gradient():
    rng = np.random.default_rng(91)
    points, truth = rng.normal(size=(5, 7, 11)), rng.normal(size=(5, 11))
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, truth)["raw_projection"]
    anchor = np.full(7, 1 / 7)
    logits = torch.tensor(rng.normal(size=(5, 7)), dtype=torch.float64, requires_grad=True)
    probability = logits.softmax(1)
    p, y = torch.tensor(points), torch.tensor(truth)
    forecast = (probability[:, :, None] * p).sum(1)
    anchor_forecast = (torch.tensor(anchor)[None, :, None] * p).sum(1)
    direct = ((forecast - y) ** 2).mean(1) + ((forecast - anchor_forecast) ** 2).mean(1)
    relative = regularized_objective(
        probability, torch.tensor(gram), torch.tensor(alignment), torch.tensor(anchor), 1.0
    )
    reference = ((torch.tensor(np.median(points, axis=1)) - y) ** 2).mean(1)
    torch.testing.assert_close(relative + reference, direct, rtol=1e-12, atol=1e-12)
    first = torch.autograd.grad(relative.sum(), logits, retain_graph=True)[0]
    second = torch.autograd.grad(direct.sum(), logits)[0]
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def test_calibration_requires_both_metrics_and_keeps_early_stop_control():
    reference = {"mae": 1.0, "mse": 2.0}
    tradeoff = {"strength": 0.0, "epoch": 25, "mae": 1.01, "mse": 1.0}
    assert choose_configuration([tradeoff], reference)["kind"] == "fixed"
    rows = [
        tradeoff,
        {"strength": 0.0, "epoch": 5, "mae": 0.9, "mse": 1.8},
        {"strength": 1.0, "epoch": 10, "mae": 0.8, "mse": 1.6},
    ]
    assert choose_configuration(rows, reference) == {"kind": "gate", "strength": 1.0, "epoch": 10}
    assert choose_configuration(rows, reference, early_only=True) == {
        "kind": "gate",
        "strength": 0.0,
        "epoch": 5,
    }


def test_training_snapshot_replays_on_unseen_rows_with_fit_only_statistics():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "family_id": ["f"] * 8,
            "dataset_id": ["d"] * 8,
            "origin_id": [str(i) for i in range(8)],
            "episode_id": [str(i) for i in range(8)],
        }
    )
    features = rng.normal(size=(8, 7, 97)).astype(np.float32)
    features[6:] += 100
    gram = np.broadcast_to(np.eye(7), (8, 7, 7)).copy()
    alignment = rng.normal(size=(8, 7))
    alignment[6:] = np.nan
    torch.set_num_threads(1)
    result = fit_snapshots(
        features, gram, alignment, frame, np.arange(6), np.ones(7) / 7, 1.0, 5101, (1,)
    )
    state = result["states"]["1"]
    np.testing.assert_array_equal(
        state["feature_mean"].numpy().ravel(),
        features[:6].astype(float).mean((0, 1)).astype(np.float32),
    )
    probability = probability_from_state(state, features[6:])
    assert probability.shape == (2, 7)
    np.testing.assert_allclose(probability.sum(1), 1)
    assert set(result["training_origins"]) == set(map(str, range(6)))
