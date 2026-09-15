import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_motm_pool import direct_features  # noqa: E402
from pool_gate_inputs import motm_coverage, pool_inputs  # noqa: E402
from pool_gate_model import fit_pool_gate, pool_probability  # noqa: E402
from r6_policy_inputs import decision_inputs  # noqa: E402
from train_metric_source_gates import fit_metric_gate  # noqa: E402

from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
)


def inputs():
    actions = ["locf", "linear_interp", "seasonal_lag", "knn_multivariate", "saits", "timemixerpp"]
    context = np.column_stack([np.arange(96) / 96, np.arange(96) / 48, np.zeros(96)])
    context[:12, 0] = np.nan
    context[:, 2] = np.nan
    candidates = np.repeat(np.nan_to_num(context)[None], 6, axis=0)
    points = np.stack(
        [
            np.column_stack([np.linspace(0, 1, 96) + i / 10, np.linspace(1, 2, 96) - i / 10])
            for i in range(7)
        ]
    )
    meta = {
        "episode_id": "e",
        "origin_id": "o",
        "family_id": "f",
        "dataset_id": "d",
        "item_id": "i",
        "episode_index": 0,
        "split": "train",
        "model_id": "test",
    }
    return context, candidates, actions, np.ones(6), points, np.zeros(3), np.ones(3), meta


def test_seven_candidate_interface_matches_both_backbone_scopes():
    *values, metadata = inputs()
    for joint in (True, False):
        _, features, vectors, _ = pool_inputs(*values, joint=joint, period=24, metadata=metadata)
        original = decision_inputs(*values, joint=joint, period=24, metadata=metadata)
        np.testing.assert_array_equal(features, original["gate_features"])
        np.testing.assert_array_equal(vectors, original["vectors"])


def test_eighth_candidate_keeps_order_coverage_and_independent_features():
    context, candidates, actions, coverage, points, mean, scale, metadata = inputs()
    completion = candidates[0].copy()
    completion[:12, 0] = 0.2
    extra = np.column_stack([np.linspace(2, 3, 96), np.linspace(0, 1, 96)])
    extended_actions = [*actions, "motm_reference"]
    extended_candidates = np.concatenate([candidates, completion[None]])
    extended_points = np.concatenate([points[:-1], extra[None], points[-1:]])
    extended_coverage = np.r_[coverage, motm_coverage(context, [2])]
    np.testing.assert_allclose(extended_coverage[-1], 12 / 108)
    for joint in (True, False):
        _, features, vectors, names = pool_inputs(
            context,
            extended_candidates,
            extended_actions,
            extended_coverage,
            extended_points,
            mean,
            scale,
            joint=joint,
            period=24,
            metadata=metadata,
        )
        sorted_points = extended_points[
            [[*extended_actions, "guarded_direct"].index(name) for name in names]
        ]
        direct = direct_features(
            context,
            extended_candidates,
            extended_actions,
            extended_coverage,
            sorted_points,
            mean,
            scale,
            24,
            joint,
        )
        np.testing.assert_array_equal(features, direct)
        assert len(names) == 8
        expected = extra.reshape(1, -1) if joint else extra.T
        np.testing.assert_array_equal(vectors[:, names.index("motm_reference")], expected)


def test_generalized_fitting_exactly_matches_original_and_supports_eight():
    torch.set_num_threads(1)
    rng = np.random.default_rng(91)
    n = 6
    frame = pd.DataFrame(
        {
            "family_id": ["f"] * n,
            "dataset_id": ["d"] * n,
            "origin_id": list(map(str, range(n))),
            "episode_id": list(map(str, range(n))),
        }
    )
    features = rng.normal(size=(n, 7, 97)).astype(np.float32)
    points = rng.normal(size=(n, 7, 9))
    target = rng.normal(size=(n, 9))
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, target)["raw_projection"]
    indices = np.arange(n)
    original = fit_metric_gate(
        frame, features, points, target, gram, alignment, indices, 5101, "joint"
    )
    current = fit_pool_gate(
        frame, features, points, target, gram, alignment, indices, 5101, "joint"
    )
    assert current["history"] == original["history"]
    for name in original["state_dict"]:
        torch.testing.assert_close(
            current["state_dict"][name], original["state_dict"][name], rtol=0, atol=0
        )
    features = np.concatenate([features, features[:, :1]], axis=1)
    points = np.concatenate([points, points[:, :1]], axis=1)
    _, _, _, gram = forecast_geometry(points)
    alignment = projection_targets(points, target)["raw_projection"]
    extended = fit_pool_gate(
        frame, features, points, target, gram, alignment, indices, 5101, "joint"
    )
    probability = pool_probability(extended["state_dict"], features)
    assert probability.shape == (n, 8)
    np.testing.assert_allclose(probability.sum(1), 1)
    assert (
        sum(
            value.numel()
            for name, value in extended["state_dict"].items()
            if name not in ("feature_mean", "feature_scale")
        )
        == 2121
    )
