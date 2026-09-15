import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from r6_policy_inputs import decision_inputs, restore_target_vectors

from tsfm_fais.routing.followup_portfolio import portfolio_feature_frame


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("horizon", [96, 192])
def test_horizons_keep_the_same_decision_scope_and_96_step_features(joint, horizon):
    actions = ["locf", "linear_interp", "seasonal_lag", "knn_multivariate", "saits", "timemixerpp"]
    rng = np.random.default_rng(5001)
    clean = rng.normal(size=(96, 4))
    context = clean.copy()
    context[10:20, 0] = np.nan
    context[30:40, 2] = np.nan
    candidates = np.repeat(clean[None], 6, axis=0)
    for index in range(6):
        candidates[index][~np.isfinite(context)] += index / 10
    points = rng.normal(size=(7, horizon, 2))
    metadata = {
        "episode_id": "test",
        "model_id": "chronos2" if joint else "timesfm2p5",
        "episode_index": 0,
        "origin_id": "origin",
        "family_id": "family",
        "dataset_id": "dataset",
        "item_id": "item",
        "split": "confirmation",
    }
    result = decision_inputs(
        context,
        candidates,
        actions,
        np.ones(6),
        points,
        np.zeros(4),
        np.ones(4),
        joint=joint,
        period=24,
        metadata=metadata,
    )
    assert result["gate_features"].shape == (1 if joint else 2, 7, 33)
    if horizon == 96:
        old, vectors, decisions, names = portfolio_feature_frame(
            result["individual"],
            points,
            [*actions, "guarded_direct"],
            candidates[0, -1, :2],
            joint=joint,
        )
        pd.testing.assert_frame_equal(result["portfolios"], old)
        np.testing.assert_array_equal(result["triple_vectors"], vectors)
        pd.testing.assert_frame_equal(result["decisions"], decisions)
        assert result["triple_names"] == names
    restored = restore_target_vectors(
        result["vectors"][:, 0], result["decisions"], horizon, joint=joint
    )
    np.testing.assert_array_equal(restored, points[6])  # guarded_direct is first alphabetically
