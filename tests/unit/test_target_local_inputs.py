import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_target_local_gate import direct_target_features  # noqa: E402
from r6_policy_inputs import decision_inputs  # noqa: E402
from target_local_inputs import split_joint_vectors, target_features  # noqa: E402

from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402


def fixture():
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
    return context, candidates, actions, np.ones(6), points, np.zeros(3), np.ones(3)


def test_target_local_keeps_joint_fallback_for_empty_non_target_channel():
    values = fixture()
    metadata = {
        "episode_id": "e",
        "origin_id": "o",
        "family_id": "f",
        "dataset_id": "d",
        "item_id": "i",
        "episode_index": 0,
        "split": "train",
        "model_id": "chronos2",
    }
    frame, joint = target_features(*values, backbone_joint=True, period=24, metadata=metadata)
    _, independent = target_features(*values, backbone_joint=False, period=24, metadata=metadata)
    names = sorted([*values[2], "guarded_direct"])
    column = FORECAST_FEATURES.index("static.fallback_target_fraction")
    np.testing.assert_array_equal(joint[:, names.index("guarded_direct"), column], [1, 1])
    np.testing.assert_array_equal(independent[:, names.index("guarded_direct"), column], [0, 0])
    assert frame.target_slot.tolist() == [0, 1]
    missing = FORECAST_FEATURES.index("static.target_missing_fraction")
    assert joint[0, 0, missing] > joint[1, 0, missing]
    sorted_points = values[4][[[*values[2], "guarded_direct"].index(name) for name in names]]
    direct = direct_target_features(*values[:4], sorted_points, *values[5:], 24, True)
    np.testing.assert_array_equal(joint, direct)


def test_independent_target_features_match_the_existing_interface_exactly():
    values = fixture()
    metadata = {
        "episode_id": "e",
        "origin_id": "o",
        "family_id": "f",
        "dataset_id": "d",
        "item_id": "i",
        "episode_index": 0,
        "split": "train",
        "model_id": "timesfm2p5",
    }
    _, features = target_features(*values, backbone_joint=False, period=24, metadata=metadata)
    existing = decision_inputs(*values, joint=False, period=24, metadata=metadata)
    np.testing.assert_array_equal(features, existing["gate_features"])


def test_joint_vector_split_preserves_each_target_and_candidate():
    values = np.arange(2 * 7 * 8).reshape(2, 7, 8)
    split = split_joint_vectors(values)
    for episode in range(2):
        for slot in (0, 1):
            np.testing.assert_array_equal(split[2 * episode + slot], values[episode, :, slot::2])
