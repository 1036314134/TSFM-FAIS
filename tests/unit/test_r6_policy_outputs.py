import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluate_r6_policies import restore, triple_vectors


@pytest.mark.parametrize("horizon", [96, 192])
def test_global_target_restore_handles_noncontiguous_decision_order(horizon):
    decisions = pd.DataFrame({"episode_index": [1, 0, 1, 0], "target_slot": [0, 1, 1, 0]})
    values = np.stack(
        [
            100 * row.episode_index + 10 * row.target_slot + np.arange(horizon)
            for row in decisions.itertuples(index=False)
        ]
    )
    result = restore(values, decisions, 2, horizon, False)
    for case in range(2):
        for target in range(2):
            np.testing.assert_array_equal(
                result[case, :, target], 100 * case + 10 * target + np.arange(horizon)
            )
    joint = pd.DataFrame({"episode_index": [1, 0], "target_slot": [-1, -1]})
    values = np.stack([result[1].reshape(-1), result[0].reshape(-1)])
    np.testing.assert_array_equal(restore(values, joint, 2, horizon, True), result)


def test_portfolio_names_and_original_rank_lists_give_the_same_prediction():
    actions = ["a", "b", "c", "d", "e", "f", "g"]
    vectors = np.arange(2 * 7 * 9).reshape(2, 7, 9).astype(float)
    decisions = pd.DataFrame({"episode_id": ["one", "two"]})
    named = {"one": "median:a+c+g", "two": "median:b+d+e"}
    ranked = {"one": ["g", "c", "a"], "two": ["e", "b", "d"]}
    expected = np.stack([vectors[0, 2], vectors[1, 3]])
    np.testing.assert_array_equal(
        triple_vectors(vectors, decisions, named, actions, named_portfolio=True), expected
    )
    np.testing.assert_array_equal(
        triple_vectors(vectors, decisions, ranked, actions, named_portfolio=False), expected
    )


def test_noncontiguous_candidate_frames_use_one_exact_inference_layout():
    import torch
    from audit_shared_forecast_gate import replay_network
    from r6_policy_inputs import pack_gate_features
    from train_shared_forecast_gate import predict_weights

    from tsfm_fais.routing.forecast_gate import SharedForecastGate

    torch.set_num_threads(1)
    torch.manual_seed(5101)
    model = SharedForecastGate().eval()
    torch.nn.init.normal_(model.score[-1].weight, std=0.1)
    values = np.random.default_rng(81).normal(size=(1025, 7, 33)).astype(np.float32)
    strided = np.concatenate([np.asfortranarray(row)[None] for row in values])
    assert not strided.flags.c_contiguous
    packed = pack_gate_features(strided)
    assert packed.flags.c_contiguous
    np.testing.assert_array_equal(packed, values)
    np.testing.assert_array_equal(
        predict_weights(model, packed), replay_network(model.state_dict(), packed)
    )
