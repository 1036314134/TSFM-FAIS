from __future__ import annotations

import joblib
import numpy as np
import pytest

from tsfm_fais.routing.neural_sequence_selectors import (
    DSelectOneSequenceSelector,
    NeuralUCBSequenceSelector,
    _Standardizer,
    smooth_step,
)


def test_smooth_step_matches_dselect_definition() -> None:
    np.testing.assert_allclose(
        smooth_step(np.asarray([-1.0, 0.0, 1.0]), gamma=1.0),
        [0.0, 0.5, 1.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_dselect_non_power_of_two_scores_preserve_reachable_mass() -> None:
    selector = DSelectOneSequenceSelector(
        candidate_ids=("a", "b", "c"),
        standardizer=_Standardizer(
            mean=np.zeros(1, dtype=float),
            scale=np.ones(1, dtype=float),
        ),
        weight=np.zeros((2, 1), dtype=float),
        bias=np.zeros(2, dtype=float),
        gamma=1.0,
    )

    raw = selector.raw_score(np.asarray([0.0]))
    scores = selector.score(np.asarray([0.0]))

    np.testing.assert_allclose(raw, [0.25, 0.25, 0.25], rtol=0.0, atol=1e-12)
    assert raw.sum() == pytest.approx(0.75)
    np.testing.assert_allclose(scores, raw, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        selector.weights(np.asarray([0.0])),
        scores,
        rtol=0.0,
        atol=1e-12,
    )


def test_dselect_fits_one_gate_per_sequence_and_round_trips(tmp_path) -> None:
    pytest.importorskip("torch")
    contexts = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    expert_outputs = tuple(
        np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float) for _ in range(len(contexts))
    )
    targets = (
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([2.0, 2.0]),
        np.asarray([2.0, 2.0]),
    )
    selector = DSelectOneSequenceSelector.fit(
        contexts,
        expert_outputs,
        targets,
        ("left", "right"),
        params={
            "epochs": 150,
            "batch_size": 4,
            "learning_rate": 0.05,
            "entropy_weight": 0.01,
            "torch_threads": 1,
        },
        seed=13,
    )

    assert selector.select(np.asarray([-2.0])) == "left"
    assert selector.select(np.asarray([2.0])) == "right"
    np.testing.assert_allclose(
        selector.score(np.asarray([-2.0])).sum(),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )

    path = tmp_path / "dselect.joblib"
    joblib.dump(selector, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        restored.score_many(contexts),
        selector.score_many(contexts),
        rtol=0.0,
        atol=1e-12,
    )


def test_dselect_fits_mixed_output_objective_from_sequence_rows() -> None:
    pytest.importorskip("torch")
    rows = []
    for episode, signal, target in (("left", -1.0, 1.0), ("right", 1.0, 2.0)):
        for candidate_id, output in (("a", 1.0), ("b", 2.0)):
            row = {
                "label_scope": "whole_series",
                "group_id": f"imputation::{episode}::__sequence__",
                "candidate_id": candidate_id,
                "prior_features": {"signal": signal} if candidate_id == "a" else {},
                "native_valid": True,
                "dselect_expert_values": [output, output],
            }
            if candidate_id == "a":
                row["dselect_target_values"] = [target, target]
            rows.append(row)

    selector = DSelectOneSequenceSelector.fit_from_rows(
        rows,
        ("signal",),
        ("a", "b"),
        params={"epochs": 120, "batch_size": 2, "learning_rate": 0.05},
        seed=29,
    )

    assert selector.select(np.asarray([-1.0])) == "a"
    assert selector.select(np.asarray([1.0])) == "b"


def test_dselect_rows_support_heterogeneous_expert_sets(monkeypatch) -> None:
    rows = [
        {
            "label_scope": "whole_series",
            "group_id": "g1",
            "candidate_id": "a",
            "prior_features": {"signal": -1.0},
            "native_valid": True,
            "dselect_expert_values": [1.0, 1.0],
        },
        {
            "label_scope": "whole_series",
            "group_id": "g1",
            "candidate_id": "b",
            "prior_features": {},
            "native_valid": True,
            "dselect_expert_values": [2.0, 2.0],
            "dselect_target_values": [1.0, 1.0],
        },
        {
            "label_scope": "whole_series",
            "group_id": "g2",
            "candidate_id": "b",
            "prior_features": {"signal": 1.0},
            "native_valid": True,
            "dselect_expert_values": [2.0, 2.0],
        },
        {
            "label_scope": "whole_series",
            "group_id": "g2",
            "candidate_id": "c",
            "prior_features": {},
            "native_valid": True,
            "dselect_expert_values": [3.0, 3.0],
            "dselect_target_values": [3.0, 3.0],
        },
    ]
    captured = {}
    sentinel = object()

    def fake_fit(
        cls,
        features,
        expert_outputs,
        targets,
        candidate_ids,
        native_valid=None,
        params=None,
        seed=20260710,
    ):
        del cls, params, seed
        captured["features"] = features
        captured["outputs"] = expert_outputs
        captured["targets"] = targets
        captured["candidate_ids"] = candidate_ids
        captured["validity"] = native_valid
        return sentinel

    monkeypatch.setattr(DSelectOneSequenceSelector, "fit", classmethod(fake_fit))

    fitted = DSelectOneSequenceSelector.fit_from_rows(
        rows,
        ("signal",),
        ("a", "b", "c"),
    )

    assert fitted is sentinel
    assert captured["candidate_ids"] == ("a", "b", "c")
    np.testing.assert_array_equal(captured["features"], [[-1.0], [1.0]])
    np.testing.assert_array_equal(
        captured["validity"],
        [[True, True, False], [False, True, True]],
    )
    np.testing.assert_array_equal(captured["outputs"][0][2], [0.0, 0.0])
    np.testing.assert_array_equal(captured["outputs"][1][0], [0.0, 0.0])
    np.testing.assert_array_equal(captured["targets"], [[1.0, 1.0], [3.0, 3.0]])


def _neural_params() -> dict[str, float | int]:
    return {
        "hidden_size": 4,
        "ridge": 0.5,
        "nu": 0.25,
        "learning_rate": 0.02,
        "training_steps": 5,
        "retrain_interval": 1,
        "regularization": 0.1,
    }


def test_neuralucb_symmetric_initialization_has_zero_prediction() -> None:
    selector = NeuralUCBSequenceSelector.initialize(
        2,
        ("a", "b"),
        _neural_params(),
        seed=7,
    )
    features = np.asarray([1.0, 0.0], dtype=float)
    contexts = selector._action_contexts(features)
    means, _ = selector._means_and_gradients(contexts)
    half_hidden = selector.hidden_size // 2
    split = selector.disjoint_input_dim

    assert contexts.shape == (2, 8)
    np.testing.assert_allclose(
        contexts[0],
        [2**-0.5, 0.0, 0.0, 0.0, 2**-0.5, 0.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(selector.weight1[:half_hidden, split:], 0.0, atol=0.0)
    np.testing.assert_allclose(selector.weight1[half_hidden:, :split], 0.0, atol=0.0)
    np.testing.assert_array_equal(
        selector.weight1[:half_hidden, :split],
        selector.weight1[half_hidden:, split:],
    )
    np.testing.assert_array_equal(
        selector.weight2[:half_hidden],
        -selector.weight2[half_hidden:],
    )
    np.testing.assert_allclose(means, 0.0, rtol=0.0, atol=1e-12)


def test_neuralucb_score_and_update_match_full_matrix_algorithm() -> None:
    selector = NeuralUCBSequenceSelector.initialize(
        2,
        ("a", "b"),
        _neural_params(),
        seed=7,
    )
    features = np.asarray([1.0, 0.0], dtype=float)
    contexts = selector._action_contexts(features)
    means, gradients = selector._means_and_gradients(contexts)
    initial_z = selector.ridge * np.eye(selector.parameter_count)
    initial_expected = means + selector.nu * np.sqrt(
        np.einsum(
            "ni,ni->n",
            gradients,
            np.linalg.solve(initial_z, gradients.T).T,
        )
        / selector.hidden_size
    )
    np.testing.assert_allclose(selector.score(features), initial_expected, rtol=0.0, atol=1e-12)

    selected_id = selector.select(features, ("a", "b"))
    selected = selector.candidate_ids.index(selected_id)
    selected_gradient = gradients[selected].copy()
    selector.observe(features, selected_id, 0.75)

    expected_z = initial_z + np.outer(selected_gradient, selected_gradient) / selector.hidden_size
    np.testing.assert_allclose(selector._precision_matrix(), expected_z, rtol=0.0, atol=1e-12)
    updated_means, updated_gradients = selector._means_and_gradients(contexts)
    updated_expected = updated_means + selector.nu * np.sqrt(
        np.einsum(
            "ni,ni->n",
            updated_gradients,
            np.linalg.solve(expected_z, updated_gradients.T).T,
        )
        / selector.hidden_size
    )
    np.testing.assert_allclose(selector.score(features), updated_expected, rtol=0.0, atol=1e-12)

    second_features = np.asarray([0.5, 1.0], dtype=float)
    second_contexts = selector._action_contexts(second_features)
    _, second_gradients = selector._means_and_gradients(second_contexts)
    second_id = selector.select(second_features, ("a", "b"))
    second = selector.candidate_ids.index(second_id)
    selector.observe(second_features, second_id, 0.25)
    expected_z += (
        np.outer(second_gradients[second], second_gradients[second]) / selector.hidden_size
    )
    np.testing.assert_allclose(selector._precision_matrix(), expected_z, rtol=0.0, atol=1e-12)
    final_means, final_gradients = selector._means_and_gradients(second_contexts)
    final_expected = final_means + selector.nu * np.sqrt(
        np.einsum(
            "ni,ni->n",
            final_gradients,
            np.linalg.solve(expected_z, final_gradients.T).T,
        )
        / selector.hidden_size
    )
    np.testing.assert_allclose(
        selector.score(second_features),
        final_expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_neuralucb_training_matches_paper_regularization_scaling() -> None:
    selector = NeuralUCBSequenceSelector.initialize(
        1,
        ("a",),
        {
            "hidden_size": 2,
            "ridge": 0.5,
            "nu": 0.25,
            "learning_rate": 0.05,
            "training_steps": 2,
            "retrain_interval": 10,
            "regularization": 0.4,
            "gradient_clip": 1e6,
        },
        seed=7,
    )
    initial_weight1 = np.asarray([[0.4, 0.1], [0.7, -0.2]], dtype=float)
    initial_weight2 = np.asarray([0.3, -0.2], dtype=float)
    selector.initial_weight1 = initial_weight1.copy()
    selector.initial_weight2 = initial_weight2.copy()
    contexts = np.stack(
        [
            selector._action_contexts(np.asarray([value], dtype=float))[0]
            for value in (1.0, -2.0, 0.5)
        ],
        axis=0,
    )
    rewards = np.asarray([0.2, 0.8, 0.4], dtype=float)
    selector.observed_contexts = [row.copy() for row in contexts]
    selector.observed_rewards = rewards.tolist()
    selector.round_count = len(contexts)

    weight1 = initial_weight1.copy()
    weight2 = initial_weight2.copy()
    count = float(len(contexts))
    regularization_scale = selector.hidden_size * selector.regularization / count
    width_scale = np.sqrt(selector.hidden_size)
    for _ in range(selector.training_steps):
        preactivation = contexts @ weight1.T
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        error = width_scale * (hidden @ weight2) - rewards
        hidden_gradient = width_scale * error[:, None] * weight2[None, :] * active / count
        gradient_weight2 = width_scale * hidden.T @ error / count
        gradient_weight1 = hidden_gradient.T @ contexts
        gradient_weight1 += regularization_scale * (weight1 - initial_weight1)
        gradient_weight2 += regularization_scale * (weight2 - initial_weight2)
        weight1 -= selector.learning_rate * gradient_weight1
        weight2 -= selector.learning_rate * gradient_weight2

    selector._train_from_history()

    np.testing.assert_allclose(selector.weight1, weight1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(selector.weight2, weight2, rtol=0.0, atol=1e-12)


def test_neuralucb_observe_updates_selected_action_only() -> None:
    selector = NeuralUCBSequenceSelector.initialize(
        2,
        ("a", "b"),
        _neural_params(),
        seed=11,
    )
    features = np.asarray([1.0, -1.0], dtype=float)
    contexts = selector._action_contexts(features)
    before = selector._precision_matrix()
    selected_id = selector.select(features, ("a", "b"))
    selected = selector.candidate_ids.index(selected_id)
    _, gradients = selector._means_and_gradients(contexts)
    selected_gradient = gradients[selected].copy()

    selector.observe(features, selected_id, 0.75)

    np.testing.assert_allclose(
        selector._precision_matrix(),
        before + np.outer(selected_gradient, selected_gradient) / selector.hidden_size,
        rtol=0.0,
        atol=1e-12,
    )
    assert selector.selected_actions == [selected]
    assert selector.observed_rewards == [0.75]
    np.testing.assert_array_equal(selector.observed_contexts[0], contexts[selected])
    assert selector.select(features, ("b",)) == "b"


def test_neuralucb_offline_replay_never_uses_unselected_loss() -> None:
    features = np.asarray([[1.0, 0.0]], dtype=float)
    probe = NeuralUCBSequenceSelector.initialize(
        2,
        ("a", "b"),
        _neural_params(),
        seed=17,
    )
    selected_id = probe.select(features[0], ("a", "b"))
    selected = probe.candidate_ids.index(selected_id)
    other = 1 - selected
    first_losses = np.full((1, 2), 0.5, dtype=float)
    second_losses = first_losses.copy()
    second_losses[0, other] = 0.9

    first = NeuralUCBSequenceSelector.fit(
        features,
        first_losses,
        ("a", "b"),
        _neural_params(),
        seed=17,
    )
    second = NeuralUCBSequenceSelector.fit(
        features,
        second_losses,
        ("a", "b"),
        _neural_params(),
        seed=17,
    )

    assert first.selected_actions == second.selected_actions == [selected]
    assert first.observed_rewards == second.observed_rewards == [2.0 / 3.0]
    np.testing.assert_array_equal(first._precision_matrix(), second._precision_matrix())
    np.testing.assert_array_equal(first.weight1, second.weight1)
    np.testing.assert_array_equal(first.weight2, second.weight2)


def test_neuralucb_offline_replay_selects_only_native_valid_actions() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    losses = np.asarray([[np.nan, 0.25], [0.5, np.nan]], dtype=float)
    validity = np.asarray([[False, True], [True, False]], dtype=bool)

    selector = NeuralUCBSequenceSelector.offline_replay(
        features,
        losses,
        ("a", "b"),
        _neural_params(),
        seed=19,
        native_valid=validity,
    )

    assert selector.selected_actions == [1, 0]
    assert selector.observed_rewards == [0.8, 2.0 / 3.0]


def test_neuralucb_online_contract_and_joblib_round_trip(tmp_path) -> None:
    selector = NeuralUCBSequenceSelector.initialize(
        2,
        ("a", "b"),
        _neural_params(),
        seed=23,
    )
    features = np.asarray([0.2, 0.8], dtype=float)

    with pytest.raises(ValueError, match="cannot be empty"):
        selector.select(features, ())
    with pytest.raises(ValueError, match="unknown candidates"):
        selector.select(features, ("missing",))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        selector.observe(features, "a", 1.1)
    selected_id = selector.select(features, ("a", "b"))
    selector.observe(features, selected_id, 0.4)

    path = tmp_path / "neuralucb.joblib"
    joblib.dump(selector, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        restored.score(features),
        selector.score(features),
        rtol=0.0,
        atol=1e-12,
    )
    assert restored.round_count == selector.round_count == 1
    assert restored.observed_rewards == selector.observed_rewards == [0.4]
    np.testing.assert_array_equal(restored._precision_matrix(), selector._precision_matrix())

    next_features = np.asarray([0.8, -0.1], dtype=float)
    next_selected = selector.select(next_features, ("a", "b"))
    assert restored.select(next_features, ("a", "b")) == next_selected
    selector.observe(next_features, next_selected, 0.6)
    restored.observe(next_features, next_selected, 0.6)
    np.testing.assert_allclose(
        restored.score(next_features),
        selector.score(next_features),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        restored._precision_matrix(),
        selector._precision_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_neuralucb_fit_replays_rounds_in_order() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
        ],
        dtype=float,
    )
    losses = np.asarray(
        [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
        dtype=float,
    )

    selector = NeuralUCBSequenceSelector.offline_replay(
        features,
        losses,
        ("a", "b"),
        _neural_params(),
        seed=29,
    )

    assert selector.round_count == 3
    assert selector.observed_count == 3
    assert len(selector.selected_actions) == 3
    for round_index, action_index in enumerate(selector.selected_actions):
        np.testing.assert_array_equal(
            selector.observed_contexts[round_index],
            selector._action_contexts(features[round_index])[action_index],
        )
        assert selector.observed_rewards[round_index] == pytest.approx(
            1.0 / (1.0 + losses[round_index, action_index])
        )
