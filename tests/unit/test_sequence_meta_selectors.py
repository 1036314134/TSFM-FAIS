from __future__ import annotations

import io
from itertools import permutations

import joblib
import numpy as np
import pytest

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.routing.meta_selectors import (
    ALORSSequenceSelector,
    MetaODSequenceSelector,
    _cofirank_gains,
    _cofirank_loss_and_gradient,
    _loss_gains,
    _smooth_ndcg_and_gradient,
)
from tsfm_fais.routing.sequence_features import (
    SEQUENCE_FEATURE_NAMES,
    SequenceFeatureExtractor,
    extract_sequence_features,
    sequence_meta_features,
)


def _incomplete_sequence() -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(24, dtype=float)
    values = np.column_stack(
        (
            np.sin(2.0 * np.pi * time / 6.0) + 0.05 * time,
            2.0 * np.sin(2.0 * np.pi * time / 6.0 + 0.4),
            np.cos(2.0 * np.pi * time / 4.0),
        )
    )
    mask = np.ones_like(values, dtype=bool)
    mask[3:6, 0] = False
    mask[:2, 1] = False
    mask[17:, 1] = False
    mask[9:13, 2] = False
    return values, mask


def test_sequence_features_have_stable_schema_and_ignore_hidden_values() -> None:
    values, mask = _incomplete_sequence()
    first = extract_sequence_features(values, mask, period=6)
    changed = values.copy()
    changed[~mask] = np.linspace(-1e9, 1e9, np.count_nonzero(~mask))
    second = extract_sequence_features(changed, mask, period=6)

    assert tuple(first) == SEQUENCE_FEATURE_NAMES
    assert first == second
    assert all(np.isfinite(tuple(first.values())))
    assert first["block_count_per_channel"] == pytest.approx(4.0 / 3.0)
    assert first["leading_block_fraction"] == pytest.approx(0.25)
    assert first["tail_block_fraction"] == pytest.approx(0.25)
    assert first["lag1_autocorrelation_valid_fraction"] > 0.0
    assert first["spectral_valid_fraction"] > 0.0
    assert first["cross_correlation_valid_fraction"] > 0.0


def test_sequence_batch_wrapper_and_vector_use_the_same_order() -> None:
    values, mask = _incomplete_sequence()
    batch = SeriesBatch(values=values[None, ...], observed_mask=mask[None, ...])
    features = sequence_meta_features(batch, period=6)
    vector = SequenceFeatureExtractor(period=6).vector(batch.values, batch.observed_mask)
    np.testing.assert_array_equal(
        vector,
        np.asarray([features[name] for name in SEQUENCE_FEATURE_NAMES]),
    )


def test_sequence_features_validate_observed_values_and_single_task_scope() -> None:
    values, mask = _incomplete_sequence()
    broken = values.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="observed values must be finite"):
        extract_sequence_features(broken, mask)
    with pytest.raises(ValueError, match="exactly one batch item"):
        extract_sequence_features(
            np.stack((values, values)),
            np.stack((mask, mask)),
        )


def _selector_training() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    signal = np.linspace(-2.0, 2.0, 32)
    contexts = np.column_stack((signal, np.square(signal), np.sin(signal)))
    losses = np.column_stack(
        (
            np.where(signal < 0.0, 0.0, 2.0),
            np.where(signal < 0.0, 2.0, 0.0),
            np.ones_like(signal),
        )
    )
    losses[::5, 2] = np.nan
    return contexts, losses, ("left", "right", "middle")


@pytest.mark.parametrize(
    "selector_class,extra_params",
    [
        (MetaODSequenceSelector, {}),
        (ALORSSequenceSelector, {"ndcg_cutoff": 2}),
    ],
)
def test_sequence_selectors_learn_one_candidate_per_task_and_round_trip(
    selector_class,
    extra_params,
) -> None:
    contexts, losses, candidate_ids = _selector_training()
    params = {
        "latent_dim": 2,
        "epochs": 180,
        "learning_rate": 0.03,
        "regularization": 1e-4,
        "n_estimators": 48,
        "max_depth": 6,
        **extra_params,
    }
    selector = selector_class(params=params, seed=19).fit(contexts, losses, candidate_ids)

    assert selector.score(contexts[2]).shape == (len(candidate_ids),)
    assert selector.rank(contexts[2])[0] == "left"
    assert selector.rank(contexts[-3])[0] == "right"
    assert np.isfinite(selector.score(contexts[2])).all()

    artifact = io.BytesIO()
    joblib.dump(selector, artifact)
    artifact.seek(0)
    restored = joblib.load(artifact)
    np.testing.assert_array_equal(restored.score(contexts[2]), selector.score(contexts[2]))
    assert restored.rank(contexts[-3]) == selector.rank(contexts[-3])


def test_selector_training_is_deterministic_for_a_fixed_seed() -> None:
    contexts, losses, candidate_ids = _selector_training()
    params = {
        "latent_dim": 2,
        "epochs": 40,
        "n_estimators": 16,
        "max_depth": 4,
    }
    first = ALORSSequenceSelector().fit(contexts, losses, candidate_ids, params, seed=7)
    second = ALORSSequenceSelector().fit(contexts, losses, candidate_ids, params, seed=7)
    np.testing.assert_array_equal(first.candidate_factors, second.candidate_factors)
    np.testing.assert_array_equal(first.score(contexts[0]), second.score(contexts[0]))
    assert first.observed_pair_count == second.observed_pair_count > 0


def test_metaod_smooth_dcg_gradient_matches_finite_difference() -> None:
    scores = np.asarray([[0.4, -0.2, 0.1]], dtype=float)
    losses = np.asarray([[0.0, 2.0, 1.0]], dtype=float)
    objective, gradient, informative_rows = _smooth_ndcg_and_gradient(
        scores,
        losses,
        temperature=0.4,
    )
    assert informative_rows == 1
    assert objective > 0.0
    epsilon = 1e-6
    numerical = np.empty_like(scores)
    for candidate in range(scores.shape[1]):
        upper = scores.copy()
        lower = scores.copy()
        upper[0, candidate] += epsilon
        lower[0, candidate] -= epsilon
        upper_value = _smooth_ndcg_and_gradient(upper, losses, 0.4)[0]
        lower_value = _smooth_ndcg_and_gradient(lower, losses, 0.4)[0]
        numerical[0, candidate] = (upper_value - lower_value) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)


def test_metaod_uses_reference_gain_and_self_comparison_offset() -> None:
    losses = np.asarray([0.0, 1.0, 2.0], dtype=float)
    gains = _loss_gains(losses)
    assert gains is not None
    np.testing.assert_allclose(
        gains,
        np.power(10.0, 1.0 / (1.0 + losses)) - 1.0,
        rtol=0.0,
        atol=1e-12,
    )

    scores = np.zeros((1, 3), dtype=float)
    objective, _, informative_rows = _smooth_ndcg_and_gradient(
        scores,
        losses[None, :],
        temperature=1.0,
    )
    # beta_j = 1.5 + two off-diagonal sigmoid(0) terms = 2.5.
    expected = float(np.sum(gains) / np.log2(2.5))
    assert informative_rows == 1
    assert objective == pytest.approx(expected)
    duplicated = _smooth_ndcg_and_gradient(
        np.vstack((scores, scores)),
        np.vstack((losses, losses)),
        1.0,
    )[0]
    assert duplicated == pytest.approx(2.0 * expected)


def test_cofirank_structured_ndcg_upper_bound_and_subgradient() -> None:
    scores = np.asarray([[-0.4, 0.8, 0.2]], dtype=float)
    losses = np.asarray([[0.0, 2.0, 1.0]], dtype=float)
    objective, gradient, informative_rows = _cofirank_loss_and_gradient(
        scores,
        losses,
        ndcg_cutoff=2,
    )
    assert informative_rows == 1
    assert objective > 0.0

    gains = _cofirank_gains(losses[0])
    assert gains is not None
    predicted = np.argsort(-scores[0], kind="stable")
    discounts = np.asarray([1.0, 1.0 / np.log2(3.0), 0.0])
    ideal = np.argsort(-gains, kind="stable")
    ideal_dcg = float(np.dot(discounts, gains[ideal]))
    predicted_regret = 1.0 - float(np.dot(discounts, gains[predicted]) / ideal_dcg)
    assert objective >= predicted_regret - 1e-12

    position_weights = np.power(np.arange(1, 4, dtype=float), -0.25)
    enumerated = []
    for ordering in permutations(range(3)):
        permutation = np.asarray(ordering, dtype=int)
        regret = 1.0 - float(np.dot(discounts, gains[permutation]) / ideal_dcg)
        margin = float(
            np.dot(position_weights, scores[0, permutation])
            - np.dot(position_weights, scores[0, ideal])
        )
        enumerated.append(regret + margin)
    assert objective == pytest.approx(max(enumerated))

    epsilon = 1e-6
    numerical = np.empty_like(scores)
    for candidate in range(scores.shape[1]):
        upper = scores.copy()
        lower = scores.copy()
        upper[0, candidate] += epsilon
        lower[0, candidate] -= epsilon
        upper_value = _cofirank_loss_and_gradient(upper, losses, 2)[0]
        lower_value = _cofirank_loss_and_gradient(lower, losses, 2)[0]
        numerical[0, candidate] = (upper_value - lower_value) / (2.0 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("selector_class", [MetaODSequenceSelector, ALORSSequenceSelector])
def test_selector_score_accepts_the_complete_feature_dictionary(selector_class) -> None:
    rng = np.random.default_rng(23)
    contexts = rng.normal(size=(8, len(SEQUENCE_FEATURE_NAMES)))
    losses = np.column_stack((np.arange(8, dtype=float), np.arange(8, 0, -1, dtype=float)))
    selector = selector_class().fit(
        contexts,
        losses,
        ("a", "b"),
        {"latent_dim": 2, "epochs": 12, "n_estimators": 8},
        seed=3,
    )
    feature_dict = {
        name: float(contexts[0, index])
        for index, name in reversed(tuple(enumerate(SEQUENCE_FEATURE_NAMES)))
    }
    np.testing.assert_allclose(
        selector.score(feature_dict),
        selector.score(contexts[0]),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        selector.score({**feature_dict, "unexpected": 1.0})


@pytest.mark.parametrize("selector_class", [MetaODSequenceSelector, ALORSSequenceSelector])
def test_selector_rejects_candidates_without_observed_training_loss(selector_class) -> None:
    contexts = np.arange(12, dtype=float).reshape(6, 2)
    losses = np.column_stack((np.zeros(6), np.ones(6), np.full(6, np.nan)))
    with pytest.raises(ValueError, match="no observed training loss"):
        selector_class().fit(contexts, losses, ("a", "b", "never"), {}, seed=1)
