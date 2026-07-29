from __future__ import annotations

from collections.abc import Mapping

import joblib
import numpy as np
import pytest

from tsfm_fais.routing.baselines import (
    BASELINE_SELECTOR_METHODS,
    DSelectOneSelector,
    RandomValidBlockSelector,
    _smooth_step_numpy,
    fit_baseline_selector,
)


def _toy_training() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    list[Mapping[str, object]],
]:
    candidate_ids = ("a", "b", "c")
    feature_names = (
        "signal",
        "start_ratio",
        "candidate_cost",
        "candidate_id::a",
        "candidate_id::b",
        "candidate_id::c",
    )
    group_candidates = (
        ("a", "b", "c"),
        ("a", "b"),
        ("a", "b", "c"),
        ("b", "c"),
        ("a", "b", "c"),
        ("a", "c"),
    )
    signals = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    rows: list[Mapping[str, object]] = []
    features: list[list[float]] = []
    losses: list[float] = []
    for group_index, (signal, available) in enumerate(zip(signals, group_candidates, strict=True)):
        for candidate_id in available:
            candidate_index = candidate_ids.index(candidate_id)
            one_hot = [0.0, 0.0, 0.0]
            one_hot[candidate_index] = 1.0
            start_ratio = float(group_index % 3) / 3.0
            features.append([signal, start_ratio, float(candidate_index + 1), *one_hot])
            if signal < 0:
                loss = {"a": 0.0, "b": 1.0, "c": 2.0}[candidate_id]
            else:
                loss = {"a": 2.0, "b": 0.0, "c": 1.0}[candidate_id]
            losses.append(loss)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "group_id": f"group-{group_index}",
                    "episode_id": f"episode-{group_index // 2}",
                    "forecaster_id": "mock",
                    "block_id": f"n0:d0:{group_index}-{group_index + 1}",
                    "prior_features": {
                        "signal": signal,
                        "start_ratio": start_ratio,
                        f"candidate_id::{candidate_id}": 1.0,
                    },
                }
            )
    return (
        np.asarray(features, dtype=float),
        np.asarray(losses, dtype=float),
        tuple(len(group) for group in group_candidates),
        feature_names,
        candidate_ids,
        rows,
    )


def _fit(method: str):
    features, losses, groups, feature_names, candidate_ids, rows = _toy_training()
    params_by_method: dict[str, dict[str, object]] = {
        "metaod": {
            "epochs": 3,
            "latent_dim": 2,
            "n_estimators": 8,
            "max_depth": 4,
        },
        "alors": {
            "epochs": 3,
            "latent_dim": 2,
            "n_estimators": 8,
            "max_depth": 4,
        },
        "dselect1": {"epochs": 3, "batch_size": 8, "torch_threads": 1},
        "neuralucb": {
            "hidden_size": 4,
            "retrain_interval": 2,
            "training_steps": 3,
        },
        "hybrid_lstm": {
            "epochs": 3,
            "hidden_size": 4,
            "batch_size": 8,
            "torch_threads": 1,
        },
        "random_valid_block": {},
    }
    return fit_baseline_selector(
        method,
        features,
        losses,
        groups,
        feature_names,
        candidate_ids,
        rows,
        params_by_method[method],
        seed=17,
    )


def test_baseline_method_registry_is_stable() -> None:
    assert BASELINE_SELECTOR_METHODS == (
        "metaod",
        "dselect1",
        "neuralucb",
        "alors",
        "hybrid_lstm",
        "random_valid_block",
    )


@pytest.mark.parametrize("method", BASELINE_SELECTOR_METHODS)
def test_all_selector_baselines_fit_predict_and_round_trip(method: str, tmp_path) -> None:
    if method in {"metaod", "dselect1", "neuralucb", "hybrid_lstm"}:
        pytest.importorskip("torch")
    features, _, _, _, _, rows = _toy_training()
    model = _fit(method)
    scores = model.predict(features)
    assert scores.shape == (len(features),)
    assert np.isfinite(scores).all()

    keys = [(str(row["block_id"]), str(row["candidate_id"])) for row in rows]
    contextual = model.predict_with_context(features, keys=keys, seed=29)
    assert contextual.shape == scores.shape
    assert np.isfinite(contextual).all()

    path = tmp_path / f"{method}.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        restored.predict_with_context(features, keys=keys, seed=29),
        contextual,
        rtol=0.0,
        atol=1e-12,
    )


def test_dselect_smooth_step_and_non_power_of_two_padding() -> None:
    pytest.importorskip("torch")
    np.testing.assert_allclose(
        _smooth_step_numpy(np.asarray([-1.0, 0.0, 1.0]), gamma=1.0),
        [0.0, 0.5, 1.0],
    )
    features, *_ = _toy_training()
    model = _fit("dselect1")
    assert isinstance(model, DSelectOneSelector)
    probabilities = model._probabilities(features)
    assert probabilities.shape == (len(features), 3)
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_random_valid_block_scores_are_deterministic_and_keyed() -> None:
    features, _, _, _, _, rows = _toy_training()
    model = _fit("random_valid_block")
    assert isinstance(model, RandomValidBlockSelector)
    keys = [(str(row["block_id"]), str(row["candidate_id"])) for row in rows]
    first = model.predict_with_context(features, keys=keys, seed=5)
    second = model.predict_with_context(features, keys=keys, seed=5)
    third = model.predict_with_context(features, keys=keys, seed=6)
    np.testing.assert_array_equal(first, second)
    assert np.any(first != third)
    assert np.all((first >= 0.0) & (first < 1.0))
    permutation = np.arange(len(features) - 1, -1, -1)
    permuted = model.predict_with_context(
        features[permutation],
        keys=[keys[index] for index in permutation],
        seed=5,
    )
    np.testing.assert_array_equal(permuted, first[permutation])
    np.testing.assert_array_equal(
        model.predict(features[permutation]), model.predict(features)[permutation]
    )


def test_hybrid_lstm_sequence_scores_do_not_depend_on_input_group_order() -> None:
    pytest.importorskip("torch")
    features, _, _, _, _, rows = _toy_training()
    model = _fit("hybrid_lstm")
    keys = [(str(row["block_id"]), str(row["candidate_id"])) for row in rows]
    expected = model.predict_with_context(features, keys=keys, seed=5)
    group_ids = tuple(dict.fromkeys(str(row["group_id"]) for row in rows))
    permutation = np.asarray(
        [
            row_index
            for group_id in reversed(group_ids)
            for row_index, row in enumerate(rows)
            if row["group_id"] == group_id
        ],
        dtype=int,
    )
    permuted = model.predict_with_context(
        features[permutation],
        keys=[keys[index] for index in permutation],
        seed=5,
    )
    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    np.testing.assert_allclose(restored, expected, rtol=0.0, atol=1e-12)


def test_neuralucb_replay_observes_one_arm_per_group() -> None:
    pytest.importorskip("torch")
    model = _fit("neuralucb")
    assert model.offline_replay is True
    assert model.replayed_group_count == 6
    assert model.observed_row_count == 6


def test_selector_training_rejects_duplicate_candidates_within_group() -> None:
    features, losses, groups, feature_names, candidate_ids, rows = _toy_training()
    broken = list(rows)
    broken[1] = {**broken[1], "candidate_id": "a"}
    with pytest.raises(ValueError, match="at most one row per candidate"):
        fit_baseline_selector(
            "alors",
            features,
            losses,
            groups,
            feature_names,
            candidate_ids,
            broken,
            {},
            seed=17,
        )


def test_selector_aliases_and_unknown_method() -> None:
    features, losses, groups, feature_names, candidate_ids, rows = _toy_training()
    alias = fit_baseline_selector(
        "random-valid-block",
        features,
        losses,
        groups,
        feature_names,
        candidate_ids,
        rows,
        {},
        seed=17,
    )
    assert isinstance(alias, RandomValidBlockSelector)
    with pytest.raises(ValueError, match="unknown baseline selector"):
        fit_baseline_selector(
            "missing",
            features,
            losses,
            groups,
            feature_names,
            candidate_ids,
            rows,
            {},
            seed=17,
        )
    with pytest.raises(ValueError, match="unsupported parameters for alors"):
        fit_baseline_selector(
            "alors",
            features,
            losses,
            groups,
            feature_names,
            candidate_ids,
            rows,
            {"epohs": 2},
            seed=17,
        )
