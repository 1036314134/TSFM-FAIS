import numpy as np
import pandas as pd
import pytest

from tsfm_fais.routing.aligned_portfolio import (
    AlignedPortfolioRegressor,
    build_option_features,
    choose_option,
    option_catalog,
    option_scores,
    option_vectors,
)
from tsfm_fais.routing.forecast_projection import projection_targets
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES
from tsfm_fais.routing.preforecast import STATIC_FEATURES

ACTIONS = (
    "locf",
    "linear_interp",
    "seasonal_lag",
    "knn_multivariate",
    "saits",
    "timemixerpp",
    "guarded_direct",
)


def test_exact_teacher_scores_recover_the_best_actual_option():
    rng = np.random.default_rng(17)
    candidates, teacher = rng.normal(size=(3, 7, 6)), rng.normal(size=(3, 6))
    _, members = option_catalog(ACTIONS)
    vectors = option_vectors(candidates, members)
    anchor = np.median(candidates, axis=1)
    np.testing.assert_array_equal(vectors[:, -1], anchor)
    # The complete symmetric catalog also retains this coordinate median.
    np.testing.assert_array_equal(np.median(vectors, axis=1), anchor)
    labels = projection_targets(vectors, teacher, anchor=anchor)
    exact = (
        np.square(vectors - teacher[:, None]).mean(axis=2)
        - np.square(anchor - teacher).mean(axis=1)[:, None]
    )
    for kind in ("unit_projection", "direct_risk"):
        scores = option_scores(vectors, labels[kind], target_kind=kind)
        np.testing.assert_allclose(scores, exact, atol=1e-12)
        chosen = choose_option(scores, "full")
        np.testing.assert_allclose(exact[np.arange(3), chosen], exact.min(axis=1), atol=1e-12)


def test_explicit_reference_is_not_replaced_by_an_option_median():
    vectors = np.array([[[0.0], [1.0], [2.0]]])
    teacher, reference = np.array([[4.0]]), np.array([[0.0]])
    labels = projection_targets(vectors, teacher, anchor=reference)
    np.testing.assert_allclose(labels["direct_risk"], [[0.0, -7.0, -12.0]])


def test_zero_direction_and_menu_boundaries():
    _, members = option_catalog(ACTIONS)
    vectors = option_vectors(np.ones((2, 7, 4)), members)
    scores = option_scores(vectors, np.full((2, 43), -123.0), target_kind="direct_risk")
    np.testing.assert_array_equal(scores, 0.0)
    np.testing.assert_array_equal(choose_option(scores, "full"), [42, 42])
    np.testing.assert_array_equal(choose_option(scores, "single"), [0, 0])
    np.testing.assert_array_equal(choose_option(scores, "triple"), [7, 7])
    np.testing.assert_array_equal(choose_option(scores, "mixed"), [0, 0])


def test_option_features_have_an_explicit_outcome_free_input_boundary():
    rng = np.random.default_rng(29)
    static = rng.normal(size=(2, 7, len(STATIC_FEATURES)))
    features, _, names = build_option_features(
        rng.normal(size=(2, 7, 6)), static, np.zeros((2, 2)), ACTIONS, horizon=3, targets=2
    )
    members = tuple("member." + name for name in ACTIONS)
    frame = pd.DataFrame(features.reshape(-1, 40), columns=[*FORECAST_FEATURES, *members])
    frame["candidate_id"] = np.tile(names, 2)
    learner = AlignedPortfolioRegressor(candidate_ids=names, member_names=members)
    expected = learner._matrix(frame)
    frame["future_mse"] = 1e9
    frame["hidden_context"] = -1e9
    pd.testing.assert_frame_equal(learner._matrix(frame), expected)
    frame["response.current_future"] = 0.0
    with pytest.raises(ValueError, match="whitelist"):
        learner._matrix(frame)


def test_shared_regressor_fits_and_predicts_the_complete_option_catalog():
    rng = np.random.default_rng(31)
    candidates = rng.normal(size=(6, 7, 6))
    features, vectors, names = build_option_features(
        candidates,
        rng.normal(size=(6, 7, len(STATIC_FEATURES))),
        np.zeros((6, 2)),
        ACTIONS,
        horizon=3,
        targets=2,
    )
    members = tuple("member." + name for name in ACTIONS)
    frame = pd.DataFrame(features.reshape(-1, 40), columns=[*FORECAST_FEATURES, *members])
    frame["candidate_id"] = np.tile(names, 6)
    frame["episode_id"] = np.repeat(np.arange(6).astype(str), 43)
    frame["family_id"] = np.repeat(["a", "a", "a", "b", "b", "b"], 43)
    frame["dataset_id"] = frame.family_id
    labels = projection_targets(vectors, rng.normal(size=(6, 6)), anchor=vectors[:, -1])
    learner = AlignedPortfolioRegressor(member_names=members).fit(
        frame, labels["unit_projection"].reshape(-1)
    )
    prediction = learner.predict(frame)
    assert prediction.shape == (6 * 43,)
    assert np.isfinite(prediction).all()
    assert learner.model.n_features_in_ == 40
