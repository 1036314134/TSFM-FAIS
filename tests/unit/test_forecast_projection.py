import numpy as np

from tsfm_fais.routing.forecast_projection import (
    compose_from_estimates,
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)


def test_projection_labels_reconstruct_the_exact_teacher_risk_difference():
    points = np.array([[[0.0, 1.0], [2.0, 3.0], [10.0, 5.0]]])
    teacher = np.array([[1.0, 4.0]])
    anchor, _, energy, _ = forecast_geometry(points)
    labels = projection_targets(points, teacher)
    expected = ((points - teacher[:, None]) ** 2).mean(axis=2) - ((anchor - teacher) ** 2).mean(
        axis=1
    )[:, None]
    np.testing.assert_allclose(labels["direct_risk"], expected)
    np.testing.assert_allclose(energy - 2 * labels["raw_projection"], expected)
    np.testing.assert_allclose(energy - 2 * np.sqrt(energy) * labels["unit_projection"], expected)


def test_simplex_solver_handles_nonunique_and_zero_gram_matrices():
    vectors = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    target = np.array([[0.2, 0.4]])
    gram = np.einsum("naq,nbq->nab", vectors, vectors) / 2
    alignment = np.mean(vectors * target[:, None], axis=2)
    weights, gap, _ = simplex_quadratic_weights(gram, alignment)
    np.testing.assert_allclose(np.einsum("na,naq->nq", weights, vectors), target, atol=1e-9)
    assert gap.max() <= 1e-7
    weights, gap, _ = simplex_quadratic_weights(np.zeros((1, 4, 4)), np.zeros((1, 4)))
    np.testing.assert_array_equal(weights, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(gap, [0])
    weights, gap, _ = simplex_quadratic_weights(
        np.array([[[0.0, 0.0], [0.0, 1.0]]]), np.array([[0.0, 1e-6]])
    )
    np.testing.assert_allclose(weights[:, 1], [1e-6], rtol=0, atol=1e-10)
    assert gap.max() <= 1e-7


def test_known_quadratic_term_prevents_capped_risk_from_overweighting_a_large_prediction():
    # The two zero forecasts fix the median anchor at zero.
    points = np.array([[[0.0], [0.0], [10.0]]])
    projected, _, _, _ = compose_from_estimates(
        points, [[0, 0, 0.5]], target_kind="unit_projection"
    )
    direct, _, _, _ = compose_from_estimates(points, [[0, 0, 0]], target_kind="direct_risk")
    np.testing.assert_allclose(projected, [[0.5]], atol=1e-9)
    np.testing.assert_allclose(direct, [[5.0]], atol=1e-9)
    raw, _, _, _ = compose_from_estimates(points, [[0, 0, 5]], target_kind="raw_projection")
    np.testing.assert_allclose(raw, projected, atol=1e-9)
