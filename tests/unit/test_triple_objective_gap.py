from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import analyze_triple_objective_gap

    return analyze_triple_objective_gap


def test_best_option_respects_joint_and_independent_targets(diagnostic):
    points = np.array([[[[0.0, 100.0]], [[100.0, 0.0]], [[40.0, 40.0]]]])
    costs = np.square(points).mean(axis=2)
    joint, joint_choice = diagnostic.best_option(points, costs, joint=True)
    independent, independent_choice = diagnostic.best_option(points, costs, joint=False)
    np.testing.assert_array_equal(joint_choice, [[2, 2]])
    np.testing.assert_array_equal(independent_choice, [[0, 1]])
    np.testing.assert_array_equal(joint, [[[40.0, 40.0]]])
    np.testing.assert_array_equal(independent, [[[0.0, 0.0]]])


def test_individually_closest_three_need_not_form_the_best_median(diagnostic):
    from itertools import combinations

    points = np.array([[[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [-10.0, 0.0], [0.0, -10.0]]])[..., None]
    teacher = np.zeros((1, 2, 1))
    individual = diagnostic.teacher_top_three(points, teacher, joint=True)
    triples = np.stack(
        [np.median(points[:, subset], axis=1) for subset in combinations(range(5), 3)], axis=1
    )
    best, _ = diagnostic.best_option(triples, np.square(triples).mean(axis=2), joint=True)
    np.testing.assert_allclose(individual, 1.1)
    np.testing.assert_array_equal(best, teacher)
