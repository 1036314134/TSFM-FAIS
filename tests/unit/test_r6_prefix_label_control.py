import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from compare_r6_prefix_labels import observed_vectors  # noqa: E402


@pytest.mark.parametrize("slot", [-1, 0, 1])
def test_weighted_vectors_match_equal_target_mse_with_unequal_observed_counts(slot):
    rng = np.random.default_rng(1201)
    points, target = rng.normal(size=(2, 7, 96, 2)), rng.normal(size=(2, 96, 2))
    observed = np.ones_like(target, dtype=bool)
    observed[0, :48, 0] = False
    observed[0, :21, 1] = False
    observed[1, 20:50, 0] = False
    observed[1, 3:38, 1] = False
    target[~observed] = np.nan
    weights = np.arange(1, 8, dtype=float) / 28
    vectors, transformed = observed_vectors(points, target, observed, slot)
    objective = ((np.sum(vectors * weights[None, :, None], axis=1) - transformed) ** 2).mean()
    forecast = np.sum(points * weights[None, :, None, None], axis=1)
    slots = (0, 1) if slot == -1 else (slot,)
    expected = np.mean(
        [
            np.mean(
                [
                    (
                        (forecast[i, observed[i, :, k], k] - target[i, observed[i, :, k], k]) ** 2
                    ).mean()
                    for k in slots
                ]
            )
            for i in range(len(target))
        ]
    )
    np.testing.assert_allclose(objective, expected, rtol=1e-12, atol=1e-12)
    target[~observed] = 1e12
    unchanged_vectors, unchanged_target = observed_vectors(points, target, observed, slot)
    np.testing.assert_array_equal(unchanged_vectors, vectors)
    np.testing.assert_array_equal(unchanged_target, transformed)
