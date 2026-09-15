import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from native_source_transfer_io import (  # noqa: E402
    fold_indices,
    holdout_group,
    masked_geometry,
    observed_errors,
    observed_weights,
)


@pytest.mark.parametrize("joint", [False, True])
def test_masked_quadratic_matches_direct_equal_target_loss(joint):
    rng = np.random.default_rng(9511)
    points = rng.normal(size=(5, 7, 12))
    truth = rng.normal(size=(5, 12))
    mask = np.ones(truth.shape, bool)
    mask[:, ::3] = False
    weights = rng.dirichlet(np.ones(7), size=5)
    gram, alignment = masked_geometry(points, truth, mask, joint=joint)
    prediction = np.einsum("na,naq->nq", weights, points)
    median = np.median(points, axis=1)
    direct = observed_errors(prediction, truth, mask, joint=joint)[1]
    reference = observed_errors(median, truth, mask, joint=joint)[1]
    relative = np.einsum("na,nab,nb->n", weights, gram, weights) - 2 * (weights * alignment).sum(1)
    np.testing.assert_allclose(relative, direct - reference, rtol=1e-12, atol=1e-12)
    truth[~mask] = np.nan
    repeated = masked_geometry(points, truth, mask, joint=joint)
    for original, changed in zip((gram, alignment), repeated, strict=True):
        np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(direct, observed_errors(prediction, truth, mask, joint=joint)[1])
    truth[~mask] = 1e250
    for original, changed in zip(
        (gram, alignment), masked_geometry(points, truth, mask, joint=joint), strict=True
    ):
        np.testing.assert_array_equal(original, changed)


def test_joint_target_weights_ignore_unequal_observation_counts():
    mask = np.array([[True, True, True, False, True, False, True, False]])
    weights = observed_weights(mask, joint=True).reshape(1, 4, 2)
    np.testing.assert_array_equal(weights.sum(1), [[0.5, 0.5]])
    prediction = np.tile([2.0, 4.0], 4)[None]
    mae, mse = observed_errors(prediction, np.zeros_like(prediction), mask, joint=True)
    np.testing.assert_array_equal(mae, [3.0])
    np.testing.assert_array_equal(mse, [10.0])


def test_fold_excludes_both_singapore_families_and_every_target():
    frame = pd.DataFrame(
        {
            "cohort": ["source", "native", "native", "native", "native"],
            "family_id": ["ett", "sg_pm25", "sg_weather", "sg_pm25", "solar"],
            "origin_id": ["a", "b", "c", "b", "d"],
        }
    )
    frame["holdout_group"] = frame.family_id.map(holdout_group)
    train, evaluation = fold_indices(frame, "singapore")
    np.testing.assert_array_equal(train, [0, 4])
    np.testing.assert_array_equal(evaluation, [1, 2, 3])


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        observed_weights(np.array([[True, False, True, False]]), joint=True)
