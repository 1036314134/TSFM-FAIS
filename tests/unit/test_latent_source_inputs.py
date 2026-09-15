import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from latent_source_inputs import combine_features, project_heads, projection_matrix  # noqa: E402
from train_latent_source_gates import condition_features  # noqa: E402


def test_chronos_projection_keeps_target_and_future_half_order():
    head = np.zeros((2, 6, 768), np.float32)
    head[0, :3] = 1
    head[0, 3:] = 2
    head[1, :3] = 3
    head[1, 3:] = 4
    matrix = np.zeros((3072, 32), np.float32)
    for index in range(4):
        matrix[index * 768, index] = 1
    np.testing.assert_array_equal(project_heads("chronos2", [head], matrix)[0, :4], [1, 2, 3, 4])


def test_timesfm_projection_keeps_sign_branch_and_target_order():
    normal = np.stack([np.full(1280, 5), np.full(1280, 7)]).astype(np.float32)
    flipped = np.stack([np.full(1280, 6), np.full(1280, 8)]).astype(np.float32)
    matrix = np.zeros((2560, 32), np.float32)
    matrix[0, 0] = matrix[1280, 1] = 1
    np.testing.assert_array_equal(
        project_heads("timesfm2p5", [normal, flipped], matrix)[:, :2], [[5, 6], [7, 8]]
    )


def test_latent_ablation_preserves_base_inputs_and_locf_reference():
    rng = np.random.default_rng(9519)
    base = rng.normal(size=(3, 7, 33)).astype(np.float32)
    latent = rng.normal(size=(3, 7, 32)).astype(np.float32)
    combined = combine_features(base, latent, 3)
    np.testing.assert_array_equal(combined[:, 3, 65:], np.zeros((3, 32)))
    for condition in ("point_teacher", "point_future"):
        values = condition_features(combined, condition)
        np.testing.assert_array_equal(values[:, :, :33], base)
        np.testing.assert_array_equal(values[:, :, 33:], np.zeros((3, 7, 64)))
    np.testing.assert_array_equal(condition_features(combined, "latent_future"), combined)
    np.testing.assert_array_equal(projection_matrix("chronos2"), projection_matrix("chronos2"))
    latent[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        combine_features(base, latent, 3)
