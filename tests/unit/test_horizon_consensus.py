import numpy as np

from tsfm_fais.forecasting.horizon_consensus import consensus_medoid_segments


def test_scalar_output_selection_exactly_recovers_odd_pool_median():
    predictions = np.random.default_rng(6101).normal(size=(4, 7, 24, 2))
    result, _ = consensus_medoid_segments(predictions, blocks=24, joint_targets=False)
    np.testing.assert_array_equal(result, np.median(predictions, axis=1))


def test_refining_blocks_reduces_consensus_distance():
    predictions = np.random.default_rng(6102).normal(size=(3, 7, 24, 2))
    median = np.median(predictions, axis=1)
    for joint in (True, False):
        errors = []
        for blocks in (1, 6, 24):
            result, choices = consensus_medoid_segments(
                predictions, blocks=blocks, joint_targets=joint
            )
            assert choices.min() >= 0 and choices.max() < 7
            errors.append(((result - median) ** 2).mean(axis=(1, 2)))
        assert np.all(np.diff(errors, axis=0) <= 1e-14)


def test_joint_and_target_selection_have_distinct_output_constraints():
    predictions = np.array([[[[0, 9]], [[1, 0]], [[9, 1]]]], dtype=float)
    target_result, _ = consensus_medoid_segments(predictions, blocks=1, joint_targets=False)
    joint_result, _ = consensus_medoid_segments(predictions, blocks=1, joint_targets=True)
    np.testing.assert_array_equal(target_result, [[[1, 1]]])
    assert not np.array_equal(joint_result, target_result)
