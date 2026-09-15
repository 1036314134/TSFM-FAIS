import numpy as np
import torch

from tsfm_fais.imputers.motm import prepare_context


def test_duplicate_context_uses_only_observations_and_preserves_global_rng():
    context = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, np.nan], [np.nan, 7.0]])
    before = torch.random.get_rng_state().clone()
    values, coordinates, grid, available = prepare_context(context)
    assert torch.equal(before, torch.random.get_rng_state())
    assert available.tolist() == [True, True]
    for channel in range(2):
        indices = (coordinates[channel, :, 0] * 3).round().long().numpy()
        assert np.isfinite(context[indices, channel]).all()
        np.testing.assert_array_equal(values[channel, :, 0].numpy(), context[indices, channel])
    assert grid.shape == values.shape == (2, 4, 1)


def test_empty_channel_is_explicit_and_single_observation_remains_valid():
    context = np.array([[np.nan, 9.0], [np.nan, np.nan], [np.nan, np.nan]])
    values, _, _, available = prepare_context(context)
    assert available.tolist() == [False, True]
    assert torch.isnan(values[0]).all()
    assert torch.equal(values[1], torch.full((3, 1), 9.0))
