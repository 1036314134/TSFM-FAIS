import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.data.masking import MaskingSpec
from tsfm_fais.imputers.runner import CandidateRunner
from tsfm_fais.stage_execution import _training_batch


@pytest.fixture
def preparation():
    path = Path(__file__).resolve().parents[2] / "scripts/prepare_native_confirmation.py"
    spec = importlib.util.spec_from_file_location("native_confirmation_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_fallback_preserves_observations_and_uses_prefix_for_empty_channels(preparation):
    context = np.array([[np.nan, np.nan], [2.0, np.nan], [np.nan, np.nan]])
    batch = SeriesBatch(context[None], np.isfinite(context[None]))
    actions = ["locf", "linear_interp"]
    results = CandidateRunner().run_many(actions, batch)
    bank, _, _ = preparation.complete_candidates(context, results, np.array([1.0, 7.0]), actions)
    # The registered LOCF method extends the first observed value to its leading edge.
    np.testing.assert_array_equal(bank[0], [[2, 7], [2, 7], [2, 7]])
    np.testing.assert_array_equal(bank[1], bank[0])


def test_native_prefix_training_is_invariant_to_all_later_values():
    values = np.arange(2048, dtype=float).reshape(1024, 2)
    values[20:30, 0] = np.nan
    changed = values.copy()
    changed[204:] = -1e8
    settings = dict(
        context_length=96,
        horizon=96,
        max_windows=4,
        dataset_id="d",
        masking_specs=[MaskingSpec("independent_block", 0.2)],
        configured_seeds=[1101],
        fit_fraction=0.2,
    )
    first = _training_batch([SimpleNamespace(item_id="item0", values=values)], **settings)
    second = _training_batch([SimpleNamespace(item_id="item0", values=changed)], **settings)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.observed_mask, second.observed_mask)
