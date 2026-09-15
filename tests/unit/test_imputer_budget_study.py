import importlib.util
from pathlib import Path

import numpy as np

from tsfm_fais.contracts import SeriesBatch


def test_larger_budget_retains_original_windows_and_does_not_duplicate(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "imputer_budget_study", scripts / "prepare_imputer_budget_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_values = np.arange(24.0).reshape(2, 6, 2)
    base_values[0, 1, 0] = np.nan
    base = SeriesBatch(base_values, np.isfinite(base_values), item_ids=("a", "b"))
    expanded_values = np.arange(72.0).reshape(6, 6, 2)
    expanded = SeriesBatch(
        expanded_values, np.isfinite(expanded_values), item_ids=("b", "c", "d", "e", "f", "g")
    )
    result = module.nested_training_batch(base, expanded, 5)
    assert result.shape == (5, 6, 2)
    assert result.item_ids[:2] == base.item_ids
    assert len(set(result.item_ids)) == 5
    np.testing.assert_array_equal(result.values[:2], base.values)
    np.testing.assert_array_equal(result.observed_mask[:2], base.observed_mask)
