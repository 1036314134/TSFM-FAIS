import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from calibrated_source_gate import fit_snapshots  # noqa: E402
from inventory_source_expansion import nested_training_origins  # noqa: E402
from train_source_expansion import fit_updates  # noqa: E402


def test_expansion_is_nested_and_uses_only_the_existing_time_grid():
    pool = tuple(range(100, 2100, 192))
    original = (pool[0], pool[4], pool[-1])
    expanded = nested_training_origins(pool, original, 7)
    assert len(expanded) == 7
    assert set(original) < set(expanded) <= set(pool)
    assert min(np.diff(expanded)) >= 192
    assert nested_training_origins(pool, original, 99) == pool


def test_expansion_rejects_an_incompatible_original_population():
    with pytest.raises(ValueError, match="outside"):
        nested_training_origins((100, 292, 484), (100, 300))


def test_fixed_updates_match_one_epoch_and_stop_at_a_partial_epoch():
    torch.set_num_threads(1)
    rng = np.random.default_rng(15)
    count = 129
    frame = pd.DataFrame(
        {
            "family_id": ["f"] * count,
            "dataset_id": ["d"] * count,
            "origin_id": [str(i) for i in range(count)],
            "episode_id": [str(i) for i in range(count)],
        }
    )
    features = rng.normal(size=(count, 7, 97)).astype(np.float32)
    gram = np.broadcast_to(np.eye(7), (count, 7, 7)).copy()
    alignment = rng.normal(size=(count, 7))
    indices = np.arange(count)
    reference = fit_snapshots(
        features, gram, alignment, frame, indices, np.ones(7) / 7, 0.0, 5101, (1,)
    )
    matched = fit_updates(frame, features, gram, alignment, indices, 5101, 2)
    assert matched["updates"] == 2 and matched["epochs_started"] == 1
    assert matched["initial_parameter_sha256"] == reference["initial_parameter_sha256"]
    for name, value in matched["state_dict"].items():
        torch.testing.assert_close(value, reference["states"]["1"][name], rtol=0, atol=0)
    partial = fit_updates(frame, features, gram, alignment, indices, 5101, 3)
    assert partial["updates"] == 3 and partial["epochs_started"] == 2
    assert partial["history"][-1]["update"] == 3
