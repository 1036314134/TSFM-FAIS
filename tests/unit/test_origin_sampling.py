import pandas as pd
import pytest

from tsfm_fais.routing.origin_sampling import nested_origin_ids


def sample_frame():
    return pd.DataFrame(
        [
            {
                "origin_id": f"{family}|{origin}",
                "family_id": family,
                "dataset_id": family,
                "item_id": "one",
                "episode_id": f"{family}|{origin}|mask={mask}",
                "loss": origin + mask,
            }
            for family in ("a", "b")
            for origin in range(8)
            for mask in range(3)
        ]
    )


def test_nested_balanced_whole_history_sampling():
    frame = sample_frame()
    small = nested_origin_ids(frame, 0.25, 6101)
    medium = nested_origin_ids(frame, 0.5, 6101)
    full = nested_origin_ids(frame, 1, 6101)
    assert set(small) < set(medium) < set(full)
    assert (len(small), len(medium), len(full)) == (4, 8, 16)
    assert frame[frame.origin_id.isin(small)].groupby("origin_id").size().tolist() == [3] * 4
    assert sum(origin.startswith("a|") for origin in small) == 2


def test_sampling_ignores_outcomes_and_row_order():
    frame = sample_frame()
    expected = nested_origin_ids(frame, 0.5, 6101)
    changed = frame.sample(frac=1, random_state=2).assign(loss=-999)
    assert nested_origin_ids(changed, 0.5, 6101) == expected
    assert nested_origin_ids(changed, 1, 6102) == nested_origin_ids(frame, 1, 6101)


def test_conflicting_origin_metadata_is_rejected():
    frame = sample_frame()
    frame.loc[0, "item_id"] = "another_series"
    with pytest.raises(ValueError, match="exactly one"):
        nested_origin_ids(frame, 0.5, 6101)
