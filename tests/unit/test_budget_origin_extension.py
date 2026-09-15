from pathlib import Path

import pytest


@pytest.fixture
def extension(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import prepare_budget_origin_extension

    return prepare_budget_origin_extension


def panel(datasets):
    episodes, cases = [], []
    for dataset in datasets:
        for origin in (100, 200, 300, 400):
            for mechanism in ("random_point", "independent_block", "value_dependent"):
                for rate in (0.1, 0.5):
                    episode_id = f"{dataset}|{origin}|{mechanism}|{rate}"
                    episodes.append(
                        {
                            "episode_id": episode_id,
                            "dataset_id": dataset,
                            "origin": origin,
                            "mechanism": mechanism,
                            "missing_rate": rate,
                            "mask_seed": 6101,
                            "split": "validation",
                        }
                    )
                    if origin == 100:
                        cases.append({"episode_id": episode_id})
        episodes.append(
            {
                "episode_id": f"{dataset}|train",
                "dataset_id": dataset,
                "origin": 50,
                "mechanism": "random_point",
                "missing_rate": 0.1,
                "mask_seed": 6101,
                "split": "train",
            }
        )
    return {"episodes": episodes}, {"cases": cases}


def test_additional_origins_exclude_old_histories_and_training(extension):
    source, prepared = panel(extension.DATASETS)
    selected, origins = extension.select_additional_episodes(source, prepared)
    assert len(selected) == 54
    assert all(row["origin"] != 100 and row["split"] == "validation" for _, row in selected)
    assert origins == {name: [200, 300, 400] for name in extension.DATASETS}
    original_ids = {row["episode_id"] for _, row in selected}
    source["episodes"].reverse()
    reordered, _ = extension.select_additional_episodes(source, prepared)
    assert {row["episode_id"] for _, row in reordered} == original_ids


def test_extension_rejects_incomplete_conditions_and_duplicate_identities(extension):
    source, prepared = panel(extension.DATASETS)
    source["episodes"] = [
        row
        for row in source["episodes"]
        if row["episode_id"] != f"{extension.DATASETS[0]}|200|random_point|0.1"
    ]
    with pytest.raises(ValueError, match="all three remaining"):
        extension.select_additional_episodes(source, prepared)
    source, prepared = panel(extension.DATASETS)
    source["episodes"].append(source["episodes"][0])
    with pytest.raises(ValueError, match="duplicate source"):
        extension.select_additional_episodes(source, prepared)
