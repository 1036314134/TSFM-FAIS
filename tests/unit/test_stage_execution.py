from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.config import load_config
from tsfm_fais.contracts import TimeSeriesItem
from tsfm_fais.stage_execution import (
    _episode_iter,
    _fit_region_end,
    _forecaster_artifacts,
    _pair_label_requests,
    _training_batch,
)
from tsfm_fais.stages import StageInputs


def _config(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    files = {
        "datasets.yaml": Path("configs/data/datasets.yaml"),
        "imputers.yaml": Path("configs/imputers/pool.yaml"),
        "forecasters.yaml": Path("configs/forecasters/pool.yaml"),
        "router.yaml": Path("configs/router/block_fais.yaml"),
    }
    for name, source in files.items():
        (config_dir / name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    path = config_dir / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "registries:",
                "  data_manifest: datasets.yaml",
                "  imputer_registry: imputers.yaml",
                "  forecaster_registry: forecasters.yaml",
                "  router_config: router.yaml",
                "experiment:",
                "  context_length: 4",
                "  horizon: 2",
                "  missing_mechanisms: [independent_block]",
                "  missing_rates: [0.25]",
                "  seeds: [7]",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(path)


def _item(length: int = 24) -> TimeSeriesItem:
    time = np.arange(length, dtype=float)
    values = np.column_stack((time, np.sin(time), np.cos(time)))
    return TimeSeriesItem(
        item_id="item-0",
        values=values,
        variate_names=("trend", "sin", "cos"),
        start=pd.Timestamp("2026-01-01"),
        freq="h",
    )


def _origins(
    config,
    partition: str,
    item: TimeSeriesItem | None = None,
) -> tuple[int, ...]:
    episodes = _episode_iter(
        config,
        SimpleNamespace(dataset_id="synthetic"),
        [item or _item()],
        partition=partition,
    )
    return tuple(episode.forecast_origin for _, episode in episodes)


def test_episode_partitions_are_chronological_disjoint_and_complete(tmp_path):
    config = _config(tmp_path)

    training = _origins(config, "train")
    evaluation = _origins(config, "eval")
    combined = _origins(config, "all")

    assert training
    assert evaluation
    assert set(training).isdisjoint(evaluation)
    assert training + evaluation == combined
    assert max(training) < min(evaluation)


def test_training_batch_uses_only_rolling_windows_before_episode_origins(tmp_path):
    config = _config(tmp_path)
    item = _item(80)
    batch = _training_batch(
        [item],
        config.experiment.context_length,
        config.experiment.horizon,
    )
    fit_end = _fit_region_end(
        len(item.values),
        config.experiment.context_length,
        config.experiment.horizon,
    )
    first_origin = _origins(config, "all", item)[0]

    assert batch.shape[0] > 1
    assert all(int(item_id.rsplit("@", 1)[1]) + batch.shape[1] <= fit_end for item_id in batch.item_ids)
    assert fit_end + config.experiment.context_length == first_origin


def test_forecaster_artifacts_resolve_model_named_directories(tmp_path):
    root = tmp_path / "checkpoints"
    chronos = root / "chronos2"
    timesfm = root / "timesfm2p5"
    chronos.mkdir(parents=True)
    timesfm.mkdir()

    resolved = _forecaster_artifacts(
        StageInputs(
            forecaster_id="chronos2,timesfm2p5",
            forecaster_artifact=root,
        )
    )

    assert resolved == (("chronos2", chronos), ("timesfm2p5", timesfm))


def test_forecaster_artifacts_resolve_json_paths_relative_to_mapping(tmp_path):
    chronos = tmp_path / "weights" / "chronos"
    timesfm = tmp_path / "weights" / "timesfm"
    chronos.mkdir(parents=True)
    timesfm.mkdir()
    mapping = tmp_path / "forecasters.json"
    mapping.write_text(
        json.dumps(
            {
                "artifacts": {
                    "chronos2": "weights/chronos",
                    "timesfm2p5": "weights/timesfm",
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = _forecaster_artifacts(
        StageInputs(
            forecaster_id="chronos2,timesfm2p5",
            forecaster_artifact=mapping,
        )
    )

    assert resolved == (
        ("chronos2", chronos.resolve()),
        ("timesfm2p5", timesfm.resolve()),
    )


def test_forecaster_artifacts_report_missing_model_path(tmp_path):
    root = tmp_path / "checkpoints"
    (root / "chronos2").mkdir(parents=True)

    with pytest.raises(ValueError, match="timesfm2p5"):
        _forecaster_artifacts(
            StageInputs(
                forecaster_id="chronos2,timesfm2p5",
                forecaster_artifact=root,
            )
        )


def test_pair_label_sampling_is_deterministic_and_covers_candidate_roles():
    edges = (
        SimpleNamespace(left="b0", right="b1"),
        SimpleNamespace(left="b1", right="b2"),
    )
    candidates = tuple(f"c{index}" for index in range(20))
    eligible = {block_id: candidates for block_id in ("b0", "b1", "b2")}

    first = _pair_label_requests(edges, eligible, candidates, seed=19, limit=64)
    second = _pair_label_requests(edges, eligible, candidates, seed=19, limit=64)

    assert first == second
    assert len(first) == 64
    assert {left for _, left, _ in first} == set(candidates)
    assert {right for _, _, right in first} == set(candidates)
    assert {(edge.left, edge.right) for edge, _, _ in first} == {
        ("b0", "b1"),
        ("b1", "b2"),
    }
