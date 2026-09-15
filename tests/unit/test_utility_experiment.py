from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tsfm_fais.forecasting.adapters import Chronos2Adapter
from tsfm_fais.forecasting.registry import ForecastRegistry
from tsfm_fais.forecasting.runner import ForecastRunner
from tsfm_fais.routing.utility import UtilitySelector, response_features
from tsfm_fais.utility_experiment import (
    UtilityExperimentConfig,
    analyze_utility_experiment,
    forecast_utility_episodes,
    prepare_utility_episodes,
    purged_origins,
)


def _config(tmp_path, **updates):
    return UtilityExperimentConfig(
        data_manifest=tmp_path / "datasets.yaml",
        output_root=tmp_path / "run",
        forecaster_artifacts={"chronos2": tmp_path / "checkpoint"},
        dataset_ids=("a", "b", "c"),
        candidate_ids=("locf", "linear_interp"),
        context_length=24,
        horizon=12,
        device="cpu",
        train_origins=2,
        validation_origins=1,
        mechanisms=("random_point",),
        missing_rates=(0.3,),
        **updates,
    )


@pytest.mark.parametrize("length", [60, 120, 480, 1000])
def test_purged_origin_intervals_do_not_overlap(tmp_path, length):
    config = _config(tmp_path)
    prefix, partitions = purged_origins(length, config)
    train, validation = partitions["train"], partitions["validation"]
    if train:
        assert min(train) - config.context_length >= prefix
    if train and validation:
        assert max(train) + config.horizon <= min(validation) - config.context_length
    for origins in partitions.values():
        assert all(
            right - left >= config.context_length + config.horizon
            for left, right in zip(origins, origins[1:], strict=False)
        )


class StaticPredictor:
    def predict(self, matrix):
        return np.asarray(matrix)[:, 0]


def _selector():
    return UtilitySelector(
        baseline_id="a",
        feature_names=("static.signal",),
        candidate_ids=("a", "b"),
        model=StaticPredictor(),
        training_groups=("past_origin",),
    )


def test_selection_does_not_read_future_loss_and_keeps_baseline_on_tie():
    frame = pd.DataFrame(
        {
            "episode_id": ["x", "x", "y", "y"],
            "candidate_id": ["a", "b", "a", "b"],
            "static.signal": [0, -1, 0, 0],
            "loss": [1, 100, 1, 100],
        }
    )
    first = _selector().select(frame)
    second = _selector().select(frame.assign(loss=[100, 1, 100, 1]))
    third = _selector().select(frame.drop(columns="loss"))
    assert (
        first.candidate_id.tolist()
        == second.candidate_id.tolist()
        == third.candidate_id.tolist()
        == ["b", "a"]
    )


def test_calibration_rejects_another_mask_of_a_training_origin():
    frame = pd.DataFrame(
        {
            "episode_id": ["new_mask", "new_mask"],
            "origin_id": ["past_origin", "past_origin"],
            "candidate_id": ["a", "b"],
            "static.signal": [0, -1],
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        _selector().calibrate(frame)


def test_response_features_use_frozen_target_scale():
    point = np.array([[10.0, 1.0], [20.0, 2.0]])
    features = response_features(
        point, np.zeros_like(point), point, np.zeros(2), np.array([10.0, 1.0])
    )
    assert features["response.mean_change"] == 1.5
    assert features["response.pool_distance"] == 0


class SmallChronosBackend:
    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, **kwargs):
        outputs = []
        for entry in inputs:
            target = np.asarray(entry["target"])
            level = np.nan_to_num(np.nanmean(target, axis=1))
            outputs.append(
                np.broadcast_to(
                    level[:, None, None], (len(level), prediction_length, len(quantile_levels))
                ).copy()
            )
        return outputs, None


def test_small_end_to_end_family_holdout_and_artifact_binding(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.forecaster_artifacts["chronos2"].mkdir()
    datasets = []
    for index, name in enumerate(config.dataset_ids):
        values = np.arange(240, dtype=float)
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=240, freq="h"),
                "x": np.sin(values / 6) + index,
                "y": np.cos(values / 9) + index,
            }
        ).to_csv(tmp_path / f"{name}.csv", index=False)
        datasets.append(
            {
                "dataset_id": name,
                "family_id": name,
                "format": "csv",
                "path": f"{name}.csv",
                "frequency": "h",
                "period": 12,
                "timestamp_column": "date",
            }
        )
    config.data_manifest.write_text(
        yaml.safe_dump({"schema_version": 1, "data_root": ".", "datasets": datasets}),
        encoding="utf-8",
    )
    prepared = prepare_utility_episodes(config)
    assert len(prepared["episodes"]) == 9
    assert prepare_utility_episodes(config)["episodes"] == prepared["episodes"]
    monkeypatch.setattr(
        ForecastRegistry,
        "build",
        lambda *_args, **_kwargs: Chronos2Adapter(backend=SmallChronosBackend()),
    )
    monkeypatch.setattr(ForecastRunner, "_cuda_profiler", staticmethod(lambda: None))
    forecasted = forecast_utility_episodes(config, "chronos2")
    assert forecasted["row_count"] == 27
    replay = forecast_utility_episodes(config, "chronos2")
    assert replay["resources_this_execution"]["forecast_call_count"] == 0
    result = analyze_utility_experiment(config, ("chronos2",))
    assert result["evidence_role"] == "development"
    folds = json.loads((config.output_root / "analysis" / "folds.json").read_text(encoding="utf-8"))
    assert len(folds) == 6
    assert all(
        row["held_family"] not in row["train_families"] + row["calibration_families"]
        for row in folds
    )
    first = config.output_root / prepared["episodes"][0]["path"]
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_utility_episodes(config)
