from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from pydantic import ValidationError

from tsfm_fais.config import ExperimentConfig, load_config
from tsfm_fais.contracts import BudgetSpec, ForecastSpec, SeriesBatch, TimeSeriesItem
from tsfm_fais.data.audit import audit_dataset
from tsfm_fais.data.catalog import DatasetManifest, DatasetSpec, load_manifest
from tsfm_fais.data.episodes import build_episode, fit_prefix_end, rolling_origins
from tsfm_fais.data.loaders import load_arrow, load_csv
from tsfm_fais.data.masking import (
    MaskingSpec,
    mask_time_series,
    no_complete_window_base_mask,
    stable_seed,
)
from tsfm_fais.data.splits import family_folds


def _item(length: int = 64, dimensions: int = 3) -> TimeSeriesItem:
    time = np.arange(length, dtype=float)
    values = np.stack([time + index for index in range(dimensions)], axis=1)
    return TimeSeriesItem(
        item_id="toy",
        values=values,
        variate_names=tuple(f"v{index}" for index in range(dimensions)),
        start=pd.Timestamp("2026-01-01"),
        freq="H",
        timestamps=pd.date_range("2026-01-01", periods=length, freq="h"),
        metadata={"period": 24},
    )


def test_series_batch_normalizes_missing_values_to_nan():
    values = np.arange(18, dtype=float).reshape(1, 6, 3)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2:4, 1] = False
    batch = SeriesBatch(values, mask)
    assert np.isnan(batch.values[:, 2:4, 1]).all()
    assert np.isfinite(batch.values[batch.observed_mask]).all()


def test_csv_loader_preserves_all_multivariate_columns_and_items(tmp_path):
    path = tmp_path / "toy.csv"
    frame = pd.DataFrame(
        {
            "item": ["a"] * 4 + ["b"] * 4,
            "timestamp": list(pd.date_range("2026-01-01", periods=4, freq="h")) * 2,
            "x": np.arange(8),
            "y": np.arange(8) + 10,
            "z": np.arange(8) + 20,
        }
    )
    frame.to_csv(path, index=False)
    spec = DatasetSpec(
        dataset_id="toy",
        family_id="toy",
        format="csv",
        path=path,
        frequency="H",
        period=24,
        timestamp_column="timestamp",
        item_id_column="item",
        expected_num_variates=3,
    )
    items = load_csv(spec)
    assert len(items) == 2
    assert items[0].values.shape == (4, 3)
    assert items[0].variate_names == ("x", "y", "z")
    assert items[0].metadata["period"] == 24
    with pytest.raises(ValueError, match="target columns"):
        load_csv(spec.model_copy(update={"target_columns": ("missing",)}))


def test_arrow_loader_preserves_item_boundaries_and_marks_implicit_time(tmp_path):
    path = tmp_path / "toy.arrow"
    table = pa.Table.from_pylist(
        [
            {
                "item_id": "a",
                "start": datetime(2026, 1, 1),
                "freq": "H",
                "target": [[1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0]],
                "variate_names": ["x", "y"],
            },
            {
                "item_id": "b",
                "start": datetime(2026, 1, 2),
                "freq": "H",
                "target": [[5.0, 20.0], [6.0, 21.0], [7.0, 22.0], [8.0, 23.0]],
                "variate_names": ["x", "y"],
            },
        ]
    )
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    spec = DatasetSpec(
        dataset_id="toy_arrow",
        family_id="toy",
        format="arrow",
        path=path,
        frequency="H",
        period=24,
        expected_num_variates=2,
        allow_implicit_regular_time=True,
    )
    items = load_arrow(spec)
    assert [item.item_id for item in items] == ["a", "b"]
    assert all(item.values.shape == (4, 2) for item in items)
    assert all(item.timestamps is None for item in items)
    assert all(item.metadata["implicit_regular_time"] for item in items)
    assert all(item.metadata["raw_variate_names"] == ("x", "y") for item in items)
    assert all(item.metadata["variate_name_normalization"] == "none" for item in items)
    assert audit_dataset(spec, items).accepted

    rejected = audit_dataset(spec.model_copy(update={"allow_implicit_regular_time": False}), items)
    assert "implicit_time_not_allowed" in {issue.code for issue in rejected.issues}


def _write_arrow_rows(path, rows):
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


def _normalizing_arrow_spec(path) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="normalized_arrow",
        family_id="toy",
        format="arrow",
        path=path,
        frequency="H",
        period=24,
        expected_num_variates=2,
        allow_implicit_regular_time=True,
        variate_name_normalization="strip_bracket_suffix",
    )


def test_arrow_loader_strips_safe_suffix_and_preserves_raw_names(tmp_path):
    path = tmp_path / "normalized.arrow"
    _write_arrow_rows(
        path,
        [
            {
                "item_id": "plain",
                "start": datetime(2026, 1, 1),
                "freq": "H",
                "target": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "variate_names": ["min_cpu", "max_cpu"],
            },
            {
                "item_id": "annotated",
                "start": datetime(2026, 1, 1),
                "freq": "H",
                "target": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                "variate_names": ["min_cpu[sp,rw]", "max_cpu[rw,drop]"],
            },
        ],
    )
    spec = _normalizing_arrow_spec(path)

    items = load_arrow(spec)

    assert all(item.variate_names == ("min_cpu", "max_cpu") for item in items)
    assert items[0].metadata["raw_variate_names"] == ("min_cpu", "max_cpu")
    assert items[1].metadata["raw_variate_names"] == (
        "min_cpu[sp,rw]",
        "max_cpu[rw,drop]",
    )
    assert audit_dataset(spec, items).accepted


@pytest.mark.parametrize(
    "names,error",
    [
        (("x[unsafe/value]", "y"), "unsafe Arrow variate name"),
        (("x", "x[tag]"), "not unique after normalization"),
        (("x", "y", "z"), "has length 3, expected D=2"),
    ],
)
def test_arrow_name_normalization_rejects_unsafe_or_ambiguous_names(tmp_path, names, error):
    path = tmp_path / "invalid_names.arrow"
    _write_arrow_rows(
        path,
        [
            {
                "item_id": "invalid",
                "start": datetime(2026, 1, 1),
                "freq": "H",
                "target": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "variate_names": list(names),
            }
        ],
    )

    with pytest.raises(ValueError, match=error):
        load_arrow(_normalizing_arrow_spec(path))


def test_variate_name_normalization_is_strict_and_arrow_only(tmp_path):
    base = {
        "dataset_id": "toy",
        "family_id": "toy",
        "format": "arrow",
        "path": tmp_path / "toy.arrow",
        "frequency": "H",
        "period": 24,
    }
    with pytest.raises(ValidationError, match="variate_name_normalization"):
        DatasetSpec(**base, variate_name_normalization="strip_any_suffix")
    with pytest.raises(ValidationError, match="supported only for Arrow"):
        DatasetSpec(
            **{**base, "format": "csv"},
            variate_name_normalization="strip_bracket_suffix",
        )


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda values: values.__setitem__((2, 0), np.nan), "nan"),
        (lambda values: values.__setitem__((2, 0), np.inf), "infinite"),
        (lambda values: values.__setitem__((2, 0), -9999), "sentinel"),
        (lambda values: values.__setitem__((slice(None), 0), 1), "constant"),
    ],
)
def test_audit_rejects_invalid_source_values(tmp_path, mutator, code):
    item = _item()
    values = item.values.copy()
    mutator(values)
    invalid = TimeSeriesItem(
        item_id=item.item_id,
        values=values,
        variate_names=item.variate_names,
        start=item.start,
        freq=item.freq,
        timestamps=item.timestamps,
    )
    spec = DatasetSpec(
        dataset_id="toy",
        family_id="toy",
        format="csv",
        path=tmp_path / "missing.csv",
        frequency="H",
        period=24,
        expected_num_variates=3,
    )
    report = audit_dataset(spec, [invalid])
    assert not report.accepted
    assert code in {issue.code for issue in report.issues}


def test_audit_rejects_time_and_item_boundary_errors(tmp_path):
    base = _item(length=8)
    timestamps = base.timestamps.copy()
    timestamps = timestamps.delete(3).insert(3, timestamps[2])
    duplicate_time = TimeSeriesItem(
        item_id="toy",
        values=base.values,
        variate_names=base.variate_names,
        start=base.start,
        freq="H",
        timestamps=timestamps,
    )
    spec = DatasetSpec(
        dataset_id="toy",
        family_id="toy",
        format="csv",
        path=tmp_path / "missing.csv",
        frequency="H",
        period=24,
        expected_num_variates=3,
    )
    report = audit_dataset(spec, [duplicate_time, duplicate_time])
    codes = {issue.code for issue in report.issues}
    assert {"duplicate_time", "duplicate_item_id"} <= codes


def test_audit_accepts_regular_weekly_axis_with_a_different_week_anchor(tmp_path):
    timestamps = pd.date_range("2026-01-05", periods=8, freq="W-MON")
    values = np.stack([np.arange(8), np.arange(8) ** 2], axis=1).astype(float)
    item = TimeSeriesItem("weekly", values, ("x", "y"), timestamps[0], "W", timestamps)
    spec = DatasetSpec(
        dataset_id="weekly",
        family_id="weekly",
        format="csv",
        path=tmp_path / "missing.csv",
        frequency="W",
        period=52,
        expected_num_variates=2,
    )
    assert audit_dataset(spec, [item]).accepted


@pytest.mark.parametrize(
    "mechanism",
    [
        "random_point",
        "independent_block",
        "synchronous_block",
        "staggered_correlated",
        "value_dependent",
        "mixed_outage",
    ],
)
def test_all_missingness_mechanisms_are_deterministic(mechanism):
    values = _item().values
    spec = MaskingSpec(mechanism=mechanism, missing_rate=0.2)
    first = mask_time_series(values, spec, 17, calibration_values=values[:32])
    second = mask_time_series(values, spec, 17, calibration_values=values[:32])
    assert np.array_equal(first.observed_mask, second.observed_mask)
    assert first.blocks == second.blocks
    assert first.realization_id == second.realization_id
    difference = abs(int((~first.observed_mask).sum()) - round(values.size * 0.2))
    assert difference <= (values.shape[1] // 2 if mechanism == "synchronous_block" else 0)


def test_synchronous_blocks_keep_channel_masks_aligned():
    values = _item().values
    result = mask_time_series(
        values,
        MaskingSpec("synchronous_block", missing_rate=0.2),
        17,
    )
    mask = result.observed_mask
    assert np.array_equal(mask[:, 0], mask[:, 1])
    assert np.array_equal(mask[:, 1], mask[:, 2])


def test_complete_synchronous_mask_preserves_established_realization_identity():
    values = _item().values
    implicit = mask_time_series(values, MaskingSpec("synchronous_block", 0.2), 17)
    explicit = mask_time_series(
        values,
        MaskingSpec("synchronous_block", 0.2),
        17,
        base_observed_mask=np.ones_like(values, dtype=bool),
    )

    assert implicit.realization_id == "410899768d79f483535f"
    assert explicit.realization_id == implicit.realization_id
    assert np.array_equal(explicit.observed_mask, implicit.observed_mask)


def test_sequence_mask_supports_the_new_point_four_rate():
    values = _item(length=96).values
    result = mask_time_series(values, MaskingSpec("mixed_outage", 0.4), 3)
    assert int((~result.observed_mask).sum()) == round(values.size * 0.4)
    assert result.metadata["target_missing_rate"] == 0.4


def test_persistent_base_mask_removes_every_complete_prefix_window():
    values = _item(length=48, dimensions=3).values
    first = no_complete_window_base_mask(values, 32, 8, 0.1, 19)
    second = no_complete_window_base_mask(values, 32, 8, 0.1, 19)

    assert np.array_equal(first, second)
    assert all(not np.all(first[start : start + 8]) for start in range(25))
    assert np.array_equal(first[32:], np.ones_like(first[32:], dtype=bool))
    assert np.all(first[:32].sum(axis=0) >= 2)


@pytest.mark.parametrize(
    ("shape", "prefix_end", "window_length", "missing_rate", "seed", "expected_missing"),
    (
        (
            (48, 3),
            32,
            8,
            0.1,
            19,
            ((1, 2), (7, 1), (8, 1), (10, 0), (11, 2), (14, 0), (15, 1), (23, 1), (27, 2), (31, 2)),
        ),
        (
            (12, 2),
            10,
            4,
            0.5,
            5,
            ((0, 0), (0, 1), (2, 0), (3, 1), (4, 1), (5, 0), (6, 0), (7, 1), (8, 0), (9, 1)),
        ),
    ),
)
def test_persistent_base_mask_preserves_legacy_row_major_sampling(
    shape,
    prefix_end,
    window_length,
    missing_rate,
    seed,
    expected_missing,
):
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)

    observed = no_complete_window_base_mask(
        values,
        prefix_end,
        window_length,
        missing_rate,
        seed,
    )

    assert tuple(map(tuple, np.argwhere(~observed))) == expected_missing


def test_native_mask_preserves_only_source_observations():
    values = _item(length=24, dimensions=3).values.copy()
    values[3:6, 1] = np.nan
    values[10, 2] = np.nan
    source_observed = np.isfinite(values)

    result = mask_time_series(
        values,
        MaskingSpec("native", 0.0),
        0,
        base_observed_mask=source_observed,
    )

    assert result.metadata["protocol"] == "native_observation_mask_v1"
    assert np.array_equal(result.observed_mask, source_observed)
    assert np.array_equal(result.base_observed_mask, source_observed)
    assert np.isnan(result.values[~source_observed]).all()


def test_native_dataset_audit_records_missingness_without_hidden_truth(tmp_path):
    item = _item(length=24, dimensions=3)
    values = item.values.copy()
    values[2:5, 0] = np.nan
    native_item = TimeSeriesItem(
        item.item_id,
        values,
        item.variate_names,
        item.start,
        item.freq,
        item.timestamps,
        item.metadata,
    )
    spec = DatasetSpec(
        dataset_id="native",
        family_id="native",
        format="csv",
        path=tmp_path / "native.csv",
        frequency="H",
        period=24,
        expected_num_variates=3,
        missingness="native",
        provenance="source_native_missing",
    )

    report = audit_dataset(spec, [native_item])

    assert report.accepted
    assert report.native_missing_values == 3
    assert report.observed_values == values.size - 3
    assert report.minimum_variate_observed_fraction == pytest.approx(21 / 24)


def test_experiment_config_rejects_missing_rates_above_evaluation_scope():
    with pytest.raises(ValidationError, match="0, 0.5"):
        ExperimentConfig(missing_rates=(0.6,))


def test_staggered_correlated_targets_the_strongest_variable_pair():
    time = np.arange(60, dtype=float)
    values = np.stack((time, 2.0 * time + 1.0, (-1.0) ** time), axis=1)
    result = mask_time_series(
        values,
        MaskingSpec("staggered_correlated", missing_rate=0.1),
        17,
        calibration_values=values[:40],
    )
    mask = result.observed_mask
    missing_by_channel = (~mask).sum(axis=0)
    assert missing_by_channel[0] > 0 and missing_by_channel[1] > 0
    assert missing_by_channel[2] == 0


@pytest.mark.parametrize(
    "mechanism",
    ("independent_block", "staggered_correlated", "value_dependent", "mixed_outage"),
)
@pytest.mark.parametrize("missing_rate", (0.1, 0.5))
def test_high_dimensional_block_mechanisms_do_not_degenerate_to_points(mechanism, missing_rate):
    time = np.arange(128, dtype=float)[:, None]
    channels = np.arange(32, dtype=float)[None, :]
    values = np.sin(time / 7.0 + channels / 11.0) + channels * 0.001
    result = mask_time_series(
        values,
        MaskingSpec(mechanism, missing_rate=missing_rate),
        23,
        calibration_values=values[:64],
    )
    mask = result.observed_mask
    blocks = result.blocks
    expected = int(round(values.size * missing_rate))
    assert int((~mask).sum()) == expected
    if mechanism != "mixed_outage":
        assert sum(block.length == 1 for block in blocks) <= max(1, len(blocks) // 4)


def test_episode_slices_a_precomputed_sequence_mask_without_regeneration():
    item = _item(length=240)
    spec = MaskingSpec("independent_block", 0.2)
    mask_seed = stable_seed("toy", item.item_id, spec, 17)
    realization = mask_time_series(
        item.values,
        spec,
        mask_seed,
        calibration_values=item.values[:96],
    )
    episode = build_episode(
        item,
        "toy",
        realization,
        forecast_origin=144,
        context_length=96,
        horizon=96,
    )
    repeated = build_episode(
        item,
        "toy",
        realization,
        forecast_origin=120,
        context_length=96,
        horizon=96,
    )
    np.testing.assert_array_equal(
        repeated.context.observed_mask[0, 24:96],
        episode.context.observed_mask[0, :72],
    )
    assert episode.mask_realization_id == repeated.mask_realization_id
    assert episode.clean_context.shape == (96, 3)
    assert episode.clean_future.shape == (96, 3)


def test_fit_prefix_and_rolling_origin_support_minimal_96_by_96_series():
    end = fit_prefix_end(196, 96, 96)
    assert end == 96
    assert rolling_origins(196, 96, 96, 96, start=end) == (96,)


def test_rolling_origins_rejects_non_positive_stride():
    with pytest.raises(ValueError, match="stride"):
        rolling_origins(100, 24, 8, stride=0)


def test_config_is_strict_resolves_paths_and_rejects_illegal_targets(tmp_path):
    config_path = tmp_path / "config.yaml"
    payload = {
        "schema_version": 1,
        "registries": {
            "data_manifest": "data.yaml",
            "imputer_registry": "imputers.yaml",
            "forecaster_registry": "forecasters.yaml",
            "router_config": "router.yaml",
        },
        "experiment": {"target_indices": [0, 2]},
        "runtime": {"output_root": "outputs"},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(config_path)
    assert config.registries.data_manifest == (tmp_path / "data.yaml").resolve()
    assert config.runtime.output_root == (tmp_path / "outputs").resolve()
    assert config.experiment.feature_policy == "legacy"
    assert config.experiment.include_family_ids == "all"
    assert config.experiment.exclude_family_ids == ()
    assert config.experiment.router_seed is None
    assert config.protocol is None

    payload["unexpected"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="unexpected"):
        load_config(config_path)

    payload.pop("unexpected")
    payload["experiment"] = {"target_indices": [-1]}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="non-negative"):
        load_config(config_path)


def test_manifest_and_contracts_reject_duplicate_or_illegal_values(tmp_path):
    spec = DatasetSpec(
        dataset_id="toy",
        family_id="toy",
        format="csv",
        path=tmp_path / "toy.csv",
        frequency="H",
        period=24,
        value_columns=("x", "y"),
    )
    with pytest.raises(ValidationError, match="dataset_id"):
        DatasetManifest(data_root=tmp_path, datasets=(spec, spec))
    with pytest.raises(ValueError, match="non-negative"):
        ForecastSpec("mock", "independent_univariate", 4, target_indices=(-1,))
    with pytest.raises(ValueError, match="max_runtime_seconds"):
        BudgetSpec(max_runtime_seconds=0)


def test_local_manifest_contains_exactly_32_multivariate_versions():
    manifest = load_manifest("configs/data/datasets.yaml")
    assert len(manifest.datasets) == 32
    assert "weather" not in {spec.dataset_id for spec in manifest.datasets}
    assert {
        spec.dataset_id for spec in manifest.datasets if spec.variate_name_normalization != "none"
    } == {"azure2019_D_5T", "azure2019_I_5T", "azure2019_U_5T"}
    assert all(
        "ori" not in {part.lower() for part in spec.path.parts} for spec in manifest.datasets
    )


def test_family_folds_keep_all_related_versions_together(tmp_path):
    base = DatasetSpec(
        dataset_id="a1",
        family_id="a",
        format="csv",
        path=tmp_path / "a1.csv",
        frequency="h",
        period=24,
        value_columns=("x", "y"),
    )
    manifest = DatasetManifest(
        data_root=tmp_path,
        datasets=(
            base,
            base.model_copy(update={"dataset_id": "a2", "path": tmp_path / "a2.csv"}),
            base.model_copy(
                update={
                    "dataset_id": "b1",
                    "family_id": "b",
                    "path": tmp_path / "b1.csv",
                }
            ),
        ),
    )
    folds = {fold.family_id: fold for fold in family_folds(manifest)}
    assert {dataset.dataset_id for dataset in folds["a"].test} == {"a1", "a2"}
    assert {dataset.family_id for dataset in folds["a"].train} == {"b"}
