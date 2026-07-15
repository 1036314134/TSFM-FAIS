from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_fais.cli import main
from tsfm_fais.label_artifacts import merge_label_artifacts


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_source(
    root: Path,
    model_id: str,
    lineage: Path,
    *,
    config: dict[str, object] | None = None,
    episodes: tuple[str, ...] = ("episode-a", "episode-b"),
    dataset_overrides: dict[str, tuple[str, str]] | None = None,
    duplicate_unary: bool = False,
) -> Path:
    root.mkdir()
    dataset_overrides = dataset_overrides or {}
    unary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for episode_id in episodes:
        dataset_id, family_id = dataset_overrides.get(
            episode_id, ("dataset-a", "family-a")
        )
        for block_id, candidate_id in (("block-0", "locf"), ("block-1", "linear_interp")):
            unary_rows.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": dataset_id,
                    "family_id": family_id,
                    "forecaster_id": model_id,
                    "group_id": f"{model_id}::{episode_id}::{block_id}",
                    "block_id": block_id,
                    "candidate_id": candidate_id,
                    "prior_features": {},
                    "unary_features": {},
                    "degradation": 0.0,
                }
            )
        pair_rows.append(
            {
                "episode_id": episode_id,
                "dataset_id": dataset_id,
                "family_id": family_id,
                "forecaster_id": model_id,
                "left_block": "block-0",
                "right_block": "block-1",
                "left_candidate": "locf",
                "right_candidate": "linear_interp",
                "features": {},
                "interaction": 0.0,
            }
        )
    if duplicate_unary:
        unary_rows.append(dict(unary_rows[0]))

    _write_jsonl(root / "teacher_labels.jsonl", unary_rows)
    _write_jsonl(root / "pair_labels.jsonl", pair_rows)
    _write_json(
        root / "stage_manifest.json",
        {"schema_version": 1, "stage": "labels", "status": "completed"},
    )
    _write_json(
        root / "resolved_config.json",
        {
            "schema_version": 1,
            "source": str(root / "config.yaml"),
            "config": config
            or {
                "schema_version": 1,
                "seed": 7,
                "experiment": {"context_length": 48, "horizon": 8},
            },
        },
    )
    _write_json(
        root / "labels_manifest.json",
        {
            "teacher_labels": str(root / "teacher_labels.jsonl"),
            "pair_labels": str(root / "pair_labels.jsonl"),
            "forecasters": [model_id],
            "imputer_artifacts": str(lineage),
            "origin_partition": "train",
            "episode_count": len(episodes),
            "unique_episode_count": len(episodes),
            "max_train_episodes_per_dataset": 12,
            "episode_sampling": {
                "partition": "train",
                "cap_per_dataset": 12,
                "selected_episode_count": len(episodes),
                "episode_execution_count": len(episodes),
            },
            "artifact_loading": {
                "strategy": "candidate_filtered_with_structured_dataset_cache_v1",
                "dataset_count": 1,
            },
            "ranking_groups": 2 * len(episodes),
            "unary_rows": len(unary_rows),
            "pair_rows": len(pair_rows),
            "selected_candidates": ["locf", "linear_interp"],
            "max_teacher_blocks_per_episode": 2,
            "max_teacher_candidates_per_episode": 2,
            "max_pair_labels_per_episode": 1,
            "csdi_num_samples": 5,
        },
    )
    return root


def test_merge_label_artifacts_writes_auditable_deterministic_output(tmp_path):
    lineage = tmp_path / "fit" / "imputer_artifacts"
    lineage.mkdir(parents=True)
    chronos = _make_source(tmp_path / "labels-chronos", "chronos2", lineage)
    timesfm = _make_source(tmp_path / "labels-timesfm", "timesfm2p5", lineage)
    output = tmp_path / "merged"

    summary = merge_label_artifacts([timesfm, chronos], output)

    assert summary["forecasters"] == ["chronos2", "timesfm2p5"]
    assert summary["source_directories"] == [str(chronos.resolve()), str(timesfm.resolve())]
    assert summary["episode_count"] == 2
    assert summary["episode_execution_count"] == 4
    assert summary["max_train_episodes_per_dataset"] == 12
    assert summary["episode_sampling"]["episode_execution_count"] == 4
    assert summary["csdi_num_samples"] == 5
    assert all(source["artifact_loading"]["dataset_count"] == 1 for source in summary["sources"])
    assert summary["unary_rows"] == 8
    assert summary["pair_rows"] == 4
    assert summary["ranking_groups"] == 8
    assert summary["dataset_ids"] == ["dataset-a"]
    assert summary["family_ids"] == ["family-a"]
    manifest = json.loads((output / "labels_manifest.json").read_text(encoding="utf-8"))
    assert manifest == summary
    unary_rows = [
        json.loads(line)
        for line in (output / "teacher_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(unary_rows) == 8
    assert unary_rows[0]["forecaster_id"] == "chronos2"
    assert unary_rows[-1]["forecaster_id"] == "timesfm2p5"
    assert (output / "resolved_config.json").is_file()


def test_labels_merge_cli(tmp_path, capsys):
    lineage = tmp_path / "fit" / "imputer_artifacts"
    lineage.mkdir(parents=True)
    first = _make_source(tmp_path / "labels-a", "chronos2", lineage)
    second = _make_source(tmp_path / "labels-b", "tirex", lineage)
    output = tmp_path / "merged-cli"

    code = main(
        [
            "labels",
            "merge",
            "--inputs",
            str(first),
            str(second),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["forecasters"] == ["chronos2", "tirex"]
    assert (output / "labels_manifest.json").is_file()


def test_merge_rejects_imputer_lineage_mismatch_before_creating_output(tmp_path):
    first_lineage = tmp_path / "fit-a"
    second_lineage = tmp_path / "fit-b"
    first_lineage.mkdir()
    second_lineage.mkdir()
    first = _make_source(tmp_path / "labels-a", "chronos2", first_lineage)
    second = _make_source(tmp_path / "labels-b", "tirex", second_lineage)
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="lineage mismatch"):
        merge_label_artifacts([first, second], output)

    assert not output.exists()


def test_merge_rejects_config_mismatch_before_creating_output(tmp_path):
    lineage = tmp_path / "fit"
    lineage.mkdir()
    first = _make_source(tmp_path / "labels-a", "chronos2", lineage)
    second = _make_source(
        tmp_path / "labels-b",
        "tirex",
        lineage,
        config={"schema_version": 1, "seed": 99},
    )
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="config mismatch"):
        merge_label_artifacts([first, second], output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("episodes", "overrides", "message"),
    (
        (("episode-a",), {}, "episode compatibility mismatch"),
        (
            ("episode-a", "episode-b"),
            {"episode-b": ("dataset-b", "family-b")},
            "dataset compatibility mismatch",
        ),
    ),
)
def test_merge_rejects_episode_or_dataset_mismatch(
    tmp_path,
    episodes,
    overrides,
    message,
):
    lineage = tmp_path / "fit"
    lineage.mkdir()
    first = _make_source(tmp_path / "labels-a", "chronos2", lineage)
    second = _make_source(
        tmp_path / "labels-b",
        "tirex",
        lineage,
        episodes=episodes,
        dataset_overrides=overrides,
    )

    with pytest.raises(ValueError, match=message):
        merge_label_artifacts([first, second], tmp_path / "merged")


def test_merge_rejects_duplicate_natural_key(tmp_path):
    lineage = tmp_path / "fit"
    lineage.mkdir()
    first = _make_source(
        tmp_path / "labels-a",
        "chronos2",
        lineage,
        duplicate_unary=True,
    )
    second = _make_source(tmp_path / "labels-b", "tirex", lineage)

    with pytest.raises(ValueError, match="duplicate unary label key"):
        merge_label_artifacts([first, second], tmp_path / "merged")


def test_merge_rejects_existing_output(tmp_path):
    lineage = tmp_path / "fit"
    lineage.mkdir()
    first = _make_source(tmp_path / "labels-a", "chronos2", lineage)
    second = _make_source(tmp_path / "labels-b", "tirex", lineage)
    output = tmp_path / "merged"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        merge_label_artifacts([first, second], output)
