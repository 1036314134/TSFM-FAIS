from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm_fais.cli import main
from tsfm_fais.label_artifacts import merge_label_artifacts
from tsfm_fais.label_resume import canonical_sha256


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
    labeled_episodes: tuple[str, ...] | None = None,
    dataset_overrides: dict[str, tuple[str, str]] | None = None,
    duplicate_unary: bool = False,
    block_prefix: str = "block",
) -> Path:
    root.mkdir()
    dataset_overrides = dataset_overrides or {}
    labeled_episodes = episodes if labeled_episodes is None else labeled_episodes
    if not set(labeled_episodes).issubset(episodes):
        raise ValueError("labeled episodes must be present in the sampling plan")
    unary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for episode_id in labeled_episodes:
        dataset_id, family_id = dataset_overrides.get(
            episode_id, ("dataset-a", "family-a")
        )
        block_ids = (f"{block_prefix}-0", f"{block_prefix}-1")
        for block_id, candidate_id in zip(
            block_ids,
            ("locf", "linear_interp"),
            strict=True,
        ):
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
                "left_block": block_ids[0],
                "right_block": block_ids[1],
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
                "experiment": {
                    "split": "rolling_origin",
                    "context_length": 48,
                    "horizon": 8,
                },
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
            "expected_episode_count": len(episodes),
            "labeled_episode_count": len(labeled_episodes),
            "no_label_episode_count": len(episodes) - len(labeled_episodes),
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
            "ranking_groups": 2 * len(labeled_episodes),
            "unary_rows": len(unary_rows),
            "pair_rows": len(pair_rows),
            "routing_target_protocol": "coherence_adjusted_marginal_v1",
            "selected_candidates": ["locf", "linear_interp"],
            "max_teacher_blocks_per_episode": 2,
            "max_teacher_candidates_per_episode": 2,
            "max_pair_labels_per_episode": 1,
            "csdi_num_samples": 5,
        },
    )
    plans: dict[str, dict[str, object]] = {}
    for episode_id in episodes:
        dataset_id, _ = dataset_overrides.get(
            episode_id, ("dataset-a", "family-a")
        )
        plan = plans.setdefault(
            dataset_id,
            {
                "dataset_id": dataset_id,
                "selection_summary": {
                    "partition": "train",
                    "selected_episode_count": 0,
                },
                "episode_ids": [],
            },
        )
        plan["episode_ids"].append(episode_id)  # type: ignore[union-attr]
        plan["selection_summary"]["selected_episode_count"] += 1  # type: ignore[index,operator]
    for plan in plans.values():
        plan["sha256"] = canonical_sha256(plan)
    entries: dict[str, dict[str, object]] = {}
    for artifact_index, episode_id in enumerate(episodes):
        dataset_id, family_id = dataset_overrides.get(
            episode_id, ("dataset-a", "family-a")
        )
        block_ids = [f"{block_prefix}-0", f"{block_prefix}-1"]
        expectation = {
            "artifact_index": artifact_index,
            "forecaster_id": model_id,
            "episode_id": episode_id,
            "dataset_id": dataset_id,
            "family_id": family_id,
            "item_id": "item-a",
            "forecast_origin": 48 + artifact_index,
            "sampling_cell": {"mechanism": "independent_block", "rate": 0.2},
            "dataset_plan_sha256": plans[dataset_id]["sha256"],
            "candidate_ids": ["locf", "linear_interp"],
            "block_ids": block_ids,
        }
        outcome = "labeled" if episode_id in labeled_episodes else "no_labels"
        entries[f"{artifact_index:08d}"] = {
            "artifact_index": artifact_index,
            "expectation": expectation,
            "outcome": outcome,
            "sidecar_file": f"label_episode_records/{artifact_index:08d}.json",
            "sidecar_sha256": "0" * 64,
            "unary_rows": 2 if outcome == "labeled" else 0,
            "unary_rows_sha256": "0" * 64,
            "pair_rows": 1 if outcome == "labeled" else 0,
            "pair_rows_sha256": "0" * 64,
            "ranking_groups": 2 if outcome == "labeled" else 0,
        }
    _write_json(
        root / "labels_progress.json",
        {
            "schema_version": 1,
            "status": "rebuilt",
            "dataset_plans": plans,
            "entries": entries,
            "completed_count": len(entries),
            "unary_rows": len(unary_rows),
            "pair_rows": len(pair_rows),
            "ranking_groups": 2 * len(labeled_episodes),
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
    assert summary["routing_target_protocol"] == "coherence_adjusted_marginal_v1"
    assert summary["split"] == "rolling_origin"
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


def test_merge_accepts_forecaster_specific_teacher_blocks(tmp_path):
    lineage = tmp_path / "fit" / "imputer_artifacts"
    lineage.mkdir(parents=True)
    chronos = _make_source(
        tmp_path / "labels-chronos",
        "chronos2",
        lineage,
        block_prefix="joint-block",
    )
    timesfm = _make_source(
        tmp_path / "labels-timesfm",
        "timesfm2p5",
        lineage,
        block_prefix="univariate-block",
    )

    summary = merge_label_artifacts(
        [chronos, timesfm],
        tmp_path / "merged-model-specific-blocks",
    )

    assert summary["forecasters"] == ["chronos2", "timesfm2p5"]
    assert summary["ranking_groups"] == 8


def test_merge_accepts_forecaster_specific_no_label_episodes(tmp_path):
    lineage = tmp_path / "fit" / "imputer_artifacts"
    lineage.mkdir(parents=True)
    chronos = _make_source(tmp_path / "labels-chronos", "chronos2", lineage)
    timesfm = _make_source(
        tmp_path / "labels-timesfm",
        "timesfm2p5",
        lineage,
        labeled_episodes=("episode-a",),
    )

    summary = merge_label_artifacts(
        [chronos, timesfm],
        tmp_path / "merged-model-specific-outcomes",
    )

    assert summary["episode_count"] == 2
    assert summary["labeled_episode_counts"] == {
        "chronos2": 2,
        "timesfm2p5": 1,
    }
    assert summary["no_label_episode_counts"] == {
        "chronos2": 0,
        "timesfm2p5": 1,
    }
    assert summary["unary_rows"] == 6
    assert summary["pair_rows"] == 3


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
        (("episode-a",), {}, "episode plan compatibility mismatch"),
        (
            ("episode-a", "episode-b"),
            {"episode-b": ("dataset-a", "family-b")},
            "episode metadata compatibility mismatch",
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
