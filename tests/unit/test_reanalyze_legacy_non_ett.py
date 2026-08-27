from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "reanalyze_legacy_non_ett.py"
    spec = importlib.util.spec_from_file_location("reanalyze_legacy_non_ett", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evaluation(
    directory: Path,
    forecaster: str,
    method: str,
    method_mase: tuple[float, float],
) -> None:
    directory.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    episodes = (
        ("episode-a", "family-a", method_mase[0]),
        ("episode-b", "family-b", method_mase[1]),
        ("episode-ett", "ett", -1000.0),
    )
    for episode_id, family_id, mase in episodes:
        rows.append(
            {
                "episode_id": episode_id,
                "dataset_id": f"dataset-{family_id}",
                "family_id": family_id,
                "forecaster_id": forecaster,
                "item_id": f"item-{episode_id}",
                "method": method,
                "method_role": "method" if method == "b_fais" else "selector_baseline",
                "metric_eligible": True,
                "native_valid": True,
                "mase": mase,
            }
        )
        if method == "b_fais":
            for candidate, offset in (("candidate-a", 0.5), ("candidate-b", 1.0)):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "dataset_id": f"dataset-{family_id}",
                        "family_id": family_id,
                        "forecaster_id": forecaster,
                        "item_id": f"item-{episode_id}",
                        "method": candidate,
                        "method_role": "baseline",
                        "metric_eligible": True,
                        "native_valid": True,
                        "mase": mase + offset,
                    }
                )
            rows.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": f"dataset-{family_id}",
                    "family_id": family_id,
                    "forecaster_id": forecaster,
                    "item_id": f"item-{episode_id}",
                    "method": "oracle",
                    "method_role": "oracle",
                    "metric_eligible": True,
                    "native_valid": True,
                    "mase": mase - 1.0,
                }
            )
    metrics = directory / "episode_metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "forecaster_id": forecaster,
        "episode_metrics_jsonl": str(metrics.resolve()),
        "episode_metrics_jsonl_sha256": _sha256(metrics),
        "total_rows": len(rows),
    }
    (directory / "evaluation_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _teacher_labels(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for forecaster in MODULE.FORECASTERS:
        for episode, family, base in (
            ("teacher-a", "family-a", 1.0),
            ("teacher-b", "family-b", 2.0),
            ("teacher-ett", "ett", -1000.0),
        ):
            for candidate, offset in (("candidate-a", 0.0), ("candidate-b", 3.0)):
                for block in ("block-0", "block-1"):
                    rows.append(
                        {
                            "forecaster_id": forecaster,
                            "episode_id": episode,
                            "family_id": family,
                            "dataset_id": f"dataset-{family}",
                            "candidate_id": candidate,
                            "block_id": block,
                            "full_candidate_loss": base + offset,
                        }
                    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (path.parent / "labels_manifest.json").write_text(
        json.dumps({"artifact_type": "merged_teacher_labels"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_top_tail_mean_uses_largest_discrete_fraction() -> None:
    assert MODULE._top_tail_mean(range(1, 11), 0.90) == 10.0
    assert MODULE._top_tail_mean(range(1, 21), 0.90) == 19.5


def test_reanalysis_excludes_ett_and_is_reproducible(tmp_path: Path) -> None:
    evaluations: dict[tuple[str, str], Path] = {}
    values = {
        "b_fais": (2.0, 4.0),
        "metaod": (3.0, 3.0),
        "alors": (5.0, 5.0),
        "dselect1": (6.0, 6.0),
        "hybrid_lstm": (7.0, 7.0),
        "neuralucb": (8.0, 8.0),
        "random_valid_series": (9.0, 9.0),
    }
    for method in MODULE.SELECTOR_METHODS:
        for forecaster in MODULE.FORECASTERS:
            directory = tmp_path / "inputs" / f"{method}-{forecaster}"
            _write_evaluation(directory, forecaster, method, values[method])
            evaluations[(method, forecaster)] = directory
    teacher = tmp_path / "teacher_labels.jsonl"
    _teacher_labels(teacher)

    first = tmp_path / "output-a"
    second = tmp_path / "output-b"
    result = MODULE.run_reanalysis(
        artifacts_root=tmp_path / "unused",
        teacher_labels=teacher,
        output_dir=first,
        evaluation_dirs=evaluations,
    )
    MODULE.run_reanalysis(
        artifacts_root=tmp_path / "unused",
        teacher_labels=teacher,
        output_dir=second,
        evaluation_dirs=evaluations,
    )

    assert result["episode_count"] == 4
    assert result["family_count"] == 2
    assert result["strongest_selector"] == "metaod"
    assert result["robust_fixed"] == {
        "chronos2": "candidate-a",
        "timesfm2p5": "candidate-a",
    }
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_version"] == "p0-0-legacy-non-ett-v001"
    assert manifest["excluded_family"] == "ett"
    assert len(manifest["source_evaluations"]) == 14
    assert all(source["metrics_sha256"] for source in manifest["source_evaluations"])
    assert all(source["manifest_sha256"] for source in manifest["source_evaluations"])
    for source in manifest["source_evaluations"]:
        source_manifest = Path(source["manifest"])
        assert source["manifest_sha256"] == _sha256(source_manifest)
        assert source["manifest_size_bytes"] == source_manifest.stat().st_size
    assert manifest["teacher_labels"]["sha256"] == _sha256(teacher)
    sibling = teacher.parent / "labels_manifest.json"
    assert manifest["teacher_labels"]["sibling_manifest"] == str(sibling.resolve())
    assert manifest["teacher_labels"]["sibling_manifest_sha256"] == _sha256(sibling)
    assert manifest["teacher_labels"]["sibling_manifest_size_bytes"] == sibling.stat().st_size
    assert len(manifest["repository_commit"]) == 40
    assert isinstance(manifest["working_tree_dirty"], bool)
    assert {entry["path"] for entry in manifest["output_files"]} == set(manifest["outputs"])
    for entry in manifest["output_files"]:
        output = first / entry["path"]
        assert entry["size_bytes"] == output.stat().st_size
        assert entry["sha256"] == _sha256(output)

    comparisons = pd.read_csv(first / "paired_tail_summary.csv")
    strongest = comparisons[
        (comparisons["scope"] == "combined")
        & (comparisons["comparator_type"] == "strongest_selector")
    ].iloc[0]
    assert strongest["pair_count"] == 4
    assert strongest["family_stratum_count"] == 4
    assert strongest["family_macro_mean_delta"] == pytest.approx(0.0)
    families = pd.read_csv(first / "family_paired_summary.csv")
    assert "ett" not in set(families["family_id"])

    for name in manifest["outputs"] + ["manifest.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
