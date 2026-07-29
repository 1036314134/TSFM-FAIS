from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_fais.contracts import ForecastSpec, SeriesBatch, TimeSeriesItem
from tsfm_fais.imputers import DEFAULT_REGISTRY, ImputerRegistry
from tsfm_fais.stage_execution import (
    _ImputeEpisodeWork,
    _load_reused_actual_candidates,
    _validate_candidate_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(root: Path) -> tuple[Path, _ImputeEpisodeWork, object, ImputerRegistry]:
    source = root / "source"
    episode_path = source / "imputations" / "toy" / "00000000.npz"
    episode_path.parent.mkdir(parents=True)
    clean = np.arange(4, dtype=float)[:, None]
    observed = np.ones_like(clean, dtype=bool)
    observed[1, 0] = False
    candidate = clean.copy()
    candidate[1, 0] = 0.0
    np.savez_compressed(
        episode_path,
        episode_id=np.asarray(["episode"]),
        dataset_id=np.asarray(["toy"]),
        family_id=np.asarray(["family"]),
        item_id=np.asarray(["item"]),
        mask_protocol=np.asarray(["sequence_mask_v2"]),
        mask_seed=np.asarray([17]),
        mask_realization_id=np.asarray(["mask"]),
        observed_mask=observed,
        clean_context=clean,
        clean_future=np.asarray([[4.0]]),
        candidate_ids=np.asarray(["locf"]),
        candidate_values=candidate[None, ...],
        candidate_native_valid=np.ones((1, *clean.shape), dtype=bool),
        candidate_status=np.asarray(["success"]),
        candidate_runtime_seconds=np.asarray([0.1]),
        candidate_peak_memory_bytes=np.asarray([10]),
    )
    assignments = source / "routing_assignments.jsonl"
    assignments.write_text("{}\n", encoding="utf-8")
    identity = {"schema_version": 1, "token": "same"}
    manifest = source / "imputation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "episode_count": 1,
                "save_all_candidate_outputs": True,
                "candidate_generation_identity": identity,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progress = {
        "status": "completed",
        "expected_episode_count": 1,
        "imputation_manifest_sha256": _sha256(manifest),
        "routing_assignments_sha256": _sha256(assignments),
        "entries": {
            "00000000": {
                "file": "toy\\00000000.npz",
                "episode_id": "episode",
                "npz_sha256": _sha256(episode_path),
            }
        },
    }
    (source / "imputation_progress.json").write_text(
        json.dumps(progress) + "\n",
        encoding="utf-8",
    )
    batch = SeriesBatch(values=clean[None, ...], observed_mask=observed[None, ...])
    item = TimeSeriesItem(
        item_id="item",
        values=clean,
        variate_names=("x",),
        start=pd.Timestamp("2020-01-01"),
        freq="h",
    )
    episode = SimpleNamespace(
        item_id="item",
        mask_seed=17,
        mask_realization_id="mask",
        clean_context=clean,
        clean_future=np.asarray([[4.0]]),
    )
    work = _ImputeEpisodeWork(
        index=0,
        episode_id="episode",
        episode=episode,
        item=item,
        spec=ForecastSpec(
            model_id="imputation",
            mode="joint_multivariate",
            horizon=1,
            target_indices=(0,),
        ),
        relative=Path("toy") / "00000000.npz",
        relative_assignment=Path("assignment_records") / "toy" / "00000000.json",
        entry_key="00000000",
        invalid_reason=None,
        pipeline_rss_before=0,
        mase_scale=np.asarray([1.0]),
        mase_scale_lag=1,
        plan=SimpleNamespace(batch=batch),
    )
    dataset = SimpleNamespace(dataset_id="toy", family_id="family")
    registry = ImputerRegistry((DEFAULT_REGISTRY.get_spec("locf"),))
    return source, work, dataset, registry


def test_candidate_source_binds_generation_identity_and_each_npz(tmp_path: Path) -> None:
    source, work, dataset, registry = _source_fixture(tmp_path)
    summary, entries = _validate_candidate_source(
        source,
        expected_generation_identity={"schema_version": 1, "token": "same"},
    )
    assert summary["episode_count"] == 1
    results = _load_reused_actual_candidates(source, work, dataset, registry, entries)
    assert tuple(results) == ("locf",)

    episode_path = source / "imputations" / "toy" / "00000000.npz"
    with episode_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="NPZ hash differs"):
        _load_reused_actual_candidates(source, work, dataset, registry, entries)


def test_candidate_source_rejects_different_generation_identity(tmp_path: Path) -> None:
    source, _work, _dataset, _registry = _source_fixture(tmp_path)
    with pytest.raises(ValueError, match="generation identity differs"):
        _validate_candidate_source(
            source,
            expected_generation_identity={"schema_version": 1, "token": "different"},
        )
