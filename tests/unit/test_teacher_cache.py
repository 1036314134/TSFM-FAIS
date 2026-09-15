import hashlib

import pytest
import torch

from tsfm_fais.routing.teacher_cache import collect_teacher_records


def save(root, episode, identity="identity"):
    path = root / (hashlib.sha256(episode.encode()).hexdigest()[:24] + ".pt")
    torch.save(
        {
            "episode_id": episode,
            "identity_sha256": identity,
            "forecaster_digest": "model",
            "prediction": torch.zeros(4, 2),
        },
        path,
    )


def test_disk_inventory_keeps_teachers_from_previous_processes(tmp_path):
    save(tmp_path, "previous_process")
    save(tmp_path, "current_process")
    rows = collect_teacher_records(
        tmp_path, {"previous_process", "current_process"}, "identity", "model", (4, 2)
    )
    assert len(rows) == 2
    assert {row["episode_id"] for row in rows} == {"previous_process", "current_process"}


def test_missing_or_mismatched_teacher_is_rejected(tmp_path):
    save(tmp_path, "a")
    with pytest.raises(ValueError, match="cover"):
        collect_teacher_records(tmp_path, {"a", "b"}, "identity", "model", (4, 2))
    save(tmp_path, "b", identity="another_run")
    with pytest.raises(ValueError, match="identity"):
        collect_teacher_records(tmp_path, {"a", "b"}, "identity", "model", (4, 2))
