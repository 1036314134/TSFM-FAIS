"""Validate the full on-disk teacher cache across interrupted training processes."""

import hashlib

import torch


def collect_teacher_records(root, expected_ids, identity_sha256, forecaster_digest, shape):
    records, seen = [], set()
    for path in sorted(root.glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        episode = saved["episode_id"]
        prediction = saved["prediction"]
        if (
            episode not in expected_ids
            or episode in seen
            or path.name != hashlib.sha256(episode.encode()).hexdigest()[:24] + ".pt"
            or saved["identity_sha256"] != identity_sha256
            or saved["forecaster_digest"] != forecaster_digest
            or not isinstance(prediction, torch.Tensor)
            or tuple(prediction.shape) != tuple(shape)
            or prediction.requires_grad
            or not bool(torch.isfinite(prediction).all())
        ):
            raise ValueError("teacher cache identity, coverage or prediction is invalid")
        seen.add(episode)
        records.append(
            {
                "episode_id": episode,
                "path": str(path.relative_to(root.parent)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if seen != set(expected_ids):
        raise ValueError("teacher cache does not cover every recorded training episode")
    return sorted(records, key=lambda row: row["episode_id"])
