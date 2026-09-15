"""Frozen forecast-head projection and integrity-checked source inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def projection_matrix(model_id):
    dimension, seed = (3072, 9521) if model_id == "chronos2" else (2560, 9522)
    return (np.random.default_rng(seed).standard_normal((dimension, 32)) / np.sqrt(32)).astype(
        np.float32
    )


def project_heads(model_id, heads, matrix):
    if model_id == "chronos2":
        if len(heads) != 1 or heads[0].shape != (2, 6, 768):
            raise ValueError("Chronos must preserve two targets and six future patches")
        raw = np.stack([heads[0][:, :3].mean(1), heads[0][:, 3:].mean(1)], axis=1).reshape(1, 3072)
    else:
        if len(heads) != 2 or any(value.shape != (2, 1280) for value in heads):
            raise ValueError("TimesFM must preserve two sign branches and two target rows")
        raw = np.concatenate(heads, axis=1)
    result = raw.astype(np.float32) @ matrix
    if (
        result.shape != ((1, 32) if model_id == "chronos2" else (2, 32))
        or not np.isfinite(result).all()
    ):
        raise ValueError("projected head inputs are incomplete")
    return result


def combine_features(base, projected, locf_index):
    base, projected = np.asarray(base, np.float32), np.asarray(projected, np.float32)
    if base.ndim != 3 or base.shape[1:] != (7, 33) or projected.shape != (*base.shape[:2], 32):
        raise ValueError("candidate features and projected representations must align")
    result = np.concatenate(
        [base, projected, projected - projected[:, locf_index : locf_index + 1]], axis=2
    )
    if not np.isfinite(result).all():
        raise ValueError("a source feature is nonfinite")
    return np.ascontiguousarray(result, dtype=np.float32)


def load_source_inputs(root, model_id):
    parent = read_json(root / "manifest.json")
    entry = next(row for row in parent["models"] if row["model_id"] == model_id)
    if parent["status"] != "completed" or file_sha256(root / entry["path"]) != entry["sha256"]:
        raise ValueError("the completed predictor collection changed")
    manifest = read_json(root / model_id / "manifest.json")
    if (
        manifest["status"] != "completed"
        or manifest["identity_sha256"] != parent["identity_sha256"]
        or len(manifest["episodes"]) != 3906
    ):
        raise ValueError("complete the fixed source collection before fitting")
    frames, arrays = [], {key: [] for key in ("features", "vectors", "teacher")}
    for record in manifest["episodes"]:
        path = root / model_id / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a collected source decision changed")
        with np.load(path, allow_pickle=False) as saved:
            frames.append(pd.DataFrame(json.loads(str(saved["decisions"]))))
            for key in arrays:
                arrays[key].append(saved[key])
    frame = pd.concat(frames, ignore_index=True)
    combined = {key: np.concatenate(parts) for key, parts in arrays.items()}
    if (
        frame.episode_id.duplicated().any()
        or frame.source_episode_id.nunique() != 3906
        or frame.origin_id.nunique() != 217
        or frame.family_id.nunique() != 15
    ):
        raise ValueError("the prespecified source population changed")
    for split, expected in (("train", (165, 2970)), ("validation", (52, 936))):
        group = frame[frame.split == split]
        if (group.origin_id.nunique(), group.source_episode_id.nunique()) != expected:
            raise ValueError("source split coverage changed")
    return manifest, frame, combined
