"""Build the established forecast-gate features for seven or eight candidates."""

import json

import numpy as np
import pandas as pd
from latent_source_inputs import read_json
from replay_preforecast_student import extend_forecast_features

from tsfm_fais.routing.forecast_response import FORECAST_FEATURES, forecast_response_inputs
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame
from tsfm_fais.utility_experiment import file_sha256


def pool_inputs(
    context, candidates, actions, coverage, points, mean, scale, *, joint, period, metadata
):
    rows = candidate_feature_frame(
        context,
        candidates,
        actions,
        coverage,
        mean,
        scale,
        [0, 1],
        joint=joint,
        period=period,
        metadata=metadata,
    )
    last = (candidates[actions.index("locf"), -1, :2] - mean[:2]) / scale[:2]
    rows = forecast_response_inputs(extend_forecast_features(rows, points, actions, last))
    names = sorted([*actions, "guarded_direct"])
    if len(names) not in (7, 8):
        raise ValueError("the registered forecast pool has seven or eight candidates")
    decisions = (
        rows.drop_duplicates("episode_id")
        .drop(columns=["candidate_id", *FORECAST_FEATURES])
        .sort_values("target_slot")
        .reset_index(drop=True)
    )
    order = pd.MultiIndex.from_product(
        [decisions.episode_id, names], names=["episode_id", "candidate_id"]
    )
    features = (
        rows.set_index(["episode_id", "candidate_id"])
        .loc[order, list(FORECAST_FEATURES)]
        .to_numpy(np.float32)
        .reshape(len(decisions), len(names), 33)
    )
    queried = [*actions, "guarded_direct"]
    bank = np.asarray(points)[[queried.index(name) for name in names]]
    vectors = np.stack(
        [
            bank.reshape(len(names), -1) if slot == -1 else bank[:, :, slot]
            for slot in decisions.target_slot
        ]
    )
    if not np.isfinite(features).all() or not np.isfinite(vectors).all():
        raise ValueError("the forecast-pool representation is incomplete")
    return decisions, np.ascontiguousarray(features), np.ascontiguousarray(vectors), names


def motm_coverage(context, fallback_columns):
    missing = ~np.isfinite(context)
    if not missing.any():
        return 1.0
    covered = missing.copy()
    covered[:, list(fallback_columns)] = False
    return float(covered.sum() / missing.sum())


def load_pool_inputs(root, model_id):
    parent = read_json(root / "manifest.json")
    entry = next(row for row in parent["models"] if row["model_id"] == model_id)
    if parent["status"] != "completed" or file_sha256(root / entry["path"]) != entry["sha256"]:
        raise ValueError("the completed pool collection changed")
    manifest = read_json(root / entry["path"])
    if (
        manifest["status"] != "completed"
        or manifest["identity_sha256"] != parent["identity_sha256"]
        or len(manifest["episodes"]) != 3906
    ):
        raise ValueError("the registered pool population or identity changed")
    frames, features, vectors = [], [], []
    for row in manifest["episodes"]:
        path = root / model_id / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("a pool decision changed")
        with np.load(path, allow_pickle=False) as saved:
            frames.append(pd.DataFrame(json.loads(str(saved["decisions"]))))
            features.append(np.pad(saved["features"], ((0, 0), (0, 0), (0, 64))))
            vectors.append(saved["vectors"])
    frame = pd.concat(frames, ignore_index=True)
    if (
        frame.source_episode_id.nunique() != 3906
        or frame.origin_id.nunique() != 217
        or frame[frame.split == "train"].origin_id.nunique() != 165
        or frame[frame.split == "validation"].origin_id.nunique() != 52
        or frame.episode_id.duplicated().any()
    ):
        raise ValueError("the original source population changed")
    return (
        manifest,
        frame,
        {"features": np.concatenate(features), "vectors": np.concatenate(vectors)},
    )
