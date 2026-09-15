"""Describe each forecast target while retaining the actual backbone fallback rules."""


import numpy as np
import pandas as pd
from latent_source_inputs import read_json
from replay_preforecast_student import extend_forecast_features

from tsfm_fais.routing.forecast_response import FORECAST_FEATURES, forecast_response_inputs
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame
from tsfm_fais.utility_experiment import file_sha256


def target_features(
    context,
    candidates,
    actions,
    coverage,
    point_z,
    mean,
    scale,
    *,
    backbone_joint,
    period,
    metadata,
):
    rows = candidate_feature_frame(
        context,
        candidates,
        actions,
        coverage,
        mean,
        scale,
        [0, 1],
        joint=False,
        period=period,
        metadata=metadata,
    )
    if backbone_joint:
        rows.loc[rows.candidate_id == "guarded_direct", "static.fallback_target_fraction"] = float(
            (~np.isfinite(context).any(axis=0)).any()
        )
    last = (candidates[actions.index("locf"), -1, :2] - mean[:2]) / scale[:2]
    rows = forecast_response_inputs(extend_forecast_features(rows, point_z, actions, last))
    ordered = sorted([*actions, "guarded_direct"])
    decisions = (
        rows.drop_duplicates("episode_id")
        .drop(columns=["candidate_id", *FORECAST_FEATURES])
        .sort_values("target_slot")
        .reset_index(drop=True)
    )
    order = pd.MultiIndex.from_product(
        [decisions.episode_id, ordered], names=["episode_id", "candidate_id"]
    )
    features = (
        rows.set_index(["episode_id", "candidate_id"])
        .loc[order, list(FORECAST_FEATURES)]
        .to_numpy(np.float32)
        .reshape(2, 7, 33)
    )
    if decisions.target_slot.tolist() != [0, 1] or not np.isfinite(features).all():
        raise ValueError("the target-local input order or coverage changed")
    return decisions, np.ascontiguousarray(features)


def split_joint_vectors(values):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1] != 7 or values.shape[2] % 2:
        raise ValueError("joint candidate vectors must preserve two interleaved targets")
    return np.ascontiguousarray(
        values.reshape(len(values), 7, -1, 2).transpose(0, 3, 1, 2).reshape(2 * len(values), 7, -1)
    )


def load_target_local(root):
    manifest = read_json(root / "manifest.json")
    if (
        manifest["status"] != "completed"
        or file_sha256(root / "decisions.parquet") != manifest["decisions_sha256"]
        or file_sha256(root / "arrays.npz") != manifest["arrays_sha256"]
    ):
        raise ValueError("target-local source preparation changed")
    frame = pd.read_parquet(root / "decisions.parquet")
    with np.load(root / "arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if (
        len(frame) != 7812
        or frame.source_episode_id.nunique() != 3906
        or frame.origin_id.nunique() != 217
        or frame.episode_id.duplicated().any()
    ):
        raise ValueError("target-local source population changed")
    np.testing.assert_array_equal(frame.target_slot.to_numpy(), np.tile([0, 1], 3906))
    return manifest, frame, arrays
