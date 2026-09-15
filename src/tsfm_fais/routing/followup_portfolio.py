"""Rebuild the frozen portfolio representation from deployment-visible inputs."""

import numpy as np
import pandas as pd

from .aligned_portfolio import build_option_features
from .forecast_response import FORECAST_FEATURES
from .preforecast import METADATA, STATIC_FEATURES


def portfolio_feature_frame(candidate_frame, point_z, queried_actions, last_locf_z, *, joint):
    """Return the 35 actual-median options in the development feature order."""
    actions = tuple(sorted(queried_actions))
    if len(actions) != 7 or len(set(actions)) != 7:
        raise ValueError("seven uniquely identified forecasts are required")
    point_z, last_locf_z = np.asarray(point_z, float), np.asarray(last_locf_z, float)
    if point_z.shape != (7, 96, 2) or last_locf_z.shape != (2,):
        raise ValueError("the frozen two-target, 96-step setting changed")
    points = point_z[[queried_actions.index(name) for name in actions]]
    slots = [-1] if joint else [0, 1]
    vectors, statics, metadata, last = [], [], [], []
    for slot in slots:
        frame = candidate_frame[candidate_frame.target_slot == slot]
        if (
            frame.episode_id.nunique() != 1
            or frame.candidate_id.duplicated().any()
            or set(frame.candidate_id) != set(actions)
        ):
            raise ValueError("each deployment decision requires its complete candidate pool")
        frame = frame.set_index("candidate_id").loc[list(actions)]
        statics.append(frame[list(STATIC_FEATURES)].to_numpy(float))
        vectors.append(points.reshape(7, -1) if joint else points[:, :, slot])
        last.append(last_locf_z if joint else last_locf_z[[slot]])
        metadata.append(
            {
                name: frame.iloc[0][name]
                for name in METADATA
                if name in frame and name != "candidate_id"
            }
        )
    features, options, names = build_option_features(
        np.stack(vectors),
        np.stack(statics),
        np.stack(last),
        actions,
        horizon=96,
        targets=2 if joint else 1,
    )
    columns = [*FORECAST_FEATURES, *["member." + name for name in actions]]
    values = pd.DataFrame(features[:, 7:42].reshape(-1, 40), columns=columns)
    decision_metadata = pd.DataFrame(metadata)
    expanded = decision_metadata.iloc[np.repeat(np.arange(len(slots)), 35)].reset_index(drop=True)
    expanded["candidate_id"] = np.tile(names[7:42], len(slots))
    return pd.concat([expanded, values], axis=1), options[:, 7:42], decision_metadata, names[7:42]


def selected_portfolio_points(choices, options, decisions, option_names, *, joint):
    by_episode = choices.set_index("episode_id")
    if (
        len(choices) != len(decisions)
        or choices.episode_id.duplicated().any()
        or set(choices.episode_id) != set(decisions.episode_id)
    ):
        raise ValueError("each target or context needs one unambiguous portfolio choice")
    selected = np.stack(
        [
            options[index, option_names.index(by_episode.loc[row.episode_id, "candidate_id"])]
            for index, row in enumerate(decisions.itertuples(index=False))
        ]
    )
    return selected[0].reshape(96, 2) if joint else selected.T
