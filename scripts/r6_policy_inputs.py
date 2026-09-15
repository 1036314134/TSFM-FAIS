"""Generate the unchanged visible decision representations at either R6 horizon."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_preforecast_student import extend_forecast_features  # noqa: E402

from tsfm_fais.routing.aligned_portfolio import build_option_features  # noqa: E402
from tsfm_fais.routing.forecast_response import (  # noqa: E402
    FORECAST_FEATURES,
    forecast_response_inputs,
)
from tsfm_fais.routing.preforecast import METADATA, STATIC_FEATURES  # noqa: E402
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame  # noqa: E402


def pack_gate_features(values):
    """Use one float32 memory layout for module and independent matrix inference."""
    result = np.ascontiguousarray(values, dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (7, 33) or not np.isfinite(result).all():
        raise ValueError("finite seven-candidate, 33-feature inputs are required")
    return result


def decision_inputs(
    context, candidates, actions, coverage, points, mean, scale, *, joint, period, metadata
):
    points = np.asarray(points, float)
    if points.shape not in ((7, 96, 2), (7, 192, 2)) or not np.isfinite(points).all():
        raise ValueError("the two-target forecast bank must have a registered horizon")
    horizon = points.shape[1]
    queried = [*actions, "guarded_direct"]
    sorted_actions = tuple(sorted(queried))
    if len(set(sorted_actions)) != 7:
        raise ValueError("the original seven candidate identities are required")
    frame = candidate_feature_frame(
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
    individual = forecast_response_inputs(extend_forecast_features(frame, points, actions, last))
    bank = points[[queried.index(name) for name in sorted_actions]]
    vectors, static, records, last_values = [], [], [], []
    for slot in [-1] if joint else [0, 1]:
        rows = (
            individual[individual.target_slot == slot]
            .set_index("candidate_id")
            .loc[list(sorted_actions)]
        )
        if len(rows) != 7 or rows.episode_id.nunique() != 1:
            raise ValueError("one complete candidate set is required for each decision")
        static.append(rows[list(STATIC_FEATURES)].to_numpy(float))
        records.append(
            {
                name: rows.iloc[0][name]
                for name in METADATA
                if name in rows and name != "candidate_id"
            }
        )
        vectors.append(bank.reshape(7, -1) if joint else bank[:, :, slot])
        last_values.append(last if joint else last[[slot]])
    vectors = np.stack(vectors)
    features, options, names = build_option_features(
        vectors,
        np.stack(static),
        np.stack(last_values),
        sorted_actions,
        horizon=horizon,
        targets=2 if joint else 1,
    )
    decisions = pd.DataFrame(records)
    expanded = decisions.iloc[np.repeat(np.arange(len(decisions)), 35)].reset_index(drop=True)
    expanded["candidate_id"] = np.tile(names[7:42], len(decisions))
    columns = [*FORECAST_FEATURES, *["member." + name for name in sorted_actions]]
    portfolios = pd.concat(
        [expanded, pd.DataFrame(features[:, 7:42].reshape(-1, 40), columns=columns)], axis=1
    )
    gate_order = pd.MultiIndex.from_product(
        [decisions.episode_id, sorted_actions], names=["episode_id", "candidate_id"]
    )
    gate_features = (
        individual.set_index(["episode_id", "candidate_id"])
        .loc[gate_order, list(FORECAST_FEATURES)]
        .to_numpy(np.float32)
        .reshape(len(decisions), 7, 33)
    )
    gate_features = pack_gate_features(gate_features)
    return {
        "individual": individual,
        "portfolios": portfolios,
        "decisions": decisions,
        "vectors": vectors,
        "triple_vectors": options[:, 7:42],
        "triple_names": names[7:42],
        "gate_features": gate_features,
        "actions": sorted_actions,
    }


def restore_target_vectors(values, decisions, horizon, *, joint):
    values = np.asarray(values, float)
    slots = decisions.target_slot.to_numpy()
    if joint:
        if values.shape != (1, horizon * 2) or slots.tolist() != [-1]:
            raise ValueError("a joint decision must preserve both forecast targets")
        return values[0].reshape(horizon, 2)
    if values.shape != (2, horizon) or sorted(slots.tolist()) != [0, 1]:
        raise ValueError("independent decisions must cover both targets")
    return values[np.argsort(slots)].T
