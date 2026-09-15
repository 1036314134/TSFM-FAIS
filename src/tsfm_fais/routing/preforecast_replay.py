"""Regenerate pre-forecast features and assemble selected deployment inputs."""

import hashlib

import numpy as np
import pandas as pd

from .preforecast import METADATA, decision_keys, preforecast_inputs
from .utility import sequence_features


def candidate_feature_frame(
    context, candidates, actions, coverage, mean, scale, targets, *, joint, period, metadata
):
    context, candidates = np.asarray(context, float), np.asarray(candidates, float)
    mean, scale = np.asarray(mean, float), np.asarray(scale, float)
    if (
        candidates.shape[1:] != context.shape
        or len(candidates) != len(actions)
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("candidate contexts and identities must agree")
    locf = candidates[actions.index("locf")]
    empty = ~np.isfinite(context).any(axis=0)
    fallback = np.full(len(targets), empty.any()) if joint else empty[list(targets)]
    slots = (
        [(-1, list(targets))]
        if joint
        else [(slot, [target]) for slot, target in enumerate(targets)]
    )
    rows = []
    for slot, selected in slots:
        for index, action in enumerate([*actions, "guarded_direct"]):
            direct = action == "guarded_direct"
            features = sequence_features(
                context - mean,
                (locf if direct else candidates[index]) - mean,
                locf - mean,
                scale,
                selected,
                period=period,
                native_coverage=1.0 if direct else float(coverage[index]),
            )
            positions = list(range(len(targets))) if slot == -1 else [slot]
            features.update(
                {
                    "static.direct_missing": float(direct),
                    "static.empty_channel_fraction": float(empty.mean()),
                    "static.empty_target_fraction": float(empty[selected].mean()),
                    "static.fallback_target_fraction": float(fallback[positions].mean())
                    if direct
                    else 0.0,
                }
            )
            rows.append(
                {
                    **{name: metadata[name] for name in METADATA if name in metadata},
                    "target_slot": slot,
                    "candidate_id": action,
                    **features,
                }
            )
    return decision_keys(preforecast_inputs(pd.DataFrame(rows)))


def assemble_selected_context(context, candidates, actions, selected_actions, targets, *, joint):
    context, candidates = np.asarray(context, float), np.asarray(candidates, float)
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != context.shape
        or len(candidates) != len(actions)
    ):
        raise ValueError("complete candidates must have [A,L,D] shape")
    if not np.isfinite(candidates).all() or len(set(actions)) != len(actions):
        raise ValueError("finite, uniquely named candidates are required")
    if not set(selected_actions) <= set(actions) | {"guarded_direct"}:
        raise ValueError("selected action is outside the supported pool")
    locf = candidates[actions.index("locf")]
    empty = ~np.isfinite(context).any(axis=0)
    if joint:
        if len(selected_actions) != 1:
            raise ValueError("a joint forecast requires one complete context choice")
        action = selected_actions[0]
        result = (
            (locf if empty.any() else context).copy()
            if action == "guarded_direct"
            else candidates[actions.index(action)].copy()
        )
    else:
        if len(selected_actions) != len(targets):
            raise ValueError("each independent target requires a choice")
        result = locf.copy()
        for target, action in zip(targets, selected_actions, strict=True):
            if action == "guarded_direct":
                result[:, target] = locf[:, target] if empty[target] else context[:, target]
            else:
                result[:, target] = candidates[actions.index(action), :, target]
    observed = np.isfinite(context)
    np.testing.assert_array_equal(result[observed], context[observed])
    return result


def unique_forecaster_inputs(contexts, targets, *, joint):
    """Deduplicate the effective float32 inputs, keeping NaN masks in the key."""
    unique, reverse, seen = [], [], {}
    for context in contexts:
        effective = context if joint else context[:, list(targets)]
        canonical = np.ascontiguousarray(effective, dtype=np.float32)
        canonical[np.isnan(canonical)] = np.nan
        if np.isinf(canonical).any():
            raise ValueError("float32 forecaster input overflow")
        key = hashlib.sha256(canonical.tobytes()).hexdigest()
        if key not in seen:
            seen[key] = len(unique)
            unique.append(context)
        reverse.append(seen[key])
    return np.stack(unique), np.asarray(reverse)
