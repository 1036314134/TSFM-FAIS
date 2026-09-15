"""Time-local input changes with explicit forecasting-dependency scope."""

import numpy as np

from .preforecast import METADATA, STATIC_FEATURES

BINS = 6
TARGET_STATS = (
    "input_mean",
    "input_std",
    "delta_mean",
    "delta_rms",
    "delta_max",
    "original_missing",
    "effective_missing",
)
COVARIATE_STATS = ("delta_mean", "delta_rms", "delta_max", "original_missing", "effective_missing")
TARGET_FEATURES = tuple(
    f"static.target_bin{index}.{name}" for index in range(BINS) for name in TARGET_STATS
)
COVARIATE_FEATURES = tuple(
    f"static.covariate_bin{index}.{name}" for index in range(BINS) for name in COVARIATE_STATS
) + (
    "static.dependency.correlation_shift_mean",
    "static.dependency.correlation_shift_max",
    "static.dependency.weighted_change",
    "static.dependency.valid_fraction",
    "static.dependency.enabled",
)
EXTRA_FEATURES = TARGET_FEATURES + COVARIATE_FEATURES
ALL_FEATURES = STATIC_FEATURES + EXTRA_FEATURES


def structured_inputs(frame, *, view="full_dependency"):
    if view not in {"target_temporal", "full_dependency"}:
        raise ValueError("unknown structured feature view")
    unknown = {name for name in frame if name.startswith("static.")} - set(ALL_FEATURES)
    if unknown or not set(ALL_FEATURES) <= set(frame):
        raise ValueError("structured inputs require the complete audited feature list")
    result = frame[[name for name in METADATA if name in frame] + list(ALL_FEATURES)].copy()
    if view == "target_temporal":
        result.loc[:, list(COVARIATE_FEATURES)] = 0.0
    return result


def target_correlations(values, targets, *, min_pairs=8):
    """Pairwise-complete Pearson coefficients; invalid pairs are explicitly flagged."""
    values = np.asarray(values, dtype=float)
    coefficients = np.zeros((len(targets), values.shape[1]))
    available = np.zeros_like(coefficients, dtype=bool)
    for slot, target in enumerate(targets):
        valid = np.isfinite(values[:, target, None]) & np.isfinite(values)
        count = valid.sum(axis=0)
        x = np.where(valid, values[:, target, None], 0.0)
        y = np.where(valid, values, 0.0)
        mean_x = x.sum(axis=0) / np.maximum(count, 1)
        mean_y = y.sum(axis=0) / np.maximum(count, 1)
        x = np.where(valid, x - mean_x, 0.0)
        y = np.where(valid, y - mean_y, 0.0)
        denominator = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
        usable = (count >= min_pairs) & (denominator > 1e-12)
        coefficients[slot] = np.divide(
            (x * y).sum(axis=0), denominator, out=np.zeros(values.shape[1]), where=usable
        ).clip(-1, 1)
        available[slot] = usable
    return coefficients, available


def input_change_features(context_z, effective_z, reference_z, targets, *, joint):
    context = np.asarray(context_z, float)
    effective, reference = np.asarray(effective_z, float), np.asarray(reference_z, float)
    targets = np.asarray(tuple(targets), dtype=int)
    if (
        context.ndim != 2
        or effective.shape != context.shape
        or reference.shape != context.shape
        or len(context) < BINS
        or not np.isfinite(reference).all()
        or np.isinf(context).any()
        or np.isinf(effective).any()
        or not len(targets)
        or len(set(targets)) != len(targets)
        or targets.min() < 0
        or targets.max() >= context.shape[1]
    ):
        raise ValueError("aligned historical inputs and distinct forecast targets are required")
    other = np.setdiff1d(np.arange(context.shape[1]), targets) if joint else np.array([], dtype=int)
    original_missing, effective_missing = ~np.isfinite(context), ~np.isfinite(effective)
    delta = np.where(effective_missing, 0.0, effective - reference)
    features = {}
    for index, times in enumerate(np.array_split(np.arange(len(context)), BINS)):
        for group, columns, names in (
            ("target", targets, TARGET_STATS),
            ("covariate", other, COVARIATE_STATS),
        ):
            stats = dict.fromkeys(names, 0.0)
            if len(columns):
                change = delta[np.ix_(times, columns)]
                stats.update(
                    delta_mean=float(change.mean()),
                    delta_rms=float(np.sqrt(np.mean(change**2))),
                    delta_max=float(np.abs(change).max()),
                    original_missing=float(original_missing[np.ix_(times, columns)].mean()),
                    effective_missing=float(effective_missing[np.ix_(times, columns)].mean()),
                )
                if group == "target":
                    block = effective[np.ix_(times, columns)]
                    mask = np.isfinite(block)
                    count = mask.sum(axis=0)
                    means = np.where(mask, block, 0).sum(axis=0) / np.maximum(count, 1)
                    residual = np.where(mask, block - means, 0.0)
                    variances = (residual**2).sum(axis=0) / np.maximum(count, 1)
                    stats.update(
                        input_mean=float(means.mean()), input_std=float(np.sqrt(variances).mean())
                    )
            features.update({f"static.{group}_bin{index}.{name}": stats[name] for name in names})
    enabled = bool(joint and context.shape[1] > 1)
    dependency = dict(
        correlation_shift_mean=0.0,
        correlation_shift_max=0.0,
        weighted_change=0.0,
        valid_fraction=0.0,
        enabled=float(enabled),
    )
    if enabled:
        current, current_valid = target_correlations(effective, targets)
        anchor, anchor_valid = target_correlations(reference, targets)
        eligible = np.ones_like(current_valid)
        eligible[np.arange(len(targets)), targets] = False
        valid = current_valid & anchor_valid & eligible
        dependency["valid_fraction"] = float(valid.sum() / eligible.sum())
        if valid.any():
            shift = np.abs(current - anchor)[valid]
            rms = np.sqrt(np.mean(delta**2, axis=0))
            dependency.update(
                correlation_shift_mean=float(shift.mean()),
                correlation_shift_max=float(shift.max()),
                weighted_change=float((np.abs(current) * rms[None])[valid].mean()),
            )
    features.update({f"static.dependency.{name}": value for name, value in dependency.items()})
    if set(features) != set(EXTRA_FEATURES) or not np.isfinite(list(features.values())).all():
        raise ValueError("invalid structured input features")
    return {name: float(np.clip(features[name], -1e8, 1e8)) for name in EXTRA_FEATURES}
