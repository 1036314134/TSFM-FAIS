"""Prefix-only peer selection and fixed HDB development inputs."""

import hashlib

import numpy as np
from dynamic_posterior_core import complete_raw, fit_dynamics, smooth_history
from sklearn.impute import KNNImputer

PREFIX, DEVELOPMENT_END, LENGTH, HORIZON = 336, 672, 192, 24
FILL_NAMES = ("median", "ffill", "linear", "seasonal24", "knn", "gaussian")


def eligible_columns(values):
    prefix = values[:PREFIX]
    good = []
    for column in range(prefix.shape[1]):
        observed = np.isfinite(prefix[:, column])
        if observed.sum() >= 256 and (observed[:-1] & observed[1:]).sum() >= 128:
            if np.std(prefix[observed, column]) > 1e-12:
                good.append(column)
    return good


def population(values, identifiers):
    if len(values) != DEVELOPMENT_END:
        raise ValueError("the experiment only accepts the frozen first 672 hours")
    eligible = eligible_columns(values)
    events, exclusions = {}, []
    for column in eligible:
        seen = np.isfinite(values[:, column])
        age, last, candidates = 0, -HORIZON, []
        for index, observed in enumerate(seen):
            age = 0 if observed else age + 1
            t = index + 1
            if t < PREFIX or t + HORIZON > DEVELOPMENT_END or age not in (6, 30, 54):
                continue
            reason = None
            if seen[t - LENGTH : t].sum() < 96:
                reason = "history_support"
            elif seen[t : t + HORIZON].sum() < 12:
                reason = "future_observed_support"
            elif t - last < HORIZON:
                reason = "overlapping_forecast"
            if reason:
                exclusions.append({"station": identifiers[column], "origin": t, "reason": reason})
                continue
            candidates.append({"origin": t, "outage_age": age})
            last = t
        if candidates:
            events[column] = candidates
    ordered = sorted(
        events,
        key=lambda c: (
            hashlib.sha256(f"r39|7401|{identifiers[c]}".encode()).hexdigest(),
            identifiers[c],
        ),
    )
    chosen, cases = ordered[:32], []
    for column in chosen:
        candidates = events[column]
        selected = np.linspace(0, len(candidates) - 1, min(2, len(candidates)), dtype=int)
        for index in selected:
            cases.append(
                {
                    "station": identifiers[column],
                    "column": column,
                    "panel": "natural_outage_h24",
                    **candidates[index],
                }
            )
        seen = np.isfinite(values[:, column])
        for t in (384, 552):
            for panel in ("native_grid_h24", "synthetic_outage_h24"):
                reason = None
                if seen[t - LENGTH : t].sum() < 96:
                    reason = "history_support"
                elif seen[t : t + HORIZON].sum() < 12:
                    reason = "future_observed_support"
                elif panel == "synthetic_outage_h24" and seen[t - 24 : t].sum() < 12:
                    reason = "artificial_tail_support"
                if reason:
                    exclusions.append(
                        {
                            "station": identifiers[column],
                            "origin": t,
                            "panel": panel,
                            "reason": reason,
                        }
                    )
                else:
                    cases.append(
                        {
                            "station": identifiers[column],
                            "column": column,
                            "panel": panel,
                            "origin": t,
                            "outage_age": 24 if panel == "synthetic_outage_h24" else 0,
                        }
                    )
    for case in cases:
        case.update(horizon=HORIZON, prefix_end=PREFIX)
        case["case_id"] = hashlib.sha256(
            f"r39|{case['panel']}|{case['station']}|{case['origin']}".encode()
        ).hexdigest()[:20]
    return cases, {
        "eligible_columns": eligible,
        "eligible_identifiers": [identifiers[c] for c in eligible],
        "event_candidates": [
            {"station": identifiers[c], "events": rows} for c, rows in events.items()
        ],
        "target_order": [identifiers[c] for c in ordered],
        "selected_columns": chosen,
        "selected_identifiers": [identifiers[c] for c in chosen],
        "exclusions": exclusions,
        "panels": {
            p: sum(r["panel"] == p for r in cases) for p in sorted({r["panel"] for r in cases})
        },
    }


def choose_peers(prefix, target, eligible, identifiers):
    if len(prefix) != PREFIX:
        raise ValueError("peer selection may only read the fitted prefix")
    candidates = []
    for column in eligible:
        if column == target:
            continue
        observed = np.isfinite(prefix[:, [target, column]]).all(1)
        if observed.sum() < 128:
            continue
        x, y = prefix[observed, target], prefix[observed, column]
        if min(np.std(x), np.std(y)) <= 1e-12:
            continue
        correlation = float(np.corrcoef(x, y)[0, 1])
        candidates.append(
            {
                "column": column,
                "station": identifiers[column],
                "correlation": correlation,
                "support": int(observed.sum()),
            }
        )
    candidates.sort(key=lambda row: (-abs(row["correlation"]), row["station"]))
    columns, decisions = [target], []
    for row in candidates:
        complete = np.isfinite(prefix[:, [*columns, row["column"]]]).all(1)
        pairs = complete[:-1] & complete[1:]
        accepted = len(columns) < 7 and min(complete.sum(), pairs.sum()) >= 128
        decisions.append(
            {
                **row,
                "accepted": bool(accepted),
                "joint_rows": int(complete.sum()),
                "joint_pairs": int(pairs.sum()),
            }
        )
        if accepted:
            columns.append(row["column"])
        if len(columns) == 7:
            break
    return columns, decisions


def hourly_statistics(prefix, median=False):
    fallback = np.nanmedian(prefix, axis=0) if median else np.nanmean(prefix, axis=0)
    result = np.empty((24, prefix.shape[1]))
    for hour in range(24):
        for column in range(prefix.shape[1]):
            values = prefix[hour::24, column]
            values = values[np.isfinite(values)]
            result[hour, column] = (
                (np.median(values) if median else np.mean(values))
                if len(values)
                else fallback[column]
            )
    return result


def periodic_forecast(context, origin, period, hourly_median):
    result = []
    for lead in range(HORIZON):
        index = len(context) + lead - period
        while index >= 0 and not np.isfinite(context[index, 0]):
            index -= period
        result.append(context[index, 0] if index >= 0 else hourly_median[(origin + lead) % 24, 0])
    return np.asarray(result)


def build_case(values, columns, row, model):
    t = row["origin"]
    if t + HORIZON > DEVELOPMENT_END or t < PREFIX:
        raise ValueError("a case crossed the registered development time boundary")
    prefix = np.asarray(values[:PREFIX, columns], float)
    context = np.array(values[t - LENGTH : t, columns], dtype=float, order="C")
    long = np.array(values[max(0, t - 336) : t, columns], dtype=float, order="C")
    if row["panel"] == "synthetic_outage_h24":
        context[-24:, 0], long[-24:, 0] = np.nan, np.nan
    mean, scale = model["mean"], model["scale"]
    z, prefix_z = (context - mean) / scale, (prefix - mean) / scale
    state = smooth_history(z, model)
    defaults = np.nanmedian(prefix, axis=0)
    hours = hourly_statistics(prefix, median=True)
    median = np.where(np.isfinite(context), context, defaults)
    forward, linear, seasonal = context.copy(), context.copy(), context.copy()
    for column in range(len(columns)):
        seen = np.flatnonzero(np.isfinite(context[:, column]))
        carry = defaults[column]
        for index in range(LENGTH):
            if np.isfinite(context[index, column]):
                carry = context[index, column]
            forward[index, column] = carry
            if not np.isfinite(seasonal[index, column]):
                previous = index - 24
                while previous >= 0 and not np.isfinite(context[previous, column]):
                    previous -= 24
                seasonal[index, column] = (
                    context[previous, column]
                    if previous >= 0
                    else hours[(t - LENGTH + index) % 24, column]
                )
        linear[:, column] = (
            np.interp(np.arange(LENGTH), seen, context[seen, column])
            if len(seen)
            else defaults[column]
        )
    knn_z = KNNImputer(n_neighbors=5, weights="uniform").fit(prefix_z).transform(z)
    fills = {
        "median": median,
        "ffill": forward,
        "linear": linear,
        "seasonal24": seasonal,
        "knn": complete_raw(context, knn_z, mean, scale),
        "gaussian": complete_raw(context, state["static_mean"], mean, scale),
    }
    observed = np.isfinite(context)
    for fill in fills.values():
        np.testing.assert_array_equal(fill[observed], context[observed])
        if not np.isfinite(fill).all():
            raise ValueError("incomplete fixed imputation")
    current, direct = state["filtered_mean"][-1].copy(), []
    for _ in range(HORIZON):
        current = model["a"] @ current + model["b"]
        direct.append(current[0])
    keep = np.isfinite(context).sum(0) >= 2
    keep[0] = True
    controls = {
        "linear_var_direct": np.asarray(direct),
        "seasonal24_direct": (periodic_forecast(context, t, 24, hours) - mean[0]) / scale[0],
        "seasonal168_direct": (periodic_forecast(context, t, 168, hours) - mean[0]) / scale[0],
        "hour_mean_direct": (hourly_statistics(prefix)[(t + np.arange(HORIZON)) % 24, 0] - mean[0])
        / scale[0],
    }
    return {
        "context": context,
        "long_context": long,
        "mean": mean,
        "scale": scale,
        "keep": keep,
        "fill_names": np.asarray(FILL_NAMES),
        "fills": np.stack([fills[name] for name in FILL_NAMES]),
        "control_names": np.asarray(list(controls)),
        "controls": np.stack(list(controls.values())),
        "filtered_mean": state["filtered_mean"],
        "filtered_covariance": state["filtered_covariance"],
    }


def queries(data):
    keep = np.flatnonzero(data["keep"])
    definitions = {
        "native_target192": (data["context"], np.asarray([0])),
        "native_peer192": (data["context"], keep),
        "native_peer336": (data["long_context"], keep),
    }
    for name, fill in zip(data["fill_names"].tolist(), data["fills"], strict=True):
        definitions["full_" + name] = (fill, keep)
        if name in ("knn", "gaussian"):
            target = data["context"].copy()
            target[:, 0] = fill[:, 0]
            definitions["target_" + name] = (target, keep)
    return {
        name: np.array(
            ((context[:, columns] - data["mean"][columns]) / data["scale"][columns]).T,
            dtype=np.float32,
            order="C",
        )
        for name, (context, columns) in definitions.items()
    }


def combine(points, data):
    points = dict(points)
    pool = [
        points["native_peer192"],
        *[points["full_" + name] for name in FILL_NAMES],
        points["target_knn"],
        points["target_gaussian"],
    ]
    points["mean9"], points["median9"] = np.mean(pool, axis=0), np.median(pool, axis=0)
    controls = dict(zip(data["control_names"].tolist(), data["controls"], strict=True))
    for name, point in list(points.items()):
        points["half_var_" + name] = 0.5 * point + 0.5 * controls["linear_var_direct"]
    points.update(controls)
    points["half_gaussian_seasonal168"] = (
        0.5 * points["full_gaussian"] + 0.5 * points["seasonal168_direct"]
    )
    if len(points) != 31 or not np.isfinite(np.stack(list(points.values()))).all():
        raise ValueError("the registered 31 finite forecasts are incomplete")
    return points


__all__ = ["fit_dynamics", "population", "choose_peers", "build_case", "queries", "combine"]
