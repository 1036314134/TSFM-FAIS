"""Frozen input roles and native quantile aggregation for the R28 development screen."""

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from chronos.utils import interpolate_quantiles, weighted_quantile
from conditioning_inputs import build_inputs, merge_recent_repair
from latent_source_inputs import ROOT, read_json
from patch_repair_eval_support import fixed_controls

from tsfm_fais.utility_experiment import file_sha256

BASE = ROOT / "artifacts/iclr27-r28"
INPUTS = ROOT / "artifacts/iclr27-r25/long-inputs-v001"
OLD = ROOT / "artifacts/iclr27-r26/mae-evaluation-v001"
PROCESS_METHODS = (
    "conditioned_repair",
    "unconditioned_repair",
    "direct_conditioned",
    "native_single_rollout",
    "native_multi_rollout",
    "budget_native192",
)


def population():
    rows = read_json(INPUTS / "manifest.json")["cases"]
    if len(rows) != 301 or len({r["group_id"] for r in rows}) != 8:
        raise ValueError("the registered development population changed")
    return rows


def load_inputs(row):
    path = INPUTS / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError("a frozen L192 imputer input changed")
    with np.load(path, allow_pickle=False) as saved:
        return {
            k: saved[k] for k in ("context", "candidate_values", "candidate_ids", "mean", "scale")
        }


def old_catalog():
    return {r["case_id"]: r for r in read_json(OLD / "predictions_frozen.json")["predictions"]}


def old_points(row, catalog):
    item = catalog[f"native_{row['case_id']}_l192"]
    if file_sha256(OLD / item["path"]) != item["sha256"]:
        raise ValueError("a frozen R26 control changed")
    with np.load(OLD / item["path"], allow_pickle=False) as saved:
        return dict(zip(saved["methods"].tolist(), saved["points"], strict=True))


def source_controls():
    old = read_json(ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json")[
        "models"
    ]["chronos2"]
    matched = read_json(
        ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001/matched_controls.json"
    )["models"]["chronos2"]
    return old, matched


def median(q, levels, start=0):
    index = [i for i, value in enumerate(levels) if abs(value - 0.5) < 1e-8]
    if len(index) != 1:
        raise ValueError("a unique native median is required")
    return q[:, index[0], start : start + 96].T


def pool_controls(bank, actions):
    return fixed_controls(bank, actions, *source_controls())


def packed_query(query, name, packed):
    return query(name, packed.context, packed.future_covariates, packed.group_ids)


def native_aggregate(pipeline, raw, dimensions, paths):
    q = torch.as_tensor(raw, device="cuda")
    levels = torch.tensor(pipeline.quantiles)
    weights = torch.outer(
        pipeline._get_prob_mass_per_quantile_level(torch.tensor(paths)),
        pipeline._get_prob_mass_per_quantile_level(levels),
    ).flatten()
    samples = (
        q.reshape(dimensions, len(paths), len(levels), 96)
        .permute(0, 3, 1, 2)
        .reshape(dimensions, 96, -1)
    )
    result = weighted_quantile(pipeline.quantiles, weights, samples).permute(0, 2, 1)
    return result.float().cpu().numpy()


def multi_inputs(pipeline, context, recent, first_quantiles, paths):
    d = context.shape[1]
    q = torch.as_tensor(first_quantiles, device="cuda")
    predicted = interpolate_quantiles(paths, pipeline.quantiles, q.permute(0, 2, 1)).permute(
        0, 2, 1
    )
    earlier = torch.tensor(np.ascontiguousarray(context[:96].T), device="cuda", dtype=torch.float32)
    expanded = torch.cat([earlier[:, None].expand(d, len(paths), 96), predicted], dim=-1)
    known = torch.tensor(np.ascontiguousarray(recent.T), device="cuda", dtype=torch.float32)[
        :, None
    ]
    expanded[..., 96:] = torch.where(torch.isnan(known), expanded[..., 96:], known)
    return expanded.reshape(d * len(paths), 192).contiguous().cpu().numpy(), np.tile(
        np.arange(len(paths)), d
    )


def build_methods(data, old, query, pipeline):
    context, candidates, ids = (
        data["context"],
        data["candidate_values"],
        data["candidate_ids"].tolist(),
    )
    observed, mean, scale = np.isfinite(context), data["mean"][:2], data["scale"][:2]
    levels, dimensions = pipeline.quantiles, context.shape[1]
    methods = dict(old)
    fallback = {}

    def normalized(raw):
        point = (raw[:, :2] - mean) / scale
        if not np.isfinite(point).all():
            raise FloatingPointError("nonfinite_required_forecast")
        return point

    def execute(name, values, horizon=96):
        packed = build_inputs(values, np.isfinite(values), repair_span=0, actual_horizon=horizon)
        return packed_query(query, name, packed)

    reason = "empty_earlier_target" if not observed[:96, :2].any(0).all() else None
    if observed[96:].all():
        reason = "recent_complete"
    try:
        if reason:
            raise FloatingPointError(reason)
        first = packed_query(query, "conditioned_first", build_inputs(context, observed))
        needed = ~observed[96:]
        recent = median(first, levels)
        if not np.isfinite(recent[needed]).all() or not np.isfinite(first).all():
            raise FloatingPointError("nonfinite_conditioned_queries")
        repaired = merge_recent_repair(context, observed, recent).values
        second = execute("conditioned_second", repaired)
        methods["conditioned_repair"] = normalized(median(second, levels))
        methods["native_single_rollout"] = normalized(
            median(native_aggregate(pipeline, second, dimensions, [0.5]), levels)
        )
        first_u = packed_query(
            query, "unconditioned_first", build_inputs(context, observed, condition_on_recent=False)
        )
        recent_u = median(first_u, levels)
        if not np.isfinite(recent_u[needed]).all():
            raise FloatingPointError("nonfinite_unconditioned_queries")
        repaired_u = merge_recent_repair(context, observed, recent_u).values
        methods["unconditioned_repair"] = normalized(
            median(execute("unconditioned_second", repaired_u), levels)
        )
        direct = packed_query(
            query, "direct_conditioned", build_inputs(context, observed, actual_horizon=96)
        )
        methods["direct_conditioned"] = normalized(median(direct, levels, 96))
        paths = [i / 10 for i in range(1, 10)]
        rolled, groups = multi_inputs(pipeline, context, context[96:], first, paths)
        raw_multi = query("native_multi_second", rolled, np.full((len(rolled), 96), np.nan), groups)
        methods["native_multi_rollout"] = normalized(
            median(native_aggregate(pipeline, raw_multi, dimensions, paths), levels)
        )
    except FloatingPointError as exc:
        for method in PROCESS_METHODS[:-1]:
            methods[method] = old["guarded_direct"].copy()
            fallback[method] = str(exc)
    try:
        methods["budget_native192"] = normalized(
            median(execute("budget_native192", context, 192), levels)
        )
    except FloatingPointError:
        methods["budget_native192"] = old["guarded_direct"].copy()
        fallback["budget_native192"] = "nonfinite_required_forecast"
    actions = sorted([*ids, "guarded_direct"])
    bank = []
    for action in actions:
        if action == "guarded_direct":
            point = old[action]
        else:
            values = context.copy()
            values[96:] = np.where(observed[96:], context[96:], candidates[ids.index(action), 96:])
            try:
                point = normalized(median(execute("recent_" + action, values), levels))
            except FloatingPointError:
                point = old["guarded_direct"].copy()
                fallback["recent_" + action] = "nonfinite_required_forecast"
        bank.append(point)
    methods.update(
        {"recent_" + name: value for name, value in pool_controls(np.stack(bank), actions).items()}
    )
    if len(methods) != 41:
        raise ValueError("the registered 41-method inventory changed")
    return methods, fallback
