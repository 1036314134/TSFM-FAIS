"""Independently reconstruct R28 input roles, native aggregation, queries and scores."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from conditioning_core import (
    BASE,
    ROOT,
    load_inputs,
    old_catalog,
    old_points,
    population,
    source_controls,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _write_json, file_sha256


def mixture_median(raw, dimensions, paths, levels):
    def mass(q):
        # Match the runtime's float32 reductions; NumPy's CPU scan orders differ.
        boundaries = torch.tensor([0.0, *q, 1.0], dtype=torch.float32)
        weights = (boundaries[2:] - boundaries[:-2]) / 2
        return weights / weights.sum()

    weights = (mass(paths)[:, None] * mass(levels)[None, :]).flatten().to("cuda")
    weights = weights / weights.sum()
    values = (
        raw.reshape(dimensions, len(paths), len(levels), 96)
        .transpose(0, 3, 1, 2)
        .reshape(dimensions, 96, -1)
    )
    order = np.argsort(values, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=-1)
    ordered_weights = weights[torch.tensor(order, device="cuda")]
    cumulative = torch.cumsum(ordered_weights, dim=-1).clamp(max=1).cpu().numpy()
    upper = (cumulative <= np.float32(0.5)).sum(-1)[..., None]
    lower = upper - 1
    a, b = (np.take_along_axis(cumulative, i, axis=-1)[..., 0] for i in (lower, upper))
    x, y = (np.take_along_axis(sorted_values, i, axis=-1)[..., 0] for i in (lower, upper))
    return (x + (np.float32(0.5) - a) / (b - a) * (y - x)).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output, forecast_root = args.output_root.resolve(), args.forecast_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning audit")
    record = json.loads((forecast_root / "manifest.json").read_text(encoding="utf-8"))
    if record["status"] != "completed" or record["smoke"] != args.smoke:
        raise ValueError("the corresponding forecasts are incomplete")
    for mapping in (record["identity"], record["runtime"]):
        for name, digest in mapping.items():
            if name != "case_ids" and file_sha256(Path(name)) != digest:
                raise ValueError("a frozen dependency changed")
    metadata = {r["case_id"]: r for r in population()}
    if not args.smoke and {r["case_id"] for r in record["cases"]} != set(metadata):
        raise ValueError("the 301-case population changed")
    levels = record["quantile_levels"]
    mid = next(i for i, q in enumerate(levels) if q == 0.5)
    controls = source_controls()
    catalog = old_catalog()
    scores = None
    if not args.smoke:
        results = BASE / "conditioning-results-v001"
        study = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
        if study["forecast_sha256"] != file_sha256(forecast_root / "manifest.json"):
            raise ValueError("readout used a different forecast panel")
        scores = pd.read_parquet(results / "case_scores.parquet").set_index(["case_id", "method"])
        targets = pd.read_parquet(results / "target_scores.parquet").set_index(
            ["case_id", "method", "target_slot"]
        )
    torch.set_num_threads(1)
    started = perf_counter()
    _, _, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    checked, max_derived, max_metric, reconstructed = set(), 0.0, 0.0, []
    for number, entry in enumerate(record["cases"]):
        row = metadata[entry["case_id"]]
        data = load_inputs(row)
        x, values, ids = data["context"], data["candidate_values"], data["candidate_ids"].tolist()
        observed, d = np.isfinite(x), x.shape[1]
        mu, sigma = data["mean"][:2], data["scale"][:2]
        old = old_points(row, catalog)
        if file_sha256(forecast_root / entry["path"]) != entry["sha256"]:
            raise ValueError("a frozen prediction file changed")
        with np.load(forecast_root / entry["path"], allow_pickle=False) as saved:
            names, points = saved["methods"].tolist(), saved["points"]
            queries = json.loads(str(saved["queries"]))
        query_data = {}
        for item in queries:
            path = forecast_root / "queries" / f"{item['key']}.npz"
            if file_sha256(path) != item["sha256"]:
                raise ValueError("a raw query changed")
            with np.load(path, allow_pickle=False) as saved:
                q = {k: saved[k] for k in ("context", "future", "groups", "quantiles")}
            query_data[item["name"]] = q
            if item["key"] not in checked:
                with torch.inference_mode():
                    replay = (
                        backbone(
                            context=torch.tensor(q["context"], device="cuda"),
                            context_mask=torch.tensor(
                                np.isfinite(q["context"]), dtype=torch.float32, device="cuda"
                            ),
                            future_covariates=torch.tensor(q["future"], device="cuda"),
                            future_covariates_mask=torch.tensor(
                                np.isfinite(q["future"]), dtype=torch.float32, device="cuda"
                            ),
                            group_ids=torch.tensor(q["groups"], device="cuda"),
                            num_output_patches=int(
                                np.ceil(q["future"].shape[1] / record["output_patch_size"])
                            ),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_array_equal(replay, q["quantiles"])
                checked.add(item["key"])

        def verify(name, context, future, groups=None, *, lookup=query_data):
            q = lookup[name]
            np.testing.assert_array_equal(q["context"], np.asarray(context, np.float32))
            np.testing.assert_array_equal(q["future"], np.asarray(future, np.float32))
            np.testing.assert_array_equal(
                q["groups"], np.zeros(len(context), np.int64) if groups is None else groups
            )
            return q["quantiles"]

        def z(q, start=0, *, center=mu, scale=sigma):
            return (q[:2, mid, start : start + 96].T - center) / scale

        def plain(name, context, horizon=96, *, dimensions=d, check=verify):
            return check(name, context.T, np.full((dimensions, horizon), np.nan))

        derived = dict(old)
        conditional_names = (
            "conditioned_repair",
            "unconditioned_repair",
            "direct_conditioned",
            "native_single_rollout",
            "native_multi_rollout",
        )
        common_fallback = any(n in entry["fallback"] for n in conditional_names)
        first = None
        if "conditioned_first" in query_data:
            first = verify("conditioned_first", x[:96].T, x[96:].T)
        if "unconditioned_first" in query_data:
            verify("unconditioned_first", x[:96].T, np.full((d, 96), np.nan))
        if not common_fallback:
            merged = np.concatenate([x[:96], np.where(observed[96:], x[96:], first[:, mid].T)])
            second = plain("conditioned_second", merged)
            derived["conditioned_repair"] = z(second)
            derived["native_single_rollout"] = (
                mixture_median(second, d, [0.5], levels)[:, :2] - mu
            ) / sigma
            qu = query_data["unconditioned_first"]["quantiles"]
            merged_u = np.concatenate([x[:96], np.where(observed[96:], x[96:], qu[:, mid].T)])
            derived["unconditioned_repair"] = z(plain("unconditioned_second", merged_u))
            direct = verify(
                "direct_conditioned",
                x[:96].T,
                np.concatenate([x[96:].T, np.full((d, 96), np.nan)], axis=1),
            )
            derived["direct_conditioned"] = z(direct, 96)
            paths = [i / 10 for i in range(1, 10)]
            indices = [levels.index(p) for p in paths]
            expanded = np.concatenate(
                [np.repeat(x[:96].T[:, None], 9, axis=1), first[:, indices]], axis=2
            ).astype(np.float32)
            expanded[..., 96:] = np.where(
                observed[96:].T[:, None], x[96:].T[:, None], expanded[..., 96:]
            )
            multi = verify(
                "native_multi_second",
                expanded.reshape(d * 9, 192),
                np.full((d * 9, 96), np.nan),
                np.tile(np.arange(9), d),
            )
            derived["native_multi_rollout"] = (
                mixture_median(multi, d, paths, levels)[:, :2] - mu
            ) / sigma
        else:
            reasons = set(entry["fallback"].get(n) for n in conditional_names)
            if "empty_earlier_target" in reasons and observed[:96, :2].any(0).all():
                raise ValueError("invalid earlier-target fallback")
            if any(str(r).startswith("nonfinite") for r in reasons):
                if not any(not np.isfinite(q["quantiles"]).all() for q in query_data.values()):
                    raise ValueError("a numerical fallback has no saved witness")
            for name in conditional_names:
                derived[name] = old["guarded_direct"]
        budget = plain("budget_native192", x, 192)
        derived["budget_native192"] = (
            old["guarded_direct"] if "budget_native192" in entry["fallback"] else z(budget)
        )
        if "native_replay" in query_data:
            np.testing.assert_array_equal(z(plain("native_replay", x)), old["guarded_direct"])
        actions = sorted([*ids, "guarded_direct"])
        regional = []
        for action in actions:
            if action == "guarded_direct":
                point = old[action]
            else:
                completed = x.copy()
                completed[96:] = np.where(observed[96:], x[96:], values[ids.index(action), 96:])
                np.testing.assert_array_equal(completed[observed], x[observed])
                q = plain("recent_" + action, completed)
                point = old["guarded_direct"] if "recent_" + action in entry["fallback"] else z(q)
            regional.append(point)
            derived["recent_" + action] = point
        bank = np.stack(regional)
        derived["recent_mean8"], derived["recent_median8"] = bank.mean(0), np.median(bank, axis=0)
        for prefix, control in zip(("source_", "matched_source_"), controls, strict=True):
            derived["recent_" + prefix + "single_mae"] = bank[control["single_index"]]
            for label, weights in (
                ("fixed_mae", control["fixed_mae"]["weights"]),
                (
                    "fixed_joint",
                    control.get(
                        "fixed_joint_weights", control.get("fixed_joint", {}).get("weights")
                    ),
                ),
            ):
                derived["recent_" + prefix + label] = (
                    bank * np.asarray(weights)[:, None, None]
                ).sum(0)
        if set(derived) != set(names):
            raise ValueError("an output is missing an independent reconstruction")
        for method, point in derived.items():
            actual = points[names.index(method)]
            np.testing.assert_allclose(point, actual, rtol=1e-5, atol=1e-6)
            max_derived = max(max_derived, float(np.abs(point - actual).max()))
            if method in old:
                np.testing.assert_array_equal(point, actual)
        if scores is not None:
            if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
                raise ValueError("the immutable target source changed")
            with np.load(row["original_path"], allow_pickle=False) as original:
                truth, valid = original["future"][:96, :2], original["future_observed"][:96, :2]
            np.testing.assert_array_equal(np.isfinite(truth), valid)
            for i, method in enumerate(names):
                metrics = []
                for slot in (0, 1):
                    err = (
                        points[i, valid[:, slot], slot]
                        - (truth[valid[:, slot], slot] - mu[slot]) / sigma[slot]
                    )
                    metric = np.array([np.abs(err).mean(), np.square(err).mean()])
                    expected = targets.loc[(row["case_id"], method, slot), ["mae", "mse"]].to_numpy(
                        float
                    )
                    np.testing.assert_allclose(metric, expected, rtol=1e-12, atol=1e-12)
                    max_metric = max(max_metric, float(np.abs(metric - expected).max()))
                    metrics.append(metric)
                metric = np.mean(metrics, axis=0)
                np.testing.assert_allclose(
                    metric,
                    scores.loc[(row["case_id"], method), ["mae", "mse"]].to_numpy(float),
                    rtol=1e-12,
                    atol=1e-12,
                )
                reconstructed.append({**row, "method": method, "mae": metric[0], "mse": metric[1]})
        if (number + 1) % 25 == 0:
            print(
                json.dumps({"audited_cases": number + 1, "raw_queries": len(checked)}), flush=True
            )
    if reconstructed:
        frame = pd.DataFrame(reconstructed)
        means = frame.groupby(["method", "group_id", "dataset_id", "item_id"])[
            ["mae", "mse"]
        ].mean()
        means = (
            means.groupby(["method", "group_id", "dataset_id"])
            .mean()
            .groupby(["method", "group_id"])
            .mean()
        )
        summary = means.groupby("method").mean().sort_index()
        actual = (
            pd.read_csv(results / "summary.csv").set_index("method").sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(summary.to_numpy(), actual.to_numpy(), rtol=1e-12, atol=1e-12)
        for excluded in means.index.get_level_values("group_id").unique():
            independent = (
                means.loc[means.index.get_level_values("group_id") != excluded]
                .groupby("method")
                .mean()
                .sort_index()
            )
            exported = (
                pd.read_csv(results / "leave_one_group_out.csv")
                .query("omitted_group == @excluded")
                .set_index("method")
                .sort_index()[["mae", "mse"]]
            )
            np.testing.assert_allclose(
                independent.to_numpy(), exported.to_numpy(), rtol=1e-12, atol=1e-12
            )
    if parameter_digest(backbone) != digest or digest != record["parameter_sha256"]:
        raise ValueError("frozen model weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "cases_checked": len(record["cases"]),
            "unique_queries_replayed": len(checked),
            "raw_replay_max_difference": 0,
            "derived_max_scaled_difference": max_derived,
            "score_max_difference": max_metric,
            "scores_checked": len(reconstructed),
            "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
