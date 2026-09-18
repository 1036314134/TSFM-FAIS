"""A fixed role-separated conditioning diagnostic using only existing historical observations."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from conditioning_core import ROOT, load_inputs, population
from evaluate_matched_replay import aggregate_groups
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

BASE = ROOT / "artifacts/iclr27-r29"
PREVIOUS = ROOT / "artifacts/iclr27-r28/conditioning-forecasts-v001"
NEW = (
    "query_role_repair",
    "helper_role_repair",
    "duplicate_unconditioned_repair",
    "query_role_direct",
)


def role_inputs(context, conditioned=True, horizon=96):
    x = np.where(np.isfinite(context), context, np.nan)
    d = x.shape[1]
    earlier = np.concatenate([x[:96].T, x[:96].T])
    future = np.full((2 * d, horizon), np.nan)
    if conditioned:
        future[d:, :96] = x[96:].T
    return earlier, future


def previous_points(case_id, catalog):
    row = catalog[case_id]
    path = PREVIOUS / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError("a frozen R28 comparator changed")
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["methods"].tolist(), data["points"], strict=True))


def repair(context, query):
    result = context.copy()
    result[96:] = np.where(np.isfinite(context[96:]), context[96:], query)
    if not np.isfinite(result[96:]).all():
        raise FloatingPointError("nonfinite_recent_query")
    return result


def predict(args):
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed role predictions")
    rows = population()
    if args.smoke:
        ids = {
            x["case_id"]
            for x in json.loads(
                (ROOT / "artifacts/iclr27-r28/conditioning-smoke-v002/manifest.json").read_text(
                    encoding="utf-8"
                )
            )["cases"]
        }
        rows = [r for r in rows if r["case_id"] in ids]
    identity = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/conditioning_core.py",
            ROOT / "docs/iclr2027/R29_QUERY_ROLE_PROTOCOL.md",
            PREVIOUS / "manifest.json",
            ROOT / "artifacts/iclr27-r28/conditioning-audit-v001/manifest.json",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists():
        if json.loads((output / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("partial role experiment definitions changed")
    else:
        _write_json(output / "identity.json", identity)
    prior_runtime = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))["runtime"]
    for filename, sha in prior_runtime.items():
        if file_sha256(Path(filename)) != sha:
            raise ValueError("the audited Chronos runtime changed")
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    started = perf_counter()
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid = pipeline.quantiles.index(0.5)
    catalog = {
        r["case_id"]: r
        for r in json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))["cases"]
    }
    counters = {
        "logical_requests": 0,
        "new_queries": 0,
        "cache_hits": 0,
        "forward_calls": 0,
        "variable_rows": 0,
    }
    entries, checks, keys = [], [], set()
    torch.cuda.reset_peak_memory_stats()
    for number, row in enumerate(rows):
        data = load_inputs(row)
        context, d = data["context"], data["context"].shape[1]
        methods = previous_points(row["case_id"], catalog)
        queries = []

        def query(name, x, future, *, records=queries):
            counters["logical_requests"] += 1
            x, f = (np.array(a, dtype=np.float32, order="C") for a in (x, future))
            x[~np.isfinite(x)], f[~np.isfinite(f)] = np.nan, np.nan
            binding = json.dumps(
                {
                    "identity": identity_sha,
                    "parameter": digest,
                    "x": list(x.shape),
                    "f": list(f.shape),
                },
                sort_keys=True,
            )
            key = hashlib.sha256(binding.encode() + x.tobytes() + f.tobytes()).hexdigest()
            path = output / "queries" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    np.testing.assert_array_equal(cached["context"], x)
                    np.testing.assert_array_equal(cached["future"], f)
                    result = cached["quantiles"]
                counters["cache_hits"] += 1
            else:
                with torch.inference_mode():
                    result = (
                        backbone(
                            context=torch.tensor(x, device="cuda"),
                            future_covariates=torch.tensor(f, device="cuda"),
                            group_ids=torch.zeros(len(x), dtype=torch.long, device="cuda"),
                            num_output_patches=int(
                                np.ceil(f.shape[1] / pipeline.model_output_patch_size)
                            ),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                _save_npz(path, context=x, future=f, quantiles=result, binding=np.asarray(binding))
                counters["new_queries"] += 1
                counters["forward_calls"] += 1
                counters["variable_rows"] += len(x)
            keys.add(key)
            records.append({"name": name, "key": key, "sha256": file_sha256(path)})
            return result

        def normalize(q, start=0, *, center=data["mean"][:2], scale=data["scale"][:2]):
            point = (q[:2, mid, start : start + 96].T - center) / scale
            if not np.isfinite(point).all():
                raise FloatingPointError("nonfinite_required_forecast")
            return point

        reason = "empty_earlier_target" if not np.isfinite(context[:96, :2]).any(0).all() else None
        if np.isfinite(context[96:]).all():
            reason = "recent_complete"
        try:
            if reason:
                raise FloatingPointError(reason)
            first = query("role_first", *role_inputs(context))
            for method, values in (
                ("query_role_repair", first[:d, mid].T),
                ("helper_role_repair", first[d:, mid].T),
            ):
                filled = repair(context, values)
                methods[method] = normalize(query(method, filled.T, np.full((d, 96), np.nan)))
            unconditioned = query("duplicate_unconditioned_first", *role_inputs(context, False))
            filled = repair(context, unconditioned[:d, mid].T)
            methods["duplicate_unconditioned_repair"] = normalize(
                query("duplicate_unconditioned_second", filled.T, np.full((d, 96), np.nan))
            )
            methods["query_role_direct"] = normalize(
                query("query_role_direct", *role_inputs(context, horizon=192)), 96
            )
            if args.smoke:
                payload = {
                    "target": context[:96].T,
                    "past_covariates": {f"self_{i:03}": context[:96, i] for i in range(d)},
                    "future_covariates": {f"self_{i:03}": context[96:, i] for i in range(d)},
                }
                public, _ = pipeline.predict_quantiles(
                    [payload],
                    prediction_length=96,
                    quantile_levels=[0.5],
                    batch_size=8,
                    predict_batches_jointly=False,
                )
                sdk = public[0].numpy()[:, :, 0]
                delta = float(np.abs((sdk - first[:d, mid]) / data["scale"][:, None]).max())
                np.testing.assert_allclose(
                    sdk / data["scale"][:, None],
                    first[:d, mid] / data["scale"][:, None],
                    rtol=0,
                    atol=1e-6,
                )
                checks.append({"case_id": row["case_id"], "public_scaled_max_difference": delta})
                counters["forward_calls"] += 1
        except FloatingPointError as error:
            reason = str(error)
            for method in NEW:
                methods[method] = methods["guarded_direct"].copy()
        if len(methods) != 45:
            raise ValueError("the 45-method inventory changed")
        names = sorted(methods)
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=np.stack([methods[m] for m in names]),
            queries=np.asarray(json.dumps(queries)),
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "fallback": reason,
            }
        )
        _write_json(
            output / "progress.json",
            {"completed": number + 1, "total": len(rows), "counters": counters},
        )
        if (number + 1) % 25 == 0:
            print(json.dumps({"predicted": number + 1, "total": len(rows)}), flush=True)
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "identity": identity,
            "parameter_sha256": digest,
            "cases": entries,
            "counters": counters,
            "unique_queries": len(keys),
            "public_checks": checks,
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "new_future_values_read": False,
            "new_training": False,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": perf_counter() - started,
        },
    )


def readout(args):
    forecasts = BASE / "role-forecasts-v001"
    manifest = json.loads((forecasts / "manifest.json").read_text(encoding="utf-8"))
    metadata = {r["case_id"]: r for r in population()}
    if manifest["smoke"] or {r["case_id"] for r in manifest["cases"]} != set(metadata):
        raise ValueError("complete all formal predictions first")
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed readout")
    rows, targets = [], []
    for entry in manifest["cases"]:
        row = metadata[entry["case_id"]]
        data = load_inputs(row)
        if (
            file_sha256(Path(row["original_path"])) != row["original_sha256"]
            or file_sha256(forecasts / entry["path"]) != entry["sha256"]
        ):
            raise ValueError("original targets or predictions changed")
        with np.load(row["original_path"], allow_pickle=False) as original:
            truth = original["future"][:96, :2]
            valid = original["future_observed"][:96, :2]
        np.testing.assert_array_equal(np.isfinite(truth), valid)
        if (valid.sum(0) < 48).any():
            raise ValueError("insufficient original future observations")
        with np.load(forecasts / entry["path"], allow_pickle=False) as predicted:
            names, points = predicted["methods"].tolist(), predicted["points"]
        z = (truth - data["mean"][:2]) / data["scale"][:2]
        for index, method in enumerate(names):
            record = {
                k: row[k]
                for k in (
                    "case_id",
                    "episode_id",
                    "group_id",
                    "family_id",
                    "dataset_id",
                    "item_id",
                    "origin",
                )
            }
            metrics = []
            for slot in (0, 1):
                e = points[index, valid[:, slot], slot] - z[valid[:, slot], slot]
                metrics.append([float(np.abs(e).mean()), float(np.square(e).mean())])
                targets.append(
                    {
                        **record,
                        "model_id": "chronos2",
                        "method": method,
                        "target_slot": slot,
                        "mae": metrics[-1][0],
                        "mse": metrics[-1][1],
                        "observed_count": int(valid[:, slot].sum()),
                    }
                )
            score = np.mean(metrics, axis=0)
            rows.append(
                {
                    **record,
                    "model_id": "chronos2",
                    "method": method,
                    "mae": score[0],
                    "mse": score[1],
                }
            )
    if len(rows) != 13545 or len(targets) != 27090:
        raise ValueError("registered score population changed")
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    tables = aggregate_groups(frame)
    for name, table in zip(("series", "datasets", "groups", "summary"), tables, strict=True):
        table.to_csv(output / f"{name}.csv", index=False)
    leave = []
    for group in tables[2].group_id.unique():
        part = (
            tables[2]
            .query("group_id != @group")
            .groupby(["model_id", "method"])[["mae", "mse"]]
            .mean()
            .reset_index()
        )
        part["omitted_group"] = group
        leave.append(part)
    pd.concat(leave, ignore_index=True).to_csv(output / "leave_one_group_out.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": NEW[0],
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
        },
    )


def audit(args):
    forecast = args.forecast_root.resolve()
    record = json.loads((forecast / "manifest.json").read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed audits")
    for filename, sha in record["identity"].items():
        if file_sha256(Path(filename)) != sha:
            raise ValueError("a frozen dependency changed")
    for filename, sha in json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))[
        "runtime"
    ].items():
        if file_sha256(Path(filename)) != sha:
            raise ValueError("the audited Chronos runtime changed")
    metadata = {r["case_id"]: r for r in population()}
    old_catalog = {
        r["case_id"]: r
        for r in json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))["cases"]
    }
    if not args.smoke and {r["case_id"] for r in record["cases"]} != set(metadata):
        raise ValueError("incomplete formal panel")
    torch.set_num_threads(1)
    started = perf_counter()
    _, _, model, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    mid = record["quantiles"].index(0.5)
    checked, metric_max, rebuilt_scores = set(), 0.0, []
    scores = None
    if not args.smoke:
        results = BASE / "role-results-v001"
        study = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
        if study["forecast_sha256"] != file_sha256(forecast / "manifest.json"):
            raise ValueError("score identity differs")
        scores = pd.read_parquet(results / "case_scores.parquet").set_index(["case_id", "method"])
        target_scores = pd.read_parquet(results / "target_scores.parquet").set_index(
            ["case_id", "method", "target_slot"]
        )
    for number, entry in enumerate(record["cases"]):
        row = metadata[entry["case_id"]]
        data = load_inputs(row)
        x, d = data["context"], data["context"].shape[1]
        mu, sigma = data["mean"][:2], data["scale"][:2]
        if file_sha256(forecast / entry["path"]) != entry["sha256"]:
            raise ValueError("a prediction changed")
        with np.load(forecast / entry["path"], allow_pickle=False) as saved:
            names, points = saved["methods"].tolist(), saved["points"]
            queries = json.loads(str(saved["queries"]))
        cache = {}
        for query in queries:
            path = forecast / "queries" / f"{query['key']}.npz"
            if file_sha256(path) != query["sha256"]:
                raise ValueError("raw query changed")
            with np.load(path, allow_pickle=False) as saved:
                q = {k: saved[k] for k in ("context", "future", "quantiles")}
            cache[query["name"]] = q
            if query["key"] not in checked:
                with torch.inference_mode():
                    again = (
                        model(
                            context=torch.tensor(q["context"], device="cuda"),
                            future_covariates=torch.tensor(q["future"], device="cuda"),
                            group_ids=torch.zeros(
                                len(q["context"]), dtype=torch.long, device="cuda"
                            ),
                            num_output_patches=int(
                                np.ceil(q["future"].shape[1] / record["patch_size"])
                            ),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_array_equal(again, q["quantiles"])
                checked.add(query["key"])
        old = previous_points(row["case_id"], old_catalog)
        rebuilt = dict(old)
        if entry["fallback"]:
            if entry["fallback"] == "empty_earlier_target" and np.isfinite(x[:96, :2]).any(0).all():
                raise ValueError("invalid fallback")
            if entry["fallback"].startswith("nonfinite") and not any(
                not np.isfinite(q["quantiles"]).all() for q in cache.values()
            ):
                raise ValueError("missing nonfinite witness")
            rebuilt.update({name: old["guarded_direct"] for name in NEW})
        else:
            for name, h, conditional in (
                ("role_first", 96, True),
                ("duplicate_unconditioned_first", 96, False),
                ("query_role_direct", 192, True),
            ):
                expected_context = np.tile(x[:96].T, (2, 1)).astype(np.float32)
                expected_future = np.full((2 * d, h), np.nan, np.float32)
                if conditional:
                    expected_future[d:, :96] = x[96:].T
                np.testing.assert_array_equal(cache[name]["context"], expected_context)
                np.testing.assert_array_equal(cache[name]["future"], expected_future)
            for name, first_name, offset, second_name in (
                ("query_role_repair", "role_first", 0, "query_role_repair"),
                ("helper_role_repair", "role_first", d, "helper_role_repair"),
                (
                    "duplicate_unconditioned_repair",
                    "duplicate_unconditioned_first",
                    0,
                    "duplicate_unconditioned_second",
                ),
            ):
                values = cache[first_name]["quantiles"][offset : offset + d, mid].T
                expected = x.copy()
                missing = ~np.isfinite(x[96:])
                expected[96:] = np.where(missing, values, x[96:])
                np.testing.assert_array_equal(
                    cache[second_name]["context"], expected.astype(np.float32).T
                )
                assert np.isnan(cache[second_name]["future"]).all()
                q = cache[second_name]["quantiles"]
                rebuilt[name] = (q[:2, mid, :96].T - mu) / sigma
            rebuilt["query_role_direct"] = (
                cache["query_role_direct"]["quantiles"][:2, mid, 96:192].T - mu
            ) / sigma
        if set(rebuilt) != set(names):
            raise ValueError("method inventory changed")
        for name, value in rebuilt.items():
            np.testing.assert_array_equal(value, points[names.index(name)])
        if scores is not None:
            if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
                raise ValueError("original target source changed")
            with np.load(row["original_path"], allow_pickle=False) as original:
                truth, valid = original["future"][:96, :2], original["future_observed"][:96, :2]
            np.testing.assert_array_equal(np.isfinite(truth), valid)
            normalized = (truth - mu) / sigma
            error = np.where(valid[None], points - normalized[None], 0.0)
            target_mae = np.abs(error).sum(1) / valid.sum(0)
            target_mse = np.square(error).sum(1) / valid.sum(0)
            mae, mse = target_mae.mean(1), target_mse.mean(1)
            for i, name in enumerate(names):
                for slot in (0, 1):
                    target = target_scores.loc[(row["case_id"], name, slot)]
                    np.testing.assert_allclose(
                        target[["mae", "mse"]].to_numpy(float),
                        [target_mae[i, slot], target_mse[i, slot]],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    if int(target["observed_count"]) != int(valid[:, slot].sum()):
                        raise ValueError("target support differs")
                actual = scores.loc[(row["case_id"], name), ["mae", "mse"]].to_numpy(float)
                np.testing.assert_allclose(actual, [mae[i], mse[i]], atol=1e-12, rtol=1e-12)
                metric_max = max(metric_max, float(np.abs(actual - [mae[i], mse[i]]).max()))
                rebuilt_scores.append(
                    {
                        "method": name,
                        "group_id": row["group_id"],
                        "dataset_id": row["dataset_id"],
                        "item_id": row["item_id"],
                        "mae": mae[i],
                        "mse": mse[i],
                    }
                )
        if (number + 1) % 25 == 0:
            print(json.dumps({"audited": number + 1, "queries": len(checked)}), flush=True)
    if rebuilt_scores:
        frame = pd.DataFrame(rebuilt_scores)
        groups = (
            frame.groupby(["method", "group_id", "dataset_id", "item_id"])[["mae", "mse"]]
            .mean()
            .groupby(["method", "group_id", "dataset_id"])
            .mean()
            .groupby(["method", "group_id"])
            .mean()
        )
        summary = groups.groupby("method").mean().sort_index()
        actual = (
            pd.read_csv(results / "summary.csv").set_index("method").sort_index()[["mae", "mse"]]
        )
        np.testing.assert_allclose(summary, actual, rtol=1e-12, atol=1e-12)
        leave_out = pd.read_csv(results / "leave_one_group_out.csv")
        for excluded in groups.index.get_level_values("group_id").unique():
            expected = (
                groups.loc[groups.index.get_level_values("group_id") != excluded]
                .groupby("method")
                .mean()
                .sort_index()
            )
            actual = (
                leave_out.query("omitted_group == @excluded")
                .set_index("method")
                .sort_index()[["mae", "mse"]]
            )
            np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    if digest != record["parameter_sha256"] or parameter_digest(model) != digest:
        raise ValueError("frozen weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "cases": len(record["cases"]),
            "replayed_queries": len(checked),
            "raw_prediction_max_difference": 0,
            "method_max_difference": 0,
            "metric_max_difference": metric_max,
            "score_rows": len(rebuilt_scores),
            "forecast_sha256": file_sha256(forecast / "manifest.json"),
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("predict", "readout", "audit"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    {"predict": predict, "readout": readout, "audit": audit}[options.phase](options)
