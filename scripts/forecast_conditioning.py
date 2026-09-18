"""Freeze R28 forecasts without accessing any evaluation future values."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from conditioning_core import (
    INPUTS,
    ROOT,
    build_methods,
    load_inputs,
    median,
    old_catalog,
    old_points,
    population,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning predictions")
    rows = population()
    if args.smoke:
        ids = {
            r["case_id"]
            for r in json.loads(
                (
                    ROOT
                    / "artifacts/pro-conditioning-review-20260916/interface-v002/selection.json"
                ).read_text(encoding="utf-8")
            )
        }
        for row in rows:
            if not np.isfinite(load_inputs(row)["context"][:96, :2]).any(0).all():
                ids.add(row["case_id"])
        rows = [r for r in rows if r["case_id"] in ids]
        if len(rows) != 9:
            raise ValueError("expected eight interface examples and one fallback example")
    files = [
        ROOT / "scripts" / name
        for name in (
            "conditioning_inputs.py",
            "conditioning_core.py",
            "forecast_conditioning.py",
            "readout_conditioning.py",
            "audit_conditioning.py",
            "r6_runtime.py",
        )
    ]
    files += [
        ROOT / "docs/iclr2027/R28_CONDITIONING_PROTOCOL.md",
        INPUTS / "manifest.json",
        ROOT / "artifacts/iclr27-r26/mae-evaluation-v001/predictions_frozen.json",
        ROOT / "artifacts/iclr27-r25/long-audit-v001/manifest.json",
        ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json",
        ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001/matched_controls.json",
    ]
    identity = {str(p): file_sha256(p) for p in files}
    identity["case_ids"] = [r["case_id"] for r in rows]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists():
        if json.loads((output / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("partial forecast code or population changed")
    else:
        _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    started = perf_counter()
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    import chronos.utils

    runtime = {
        str(Path(inspect.getfile(c))): file_sha256(Path(inspect.getfile(c)))
        for c in (type(backbone), type(pipeline), chronos.utils.weighted_quantile)
    }
    torch.cuda.reset_peak_memory_stats()
    counters = {
        "logical_requests": 0,
        "new_queries": 0,
        "cache_hits": 0,
        "forward_calls": 0,
        "variable_rows": 0,
        "input_patches": 0,
        "output_patches": 0,
    }
    keys = set()
    catalog = old_catalog()
    entries, smoke_checks = [], []
    print(json.dumps({"stage": "model_loaded", "cases": len(rows)}), flush=True)
    for index, row in enumerate(rows):
        data = load_inputs(row)
        old = old_points(row, catalog)
        queries = []

        def query(name, context, future, groups, *, query_records=queries):
            counters["logical_requests"] += 1
            x, f = (np.array(v, np.float32, order="C") for v in (context, future))
            x[~np.isfinite(x)], f[~np.isfinite(f)] = np.nan, np.nan
            g = np.asarray(groups, np.int64)
            patches = int(np.ceil(f.shape[1] / pipeline.model_output_patch_size))
            binding = json.dumps(
                {
                    "identity": identity_sha,
                    "parameter": digest,
                    "runtime": runtime,
                    "x_shape": list(x.shape),
                    "f_shape": list(f.shape),
                    "patches": patches,
                },
                sort_keys=True,
            )
            key = hashlib.sha256(
                binding.encode() + x.tobytes() + f.tobytes() + g.tobytes()
            ).hexdigest()
            path = output / "queries" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    for field, value in (("context", x), ("future", f), ("groups", g)):
                        np.testing.assert_array_equal(saved[field], value)
                    q = saved["quantiles"]
                counters["cache_hits"] += 1
            else:
                with torch.inference_mode():
                    q = (
                        backbone(
                            context=torch.tensor(x, device="cuda"),
                            context_mask=torch.tensor(
                                np.isfinite(x), device="cuda", dtype=torch.float32
                            ),
                            future_covariates=torch.tensor(f, device="cuda"),
                            future_covariates_mask=torch.tensor(
                                np.isfinite(f), device="cuda", dtype=torch.float32
                            ),
                            group_ids=torch.tensor(g, device="cuda"),
                            num_output_patches=patches,
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                _save_npz(
                    path, context=x, future=f, groups=g, quantiles=q, binding=np.asarray(binding)
                )
                counters["new_queries"] += 1
                counters["forward_calls"] += 1
                counters["variable_rows"] += len(x)
                counters["input_patches"] += len(x) * int(
                    np.ceil(x.shape[1] / backbone.chronos_config.input_patch_size)
                )
                counters["output_patches"] += len(x) * patches
            keys.add(key)
            query_records.append({"name": name, "key": key, "sha256": file_sha256(path)})
            return q

        methods, fallback = build_methods(data, old, query, pipeline)
        if args.smoke:
            context, d = data["context"], data["context"].shape[1]
            q = query("native_replay", context.T, np.full((d, 96), np.nan), np.zeros(d, np.int64))
            native = (median(q, pipeline.quantiles)[:, :2] - data["mean"][:2]) / data["scale"][:2]
            np.testing.assert_array_equal(native, old["guarded_direct"])
            if not fallback:
                for paths, method in (
                    ([0.5], "native_single_rollout"),
                    ([i / 10 for i in range(1, 10)], "native_multi_rollout"),
                ):
                    with torch.inference_mode():
                        q = pipeline._predict_batch(
                            context=torch.tensor(
                                np.ascontiguousarray(context[:96].T), dtype=torch.float32
                            ),
                            group_ids=torch.zeros(d, dtype=torch.long),
                            future_covariates=torch.tensor(
                                np.concatenate([context[96:].T, np.full((d, 96), np.nan)], axis=1),
                                dtype=torch.float32,
                            ),
                            unrolled_quantiles_tensor=torch.tensor(paths),
                            prediction_length=192,
                            max_output_patches=96 // pipeline.model_output_patch_size,
                            target_idx_ranges=[(0, d)],
                        )[0].numpy()
                    expected = (median(q, pipeline.quantiles, 96)[:, :2] - data["mean"][:2]) / data[
                        "scale"
                    ][:2]
                    diff = float(np.abs(expected - methods[method]).max())
                    np.testing.assert_allclose(expected, methods[method], atol=1e-6, rtol=0)
                    smoke_checks.append(
                        {"case_id": row["case_id"], "method": method, "scaled_max_difference": diff}
                    )
                    counters["forward_calls"] += 2
        names = sorted(methods)
        points = np.stack([methods[n] for n in names])
        if points.shape != (41, 96, 2) or not np.isfinite(points).all():
            raise ValueError("invalid complete method output")
        observed = np.isfinite(data["context"])
        metadata = {
            "fallback": fallback,
            "recent_target_missing_rate": float((~observed[96:, :2]).mean()),
            "target_tail_gap": bool((~observed[-1, :2]).any()),
            "recent_aux_observed_rate": float(observed[96:, 2:].mean())
            if observed.shape[1] > 2
            else None,
        }
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=points,
            queries=np.asarray(json.dumps(queries)),
            metadata=np.asarray(json.dumps(metadata)),
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                **metadata,
            }
        )
        _write_json(
            output / "progress.json",
            {"completed": index + 1, "total": len(rows), "counters": counters},
        )
        if (index + 1) % 25 == 0:
            print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "identity": identity,
            "runtime": runtime,
            "parameter_sha256": digest,
            "cases": entries,
            "counters": counters,
            "unique_queries": len(keys),
            "quantile_levels": list(pipeline.quantiles),
            "output_patch_size": pipeline.model_output_patch_size,
            "smoke_native_checks": smoke_checks,
            "wall_seconds": perf_counter() - started,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "future_values_read": False,
            "new_training": False,
        },
    )


if __name__ == "__main__":
    main()
