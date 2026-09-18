"""Check supplied conditioning inputs against the frozen Chronos runtime without scores."""

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parents[1]
SUPPLIED = ROOT / "artifacts/pro-conditioning-review-20260916/supplied"
sys.path.insert(0, str(SUPPLIED))
from conditioning_inputs import build_inputs, median_output_slice, merge_recent_repair  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from r6_runtime import make_forecaster  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError("preserve any previous interface diagnostic")
    source = ROOT / "artifacts/iclr27-r25/long-inputs-v001"
    prepared = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    helper_sha = file_sha256(SUPPLIED / "conditioning_inputs.py")
    if helper_sha != "133bfce95c5997a27e9b29886895fc299a3731478b9d476c2cc7bfaba0a26886":
        raise ValueError("the reviewed supplied helper changed")
    inventory, selected, seen = [], [], set()
    for row in sorted(
        prepared["cases"], key=lambda r: (r["group_id"], r["dataset_id"], r["item_id"], r["origin"])
    ):
        path = source / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("an existing development input changed")
        with np.load(path, allow_pickle=False) as saved:
            context = saved["context"]
        observed = np.isfinite(context)
        item = {
            "case_id": row["case_id"],
            "group_id": row["group_id"],
            "dimensions": context.shape[1],
            "empty_prefix_targets": np.flatnonzero(~observed[:96, :2].any(0)).tolist(),
            "empty_prefix_channels": np.flatnonzero(~observed[:96].any(0)).tolist(),
            "empty_full_channels": np.flatnonzero(~observed.any(0)).tolist(),
            "recent_missing_cells": int((~observed[96:]).sum()),
        }
        inventory.append(item)
        if row["group_id"] not in seen and not item["empty_prefix_targets"]:
            selected.append(row)
            seen.add(row["group_id"])
    if len(selected) > 16 or len(seen) != 8:
        raise ValueError("expected one input-only-selected case from each of eight groups")
    output.mkdir(parents=True)
    _write_json(output / "input_inventory.json", inventory)
    _write_json(output / "selection.json", selected)
    print(
        json.dumps(
            {"stage": "inventory_ready", "cases": len(inventory), "selected": len(selected)}
        ),
        flush=True,
    )
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    runner, adapter, backbone, before_digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    load_seconds = perf_counter() - started
    print(json.dumps({"stage": "model_loaded", "seconds": load_seconds}), flush=True)
    patch_size = pipeline.model_output_patch_size
    levels = pipeline.quantiles
    spec = ForecastSpec(
        "chronos2", "joint_multivariate", 96, context_length=192, target_indices=[0, 1]
    )
    calls = 0

    def raw(packed):
        nonlocal calls
        tensors = {
            name: torch.tensor(
                np.ascontiguousarray(getattr(packed, name)), device="cuda", dtype=torch.float32
            )
            for name in ("context", "context_mask", "future_covariates", "future_covariates_mask")
        }
        tensors["group_ids"] = torch.tensor(packed.group_ids, device="cuda", dtype=torch.long)
        patches = math.ceil((packed.repair_span + packed.actual_horizon) / patch_size)
        traces = []

        def capture(module, inputs):
            traces.append(inputs[0].detach().cpu().numpy().copy())

        handle = backbone.input_patch_embedding.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                result = backbone(**tensors, num_output_patches=patches).quantile_preds
            calls += 1
        finally:
            handle.remove()
        if len(traces) != 2:
            raise ValueError("expected historical and future embedding inputs")
        future_mask = np.zeros((packed.context.shape[0], patches * patch_size), np.float32)
        future_mask[:, : packed.future_covariates_mask.shape[1]] = packed.future_covariates_mask
        np.testing.assert_array_equal(
            traces[1][..., -patch_size:].reshape(future_mask.shape), future_mask
        )
        prediction = result.float().cpu().numpy()
        return prediction, {
            "history_mask": traces[0][..., -backbone.chronos_config.input_patch_size :],
            "future_mask": traces[1][..., -patch_size:],
        }

    old_root = ROOT / "artifacts/iclr27-r25/long-forecasts-v001"
    old_cases = {
        r["case_id"]: r
        for r in json.loads((old_root / "manifest.json").read_text(encoding="utf-8"))["cases"]
    }
    rows = []
    infer_started = perf_counter()
    for row in selected:
        with np.load(source / row["path"], allow_pickle=False) as saved:
            context, mean, scale = saved["context"], saved["mean"], saved["scale"]
        observed = np.isfinite(context)
        base = build_inputs(context, observed, repair_span=0, actual_horizon=96)
        q_base, _ = raw(base)
        native = median_output_slice(q_base, levels, start=0, length=96)
        public = runner.predict_missing(context[None], spec).point[0]
        calls += 1
        public_delta = float(np.max(np.abs(native[:, :2] - public) / scale[:2]))
        np.testing.assert_allclose(native[:, :2] / scale[:2], public / scale[:2], rtol=0, atol=1e-6)
        cache_delta = None
        if observed.any(0).all():
            old = old_cases[row["case_id"]]
            if file_sha256(old_root / old["path"]) != old["sha256"]:
                raise ValueError("an old forecast cache changed")
            with np.load(old_root / old["path"], allow_pickle=False) as saved:
                names = saved["methods"].tolist()
                old_point = saved["points"][names.index("guarded_direct")]
            cache_delta = float(np.abs((native[:, :2] - mean[:2]) / scale[:2] - old_point).max())
            np.testing.assert_allclose(
                (native[:, :2] - mean[:2]) / scale[:2], old_point, rtol=0, atol=1e-5
            )
        conditioned = build_inputs(context, observed, repair_span=96)
        q_cond, cond_trace = raw(conditioned)
        recent = median_output_slice(q_cond, levels, start=0, length=96)
        unconditioned = build_inputs(context, observed, repair_span=96, condition_on_recent=False)
        q_uncond, _ = raw(unconditioned)
        recent_uncond = median_output_slice(q_uncond, levels, start=0, length=96)
        merged = merge_recent_repair(context, observed, recent)
        np.testing.assert_array_equal(merged.values[observed], context[observed])
        np.testing.assert_array_equal(np.isnan(merged.values[:96]), ~observed[:96])
        second = build_inputs(
            merged.values, merged.model_observed, repair_span=0, actual_horizon=96
        )
        q_second, second_trace = raw(second)
        np.testing.assert_array_equal(
            second_trace["history_mask"].reshape(context.shape[1], 192), merged.model_observed.T
        )
        direct = build_inputs(context, observed, repair_span=96, actual_horizon=96)
        q_direct, _ = raw(direct)
        direct_point = median_output_slice(q_direct, levels, start=96, length=96)
        second_point = median_output_slice(q_second, levels, start=0, length=96)
        if not np.isfinite(second_point).all() or not np.isfinite(direct_point).all():
            raise ValueError("the native conditional interface returned a non-finite forecast")
        needed = ~observed[96:]
        influence = float(np.max(np.abs((recent - recent_uncond) / scale)[needed]))
        _save_npz(
            output / "cases" / f"{row['case_id']}.npz",
            context=context,
            observed=observed,
            first_context=conditioned.context,
            first_context_mask=conditioned.context_mask,
            first_future=conditioned.future_covariates,
            first_future_mask=conditioned.future_covariates_mask,
            recent_conditioned=recent,
            recent_unconditioned=recent_uncond,
            second_context=merged.values,
            second_mask=merged.model_observed,
            native=native,
            repaired=second_point,
            direct=direct_point,
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "group_id": row["group_id"],
                "dimensions": context.shape[1],
                "public_scaled_max_difference": public_delta,
                "old_cache_scaled_max_difference": cache_delta,
                "known_recent_mask_cells": int(cond_trace["future_mask"].sum()),
                "filled_recent_cells": int(needed.sum()),
                "conditioning_scaled_max_output_change_on_missing": influence,
                "all_outputs_finite": True,
            }
        )
        print(
            json.dumps(
                {"stage": "case_checked", "case_id": row["case_id"], "group_id": row["group_id"]}
            ),
            flush=True,
        )
    after_digest = parameter_digest(backbone)
    if before_digest != after_digest or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("the previously frozen forecasting weights changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed_interface_only",
            "accuracy_evaluated": False,
            "prediction_future_values_read": False,
            "new_training": False,
            "case_count": len(rows),
            "logical_forecast_calls": calls,
            "helper_sha256": helper_sha,
            "script_sha256": file_sha256(Path(__file__)),
            "local_model_source_sha256": file_sha256(Path(inspect.getfile(type(backbone)))),
            "backbone_parameter_sha256": before_digest,
            "output_patch_size": patch_size,
            "quantiles": list(levels),
            "load_seconds": load_seconds,
            "probe_seconds": perf_counter() - infer_started,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "inventory_cases": len(inventory),
            "empty_prefix_target_cases": sum(bool(x["empty_prefix_targets"]) for x in inventory),
            "empty_prefix_channel_cases": sum(bool(x["empty_prefix_channels"]) for x in inventory),
            "empty_full_channel_cases": sum(bool(x["empty_full_channels"]) for x in inventory),
            "cases": rows,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed_interface_only",
                "cases": len(rows),
                "calls": calls,
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
