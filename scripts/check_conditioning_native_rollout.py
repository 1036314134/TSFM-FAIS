"""Compare the proposed repair with Chronos's existing median-path rollout, without labels."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError("preserve previous comparison results")
    probe = ROOT / "artifacts/pro-conditioning-review-20260916/interface-v002"
    manifest = json.loads((probe / "manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((probe / "selection.json").read_text(encoding="utf-8"))
    selected = {row["case_id"]: row for row in selected}
    source = ROOT / "artifacts/iclr27-r25/long-inputs-v001"
    torch.set_num_threads(1)
    started = perf_counter()
    _, adapter, model, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    median = next(i for i, q in enumerate(pipeline.quantiles) if q == 0.5)
    output.mkdir(parents=True)
    rows = []
    print(json.dumps({"stage": "model_loaded", "seconds": perf_counter() - started}), flush=True)
    for row in manifest["cases"]:
        case = row["case_id"]
        with np.load(probe / "cases" / f"{case}.npz", allow_pickle=False) as saved:
            context = saved["first_context"]
            recent = saved["first_future"]
            expected = saved["repaired"]
            expected_context = saved["second_context"]
        with np.load(source / selected[case]["path"], allow_pickle=False) as saved:
            scale = saved["scale"]
        future = np.concatenate([recent, np.full_like(recent, np.nan)], axis=-1)
        traces, raw_outputs = [], []

        def capture(module, positional, kwargs, *, sink=traces):
            sink.append(kwargs["context"].detach().cpu().numpy().copy())

        def capture_output(module, positional, result, *, sink=raw_outputs):
            sink.append(result.quantile_preds.detach().cpu().numpy().copy())

        hook = model.register_forward_pre_hook(capture, with_kwargs=True)
        output_hook = model.register_forward_hook(capture_output)
        try:
            with torch.inference_mode():
                result = pipeline._predict_batch(
                    context=torch.tensor(np.ascontiguousarray(context), dtype=torch.float32),
                    group_ids=torch.zeros(context.shape[0], dtype=torch.long),
                    future_covariates=torch.tensor(future, dtype=torch.float32),
                    unrolled_quantiles_tensor=torch.tensor([0.5]),
                    prediction_length=192,
                    max_output_patches=96 // pipeline.model_output_patch_size,
                    target_idx_ranges=[(0, context.shape[0])],
                )[0]
        finally:
            hook.remove()
            output_hook.remove()
        if len(traces) != 2:
            raise ValueError("expected two forwards in the native single-path rollout")
        actual = result[:, median, 96:192].numpy().T
        raw_second = raw_outputs[1][:, median, :96].T
        context_equal = np.array_equal(
            traces[1], expected_context.astype(np.float32).T, equal_nan=True
        )
        delta = float(np.max(np.abs(actual - expected) / scale))
        _save_npz(
            output / "cases" / f"{case}.npz",
            native_rollout=actual,
            proposed=expected,
            native_second_context=traces[1],
            proposed_second_context=expected_context,
            native_second_raw_median=raw_second,
        )
        rows.append(
            {
                "case_id": case,
                "group_id": row["group_id"],
                "forwards": len(traces),
                "second_context_exact": context_equal,
                "prediction_exact": np.array_equal(actual, expected),
                "max_scaled_prediction_difference": delta,
                "raw_second_prediction_exact": np.array_equal(raw_second, expected),
                "raw_second_max_scaled_difference": float(
                    np.max(np.abs(raw_second - expected) / scale)
                ),
            }
        )
    if parameter_digest(model) != digest:
        raise ValueError("frozen model weights changed")
    result = {
        "status": "completed_no_accuracy_readout",
        "new_future_values_read": False,
        "cases": rows,
        "model_forwards": sum(r["forwards"] for r in rows),
        "wall_seconds": perf_counter() - started,
        "script_sha256": file_sha256(Path(__file__)),
        "basis_manifest_sha256": file_sha256(probe / "manifest.json"),
        "native_unrolled_quantiles": [0.5],
        "native_max_output_patches": 6,
    }
    _write_json(output / "manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
