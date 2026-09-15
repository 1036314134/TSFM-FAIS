"""Compare cached, singleton SDK and raw Chronos calls on a fixed failing task."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed interface diagnostic")
    source_root = ROOT / "artifacts/iclr27-r3/development-expanded-v001"
    accuracy_root = ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    controls_root = ROOT / "artifacts/iclr27-r4/history-controls-screening-v001/chronos2"
    motm_root = ROOT / "artifacts/iclr27-r5/motm-development-v001"
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    config = source["identity"]["config"]
    controls = json.loads((controls_root / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        row
        for row in source["episodes"]
        if row["episode_id"] == controls["episodes"][28]["episode_id"]
    )
    scaler = next(
        row
        for row in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
        if row["dataset_id"] == record["dataset_id"] and row["item_id"] == record["item_id"]
    )
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    with np.load(source_root / record["path"], allow_pickle=False) as saved:
        values = ((saved["candidate_values"] - mean) / scale).astype(np.float32)
    with np.load(controls_root / controls["episodes"][28]["path"], allow_pickle=False) as saved:
        cached = saved["point_z"][:6]
    prepared = json.loads((motm_root / "prepared_manifest.json").read_text(encoding="utf-8"))
    p = next(row for row in prepared["records"] if row["episode_id"] == record["episode_id"])
    forecast_manifest = json.loads(
        (motm_root / "chronos2/manifest.json").read_text(encoding="utf-8")
    )
    f = next(
        row for row in forecast_manifest["predictions"] if row["episode_id"] == record["episode_id"]
    )
    with np.load(motm_root / p["path"], allow_pickle=False) as saved:
        motm_values = saved["completed_z"].astype(np.float32)
    with np.load(motm_root / f["path"], allow_pickle=False) as saved:
        motm_cached = saved["point_z"]
    from chronos import BaseChronosPipeline

    torch.set_num_threads(1)
    pipeline = BaseChronosPipeline.from_pretrained(
        config["forecaster_artifacts"]["chronos2"], device_map="cuda"
    )
    model = pipeline.model.eval().requires_grad_(False)
    median = int(np.flatnonzero(np.isclose(pipeline.quantiles, 0.5))[0])
    horizon, targets = config["horizon"], list(config["target_indices"])
    patches = math.ceil(horizon / pipeline.model_output_patch_size)

    def sdk(bank, batch_size):
        q, _ = pipeline.predict_quantiles(
            inputs=[{"target": value.T} for value in bank],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=batch_size,
            predict_batches_jointly=False,
        )
        return np.stack([value.numpy()[:, :, 1].T[:, targets] for value in q])

    def raw(bank, future, contiguous=False):
        count, length, dims = bank.shape
        context = torch.tensor(bank.transpose(0, 2, 1).reshape(count * dims, length), device="cuda")
        if contiguous:
            context = context.contiguous()
        groups = torch.arange(count, device="cuda").repeat_interleave(dims)
        kwargs = (
            {
                "future_covariates": torch.full(
                    (count * dims, patches * pipeline.model_output_patch_size),
                    float("nan"),
                    device="cuda",
                )
            }
            if future
            else {}
        )
        with torch.no_grad():
            q = model(
                context=context, group_ids=groups, num_output_patches=patches, **kwargs
            ).quantile_preds
        return (
            q[:, median, :horizon]
            .reshape(count, dims, horizon)
            .transpose(1, 2)[:, :, targets]
            .cpu()
            .numpy()
        )

    arrays, rows = {}, []
    for name, bank, reference in (("finite", values, cached), ("motm", motm_values, motm_cached)):
        arrays[name + "_inputs"] = bank
        arrays[name + "_cached"] = reference
        versions = {
            "sdk_batch8": sdk(bank, 8),
            "sdk_batch1": sdk(bank, 1),
            "raw_single_none": np.concatenate([raw(item[None], False) for item in bank]),
            "raw_single_nan": np.concatenate([raw(item[None], True) for item in bank]),
            "raw_single_contiguous": np.concatenate(
                [raw(item[None], False, True) for item in bank]
            ),
            "raw_all_none": raw(bank, False),
            "raw_all_nan": raw(bank, True),
        }
        for variant, point in versions.items():
            arrays[name + "_" + variant] = point
            rows.append(
                {
                    "bank": name,
                    "variant": variant,
                    "max_diff_cached": float(np.abs(point - reference).max()),
                    "max_diff_single_sdk": float(np.abs(point - versions["sdk_batch1"]).max()),
                }
            )
    np.savez_compressed(output / "arrays.npz", **arrays)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "episode_id": record["episode_id"],
            "comparisons": rows,
            "torch_version": torch.__version__,
            "model_dtype": str(model.dtype),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "script_sha256": file_sha256(Path(__file__)),
            "arrays_sha256": file_sha256(output / "arrays.npz"),
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
