"""Verify canonical forward values, source gradients and complete-input identity."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT
from learned_patch_repair import PatchRepair, RepairHook, repair_dimensions
from patch_repair_inputs import load_case, source_identity, source_population
from patch_repair_runtime import differentiable_point, repair_loss
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed learned-repair diagnostics")
    rows, _ = source_population()
    training = [r for r in rows if r["split"] == "train"]
    selected = [training[0], max(training, key=lambda r: r["dimensions"])]
    torch.set_num_threads(1)
    records = []
    for model_id in ("chronos2", "timesfm2p5"):
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        width, patch = repair_dimensions(backbone, model_id)
        for row in selected:
            raw, observed, truth = load_case(row)
            values = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
            scale = torch.tensor(row["scaler"]["scale"][:2], device="cuda", dtype=torch.float64)
            mean = torch.tensor(row["scaler"]["mean"][:2], device="cuda", dtype=torch.float64)
            targets = torch.as_tensor(truth, device="cuda", dtype=torch.float64)
            spec = ForecastSpec(
                model_id,
                "joint_multivariate" if joint else "independent_univariate",
                96,
                context_length=96,
                target_indices=[0, 1],
            )
            public = runner.predict_missing(raw[None], spec).point[0]
            baseline = differentiable_point(model_id, adapter, backbone, values)
            difference = float(
                (
                    abs(baseline.detach().cpu().numpy() - public)
                    / np.asarray(row["scaler"]["scale"][:2])
                ).max()
            )
            if difference > 1e-6:
                raise ValueError(f"canonical differentiable parity failed: {difference}")
            torch.manual_seed(5101)
            repair = PatchRepair(width, patch).to("cuda")
            optimizer = torch.optim.AdamW(
                repair.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False
            )
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            with RepairHook(backbone, model_id, observed, repair, "pattern") as hook:
                point = differentiable_point(model_id, adapter, backbone, values)
                np.testing.assert_array_equal(point.detach().cpu(), baseline.detach().cpu())
                loss = repair_loss(point, targets, mean, scale, hook.penalty)
                loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0))
            if (
                not np.isfinite(norm)
                or norm <= 0
                or any(p.grad is not None for p in backbone.parameters())
            ):
                raise ValueError("source gradients did not remain confined to the new module")
            optimizer.step()
            elapsed = perf_counter() - started
            with (
                torch.no_grad(),
                RepairHook(backbone, model_id, np.ones_like(observed), repair, "pattern"),
            ):
                complete = differentiable_point(model_id, adapter, backbone, values)
            np.testing.assert_array_equal(complete.cpu(), baseline.detach().cpu())
            long_raw, long_observed = np.tile(raw, (2, 1)), np.tile(observed, (2, 1))
            long_tensor = torch.as_tensor(long_raw, device="cuda", dtype=torch.float32)
            long_baseline = differentiable_point(model_id, adapter, backbone, long_tensor)
            with (
                torch.no_grad(),
                RepairHook(backbone, model_id, np.ones_like(long_observed), repair, "fraction"),
            ):
                long_complete = differentiable_point(model_id, adapter, backbone, long_tensor)
            np.testing.assert_array_equal(long_complete.cpu(), long_baseline.detach().cpu())
            records.append(
                {
                    "model_id": model_id,
                    "episode_id": row["episode_id"],
                    "dimensions": row["dimensions"],
                    "parameter_count": sum(p.numel() for p in repair.parameters()),
                    "embedding_width": width,
                    "patch_size": patch,
                    "canonical_standardized_max_difference": difference,
                    "source_gradient_norm_before_clip": norm,
                    "one_training_step_seconds": elapsed,
                    "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "long_context_test_is_synthetic_interface_only": True,
                    "backbone_sha256": digest,
                }
            )
            print(f"verified {model_id} D={row['dimensions']}", flush=True)
            del (
                optimizer,
                repair,
                point,
                loss,
                hook,
                baseline,
                complete,
                long_complete,
                long_baseline,
            )
            gc.collect()
            torch.cuda.empty_cache()
        if parameter_digest(backbone) != digest:
            raise ValueError("the backbone changed during adapter-only gradients")
        del runner, adapter, backbone
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_identity": source_identity(),
            "records": records,
            "files": {
                str(p.relative_to(ROOT)): file_sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/learned_patch_repair.py",
                    ROOT / "scripts/patch_repair_runtime.py",
                    ROOT / "docs/iclr2027/R24_LEARNED_PATCH_REPAIR_PLAN.md",
                )
            },
            "evaluation_future_labels_read": False,
            "source_training_labels_used": True,
            "accuracy_improvement_established": False,
        },
    )


if __name__ == "__main__":
    main()
