"""Verify differentiable TimesFM medians against the configured public backend."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probe_differentiable_imputation import compose, parameter_digest  # noqa: E402

from tsfm_fais.forecasting.adapters.timesfm import TimesFM2p5Adapter  # noqa: E402
from tsfm_fais.forecasting.timesfm_differentiable import timesfm_median  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(6101)
    source_root, accuracy_root, output = (
        args.source_root.resolve(),
        args.accuracy_root.resolve(),
        args.output_root.resolve(),
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed TimesFM feasibility evidence")
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if accuracy["source_episode_manifest_sha256"] != file_sha256(
        source_root / "episodes_manifest.json"
    ):
        raise ValueError("source episodes and standardizers do not match")
    config = source["identity"]["config"]
    targets, horizon = list(config["target_indices"]), config["horizon"]
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
    }
    cases = []
    selected = []
    for dataset in ("ETTh1", "exchange_rate", "Coastal_T_S_H"):
        options = [
            row
            for row in source["episodes"]
            if row["dataset_id"] == dataset
            and row["split"] == "train"
            and row["mechanism"] == "independent_block"
            and np.isclose(row["missing_rate"], 0.3)
            and row["mask_seed"] == min(config["mask_seeds"])
        ]
        record = min(options, key=lambda row: row["origin"])
        selected.append(record["episode_id"])
        path = source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("training probe cache changed")
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        with np.load(path, allow_pickle=False) as episode:
            context = torch.tensor((episode["context"] - mean) / scale, dtype=torch.float32)
            candidates = torch.tensor(
                (episode["candidate_values"] - mean) / scale, dtype=torch.float32
            )
            truth = torch.tensor(
                ((episode["future"] - mean) / scale)[:, targets], dtype=torch.float64
            )
        initial, _ = compose(candidates, context, torch.zeros(len(candidates), 1))
        cases.append((record["episode_id"], initial, truth, targets))
    first = cases[0]
    cases.append(("trimmed_80_step_history", first[1][-80:], first[2], first[3]))
    cases.append(
        (
            "constant_target_history",
            torch.tensor([[1.0, -2.0]]).expand(96, -1).clone(),
            torch.zeros(horizon, 2, dtype=torch.float64),
            [0, 1],
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "source_manifest_sha256": file_sha256(source_root / "episodes_manifest.json"),
        "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "gradient_module_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/timesfm_differentiable.py"
        ),
        "episodes": selected,
        "checkpoint": config["forecaster_artifacts"]["timesfm2p5"],
        "horizon": horizon,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("TimesFM probe identity changed; choose a new output directory")
    _write_json(identity_path, identity)
    for name, path in (
        ("script_snapshot.py", Path(__file__)),
        ("gradient_snapshot.py", ROOT / "src/tsfm_fais/forecasting/timesfm_differentiable.py"),
    ):
        (output / name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    adapter = TimesFM2p5Adapter(model_name=identity["checkpoint"], device="cuda", batch_size=2)
    backend = adapter._ensure_backend()
    core = backend.model.eval().requires_grad_(False)
    before = parameter_digest(core)
    records = []
    started = monotonic()
    for name, cpu_context, cpu_truth, case_targets in cases:
        adapter._maybe_compile(backend, len(cpu_context), horizon)
        official, _ = backend.forecast(
            horizon=horizon,
            inputs=[cpu_context[:, target].numpy().copy() for target in case_targets],
        )
        context = cpu_context.cuda().detach().requires_grad_(True)
        point = timesfm_median(core, context, horizon, case_targets)
        error = float(np.max(np.abs(point.detach().cpu().numpy() - official.T)))
        np.testing.assert_allclose(point.detach().cpu().numpy(), official.T, rtol=2e-4, atol=2e-4)
        loss = (point - cpu_truth.cuda()).square().mean()
        loss.backward()
        if context.grad is None or not bool(torch.isfinite(context.grad).all()):
            raise ValueError("TimesFM input gradients are missing or nonfinite")
        auxiliary = [index for index in range(context.shape[1]) if index not in case_targets]
        if auxiliary and not torch.equal(
            context.grad[:, auxiliary], torch.zeros_like(context.grad[:, auxiliary])
        ):
            raise ValueError("independent TimesFM unexpectedly depends on auxiliary columns")
        result = {
            "case": name,
            "maximum_forward_difference": error,
            "gradient_norm": float(context.grad.norm()),
            "finite_gradients": True,
            "auxiliary_gradient_zero": True,
            "source_mse": float(loss.detach()),
        }
        records.append(result)
        _write_json(
            output / "progress.json",
            {"completed": len(records), "total": len(cases), "records": records},
        )
        print(json.dumps(result), flush=True)
    if parameter_digest(core) != before or any(
        parameter.grad is not None or parameter.requires_grad for parameter in core.parameters()
    ):
        raise ValueError("TimesFM forecasting parameters changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "records": records,
            "parameter_digest_unchanged": True,
            "forecaster_parameter_sha256": before,
            "sdk_model_sha256": file_sha256(Path(inspect.getfile(type(core)))),
            "elapsed_seconds": monotonic() - started,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "interpretation": "technical forward and gradient parity only; no deployment policy is trained and no method superiority is implied",
        },
    )


if __name__ == "__main__":
    main()
