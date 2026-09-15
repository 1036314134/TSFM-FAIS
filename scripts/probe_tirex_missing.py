"""Check the pinned TiRex native API before adding a new evaluation backend."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting.adapters.tirex import TiRexAdapter  # noqa: E402
from tsfm_fais.forecasting.base import ForecastCapabilities  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


class TiRexMissingProbe(TiRexAdapter):
    """Local diagnostic override; the production adapter is not changed here."""

    capabilities = ForecastCapabilities(
        frozenset({"independent_univariate"}), 2048, supports_missing_context=True
    )


def digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        result.update(
            parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed TiRex probes")
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.reference_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = args.reference_root / "model/model.ckpt"
    if (
        reference["status"] != "completed"
        or file_sha256(checkpoint) != reference["identity"]["checkpoint_sha256"]
    ):
        raise ValueError("the pinned checkpoint changed")
    torch.set_num_threads(1)
    adapter = TiRexMissingProbe(str(checkpoint), device="cuda", batch_size=8, backend_name="torch")
    backend = adapter._ensure_backend().eval().requires_grad_(False)
    before = digest(backend)
    time = np.arange(96, dtype=float)
    base = np.sin(time * (2 * np.pi / 24)) + 0.01 * time
    names = [
        "complete",
        "interior_missing",
        "tail_missing",
        "leading_missing",
        "all_missing",
        "constant",
    ]
    contexts = np.tile(base, (6, 1))
    contexts[1, 24:48] = np.nan
    contexts[2, 64:] = np.nan
    contexts[3, :32] = np.nan
    contexts[4] = np.nan
    contexts[5] = 3.0
    spec = ForecastSpec(
        "tirex", "independent_univariate", 96, context_length=96, target_indices=(0,)
    )
    initial_precision = {
        "matmul": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    diagnostics = []
    tensor = torch.as_tensor(contexts, dtype=torch.float32, device="cuda")
    for precision in ("initial", "highest"):
        if precision == "highest":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        outputs = {}
        for batch_size in (1, 8):
            raw_quantiles, raw_median = backend.forecast(
                context=tensor, prediction_length=96, output_type="numpy", batch_size=batch_size
            )
            raw_quantiles, raw_median = np.asarray(raw_quantiles), np.asarray(raw_median)
            if (
                raw_quantiles.shape != (6, 96, 9)
                or raw_median.shape != (6, 96)
                or not np.isfinite(raw_quantiles).all()
            ):
                raise ValueError("the native TiRex shape or finiteness contract failed")
            np.testing.assert_allclose(raw_median, raw_quantiles[:, :, 4], rtol=0, atol=1e-7)
            adapter.batch_size = batch_size
            wrapped = adapter.predict_missing(contexts[:, :, None], spec).point[:, :, 0]
            np.testing.assert_allclose(wrapped, raw_median, rtol=0, atol=1e-7)
            outputs[batch_size] = raw_median
        adapter.batch_size = 1
        maximum_single_difference = 0.0
        for index in range(len(names)):
            single = adapter.predict_missing(contexts[index : index + 1, :, None], spec).point[
                0, :, 0
            ]
            np.testing.assert_allclose(single, outputs[1][index], rtol=0, atol=1e-7)
            maximum_single_difference = max(
                maximum_single_difference, float(np.abs(single - outputs[1][index]).max())
            )
        diagnostics.append(
            {
                "precision": precision,
                "single_vs_batched_one_maximum_difference": maximum_single_difference,
                "batch_one_vs_eight_maximum_difference": float(
                    np.abs(outputs[1] - outputs[8]).max()
                ),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)
    if digest(backend) != before:
        raise ValueError("the TiRex parameters changed during native inference")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "reference_manifest_sha256": file_sha256(args.reference_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "adapter_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/adapters/tirex.py"),
            "cases": names,
            "quantile_shape": list(raw_quantiles.shape),
            "median_quantile_parity": True,
            "precision_before_probe": initial_precision,
            "packing_diagnostics": diagnostics,
            "evaluation_policy": {
                "batch_size": 1,
                "matmul_precision": "highest",
                "matmul_tf32": False,
                "cudnn_tf32": False,
            },
            "parameters_unchanged": True,
            "parameter_sha256": before,
            "backend": "torch; compilation disabled",
            "limits": "synthetic native-interface and parameter check only; no real-data accuracy conclusion",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
