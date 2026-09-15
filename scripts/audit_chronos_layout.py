"""Check the Chronos-2 SDK contract and legacy median recovery on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting.accuracy import recover_legacy_chronos_median  # noqa: E402
from tsfm_fais.forecasting.adapters import Chronos2Adapter  # noqa: E402
from tsfm_fais.forecasting.adapters._utils import stack_payload  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[-1]])
    import torch

    torch.set_num_threads(1)
    root = args.source_root
    manifest = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "chronos2/forecast_identity.json").read_text(encoding="utf-8"))
    config = manifest["identity"]["config"]
    targets = config["target_indices"]
    spec = ForecastSpec(
        "chronos2", "joint_multivariate", config["horizon"], target_indices=tuple(targets)
    )
    selected = [
        next(
            record
            for record in manifest["episodes"]
            if record["episode_id"]
            == "Coastal_T_S_H|BMP120|6170|validation|value_dependent|0.3|6101"
        )
    ]
    for dataset in ("Coastal_T_S_H", "azure2019_D_5T", "ETTh1"):
        selected.append(
            next(
                record
                for record in manifest["episodes"]
                if record["dataset_id"] == dataset
                and record["mechanism"] == "random_point"
                and record["split"] == "validation"
            )
        )
    adapter = Chronos2Adapter(model_name=identity["checkpoint"], device=args.device, batch_size=32)
    backend = adapter._ensure_backend()
    output = []
    for record in selected:
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("episode cache changed")
        with (
            np.load(path, allow_pickle=False) as e,
            np.load(root / "chronos2/predictions" / path.name, allow_pickle=False) as cached,
        ):
            ids = e["candidate_ids"].tolist()
            action_indices = [ids.index("locf"), ids.index("seasonal_lag"), len(ids)]
            contexts = [e["candidate_values"][index] for index in action_indices[:-1]] + [
                e["context"]
            ]
            complete = [*e["candidate_values"], e["clean_context"]]
            all_native, _ = backend.predict_quantiles(
                inputs=[{"target": values.astype(np.float32).T} for values in complete],
                prediction_length=spec.horizon,
                quantile_levels=list(spec.quantile_levels),
                batch_size=32,
                predict_batches_jointly=False,
            )
            missing_native, _ = backend.predict_quantiles(
                inputs=[{"target": e["context"].astype(np.float32).T}],
                prediction_length=spec.horizon,
                quantile_levels=list(spec.quantile_levels),
                batch_size=32,
                predict_batches_jointly=False,
            )
            native = [all_native[index] for index in action_indices[:-1]] + [missing_native[0]]
            raw = stack_payload(native)
            assert raw.shape == (3, e["context"].shape[1], spec.horizon, 3)
            expected = np.take(raw[..., 1].transpose(0, 2, 1), targets, axis=-1)
            converted = np.concatenate(
                [
                    adapter.predict(np.stack(contexts[:2]), spec).point,
                    adapter.predict_missing(contexts[2][None], spec).point,
                ]
            )
            np.testing.assert_allclose(converted, expected, atol=1e-4, rtol=1e-5)
            d = e["context"].shape[1]
            recovered = (
                recover_legacy_chronos_median(
                    cached["quantiles"], targets, list(spec.quantile_levels), d
                )
                if d == len(spec.quantile_levels)
                else cached["point"]
            )
            recovered = recovered[action_indices]
            agrees = bool(np.allclose(recovered, expected, atol=1e-3, rtol=1e-4))
            result = {
                "episode_id": record["episode_id"],
                "dimensions": d,
                "sdk_layout": list(raw.shape),
                "adapter_max_abs_difference": float(np.max(np.abs(converted - expected))),
                "old_cache_max_abs_difference": float(
                    np.max(np.abs(cached["point"][action_indices] - expected))
                ),
                "recovered_cache_max_abs_difference": float(np.max(np.abs(recovered - expected))),
                "cpu_gpu_cache_agrees": agrees,
            }
            output.append(result)
            print(json.dumps(result), flush=True)
    _write_json(
        args.output,
        {
            "status": "verified"
            if all(case["cpu_gpu_cache_agrees"] for case in output)
            else "sdk_verified_cache_numerics_differ",
            "device": args.device,
            "threads": 1,
            "gpu_calls": 0 if args.device == "cpu" else len(selected) * 4,
            "script_sha256": file_sha256(Path(__file__)),
            "adapter_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/adapters/chronos.py"),
            "source_manifest_sha256": file_sha256(root / "episodes_manifest.json"),
            "cpu_gpu_tolerance": {"atol": 1e-3, "rtol": 1e-4},
            "cases": output,
        },
    )
    if args.device == "cuda" and not all(case["cpu_gpu_cache_agrees"] for case in output):
        raise ValueError(
            "same-device reproduction differs from recovered cache; audit before reuse"
        )


if __name__ == "__main__":
    main()
