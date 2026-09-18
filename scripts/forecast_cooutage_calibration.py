"""Freeze new H24 forecasts with source-trained weights and unchanged R30 controls."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from cooutage_calibration_core import (
    LEARNED,
    NEW_METHODS,
    PARENT,
    ROOT,
    load_npz,
    mixed_context,
    new_gate,
    normalized_features,
    read_json,
    tensor_inputs,
)
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.forecasting.chronos_differentiable import chronos_median
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    training, output = args.training_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete calibrated forecasts")
    trained = read_json(training / "manifest.json")
    if trained["status"] != "completed":
        raise ValueError("finish source learning before evaluation forecasting")
    identity = {
        str(p): file_sha256(p)
        for p in (
            training / "manifest.json",
            Path(__file__),
            ROOT / "scripts/forecast_calibration_core.py",
            ROOT / "scripts/cooutage_calibration_core.py",
            PARENT / "peer-forecasts-v001/manifest.json",
            PARENT / "peer-inputs-v001/manifest.json",
            ROOT / "docs/iclr2027/R32_COOUTAGE_CALIBRATION_PROTOCOL.md",
        )
    }
    for name, sha in trained["files"].items():
        if file_sha256(training / name) != sha:
            raise ValueError("a source-trained artifact changed")
    scaler = load_npz(training / "feature_scaler.npz")
    fixed = read_json(training / "fixed_output.json")
    parent = read_json(PARENT / "peer-forecasts-v001/manifest.json")
    old = {r["case_id"]: r for r in parent["cases"]}
    torch.set_num_threads(1)
    started = perf_counter()
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    backbone.eval().requires_grad_(False)
    gates = {}
    for record in trained["learned"]:
        path = training / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a learned gate changed")
        gate = new_gate(record["method"])
        gate.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model"])
        gates[record["method"]] = gate.eval()
    entries = []
    for number, row in enumerate(trained["evaluation"]):
        d = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        f = load_npz(
            PARENT / "peer-forecasts-v001" / old[row["case_id"]]["path"],
            old[row["case_id"]]["sha256"],
        )
        features = load_npz(training / row["feature_path"], row["feature_sha256"])["features"]
        feature_tensor = normalized_features(features, scaler["mean"], scaler["scale"])
        tensor = tensor_inputs(d)
        methods = dict(zip(f["methods"].tolist(), f["points"], strict=True))
        alpha_rows, contexts = [], []
        input_methods = [*LEARNED, "half_input_mix"]
        with torch.inference_mode():
            for name in input_methods:
                alpha = (
                    gates[name](feature_tensor)
                    if name in gates
                    else torch.full((2,), 0.5, device="cuda")
                )
                context = mixed_context(tensor, alpha)
                methods[name] = (
                    chronos_median(pipeline, context, 24, [0, 1]).cpu().numpy().astype(float)
                )
                alpha_rows.append(alpha.cpu().numpy())
                contexts.append(context.cpu().numpy())
        methods["half_output_mix"] = (methods["local_ridge"] + methods["peer_ridge"]) * 0.5
        bank = np.stack([methods[n] for n in fixed["methods"]])
        for name, weights in (
            ("calibrated_output_global", fixed["global"]),
            ("calibrated_output_station", fixed["stations"][row["station"]]),
        ):
            methods[name] = np.stack(
                [np.asarray(weights[t]["weights"]) @ bank[:, :, t] for t in (0, 1)], axis=1
            )
        names = sorted(methods)
        points = np.stack([methods[n] for n in names])
        if len(names) != 44 or not np.isfinite(points).all():
            raise ValueError("the registered evaluation output set is incomplete")
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path,
            methods=np.asarray(names),
            points=points,
            input_methods=np.asarray(input_methods),
            alphas=np.stack(alpha_rows),
            contexts_z=np.stack(contexts),
            features=features,
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "panel": row["panel"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json", {"completed": number + 1, "total": len(trained["evaluation"])}
        )
    if (
        digest != trained["parameter_sha256"]
        or digest != parent["parameter_sha256"]
        or parameter_digest(backbone) != digest
    ):
        raise ValueError("the shared backbone changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": trained["smoke"],
            "identity": identity,
            "training_root": str(training),
            "cases": entries,
            "new_methods": list(NEW_METHODS),
            "parameter_sha256": digest,
            "new_model_forwards": 4 * len(entries),
            "evaluation_future_values_read": False,
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
