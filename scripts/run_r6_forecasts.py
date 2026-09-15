"""Generate all frozen R6 candidate forecasts at both registered horizons."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from r6_runtime import forecast_spec, make_forecaster, query_r6_candidates  # noqa: E402

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "input-audit-root",
        "horizon-probe",
        "method-freeze",
        "legacy-bundle",
        "previous-forecasts",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed R6 forecasts")
    prep_path = args.prepared_root / "manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    audit = json.loads((args.input_audit_root / "manifest.json").read_text(encoding="utf-8"))
    probe = json.loads((args.horizon_probe / "manifest.json").read_text(encoding="utf-8"))
    method = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    if (
        any(row["status"] != "completed" for row in (prep, audit, probe, method))
        or audit["failed_imputer_fits"]
    ):
        raise ValueError("finish the input and horizon checks before confirmation forecasting")
    if audit["prepared_sha256"] != file_sha256(prep_path) or probe[
        "prepared_sha256"
    ] != file_sha256(prep_path):
        raise ValueError("the checks refer to different input preparation")
    if probe["runtime_sha256"] != file_sha256(ROOT / "scripts/r6_runtime.py"):
        raise ValueError("the probed runtime changed")
    if prep["identity"]["method_freeze_sha256"] != file_sha256(args.method_freeze):
        raise ValueError("the primary method changed after cohort/input construction")
    for record in method["source_models"]:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("a frozen primary or matched-control source gate changed")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("prefix standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "horizons": [96, 192],
        "prepared_sha256": file_sha256(prep_path),
        "input_audit_sha256": file_sha256(args.input_audit_root / "manifest.json"),
        "horizon_probe_sha256": file_sha256(args.horizon_probe / "manifest.json"),
        "method_freeze_sha256": file_sha256(args.method_freeze),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
        "candidate_ids": [*prep["identity"]["candidate_ids"], "guarded_direct", "motm_reference"],
        "context_length": 96,
        "input_scale": "raw",
        "target_indices": [0, 1],
        "information_boundary": "current prepared histories only; no future arrays or prediction errors are read",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial R6 forecasting identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    runner, adapter, backbone, digest, joint = make_forecaster(
        args.model, args.legacy_bundle, args.previous_forecasts
    )
    if digest != next(
        row["parameter_sha256"] for row in probe["models"] if row["model_id"] == args.model
    ):
        raise ValueError("the forecaster differs from the verified horizon probe")
    horizon_records = []
    for horizon in (96, 192):
        directory = output / f"h{horizon}"
        directory.mkdir(exist_ok=True)
        marker = directory / "manifest.json"
        if marker.exists():
            completed = json.loads(marker.read_text(encoding="utf-8"))
            if (
                completed["identity_sha256"] != identity_sha
                or completed["parameter_sha256"] != digest
            ):
                raise ValueError("a completed horizon belongs to another runtime")
            for row in completed["predictions"]:
                if file_sha256(directory / row["path"]) != row["sha256"]:
                    raise ValueError("a completed forecast changed")
            horizon_records.append(
                {
                    "horizon": horizon,
                    "path": str(marker.relative_to(output)),
                    "sha256": file_sha256(marker),
                }
            )
            continue
        spec = forecast_spec(args.model, horizon, joint)
        files = []
        for index, record in enumerate(prep["episodes"]):
            source = args.prepared_root / record["path"]
            if file_sha256(source) != record["sha256"]:
                raise ValueError("a frozen candidate input changed")
            path = directory / "predictions" / source.name
            if not path.exists():
                with np.load(source, allow_pickle=False) as saved:
                    context, candidates = saved["context"], saved["candidate_values"]
                    actions, motm = saved["candidate_ids"].tolist(), saved["motm_values"]
                if [*actions, "guarded_direct", "motm_reference"] != identity["candidate_ids"]:
                    raise ValueError("candidate identities or order changed")
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                points, diagnostics = query_r6_candidates(
                    runner,
                    spec,
                    context,
                    candidates,
                    actions,
                    motm,
                    np.asarray(scaler["mean"]),
                    np.asarray(scaler["scale"]),
                    joint=joint,
                )
                _save_npz(
                    path,
                    point_z=points,
                    candidate_ids=np.asarray(identity["candidate_ids"]),
                    diagnostics=np.asarray(json.dumps(diagnostics)),
                    source_sha256=np.asarray(record["sha256"]),
                    parameter_sha256=np.asarray(digest),
                    identity_sha256=np.asarray(identity_sha),
                )
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["source_sha256"]) != record["sha256"]
                    or str(saved["identity_sha256"]) != identity_sha
                    or str(saved["parameter_sha256"]) != digest
                ):
                    raise ValueError("partial forecast provenance changed")
                if (
                    saved["point_z"].shape != (8, horizon, 2)
                    or not np.isfinite(saved["point_z"]).all()
                ):
                    raise ValueError("a forecast is truncated or nonfinite")
                diagnostics = json.loads(str(saved["diagnostics"]))
            files.append(
                {
                    "episode_id": record["episode_id"],
                    "path": str(path.relative_to(directory)),
                    "sha256": file_sha256(path),
                    **diagnostics,
                }
            )
            _write_json(
                output / "progress.json",
                {
                    "status": "forecasting",
                    "horizon": horizon,
                    "completed_inputs_in_horizon": index + 1,
                    "total_inputs": len(prep["episodes"]),
                },
            )
            if (index + 1) % 100 == 0 or index + 1 == len(prep["episodes"]):
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "horizon": horizon,
                            "completed_inputs": index + 1,
                            "total_inputs": len(prep["episodes"]),
                        }
                    ),
                    flush=True,
                )
        if len(files) != len(prep["episodes"]) or {row["episode_id"] for row in files} != {
            row["episode_id"] for row in prep["episodes"]
        }:
            raise ValueError("a horizon lost or duplicated input tasks")
        _write_json(
            marker,
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "horizon": horizon,
                "parameter_sha256": digest,
                "predictions": files,
                "future_arrays_read": False,
            },
        )
        horizon_records.append(
            {
                "horizon": horizon,
                "path": str(marker.relative_to(output)),
                "sha256": file_sha256(marker),
            }
        )
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen forecasting parameters changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "horizons": horizon_records,
            "parameter_sha256": digest,
            "parameters_unchanged": True,
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": "candidate forecasts only; no R6 policy scores or accuracy conclusions yet",
        },
    )


if __name__ == "__main__":
    main()
