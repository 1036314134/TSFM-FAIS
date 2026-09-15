"""Collect frozen complete-history teachers for the 32 used R6 synthetic origins."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from r6_runtime import forecast_spec, make_forecaster  # noqa: E402

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def synthetic_origins(episodes):
    groups = {}
    for record in episodes:
        if record["panel"] != "new_synthetic":
            continue
        key = (record["dataset_id"], record["item_id"], record["window"]["origin"])
        group = groups.setdefault(record["origin_id"], {"key": key, "count": 0})
        if group["key"] != key:
            raise ValueError("one origin identifier describes different histories")
        group["count"] += 1
    if len(groups) != 32 or any(row["count"] != 36 for row in groups.values()):
        raise ValueError("the registered synthetic origin coverage changed")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort-root",
        "prepared-root",
        "audit-root",
        "legacy-bundle",
        "previous-forecasts",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed teacher collections")
    cohort_path, prep_path = (
        args.cohort_root / "manifest.json",
        args.prepared_root / "manifest.json",
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    audit = json.loads((args.audit_root / "manifest.json").read_text(encoding="utf-8"))
    if any(row["status"] != "completed" for row in (cohort, prep, audit)) or prep["identity"][
        "cohort_sha256"
    ] != file_sha256(cohort_path):
        raise ValueError("finish the matching input and confirmation audits first")
    sources = {(row["dataset_id"], row["item_id"]): row for row in cohort["sources"]}
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("the original prefix statistics changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    origins = synthetic_origins(prep["episodes"])
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "cohort_sha256": file_sha256(cohort_path),
        "prepared_sha256": file_sha256(prep_path),
        "confirmation_audit_sha256": file_sha256(args.audit_root / "manifest.json"),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
        "model_id": args.model,
        "context_length": 96,
        "horizons": [96, 192],
        "origin_count": 32,
        "limits": "complete histories unavailable at deployment; explanatory reuse after R6 confirmation",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial teacher collection changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    runner, adapter, backbone, digest, joint = make_forecaster(
        args.model, args.legacy_bundle, args.previous_forecasts
    )
    records, verified_sources = [], set()
    for origin_id, group in sorted(origins.items()):
        dataset, item, origin = group["key"]
        source = sources[(dataset, item)]
        source_path = Path(source["path"])
        if (dataset, item) not in verified_sources:
            if file_sha256(source_path) != source["sha256"]:
                raise ValueError("an original source trajectory changed")
            verified_sources.add((dataset, item))
        clean = np.load(source_path, mmap_mode="r")[origin - 96 : origin].copy(order="K")
        if clean.shape != (96, source["shape"][1]) or not np.isfinite(clean).all():
            raise ValueError("a teacher history is incomplete or has changed dimensions")
        scaler = scalers[(dataset, item)]
        mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
        key = hashlib.sha256(origin_id.encode()).hexdigest()[:24]
        clean_sha = hashlib.sha256(np.ascontiguousarray(clean).tobytes()).hexdigest()
        for horizon in (96, 192):
            path = output / f"h{horizon}" / f"{key}.npz"
            if not path.exists():
                point = (
                    runner.predict(clean[None], forecast_spec(args.model, horizon, joint)).point[0]
                    - mean
                ) / scale
                _save_npz(
                    path,
                    point_z=point,
                    clean_context=clean,
                    clean_sha256=np.asarray(clean_sha),
                    identity_sha256=np.asarray(identity_sha),
                    parameter_sha256=np.asarray(digest),
                )
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["identity_sha256"]) != identity_sha
                    or str(saved["parameter_sha256"]) != digest
                    or str(saved["clean_sha256"]) != clean_sha
                    or saved["point_z"].shape != (horizon, 2)
                    or not np.isfinite(saved["point_z"]).all()
                ):
                    raise ValueError("a cached teacher is invalid or from a different runtime")
                np.testing.assert_array_equal(saved["clean_context"], clean)
            records.append(
                {
                    "origin_id": origin_id,
                    "dataset_id": dataset,
                    "item_id": item,
                    "origin": origin,
                    "horizon": horizon,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "clean_sha256": clean_sha,
                }
            )
        print(
            json.dumps(
                {"model": args.model, "completed_origins": len(records) // 2, "total_origins": 32}
            ),
            flush=True,
        )
    if len(records) != 64 or parameter_digest(backbone) != digest:
        raise ValueError("teacher coverage or forecaster parameters changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "parameter_sha256": digest,
            "parameters_unchanged": True,
            "teachers": records,
            "historical_prediction_requests": len(records),
            "evaluation_future_arrays_read": False,
            "limits": "teachers collected without scoring; all 32 complete inputs checked against original trajectories",
        },
    )


if __name__ == "__main__":
    main()
