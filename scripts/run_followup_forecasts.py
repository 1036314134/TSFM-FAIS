"""Query the registered seven candidate inputs before reading follow-up futures."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from replay_preforecast_student import query_candidate_points  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "input-audit-root",
        "source-bundle",
        "old-bundle",
        "old-evaluation-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed follow-up forecasts")
    prep_path = args.prepared_root / "manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    audit = json.loads((args.input_audit_root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((args.old_bundle / "manifest.json").read_text(encoding="utf-8"))
    old_eval = json.loads(
        (args.old_evaluation_root / args.model / "manifest.json").read_text(encoding="utf-8")
    )
    if any(row["status"] != "completed" for row in (prep, audit, bundle, old, old_eval)):
        raise ValueError("complete the input audit and source freezes first")
    if audit["prepared_sha256"] != file_sha256(prep_path) or audit["verified_episodes"] != len(
        prep["episodes"]
    ):
        raise ValueError("the input audit does not cover this preparation")
    if audit["failed_imputer_fits"]:
        raise ValueError("review failed prefix fits before confirming the method's accuracy")
    if prep["identity"]["source_bundle_sha256"] != file_sha256(
        args.source_bundle / "manifest.json"
    ):
        raise ValueError("the source method changed after input preparation")
    if Path(bundle["identity"]["confirmation_root"]).resolve() != args.output_root.resolve():
        raise ValueError("use the confirmation destination fixed with the source methods")
    for name, sha in old["identity"]["runtime_code_sha256"].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError("the original frozen forecaster runtime changed")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("follow-up standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "prepared_manifest_sha256": file_sha256(prep_path),
        "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
        "old_bundle_sha256": file_sha256(args.old_bundle / "manifest.json"),
        "input_audit_sha256": file_sha256(args.input_audit_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "context_length": 96,
        "horizon": 96,
        "targets": [0, 1],
        "input_scale": "raw",
        "candidate_ids": [*prep["identity"]["candidate_ids"], "guarded_direct"],
        "expected_parameter_sha256": old_eval["parameter_sha256"],
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/replay_preforecast_student.py",
                "scripts/evaluate_timesfm_vendor_missing.py",
                "src/tsfm_fais/routing/preforecast_replay.py",
                "src/tsfm_fais/forecasting/accuracy.py",
            )
        },
        "feature_outcome_policy": "future arrays and predictive metrics are not read in this stage",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial forecast stage changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    registry = default_forecast_registry()
    joint = registry.get(args.model).mode == "joint_multivariate"
    model_path = old["identity"]["forecaster_artifacts"][args.model]
    adapter = (
        registry.build(args.model, model_name=model_path, device="cuda", batch_size=8)
        if joint
        else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    parameter_sha = parameter_digest(backbone)
    if parameter_sha != identity["expected_parameter_sha256"]:
        raise ValueError("forecaster parameters differ from the prior audited runtime")
    spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=96, target_indices=[0, 1]
    )
    predictions = []
    for record in prep["episodes"]:
        source = args.prepared_root / record["path"]
        if file_sha256(source) != record["sha256"]:
            raise ValueError("a registered candidate input changed")
        path = output / "predictions" / source.name
        if not path.exists():
            with np.load(source, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                actions = saved["candidate_ids"].tolist()
            if [*actions, "guarded_direct"] != identity["candidate_ids"]:
                raise ValueError("candidate input order changed")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            points, distinct = query_candidate_points(
                runner,
                spec,
                context,
                candidates,
                actions,
                [0, 1],
                np.asarray(scaler["mean"]),
                np.asarray(scaler["scale"]),
                joint=joint,
            )
            if points.shape != (7, 96, 2) or not np.isfinite(points).all():
                raise ValueError("the supported candidate pool produced invalid forecasts")
            _save_npz(
                path,
                point_z=points,
                candidate_ids=np.asarray(identity["candidate_ids"]),
                distinct_contexts=np.asarray(distinct),
                source_sha256=np.asarray(record["sha256"]),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(parameter_sha),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != record["sha256"]
                or str(saved["parameter_sha256"]) != parameter_sha
            ):
                raise ValueError("a partial forecast artifact changed identity")
            distinct = int(saved["distinct_contexts"])
        predictions.append(
            {
                "episode_id": record["episode_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "distinct_contexts": distinct,
            }
        )
        _write_json(
            output / "progress.json",
            {
                "status": "forecasting",
                "completed_episodes": len(predictions),
                "total_episodes": len(prep["episodes"]),
            },
        )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "completed_episodes": len(predictions),
                    "total_episodes": len(prep["episodes"]),
                }
            ),
            flush=True,
        )
    if parameter_digest(backbone) != parameter_sha:
        raise ValueError("the forecasting model changed during evaluation")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "predictions": predictions,
            "parameters_unchanged": True,
            "parameter_sha256": parameter_sha,
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": "candidate forecasting completed; follow-up policy decisions, metrics and audit are separate stages",
        },
    )


if __name__ == "__main__":
    main()
