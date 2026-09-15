"""Evaluate the late MoTM comparator without changing frozen primary policies."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apply_followup_policies import result_panels  # noqa: E402
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from run_native_confirmation import hierarchical_metrics  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.observed_accuracy import observed_future_errors  # noqa: E402
from tsfm_fais.routing.preforecast_replay import (  # noqa: E402
    assemble_selected_context,
    unique_forecaster_inputs,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402

METHODS = ("median_risk", "forecast_median_guarded", "motm_reference", "forecast_median_with_motm")


def matching_candidate(context, candidates, actions, completed, *, joint):
    """Compare exactly the float32 inputs consumed by the frozen forecaster."""
    order = [*actions, "guarded_direct"]
    contexts = [completed] + [
        assemble_selected_context(
            context, candidates, actions, [name] if joint else [name, name], [0, 1], joint=joint
        )
        for name in order
    ]
    _, reverse = unique_forecaster_inputs(contexts, [0, 1], joint=joint)
    matches = np.flatnonzero(reverse[1:] == reverse[0])
    return int(matches[0]) if len(matches) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "motm-root",
        "forecast-root",
        "policy-root",
        "old-bundle",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed supplementary evaluations")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    motm = json.loads((args.motm_root / "manifest.json").read_text(encoding="utf-8"))
    forecast_dir, policy_dir = args.forecast_root / args.model, args.policy_root / args.model
    forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
    policy = json.loads((policy_dir / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((args.old_bundle / "manifest.json").read_text(encoding="utf-8"))
    if any(row["status"] != "completed" for row in (prep, motm, forecast, policy, old)):
        raise ValueError("complete primary evaluations and the reference imputer inputs first")
    if motm["identity"]["prepared_manifest_sha256"] != file_sha256(
        args.prepared_root / "manifest.json"
    ):
        raise ValueError("MoTM input preparation uses a different cohort")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "model_id": args.model,
        "protocol_sha256": file_sha256(args.protocol),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "motm_sha256": file_sha256(args.motm_root / "manifest.json"),
        "forecast_sha256": file_sha256(forecast_dir / "manifest.json"),
        "primary_policy_sha256": file_sha256(policy_dir / "manifest.json"),
        "old_bundle_sha256": file_sha256(args.old_bundle / "manifest.json"),
        "methods": METHODS,
        "scope": "late supplementary comparator; primary method and cohort unchanged",
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/evaluate_timesfm_vendor_missing.py",
                "scripts/apply_followup_policies.py",
                "src/tsfm_fais/routing/preforecast_replay.py",
                "src/tsfm_fais/forecasting/observed_accuracy.py",
            )
        },
    }
    identity = json.loads(json.dumps(identity))
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("supplementary evaluation identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    primary_record = json.loads(
        (policy_dir / "policy_predictions.json").read_text(encoding="utf-8")
    )
    primary_path = policy_dir / "policy_predictions.npz"
    if file_sha256(primary_path) != primary_record["prediction_sha256"]:
        raise ValueError("primary policy predictions changed")
    with np.load(primary_path, allow_pickle=False) as saved:
        primary, primary_methods = saved["point_z"], saved["methods"].tolist()
        if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
            raise ValueError("primary episode order changed")
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("prefix standardizers changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    reference_map = {row["episode_id"]: row for row in motm["episodes"]}
    forecast_map = {row["episode_id"]: row for row in forecast["predictions"]}
    expected_ids = {row["episode_id"] for row in prep["episodes"]}
    if set(reference_map) != expected_ids or set(forecast_map) != expected_ids:
        raise ValueError("the supplementary comparator lost cohort coverage")
    torch.set_num_threads(1)
    registry = default_forecast_registry()
    joint = args.model == "chronos2"
    model_path = old["identity"]["forecaster_artifacts"][args.model]
    adapter = (
        registry.build(args.model, model_name=model_path, device="cuda", batch_size=8)
        if joint
        else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    parameter_sha = parameter_digest(backbone)
    if parameter_sha != forecast["parameter_sha256"]:
        raise ValueError("the supplementary forecaster differs from the primary run")
    spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=96, target_indices=[0, 1]
    )
    files = []
    for index, record in enumerate(prep["episodes"]):
        source = args.prepared_root / record["path"]
        reference = reference_map[record["episode_id"]]
        ref_path = args.motm_root / reference["path"]
        pred = forecast_map[record["episode_id"]]
        base_path = forecast_dir / pred["path"]
        if (
            file_sha256(source) != record["sha256"]
            or file_sha256(ref_path) != reference["sha256"]
            or file_sha256(base_path) != pred["sha256"]
        ):
            raise ValueError("a supplemental input or base forecast changed")
        path = output / "predictions" / source.name
        if not path.exists():
            with np.load(source, allow_pickle=False) as saved:
                context, candidates, actions = (
                    saved["context"],
                    saved["candidate_values"],
                    saved["candidate_ids"].tolist(),
                )
            with np.load(ref_path, allow_pickle=False) as saved:
                completed = saved["values"]
            with np.load(base_path, allow_pickle=False) as saved:
                base = saved["point_z"]
                if saved["candidate_ids"].tolist() != [*actions, "guarded_direct"]:
                    raise ValueError("base forecast order changed")
            np.testing.assert_array_equal(
                completed[np.isfinite(context)], context[np.isfinite(context)]
            )
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            reused = matching_candidate(context, candidates, actions, completed, joint=joint)
            if reused is not None:
                point = base[reused]
            else:
                point = (
                    runner.predict(completed[None], spec).point[0] - np.asarray(scaler["mean"])[:2]
                ) / np.asarray(scaler["scale"])[:2]
            points = np.stack(
                [
                    primary[index, primary_methods.index("median_risk")],
                    primary[index, primary_methods.index("forecast_median_guarded")],
                    point,
                    np.median(np.concatenate([base, point[None]]), axis=0),
                ]
            )
            if points.shape != (4, 96, 2) or not np.isfinite(points).all():
                raise ValueError("the supplementary prediction is incomplete")
            _save_npz(
                path,
                point_z=points,
                methods=np.asarray(METHODS),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(parameter_sha),
                reused_candidate_index=np.asarray(-1 if reused is None else reused),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != parameter_sha
            ):
                raise ValueError("a partial supplementary forecast changed identity")
        files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json",
            {
                "status": "forecasting",
                "completed_episodes": len(files),
                "total_episodes": len(expected_ids),
            },
        )
    if parameter_digest(backbone) != parameter_sha:
        raise ValueError("forecaster parameters changed")
    _write_json(
        output / "predictions_frozen.json",
        {"identity_sha256": identity_sha, "files": files, "future_arrays_read": False},
    )
    rows = []
    for record, prediction in zip(prep["episodes"], files, strict=True):
        with np.load(output / prediction["path"], allow_pickle=False) as saved:
            points = saved["point_z"]
        with np.load(args.prepared_root / record["path"], allow_pickle=False) as saved:
            future, mask = saved["future"], saved["future_observed"]
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
        errors, _ = observed_future_errors(points * scale + mean, future, mask, scale)
        for index, method in enumerate(METHODS):
            rows.append(
                {
                    **{
                        key: record[key]
                        for key in (
                            "episode_id",
                            "origin_id",
                            "dataset_id",
                            "family_id",
                            "item_id",
                            "panel",
                        )
                    },
                    "model_id": args.model,
                    "method": method,
                    "native_missing_context": record["window"]["context_has_missing"],
                    **{name: float(value[index].mean()) for name, value in errors.items()},
                }
            )
    scores = pd.DataFrame(rows)
    scores.to_parquet(output / "episode_results.parquet", index=False)
    summaries = []
    for name, panel in result_panels(scores):
        _, families, summary = hierarchical_metrics(panel)
        families.to_csv(output / f"{name}_family_metrics.csv", index=False)
        summaries.append(summary.assign(panel=name, families=panel.family_id.nunique()))
    pd.concat(summaries, ignore_index=True).to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "predictions": files,
            "parameter_sha256": parameter_sha,
            "parameters_unchanged": True,
            "scores_sha256": file_sha256(output / "episode_results.parquet"),
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": "late supplemental comparison; eight-candidate median has one additional possible query; independent audit pending; Solar pretraining overlap disclosed",
        },
    )


if __name__ == "__main__":
    main()
