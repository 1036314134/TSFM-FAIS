"""Cross target and other-variable imputations in a fixed development budget study."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.input_scope_controls import VARIANTS, crossed_inputs  # noqa: E402
from tsfm_fais.routing.preforecast_replay import unique_forecaster_inputs  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared-root",
        "budget-evaluation-root",
        "budget-audit-root",
        "source-root",
        "accuracy-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed input-scope interventions")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.budget_audit_root / "manifest.json").read_text(encoding="utf-8"))
    reference_root = args.budget_evaluation_root / args.model
    reference = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        prepared["status"] != "completed"
        or audit["status"] != "completed"
        or file_sha256(reference_root / "manifest.json") != audit["source_manifests"][args.model]
    ):
        raise ValueError("complete and audit the underlying budget study first")
    source_path = args.source_root / "episodes_manifest.json"
    if file_sha256(source_path) != prepared["identity"]["source_manifest_sha256"]:
        raise ValueError("the development histories changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    original = {record["episode_id"]: record for record in source["episodes"]}
    cases = {(record["episode_id"], record["budget"]): record for record in prepared["cases"]}
    references = {
        (record["episode_id"], record["budget"]): record for record in reference["predictions"]
    }
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "budget_audit_sha256": file_sha256(args.budget_audit_root / "manifest.json"),
        "budget_reference_sha256": file_sha256(reference_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/input_scope_controls.py"),
        "model_id": args.model,
        "base_budget": "epochs10_windows64",
        "changed_budgets": ["epochs50_windows64", "epochs50_windows512"],
        "imputers": ["saits", "timemixerpp"],
        "variants": list(VARIANTS),
        "target_indices": [0, 1],
        "scope": "post-budget exploratory development interventions; no new training or confirmation claim",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("input-scope study identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "module_snapshot.py").write_bytes(
        (ROOT / "src/tsfm_fais/forecasting/input_scope_controls.py").read_bytes()
    )
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    registry = default_forecast_registry()
    joint = args.model == "chronos2"
    model_path = source["identity"]["config"]["forecaster_artifacts"][args.model]
    torch.set_num_threads(1)
    adapter = (
        registry.build(args.model, model_name=model_path, device="cuda", batch_size=8)
        if joint
        else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
    parameter_sha = parameter_digest(backbone)
    spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=96, target_indices=[0, 1]
    )
    rows, interactions, files = [], [], []
    for episode_id in sorted({key[0] for key in cases}):
        episode = original[episode_id]
        source_file = args.source_root / episode["path"]
        if file_sha256(source_file) != episode["sha256"]:
            raise ValueError("a source context changed")
        with np.load(source_file, allow_pickle=False) as saved:
            context = saved["context"]
        scaler = scalers[(episode["dataset_id"], episode["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        for budget in identity["changed_budgets"]:
            banks, reference_points = [], []
            for selected_budget in (identity["base_budget"], budget):
                record, ref = (
                    cases[(episode_id, selected_budget)],
                    references[(episode_id, selected_budget)],
                )
                input_path, ref_path = (
                    args.prepared_root / record["path"],
                    reference_root / ref["path"],
                )
                if (
                    file_sha256(input_path) != record["sha256"]
                    or file_sha256(ref_path) != ref["sha256"]
                ):
                    raise ValueError("a budget input or endpoint prediction changed")
                with np.load(input_path, allow_pickle=False) as saved:
                    banks.append(saved["candidate_values"])
                    actions = saved["candidate_ids"].tolist()
                with np.load(ref_path, allow_pickle=False) as saved:
                    methods = saved["methods"].tolist()
                    reference_points.append(saved["point_z"])
            for imputer in identity["imputers"]:
                key = hashlib.sha256(f"{episode_id}|{budget}|{imputer}".encode()).hexdigest()[:24]
                cache = output / "predictions" / f"{key}.npz"
                if not cache.exists():
                    action = actions.index(imputer)
                    inputs = crossed_inputs(context, banks[0][action], banks[1][action], [0, 1])
                    unique, reverse = unique_forecaster_inputs(inputs, [0, 1], joint=joint)
                    point = runner.predict_missing(unique, spec).point[reverse]
                    point = (point - mean[:2]) / scale[:2]
                    for index, ref in ((0, reference_points[0]), (3, reference_points[1])):
                        np.testing.assert_allclose(
                            point[index], ref[methods.index(imputer)], rtol=2e-4, atol=2e-4
                        )
                    if not joint:
                        np.testing.assert_array_equal(point[0], point[2])
                        np.testing.assert_array_equal(point[1], point[3])
                    _save_npz(
                        cache,
                        point_z=point,
                        identity_sha256=np.asarray(identity_sha),
                        parameter_sha256=np.asarray(parameter_sha),
                        distinct_inputs=np.asarray(len(unique)),
                    )
                with np.load(cache, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["parameter_sha256"]) != parameter_sha
                    ):
                        raise ValueError("an intervention cache changed identity")
                    point = saved["point_z"]
                with np.load(source_file, allow_pickle=False) as saved:
                    truth = (saved["future"][:, :2] - mean[:2]) / scale[:2]
                mae = np.abs(point - truth).mean(axis=(1, 2))
                mse = np.square(point - truth).mean(axis=(1, 2))
                for index, variant in enumerate(VARIANTS):
                    rows.append(
                        {
                            "model_id": args.model,
                            "dataset_id": episode["dataset_id"],
                            "episode_id": episode_id,
                            "budget": budget,
                            "imputer": imputer,
                            "variant": variant,
                            "mae": float(mae[index]),
                            "mse": float(mse[index]),
                        }
                    )
                interactions.append(
                    {
                        "model_id": args.model,
                        "dataset_id": episode["dataset_id"],
                        "episode_id": episode_id,
                        "budget": budget,
                        "imputer": imputer,
                        "forecast_interaction_rms_z": float(
                            np.sqrt(np.mean((point[3] - point[1] - point[2] + point[0]) ** 2))
                        ),
                        "delta_mse_targets": float(mse[1] - mse[0]),
                        "delta_mse_covariates": float(mse[2] - mse[0]),
                        "delta_mse_both": float(mse[3] - mse[0]),
                        "mse_interaction": float(mse[3] - mse[1] - mse[2] + mse[0]),
                    }
                )
                files.append(
                    {
                        "episode_id": episode_id,
                        "budget": budget,
                        "imputer": imputer,
                        "path": str(cache.relative_to(output)),
                        "sha256": file_sha256(cache),
                    }
                )
        print(
            json.dumps(
                {
                    "model_id": args.model,
                    "completed_interventions": len(files),
                    "total_interventions": 72,
                }
            ),
            flush=True,
        )
    if parameter_digest(backbone) != parameter_sha or len(files) != 72:
        raise ValueError("model parameters or intervention coverage changed")
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summary = (
        frame.groupby(["model_id", "dataset_id", "budget", "imputer", "variant"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(interactions).to_csv(output / "episode_interactions.csv", index=False)
    pd.DataFrame(interactions).groupby(["model_id", "dataset_id", "budget", "imputer"])[
        [
            "forecast_interaction_rms_z",
            "delta_mse_targets",
            "delta_mse_covariates",
            "delta_mse_both",
            "mse_interaction",
        ]
    ].mean().reset_index().to_csv(output / "interactions.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "predictions": files,
            "parameters_unchanged": True,
            "endpoint_prediction_parity": True,
            "independent_covariate_invariance": not joint,
            "summary": summary.to_dict("records"),
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": "exploratory intervention on three fixed development histories; error interaction includes loss nonlinearity, while forecast-vector interaction is recorded separately",
        },
    )


if __name__ == "__main__":
    main()
