"""Verify MoTM's reference parity and downstream errors on fixed source windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.accuracy import guarded_direct_forecast  # noqa: E402
from tsfm_fais.imputers.motm import MOTMReference, prepare_context  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def official_prediction(imputer, context, directory, dummy_value):
    from src.data.dataloader.base import ImputationDatasetTest
    from src.tools.inference_mixture import infer_fn_mixture

    values, coordinates, grid, available = prepare_context(context)
    if not bool(available.all()):
        raise ValueError("official parity cases must contain an observation in every variable")
    raw = torch.tensor(context.T.copy(), dtype=torch.float32).unsqueeze(-1)
    with torch.random.fork_rng():
        dataset = ImputationDatasetTest(
            raw, grid, latent_dim=128, ground_truths=torch.full_like(raw, dummy_value)
        )
    torch.testing.assert_close(dataset.values_with_replacement, values, rtol=0, atol=0)
    torch.testing.assert_close(dataset.grids_with_replacement, coordinates, rtol=0, atol=0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=imputer.batch_size, num_workers=0, shuffle=False
    )
    directory.mkdir(parents=True, exist_ok=True)
    captured = {}
    previous_profile = sys.getprofile()

    def capture(frame, event, _):
        if event == "return" and frame.f_code is infer_fn_mixture.__code__:
            materials = frame.f_locals.get("materials", {})
            if torch.is_tensor(materials.get("interpo_Ridge")):
                captured["prediction"] = materials["interpo_Ridge"].detach().cpu().numpy().copy()

    try:
        sys.setprofile(capture)
        infer_fn_mixture(
            imputer.models,
            loader,
            {
                name: [setting[name] for setting in imputer.settings]
                for name in ("inner_steps", "inner_lr", "loss_type")
            },
            directory,
            inr_last_layers=1,
            lambda_ridge=imputer.ridge,
            share_ridge_weights=False,
            learn_lambda=False,
            plot_imputation=False,
            run_baselines=False,
        )
    finally:
        sys.setprofile(previous_profile)
    if "prediction" not in captured:
        raise ValueError("the unmodified reference did not expose completed imputation arrays")
    return captured["prediction"].reshape(context.shape[1], context.shape[0]).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MoTM feasibility evidence")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    source_sha = file_sha256(args.source_root / "episodes_manifest.json")
    if accuracy["source_episode_manifest_sha256"] != source_sha:
        raise ValueError("source windows and standardizers do not match")
    config = source["identity"]["config"]
    selected = []
    for dataset in ("ETTh1", "exchange_rate", "Coastal_T_S_H"):
        choices = [
            row
            for row in source["episodes"]
            if row["dataset_id"] == dataset
            and row["split"] == "train"
            and row["mechanism"] == "independent_block"
            and np.isclose(row["missing_rate"], 0.3)
            and row["mask_seed"] == min(config["mask_seeds"])
        ]
        selected.append(min(choices, key=lambda row: row["origin"]))
    identity = {
        "source_manifest_sha256": source_sha,
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "reference_manifest_sha256": file_sha256(args.reference_root / "manifest.json"),
        "runtime_manifest_sha256": file_sha256(args.runtime_root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/motm.py"),
        "episodes": [row["episode_id"] for row in selected],
        "ridge": 0.5,
        "context_duplication_seed": 42,
        "batch_size": 32,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("MoTM probe identity changed")
    _write_json(identity_path, identity)
    (output / "script_snapshot.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "module_snapshot.py").write_text(
        (ROOT / "src/tsfm_fais/imputers/motm.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    imputer = MOTMReference(args.reference_root, args.runtime_root)
    imputation_started = monotonic()
    cases, parity = [], []
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    for record in selected:
        path = args.source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("source candidate cache changed")
        with np.load(path, allow_pickle=False) as saved:
            context = saved["context"]
            candidates = saved["candidate_values"]
            future = saved["future"]
            actions = saved["candidate_ids"].tolist()
        completed, metadata = imputer.impute(context, candidates[actions.index("locf")])
        values, coordinates, grid, available = prepare_context(context)
        direct = imputer.predict_prepared(values, coordinates, grid)[..., 0].numpy().T
        case_name = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:20]
        # These official-reference metric files contain dummy labels and are never evaluation inputs.
        reference = official_prediction(
            imputer, context, output / "reference-dummy-labels" / case_name / "first", 1.0
        )
        changed = official_prediction(
            imputer, context, output / "reference-dummy-labels" / case_name / "changed", 100.0
        )
        np.testing.assert_allclose(direct, reference, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(reference, changed)
        np.testing.assert_array_equal(
            completed[np.isfinite(context)], context[np.isfinite(context)]
        )
        _save_npz(
            output / f"{case_name}-imputation.npz",
            context=context,
            completed=completed,
            raw_motm=direct,
            official=reference,
            changed_dummy=changed,
        )
        parity.append(
            {
                "episode_id": record["episode_id"],
                "max_absolute_error": float(np.max(np.abs(direct - reference))),
                "dummy_label_invariant": True,
                **metadata,
            }
        )
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        cases.append((record, context, candidates, completed, future, mean, scale, case_name))
    edge = cases[0][1].copy()
    edge[:, 0] = np.nan
    edge[1:, 1] = np.nan
    fallback = cases[0][2][config["candidate_ids"].index("locf")]
    edge_completed, edge_metadata = imputer.impute(edge, fallback)
    np.testing.assert_array_equal(edge_completed[:, 0], fallback[:, 0])
    if not np.isfinite(edge_completed).all():
        raise ValueError("empty and single-observation boundary cases did not remain finite")
    imputation_seconds = monotonic() - imputation_started
    imputer.verify_frozen()
    rows, frozen_forecasters = [], {}
    registry = default_forecast_registry()
    for model_id in ("chronos2", "timesfm2p5"):
        adapter = (
            TimesFMVendorMissingAdapter(
                model_name=str(config["forecaster_artifacts"][model_id]),
                device="cuda",
                batch_size=8,
            )
            if model_id == "timesfm2p5"
            else registry.build(
                model_id,
                model_name=str(config["forecaster_artifacts"][model_id]),
                device="cuda",
                batch_size=8,
            )
        )
        runner = ForecastRunner(registry, {model_id: adapter})
        backbone = adapter._ensure_backend().model
        before = parameter_digest(backbone)
        spec = ForecastSpec(
            model_id,
            registry.get(model_id).mode,
            config["horizon"],
            context_length=config["context_length"],
            target_indices=config["target_indices"],
        )
        for record, context, candidates, completed, future, mean, scale, case_name in cases:
            inputs = (np.concatenate([candidates, completed[None]]) - mean) / scale
            point = runner.predict(inputs, spec).point
            native = runner.predict_missing(((context - mean) / scale)[None], spec).point[0]
            guarded, _ = guarded_direct_forecast(
                context,
                config["target_indices"],
                native,
                point[config["candidate_ids"].index("locf")],
                joint=model_id == "chronos2",
            )
            forecasts = np.concatenate(
                [
                    point,
                    guarded[None],
                    np.median(np.concatenate([point[:-1], guarded[None]]), axis=0)[None],
                    np.median(np.concatenate([point, guarded[None]]), axis=0)[None],
                ]
            )
            names = [
                *config["candidate_ids"],
                "motm",
                "guarded_direct",
                "forecast_median_guarded",
                "forecast_median_guarded_with_motm",
            ]
            truth = ((future - mean) / scale)[:, list(config["target_indices"])]
            residual = forecasts - truth
            if not np.isfinite(residual).all():
                raise ValueError("source forecast errors must be finite")
            for index, name in enumerate(names):
                rows.append(
                    {
                        "episode_id": record["episode_id"],
                        "dataset_id": record["dataset_id"],
                        "model_id": model_id,
                        "method": name,
                        "mae": float(np.abs(residual[index]).mean()),
                        "mse": float((residual[index] ** 2).mean()),
                    }
                )
            _save_npz(
                output / f"{case_name}-{model_id}.npz",
                method_ids=np.asarray(names),
                point_z=forecasts,
                truth_z=truth,
            )
        frozen_forecasters[model_id] = before == parameter_digest(backbone)
        if not frozen_forecasters[model_id]:
            raise ValueError("forecasting parameters changed")
    pd.DataFrame(rows).to_csv(output / "source_forecast_errors.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "source_feasibility",
            "identity": identity,
            "parity": parity,
            "edge_case": edge_metadata,
            "imputer_parameters_unchanged": imputer.verify_frozen(),
            "forecaster_parameters_unchanged": frozen_forecasters,
            "imputation_and_reference_seconds": imputation_seconds,
            "interpretation": "three fixed source windows; official forward parity and dummy-label independence; source errors do not establish held-out forecasting superiority; reference-dummy-labels files are excluded from evaluation",
        },
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
