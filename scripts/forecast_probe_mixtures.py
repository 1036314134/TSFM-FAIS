"""Forecast actual mixtures of imputed contexts; keep output ensembles separate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.routing.recent_feedback import combine_imputations  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    source_root, probe_root, accuracy_root = (
        args.source_root.resolve(),
        args.probe_root.resolve(),
        args.accuracy_root.resolve(),
    )
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    analysis_root = probe_root / "analysis-v001"
    analysis = json.loads((analysis_root / "manifest.json").read_text(encoding="utf-8"))
    if analysis["accuracy_manifest_sha256"] != file_sha256(
        accuracy_root / "manifest.json"
    ) or accuracy["source_episode_manifest_sha256"] != file_sha256(
        source_root / "episodes_manifest.json"
    ):
        raise ValueError("mixture decisions and current forecasts have different source histories")
    model = args.model
    weights_file = analysis_root / "input_mixture_weights.parquet"
    choices = pd.read_parquet(weights_file)
    choices = choices[choices.model_id == model]
    decision_ids = set(
        json.loads((probe_root / "plan.json").read_text(encoding="utf-8"))["decision_episode_ids"]
    )
    if set(choices.episode_id) != decision_ids:
        raise ValueError("mixture weights do not cover the planned decisions")
    if model == "chronos2" and choices.per_target.any():
        raise ValueError("joint Chronos contexts require a shared sequence mixture")
    config = source["identity"]["config"]
    targets = config["target_indices"]
    scalers = {
        (entry["dataset_id"], entry["item_id"]): entry
        for entry in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
    }
    output = probe_root / (model + "-input-mixtures")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "analysis_manifest_sha256": file_sha256(analysis_root / "manifest.json"),
        "weights_sha256": file_sha256(weights_file),
        "script_sha256": file_sha256(Path(__file__)),
        "mixture_module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/recent_feedback.py"),
        "model_id": model,
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("mixture forecasting identity changed")
    _write_json(identity_path, identity)
    registry = default_forecast_registry()
    adapter = registry.build(
        model, model_name=config["forecaster_artifacts"][model], device="cuda", batch_size=8
    )
    runner = ForecastRunner(registry, {model: adapter})
    spec = ForecastSpec(
        model,
        registry.get(model).mode,
        config["horizon"],
        context_length=config["context_length"],
        target_indices=tuple(targets),
    )
    original = np.load(accuracy_root / f"{model}_point_z.npy", mmap_mode="r")
    truth = np.load(accuracy_root / "truth_z.npy", mmap_mode="r")
    original_ids = accuracy["action_orders"][model]
    rows, files = [], []
    started = monotonic()
    for index, current in enumerate(source["episodes"]):
        if current["episode_id"] not in decision_ids:
            continue
        current_choices = choices[choices.episode_id == current["episode_id"]]
        if current_choices.empty:
            raise ValueError("input-mixture plan omits an evaluation episode")
        source_path = source_root / current["path"]
        if file_sha256(source_path) != current["sha256"]:
            raise ValueError("current imputation cache changed")
        cache = output / "predictions" / source_path.name
        with np.load(source_path, allow_pickle=False) as e:
            ids = e["candidate_ids"].tolist()
            imputed = e["candidate_values"]
            context = e["context"]
            predictions = []
            descriptions = []
            uniform = np.full(len(ids), 1 / len(ids))
            settings = [
                (
                    {
                        "method": "imputation_mean",
                        "objective": "none",
                        "probe_horizon": 0,
                        "probe_count": 0,
                        "shrinkage": 0.0,
                    },
                    combine_imputations(context, imputed, uniform),
                )
            ]
            median = np.median(imputed, axis=0)
            median[np.isfinite(context)] = context[np.isfinite(context)]
            settings.append(
                (
                    {
                        "method": "imputation_median",
                        "objective": "none",
                        "probe_horizon": 0,
                        "probe_count": 0,
                        "shrinkage": 0.0,
                    },
                    median,
                )
            )
            for choice in current_choices.itertuples():
                action_ids = json.loads(choice.action_ids)
                positions = [ids.index(action) for action in action_ids]
                values = combine_imputations(
                    context,
                    imputed[positions],
                    np.asarray(json.loads(choice.weights)),
                    targets if choice.per_target else None,
                )
                settings.append(
                    (
                        {
                            "method": choice.method,
                            "objective": choice.objective,
                            "probe_horizon": int(choice.probe_horizon),
                            "probe_count": int(choice.probe_count),
                            "shrinkage": float(choice.shrinkage),
                        },
                        values,
                    )
                )
            descriptions = [description for description, _ in settings]
            if not cache.exists():
                pending = []
                for _, values in settings:
                    # Exact model-input matches can reuse a paid candidate query.
                    match = next(
                        (
                            i
                            for i, candidate in enumerate(imputed)
                            if np.array_equal(
                                values.astype(np.float32), candidate.astype(np.float32)
                            )
                        ),
                        None,
                    )
                    if match is not None:
                        predictions.append(
                            np.asarray(original[index, original_ids.index(ids[match])])
                        )
                    else:
                        predictions.append(None)
                        pending.append((len(predictions) - 1, values))
                if pending:
                    actual = runner.predict(np.stack([values for _, values in pending]), spec).point
                    scaler = scalers[(current["dataset_id"], current["item_id"])]
                    mean, scale = (
                        np.asarray(scaler["mean"])[targets],
                        np.asarray(scaler["scale"])[targets],
                    )
                    for (position, _), point in zip(pending, actual, strict=True):
                        predictions[position] = (point - mean) / scale
                _save_npz(
                    cache,
                    point_z=np.stack(predictions),
                    source_sha256=np.asarray(current["sha256"]),
                    descriptions=np.asarray(json.dumps(descriptions, sort_keys=True)),
                )
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["source_sha256"]) != current["sha256"] or str(
                    saved["descriptions"]
                ) != json.dumps(descriptions, sort_keys=True):
                    raise ValueError("cached mixture predictions belong to another decision")
                predictions = saved["point_z"]
            for description, prediction in zip(descriptions, predictions, strict=True):
                error = prediction - truth[index]
                rows.append(
                    {
                        key: current[key]
                        for key in ("episode_id", "origin_id", "family_id", "dataset_id")
                    }
                    | {"model_id": model}
                    | description
                    | {"mae": float(np.mean(np.abs(error))), "mse": float(np.mean(error**2))}
                )
        files.append(
            {
                "episode_id": current["episode_id"],
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        if len(files) % 25 == 0:
            state = {"episodes": len(files), "elapsed_seconds": monotonic() - started}
            _write_json(output / "progress.json", state)
            print(json.dumps(state), flush=True)
    frame = pd.DataFrame(rows)
    keys = ["model_id", "method", "objective", "probe_horizon", "probe_count", "shrinkage"]
    if frame.duplicated(keys + ["episode_id"]).any():
        raise ValueError("duplicate input mixture results")
    family = (
        frame.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "episodes": files,
            "resources_this_execution": runner.resource_metrics(),
            "elapsed_seconds": monotonic() - started,
            "semantics": "actual completed contexts are passed to the frozen model; predictions are reused only for exactly matching float32 candidate inputs",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
