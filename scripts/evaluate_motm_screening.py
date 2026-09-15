"""Evaluate two fixed MoTM input protocols against exactly matched cached controls."""

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
from tsfm_fais.imputers.motm import MOTMReference  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, file_sha256  # noqa: E402
from tsfm_fais.utility_experiment import _write_json as _write_plain_json  # noqa: E402


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    _write_plain_json(temporary, value)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "accuracy-root",
        "plan",
        "controls-root",
        "reference-root",
        "runtime-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MoTM screening")
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source_sha = file_sha256(args.source_root / "episodes_manifest.json")
    if (
        accuracy["source_episode_manifest_sha256"] != source_sha
        or plan["source_manifest_sha256"] != source_sha
    ):
        raise ValueError("the source, prediction export and plan differ")
    config = source["identity"]["config"]
    index = {row["episode_id"]: (position, row) for position, row in enumerate(source["episodes"])}
    selected = [index[episode][1] for episode in plan["decision_episode_ids"]]
    if len(selected) != 90 or any(row["split"] != "validation" for row in selected):
        raise ValueError("this study requires the predeclared 90 development tasks")
    variants = ("motm_raw", "motm_prefix_z")
    identity = {
        "source_manifest_sha256": source_sha,
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "plan_sha256": file_sha256(args.plan),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/motm.py"),
        "reference_manifest_sha256": file_sha256(args.reference_root / "manifest.json"),
        "runtime_manifest_sha256": file_sha256(args.runtime_root / "manifest.json"),
        "views": variants,
        "ridge": 0.5,
        "context_duplication_seed": 42,
        "batch_size": 32,
    }
    identity = json.loads(json.dumps(identity))
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("screening identity changed; use a new output directory")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "module_snapshot.py").write_text(
        (ROOT / "src/tsfm_fais/imputers/motm.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    torch.set_num_threads(1)
    prepared_path = output / "prepared_manifest.json"
    if not prepared_path.exists():
        imputer = MOTMReference(args.reference_root, args.runtime_root)
        preparation = []
        for number, record in enumerate(selected):
            name = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24]
            path = output / "imputations" / f"{name}.npz"
            if not path.exists():
                source_path = args.source_root / record["path"]
                if file_sha256(source_path) != record["sha256"]:
                    raise ValueError("source candidate cache changed")
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
                with np.load(source_path, allow_pickle=False) as saved:
                    context = saved["context"]
                    fallback = saved["candidate_values"][
                        saved["candidate_ids"].tolist().index("locf")
                    ]
                    observed = np.isfinite(context)
                    context_z = (context - mean) / scale
                    completed, diagnostics = [], []
                    for view in variants:
                        model_context = context if view == "motm_raw" else context_z
                        model_fallback = (
                            fallback if view == "motm_raw" else (fallback - mean) / scale
                        )
                        torch.cuda.synchronize()
                        beginning = monotonic()
                        values, metadata = imputer.impute(model_context, model_fallback)
                        torch.cuda.synchronize()
                        metadata["seconds"] = monotonic() - beginning
                        values_z = (values - mean) / scale if view == "motm_raw" else values
                        np.testing.assert_array_equal(values_z[observed], context_z[observed])
                        completed.append(values_z)
                        diagnostics.append(metadata)
                    # Reconstruction labels are opened only after both completed inputs exist.
                    clean = (saved["clean_context"] - mean) / scale
                cells = ~observed & np.isfinite(clean)
                for values, metadata in zip(completed, diagnostics, strict=True):
                    error = (values - clean)[cells]
                    metadata.update(
                        imputation_scored_cells=int(cells.sum()),
                        imputation_mae_z=float(np.abs(error).mean()) if len(error) else None,
                        imputation_mse_z=float((error**2).mean()) if len(error) else None,
                    )
                imputer.verify_frozen()
                _save_npz(
                    path,
                    completed_z=np.stack(completed),
                    identity_sha256=np.asarray(identity_sha),
                    source_sha256=np.asarray(record["sha256"]),
                    metadata=np.asarray(json.dumps(diagnostics)),
                )
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["identity_sha256"]) != identity_sha
                    or str(saved["source_sha256"]) != record["sha256"]
                ):
                    raise ValueError("prepared imputation belongs to another protocol")
                metadata = json.loads(str(saved["metadata"]))
            preparation.append(
                {
                    "episode_id": record["episode_id"],
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "views": metadata,
                }
            )
            if (number + 1) % 10 == 0:
                _write_json(
                    output / "progress.json",
                    {"phase": "imputation", "completed": number + 1, "total": len(selected)},
                )
                print(json.dumps({"phase": "imputation", "completed": number + 1}), flush=True)
        _write_json(
            prepared_path,
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "imputer_parameters_unchanged": imputer.verify_frozen(),
                "records": preparation,
            },
        )
        del imputer
        torch.cuda.empty_cache()
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["identity_sha256"] != identity_sha or not prepared["imputer_parameters_unchanged"]:
        raise ValueError("prepared imputation identity or frozen-parameter check failed")
    prepared_records = {row["episode_id"]: row for row in prepared["records"]}
    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    all_rows, forecast_sources = [], {}
    registry = default_forecast_registry()
    for model in ("chronos2", "timesfm2p5"):
        controls = args.controls_root / model
        control = json.loads((controls / "manifest.json").read_text(encoding="utf-8"))
        if (
            control["status"] != "completed"
            or control["identity"]["accuracy_manifest_sha256"]
            != identity["accuracy_manifest_sha256"]
        ):
            raise ValueError("cached controls use different prediction units or source tasks")
        control_records = {row["episode_id"]: row for row in control["episodes"]}
        forecast_sources[model] = file_sha256(controls / "manifest.json")
        model_root = output / model
        model_root.mkdir(exist_ok=True)
        model_manifest_path = model_root / "manifest.json"
        adapter = None
        if not model_manifest_path.exists():
            adapter = (
                TimesFMVendorMissingAdapter(
                    model_name=str(config["forecaster_artifacts"][model]),
                    device="cuda",
                    batch_size=8,
                )
                if model == "timesfm2p5"
                else registry.build(
                    model,
                    model_name=str(config["forecaster_artifacts"][model]),
                    device="cuda",
                    batch_size=8,
                )
            )
            runner = ForecastRunner(registry, {model: adapter})
            before = parameter_digest(adapter._ensure_backend().model)
        else:
            previous = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            if (
                previous["status"] != "completed"
                or previous["identity_sha256"] != identity_sha
                or not previous["parameters_unchanged"]
            ):
                raise ValueError("completed model metadata does not match the current protocol")
            before = previous["forecaster_parameter_sha256"]
        rows, predictions = [], []
        spec = ForecastSpec(
            model,
            registry.get(model).mode,
            config["horizon"],
            context_length=config["context_length"],
            target_indices=config["target_indices"],
        )
        for number, record in enumerate(selected):
            ready = prepared_records[record["episode_id"]]
            input_path = output / ready["path"]
            if file_sha256(input_path) != ready["sha256"]:
                raise ValueError("prepared completion changed")
            prediction_path = model_root / Path(ready["path"]).name
            if not prediction_path.exists():
                if adapter is None:
                    raise ValueError("completed forecasting manifest has a missing prediction")
                with np.load(input_path, allow_pickle=False) as saved:
                    point = runner.predict(saved["completed_z"], spec).point
                _save_npz(
                    prediction_path,
                    point_z=point,
                    input_sha256=np.asarray(ready["sha256"]),
                    identity_sha256=np.asarray(identity_sha),
                    model_id=np.asarray(model),
                    forecaster_parameter_sha256=np.asarray(before),
                )
            with np.load(prediction_path, allow_pickle=False) as saved:
                if (
                    str(saved["input_sha256"]) != ready["sha256"]
                    or str(saved["identity_sha256"]) != identity_sha
                    or str(saved["model_id"]) != model
                    or str(saved["forecaster_parameter_sha256"]) != before
                ):
                    raise ValueError("forecast cache belongs to another completed context")
                added = saved["point_z"]
                if added.shape != (len(variants), config["horizon"], len(config["target_indices"])):
                    raise ValueError("MoTM forecast cache has an invalid target layout")
            baseline_record = control_records[record["episode_id"]]
            baseline_path = controls / baseline_record["path"]
            if file_sha256(baseline_path) != baseline_record["sha256"]:
                raise ValueError("matched baseline prediction changed")
            with np.load(baseline_path, allow_pickle=False) as baseline:
                methods, point = baseline["methods"].tolist(), baseline["point_z"]
                positions = [
                    methods.index("prefix_input_z_" + action) for action in config["candidate_ids"]
                ]
                positions.append(methods.index("prefix_input_z_guarded_direct"))
                base_pool = point[positions]
                comparison = {name: added[slot] for slot, name in enumerate(variants)}
                comparison.update(
                    {
                        "forecast_median_guarded": np.median(base_pool, axis=0),
                        "forecast_mean_guarded": base_pool.mean(axis=0),
                        "native_input_z": point[methods.index("prefix_input_z_direct")],
                        "guarded_direct": base_pool[-1],
                    }
                )
                for slot, view in enumerate(variants):
                    comparison["forecast_median_with_" + view] = np.median(
                        np.concatenate([base_pool, added[slot : slot + 1]]), axis=0
                    )
                for action in config["candidate_ids"]:
                    comparison["fixed_" + action] = point[methods.index("prefix_input_z_" + action)]
            outcome = truth[index[record["episode_id"]][0]]
            for method, prediction in comparison.items():
                residual = prediction - outcome
                if not np.isfinite(residual).all():
                    raise ValueError("every candidate must return finite forecasts")
                rows.append(
                    {
                        key: record[key]
                        for key in (
                            "episode_id",
                            "origin_id",
                            "family_id",
                            "dataset_id",
                            "mechanism",
                            "missing_rate",
                        )
                    }
                    | {
                        "model_id": model,
                        "method": method,
                        "mae": float(np.abs(residual).mean()),
                        "mse": float((residual**2).mean()),
                    }
                )
            predictions.append(
                {
                    "episode_id": record["episode_id"],
                    "path": str(prediction_path.relative_to(output)),
                    "sha256": file_sha256(prediction_path),
                }
            )
            if (number + 1) % 10 == 0:
                _write_json(
                    output / "progress.json",
                    {
                        "phase": "forecast",
                        "model": model,
                        "completed": number + 1,
                        "total": len(selected),
                    },
                )
                print(
                    json.dumps({"phase": "forecast", "model": model, "completed": number + 1}),
                    flush=True,
                )
        if adapter is not None:
            if before != parameter_digest(adapter._ensure_backend().model):
                raise ValueError("forecasting parameters changed")
            _write_json(
                model_manifest_path,
                {
                    "status": "completed",
                    "identity_sha256": identity_sha,
                    "forecaster_parameter_sha256": before,
                    "parameters_unchanged": True,
                    "predictions": predictions,
                },
            )
            del runner, adapter
            torch.cuda.empty_cache()
        else:
            previous = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            if (
                previous["identity_sha256"] != identity_sha
                or not previous["parameters_unchanged"]
                or previous["predictions"] != predictions
            ):
                raise ValueError("completed forecast stage does not match its predictions")
        all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    keys = ["model_id", "method"]
    for _, group in frame.groupby(keys):
        if len(group) != len(selected) or group.episode_id.duplicated().any():
            raise ValueError("all methods must cover the same 90 tasks")
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
            "decision_count": len(selected),
            "prepared_manifest_sha256": file_sha256(prepared_path),
            "control_manifests": forecast_sources,
            "information": "all forecaster inputs and forecast metrics use common prefix units; raw and prefix-z imputer input protocols are predeclared separately",
            "interpretation": "fixed pretrained baseline expansion on the registered development panel; not independent confirmation or evidence for a new selector",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
