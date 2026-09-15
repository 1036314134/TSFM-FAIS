"""Measure downstream forecast accuracy after matched imputer-budget changes."""

import argparse
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
from probe_tirex_missing import digest as tirex_parameter_digest  # noqa: E402
from replay_preforecast_student import query_candidate_points  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.routing.preforecast_replay import assemble_selected_context  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "source-root", "accuracy-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5", "tirex"), required=True)
    parser.add_argument("--tirex-reference", type=Path)
    parser.add_argument("--tirex-probe", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed budget evaluations")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        prepared["status"] != "completed"
        or prepared["identity"]["source_manifest_sha256"]
        != file_sha256(args.source_root / "episodes_manifest.json")
        or accuracy["source_episode_manifest_sha256"]
        != prepared["identity"]["source_manifest_sha256"]
    ):
        raise ValueError("the budget inputs and forecast references have different provenance")
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "model_id": args.model,
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/replay_preforecast_student.py",
                "scripts/evaluate_timesfm_vendor_missing.py",
                "src/tsfm_fais/forecasting/adapters/chronos.py",
            )
        },
        "scope": prepared.get(
            "panel_description",
            "three development datasets, 18 fixed episodes; all imputer fitting remains in original training prefixes",
        ),
    }
    if args.model == "tirex":
        if args.tirex_reference is None or args.tirex_probe is None:
            parser.error("TiRex requires the pinned reference and native-interface probe")
        tirex_reference = json.loads(
            (args.tirex_reference / "manifest.json").read_text(encoding="utf-8")
        )
        tirex_probe = json.loads((args.tirex_probe / "manifest.json").read_text(encoding="utf-8"))
        if (
            tirex_reference["status"] != "completed"
            or tirex_probe["status"] != "completed"
            or tirex_probe["reference_manifest_sha256"]
            != file_sha256(args.tirex_reference / "manifest.json")
        ):
            raise ValueError("the TiRex reference has not passed its declared interface check")
        checkpoint = args.tirex_reference / "model/model.ckpt"
        if file_sha256(checkpoint) != tirex_reference["identity"]["checkpoint_sha256"]:
            raise ValueError("the TiRex checkpoint changed")
        identity["tirex_reference_sha256"] = file_sha256(args.tirex_reference / "manifest.json")
        identity["tirex_probe_sha256"] = file_sha256(args.tirex_probe / "manifest.json")
        identity["tirex_policy"] = tirex_probe["evaluation_policy"]
        identity["source_sha256"]["src/tsfm_fais/forecasting/adapters/tirex.py"] = file_sha256(
            ROOT / "src/tsfm_fais/forecasting/adapters/tirex.py"
        )
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("the budget evaluation identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    registry = default_forecast_registry()
    joint = args.model == "chronos2"
    model_path = (
        str(checkpoint)
        if args.model == "tirex"
        else source["identity"]["config"]["forecaster_artifacts"][args.model]
    )
    torch.set_num_threads(1)
    if args.model == "tirex":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = registry.build(
            "tirex", model_name=model_path, device="cuda", batch_size=1, backend_name="torch"
        )
        get_parameter_digest = tirex_parameter_digest
    else:
        get_parameter_digest = parameter_digest
        adapter = (
            registry.build(args.model, model_name=model_path, device="cuda", batch_size=8)
            if joint
            else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
        )
    runner = ForecastRunner(registry, {args.model: adapter})
    backend = adapter._ensure_backend()
    backbone = (backend if args.model == "tirex" else backend.model).eval().requires_grad_(False)
    parameter_sha = get_parameter_digest(backbone)
    spec = ForecastSpec(
        args.model, registry.get(args.model).mode, 96, context_length=96, target_indices=[0, 1]
    )
    references = None
    if args.model != "tirex":
        reference_path = args.accuracy_root / f"{args.model}_point_z.npy"
        if file_sha256(reference_path) != accuracy["prediction_arrays"][reference_path.name]:
            raise ValueError("the original forecast references changed")
        references = np.load(reference_path, mmap_mode="r")
    cases = list(prepared["cases"])
    unique = {row["episode_id"]: row for row in cases}
    if len(unique) != prepared.get("evaluation_episode_count", 18):
        raise ValueError("the development budget panel changed")
    cases.extend({**row, "budget": "legacy"} for row in unique.values())
    rows, files = [], []
    for case in cases:
        original = source["episodes"][case["episode_index"]]
        source_path = args.source_root / original["path"]
        if (
            original["episode_id"] != case["episode_id"]
            or file_sha256(source_path) != original["sha256"]
        ):
            raise ValueError("a development input changed")
        input_path = (
            source_path if case["budget"] == "legacy" else args.prepared_root / case["path"]
        )
        if case["budget"] != "legacy" and file_sha256(input_path) != case["sha256"]:
            raise ValueError("a budget-specific imputation changed")
        scaler = scalers[(original["dataset_id"], original["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        path = output / "predictions" / case["budget"] / source_path.name
        if not path.exists():
            with np.load(source_path, allow_pickle=False) as saved:
                context = saved["context"]
            with np.load(input_path, allow_pickle=False) as saved:
                bank, actions = saved["candidate_values"], saved["candidate_ids"].tolist()
            points, distinct = query_candidate_points(
                runner, spec, context, bank, actions, [0, 1], mean, scale, joint=joint
            )
            methods = [*actions, "guarded_direct"]
            if case["budget"] == "legacy" and references is not None:
                expected = references[
                    case["episode_index"],
                    [accuracy["action_orders"][args.model].index(name) for name in methods],
                ]
                np.testing.assert_allclose(points, expected, rtol=2e-4, atol=2e-4)
            if case["budget"] == "legacy" and args.model == "tirex":
                served = np.stack(
                    [
                        assemble_selected_context(
                            context, bank, actions, [name, name], [0, 1], joint=False
                        )
                        for name in methods
                    ]
                )
                individual = served[:, :, :2].transpose(0, 2, 1).reshape(-1, 96)
                _, native_point = backbone.forecast(
                    context=torch.as_tensor(individual, dtype=torch.float32, device="cuda"),
                    prediction_length=96,
                    output_type="numpy",
                    batch_size=1,
                )
                expected = np.asarray(native_point).reshape(7, 2, 96).transpose(0, 2, 1)
                expected = (expected - mean[:2]) / scale[:2]
                np.testing.assert_allclose(points, expected, rtol=0, atol=1e-7)
            predictions = dict(zip(methods, points, strict=True))
            predictions["forecast_median_guarded"] = np.median(points, axis=0)
            predictions["forecast_mean_guarded"] = np.mean(points, axis=0)
            _save_npz(
                path,
                methods=np.asarray(list(predictions)),
                point_z=np.stack(list(predictions.values())),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(parameter_sha),
                distinct_inputs=np.asarray(distinct),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != parameter_sha
            ):
                raise ValueError("a budget forecast cache changed identity")
            methods, points = saved["methods"].tolist(), saved["point_z"]
        with np.load(source_path, allow_pickle=False) as saved:
            truth = (saved["future"][:, :2] - mean[:2]) / scale[:2]
        for method, point in zip(methods, points, strict=True):
            residual = point - truth
            rows.append(
                {
                    "model_id": args.model,
                    "dataset_id": original["dataset_id"],
                    "episode_id": original["episode_id"],
                    "budget": case["budget"],
                    "method": method,
                    "mae": float(np.abs(residual).mean()),
                    "mse": float(np.square(residual).mean()),
                }
            )
        files.append(
            {
                "episode_id": original["episode_id"],
                "budget": case["budget"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    if get_parameter_digest(backbone) != parameter_sha:
        raise ValueError("the fixed forecasting model changed")
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_results.parquet", index=False)
    summary = (
        frame.groupby(["model_id", "dataset_id", "budget", "method"])[["mae", "mse"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "predictions": files,
            "parameters_unchanged": True,
            "parameter_sha256": parameter_sha,
            "summary": summary.to_dict("records"),
            "runtime_current_process_only": runner.resource_metrics(),
            "limits": [
                "development-only budget sensitivity",
                prepared.get(
                    "panel_description",
                    "one fixed original validation origin per dataset, multiple masks",
                ),
                "same imputer structures; not a claim about fully tuned algorithm capacity",
            ],
        },
    )
    print(
        summary[summary.method.isin(["saits", "timemixerpp", "forecast_median_guarded"])].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
