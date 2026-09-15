"""Test observable-consensus input optimization on the fixed development panel."""

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
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.forecasting.adapters.timesfm import TimesFM2p5Adapter  # noqa: E402
from tsfm_fais.forecasting.chronos_differentiable import chronos_median  # noqa: E402
from tsfm_fais.forecasting.timesfm_differentiable import timesfm_median  # noqa: E402
from tsfm_fais.routing.consensus_projection import project_consensus  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "controls-root", "motm-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed input projection results")
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    config = source["identity"]["config"]
    motm = json.loads((args.motm_root / "manifest.json").read_text(encoding="utf-8"))
    controls_root = args.controls_root / args.model
    controls = json.loads((controls_root / "manifest.json").read_text(encoding="utf-8"))
    motm_forecast = json.loads(
        (args.motm_root / args.model / "manifest.json").read_text(encoding="utf-8")
    )
    prepared = json.loads((args.motm_root / "prepared_manifest.json").read_text(encoding="utf-8"))
    accuracy_sha = file_sha256(args.accuracy_root / "manifest.json")
    if (
        accuracy["source_episode_manifest_sha256"] != file_sha256(source_path)
        or controls["identity"]["accuracy_manifest_sha256"] != accuracy_sha
        or motm["identity"]["accuracy_manifest_sha256"] != accuracy_sha
    ):
        raise ValueError("projection inputs and forecasts do not share the source protocol")
    if any(item["status"] != "completed" for item in (motm, controls, motm_forecast, prepared)):
        raise ValueError("complete all input and teacher forecasts before projection")
    records = {row["episode_id"]: (index, row) for index, row in enumerate(source["episodes"])}
    completions = {row["episode_id"]: row for row in prepared["records"]}
    extra_forecasts = {row["episode_id"]: row for row in motm_forecast["predictions"]}
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    identity = {
        "source_manifest_sha256": file_sha256(source_path),
        "accuracy_manifest_sha256": accuracy_sha,
        "controls_manifest_sha256": file_sha256(controls_root / "manifest.json"),
        "motm_manifest_sha256": file_sha256(args.motm_root / "manifest.json"),
        "motm_forecast_manifest_sha256": file_sha256(args.motm_root / args.model / "manifest.json"),
        "prepared_manifest_sha256": file_sha256(args.motm_root / "prepared_manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "projection_module_sha256": file_sha256(
            ROOT / "src/tsfm_fais/routing/consensus_projection.py"
        ),
        "mixture_module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/differentiable.py"),
        "timesfm_gradient_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/timesfm_differentiable.py"
        ),
        "chronos_gradient_sha256": file_sha256(
            ROOT / "src/tsfm_fais/forecasting/chronos_differentiable.py"
        ),
        "model_id": args.model,
        "steps": 20,
        "learning_rate": 0.05,
        "checkpoints": [0, 1, 5, 20],
        "teacher": "median of seven finite candidate forecasts and guarded native forecast",
        "candidate_ids": config["candidate_ids"] + ["motm_prefix_z"],
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("input projection identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "projection_snapshot.py").write_text(
        (ROOT / "src/tsfm_fais/routing/consensus_projection.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    torch.set_num_threads(1)
    torch.manual_seed(6101)
    if args.model == "chronos2":
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(
            config["forecaster_artifacts"][args.model], device_map="cuda"
        )
    else:
        adapter = TimesFM2p5Adapter(
            model_name=config["forecaster_artifacts"][args.model], device="cuda", batch_size=2
        )
        pipeline = adapter._ensure_backend()
    model = pipeline.model.eval().requires_grad_(False)
    before = parameter_digest(model)
    horizon, targets = config["horizon"], list(config["target_indices"])

    def forecast(context):
        if args.model == "timesfm2p5":
            return timesfm_median(model, context, horizon, targets)
        return chronos_median(pipeline, context, horizon, targets)

    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    rows, traces, files = [], [], []
    for number, control_record in enumerate(controls["episodes"]):
        episode = control_record["episode_id"]
        source_index, record = records[episode]
        if record["split"] != "validation":
            raise ValueError("input projection must use the fixed development tasks")
        case_name = hashlib.sha256(episode.encode()).hexdigest()[:24]
        cache = output / "predictions" / f"{case_name}.npz"
        if not cache.exists():
            paths = [
                (args.source_root / record["path"], record["sha256"]),
                (controls_root / control_record["path"], control_record["sha256"]),
                (args.motm_root / completions[episode]["path"], completions[episode]["sha256"]),
                (
                    args.motm_root / extra_forecasts[episode]["path"],
                    extra_forecasts[episode]["sha256"],
                ),
            ]
            if any(file_sha256(path) != digest for path, digest in paths):
                raise ValueError("a source completion or teacher prediction changed")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            with (
                np.load(paths[0][0], allow_pickle=False) as data,
                np.load(paths[1][0], allow_pickle=False) as reference,
                np.load(paths[2][0], allow_pickle=False) as imputed,
                np.load(paths[3][0], allow_pickle=False) as predicted,
            ):
                context = torch.tensor(
                    (data["context"] - mean) / scale, dtype=torch.float32, device="cuda"
                )
                extra_slot = motm["identity"]["views"].index("motm_prefix_z")
                candidate_values = np.concatenate(
                    [
                        (data["candidate_values"] - mean) / scale,
                        imputed["completed_z"][extra_slot : extra_slot + 1],
                    ]
                )
                candidates = torch.tensor(candidate_values, dtype=torch.float32, device="cuda")
                methods = reference["methods"].tolist()
                positions = [
                    methods.index("prefix_input_z_" + action) for action in config["candidate_ids"]
                ]
                candidate_predictions = torch.tensor(
                    np.concatenate(
                        [
                            reference["point_z"][positions],
                            predicted["point_z"][extra_slot : extra_slot + 1],
                        ]
                    ),
                    dtype=torch.float64,
                    device="cuda",
                )
                native = torch.tensor(
                    reference["point_z"][methods.index("prefix_input_z_guarded_direct")],
                    dtype=torch.float64,
                    device="cuda",
                )
            with torch.no_grad():
                teacher = torch.cat([candidate_predictions, native[None]]).quantile(0.5, dim=0)
                anchor = int(((candidate_predictions - teacher) ** 2).mean(dim=(1, 2)).argmin())
                actual = forecast(candidates[anchor])
                np.testing.assert_allclose(
                    actual.cpu().numpy(),
                    candidate_predictions[anchor].cpu().numpy(),
                    rtol=2e-4,
                    atol=2e-4,
                )
            torch.cuda.synchronize()
            beginning = monotonic()
            result = project_consensus(candidates, context, candidate_predictions, native, forecast)
            torch.cuda.synchronize()
            metrics = [
                {
                    "step": step,
                    "teacher_mse": result["checkpoints"][step]["teacher_mse"],
                    "kind": result["checkpoints"][step]["kind"],
                    "projection_forward_calls": result["checkpoints"][step]["forward_calls"],
                    "backward_calls": result["checkpoints"][step]["backward_calls"],
                }
                for step in identity["checkpoints"]
            ]
            details = {
                "stop_reason": result["stop_reason"],
                "projection_forward_calls": result["forward_calls"],
                "backward_calls": result["backward_calls"],
                "projection_seconds": monotonic() - beginning,
                "logical_teacher_forecasts": 8,
                "parity_forward_calls": 1,
                "internal_passes_per_forecast": 2 if args.model == "timesfm2p5" else 1,
                "checkpoints": metrics,
            }
            _save_npz(
                cache,
                identity_sha256=np.asarray(identity_sha),
                forecaster_parameter_sha256=np.asarray(before),
                point_z=np.stack(
                    [
                        result["checkpoints"][step]["prediction"].cpu().numpy()
                        for step in identity["checkpoints"]
                    ]
                ),
                teacher_z=result["teacher"].cpu().numpy(),
                metadata=np.asarray(json.dumps(details)),
                **{
                    f"weights_step_{step}": result["checkpoints"][step]["weights"].cpu().numpy()
                    if result["checkpoints"][step]["weights"] is not None
                    else np.empty((0, candidates.shape[2]))
                    for step in identity["checkpoints"]
                },
            )
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["forecaster_parameter_sha256"]) != before
            ):
                raise ValueError("cached projection belongs to a different protocol or model")
            points, teacher = saved["point_z"], saved["teacher_z"]
            metadata = json.loads(str(saved["metadata"]))
        common = {
            key: record[key]
            for key in ("episode_id", "family_id", "dataset_id", "mechanism", "missing_rate")
        }
        # Actual future values first enter after projection and selection are complete.
        outcome = truth[source_index]
        for step, prediction in [(-1, teacher), *zip(identity["checkpoints"], points, strict=True)]:
            error = prediction - outcome
            if not np.isfinite(error).all():
                raise ValueError("projection evaluation produced an invalid error")
            rows.append(
                common
                | {
                    "model_id": args.model,
                    "step": step,
                    "mae": float(np.abs(error).mean()),
                    "mse": float((error**2).mean()),
                }
            )
        traces.append(common | metadata)
        files.append(
            {
                "episode_id": episode,
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        if (number + 1) % 10 == 0:
            _write_json(
                output / "progress.json",
                {"completed": number + 1, "total": len(controls["episodes"]), "model": args.model},
            )
            print(json.dumps({"completed": number + 1, "model": args.model}), flush=True)
    if before != parameter_digest(model) or any(
        parameter.grad is not None or parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("the forecasting model was modified")
    frame = pd.DataFrame(rows)
    family = (
        frame.groupby(["step", "family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=["step", "family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby("step")[["mae", "mse"]].mean().reset_index()
    frame.to_parquet(output / "episode_results.parquet", index=False)
    pd.DataFrame(traces).to_json(output / "traces.json", orient="records", indent=2)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "forecaster_parameters_unchanged": True,
            "forecaster_parameter_sha256": before,
            "decision_count": len(files),
            "predictions": files,
            "interpretation": "instance-level teacher fitting is deployable as a procedure and needs additional gradient calls; cached teacher forecasts are not zero-cost; weights are fitted without actual future labels; no guarantee of improvement or global optimization",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
