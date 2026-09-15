"""Re-evaluate frozen selected composer checkpoints through the public Chronos SDK."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed public-interface audit")
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    config = source["identity"]["config"]
    source_sha = file_sha256(args.source_root / "episodes_manifest.json")
    accuracy_sha = file_sha256(args.accuracy_root / "manifest.json")
    source_records = {
        row["episode_id"]: (index, row) for index, row in enumerate(source["episodes"])
    }
    scales = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    torch.set_num_threads(1)
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        config["forecaster_artifacts"]["chronos2"], device_map="cuda"
    )
    before = parameter_digest(pipeline.model)
    rows, provenance = [], {}
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if (
            manifest["status"] != "completed"
            or not manifest["parameter_digest_unchanged"]
            or identity["source_manifest_sha256"] != source_sha
            or identity["accuracy_manifest_sha256"] != accuracy_sha
        ):
            raise ValueError("the selected training run has a different or incomplete protocol")
        snapshot = run / "composer_snapshot.py"
        raw_snapshot_sha = file_sha256(snapshot)
        lf_snapshot_sha = hashlib.sha256(snapshot.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if identity["module_sha256"] not in (raw_snapshot_sha, lf_snapshot_sha):
            raise ValueError("original composer implementation changed")
        spec = importlib.util.spec_from_file_location(
            "composer_audit_" + run.name.replace("-", "_"), snapshot
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        prior = json.loads((run / "training_prior.json").read_text(encoding="utf-8"))[
            "probabilities"
        ]
        kwargs = {}
        if identity.get("forecaster_features"):
            kwargs["forecaster_dim"] = pipeline.model.model_dim
        if identity.get("target_conditioned"):
            kwargs["target_conditioned"] = True
        models = {
            "fixed": module.FixedBlockComposer(prior),
            "adaptive": module.ContextBlockComposer(prior, **kwargs),
        }
        for method, composer in models.items():
            saved = torch.load(run / f"best-{method}.pt", map_location="cpu", weights_only=True)
            if saved["identity_sha256"] != file_sha256(run / "identity.json"):
                raise ValueError("selected weights belong to a different training run")
            composer.load_state_dict(saved["model"], strict=True)
            composer.cuda().eval().requires_grad_(False)
        stored = pd.read_parquet(run / "selected_validation.parquet").set_index(
            ["episode_id", "method"]
        )
        with torch.no_grad():
            for episode in identity["validation_ids"]:
                index, record = source_records[episode]
                path = args.source_root / record["path"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("source candidate context changed")
                scaler = scales[(record["dataset_id"], record["item_id"])]
                mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
                with np.load(path, allow_pickle=False) as saved:
                    context = torch.tensor(
                        (saved["context"] - mean) / scale, dtype=torch.float32, device="cuda"
                    )
                    candidates = torch.tensor(
                        (saved["candidate_values"] - mean) / scale,
                        dtype=torch.float32,
                        device="cuda",
                    )
                patches = None
                if identity.get("forecaster_features") and not bool(torch.isfinite(context).all()):
                    count, length, dims = candidates.shape
                    flattened = candidates.transpose(1, 2).reshape(count * dims, length)
                    patched, _, _ = pipeline.model._prepare_patched_context(flattened)
                    embedded = pipeline.model.input_patch_embedding(patched)
                    patches = embedded.reshape(count, dims, embedded.shape[1], -1).float()
                for method, composer in models.items():
                    options = {}
                    if method == "adaptive" and identity.get("forecaster_features"):
                        options["forecaster_patches"] = patches
                    if method == "adaptive" and identity.get("target_conditioned"):
                        options["targets"] = list(config["target_indices"])
                    completed = composer(candidates, context, record["period"], **options).values
                    predicted, _ = pipeline.predict_quantiles(
                        inputs=[{"target": completed.cpu().numpy().T}],
                        prediction_length=config["horizon"],
                        quantile_levels=[0.1, 0.5, 0.9],
                        batch_size=1,
                        predict_batches_jointly=False,
                    )
                    point = predicted[0].numpy()[:, :, 1].T[:, list(config["target_indices"])]
                    error = point - truth[index]
                    previous = stored.loc[(episode, method)]
                    rows.append(
                        {
                            "run": run.name,
                            "method": method,
                            "episode_id": episode,
                            "family_id": record["family_id"],
                            "dataset_id": record["dataset_id"],
                            "mae": float(np.abs(error).mean()),
                            "mse": float((error**2).mean()),
                            "stored_mae": float(previous.mae),
                            "stored_mse": float(previous.mse),
                            "model_input_contiguous_before_sdk": completed.T.is_contiguous(),
                            "complete_context": bool(torch.isfinite(context).all()),
                        }
                    )
        provenance[run.name] = {
            "run_manifest_sha256": file_sha256(run / "manifest.json"),
            "snapshot_raw_sha256": raw_snapshot_sha,
            "snapshot_lf_sha256": lf_snapshot_sha,
            "line_endings_only": raw_snapshot_sha != identity["module_sha256"],
        }
        print(json.dumps({"completed_run": run.name}), flush=True)
    if before != parameter_digest(pipeline.model):
        raise ValueError("the forecasting parameters changed during evaluation")
    frame = pd.DataFrame(rows)
    keys = ["run", "method"]
    family = (
        frame.groupby(keys + ["family_id", "dataset_id"])[
            ["mae", "mse", "stored_mae", "stored_mse"]
        ]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse", "stored_mae", "stored_mse"]].mean().reset_index()
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development_evaluation_audit",
            "source_runs": provenance,
            "script_sha256": file_sha256(Path(__file__)),
            "forecaster_parameters_unchanged": True,
            "interpretation": "public SDK evaluation of the already-selected frozen checkpoints; no retraining or new checkpoint selection; old teacher-generation accuracy is not validated by this audit",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
