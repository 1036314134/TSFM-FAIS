"""Train only low-rank repair modules on frozen source forecasting tasks."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from learned_patch_repair import PatchRepair, RepairHook, repair_dimensions
from patch_repair_inputs import load_case, source_identity, source_population, training_weights
from patch_repair_runtime import differentiable_point
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.utility_experiment import _write_json, file_sha256


def repair_loss(point, truth, mean, scale, penalty, weight=1.0):
    error = (point.double() - truth.double()) / scale
    return weight * torch.sqrt(error.square() + 1e-6).mean() + 1e-3 * penalty


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def save_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    torch.load(temporary, map_location="cpu", weights_only=True)
    temporary.replace(path)


def run_step(model_id, adapter, backbone, repair, optimizer, condition, row, arrays, weight):
    raw, observed, truth = arrays
    values = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
    target = torch.as_tensor(truth, device="cuda", dtype=torch.float64)
    mean = torch.tensor(row["scaler"]["mean"][:2], device="cuda", dtype=torch.float64)
    scale = torch.tensor(row["scaler"]["scale"][:2], device="cuda", dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad(), RepairHook(backbone, model_id, observed, repair, condition) as hook:
        point = differentiable_point(model_id, adapter, backbone, values)
        loss = repair_loss(point, target, mean, scale, hook.penalty, weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError("the registered source loss became nonfinite")
        updated = bool(loss.requires_grad)
        norm = 0.0
        if updated:
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(repair.parameters(), 1.0))
            if not np.isfinite(norm):
                raise ValueError("the source gradient became nonfinite")
            optimizer.step()
    return float(loss.detach()), norm, updated


def verify_deployment(model_id, runner, adapter, backbone, repair, condition, arrays, row):
    raw, observed, _ = arrays
    records = []
    for length in (96, 192):
        # The repeated L192 case is an interface test, never a forecast-accuracy sample.
        values = raw if length == 96 else np.tile(raw, (2, 1))
        mask = observed if length == 96 else np.tile(observed, (2, 1))
        tensor = torch.as_tensor(values, device="cuda", dtype=torch.float32)
        spec = ForecastSpec(
            model_id,
            "joint_multivariate" if model_id == "chronos2" else "independent_univariate",
            96,
            context_length=length,
            target_indices=[0, 1],
        )
        with torch.no_grad(), RepairHook(backbone, model_id, mask, repair, condition):
            direct = differentiable_point(model_id, adapter, backbone, tensor).cpu().numpy()
        with RepairHook(backbone, model_id, mask, repair, condition):
            public = runner.predict_missing(values[None], spec).point[0]
        difference = float((abs(direct - public) / np.asarray(row["scaler"]["scale"][:2])).max())
        if difference > 1e-6:
            raise ValueError(f"trained repair deployment parity failed: {difference}")
        baseline = runner.predict_missing(values[None], spec).point[0]
        with RepairHook(backbone, model_id, np.ones_like(mask), repair, condition):
            clean = runner.predict_missing(values[None], spec).point[0]
        np.testing.assert_array_equal(clean, baseline)
        records.append(
            {
                "context_length": length,
                "active_standardized_max_difference": difference,
                "complete_identity_exact": True,
                "synthetic_length_interface_only": length != 96,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair training")
    population, excluded = source_population()
    rows = [r for r in population if r["split"] == "train"]
    validation = [r for r in population if r["split"] == "validation"]
    if len(rows) != 2682 or len(validation) != 864 or len(excluded) != 360:
        raise ValueError("the registered source population changed")
    identity = {
        "files": {
            **source_identity(),
            **{
                str(p.relative_to(ROOT)): file_sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/learned_patch_repair.py",
                    ROOT / "scripts/patch_repair_runtime.py",
                    ROOT / "docs/iclr2027/R26_MAE_REPAIR_PROTOCOL.md",
                )
            },
        },
        "prediction_loss": "source_smooth_mae",
        "seed": 5101,
        "epochs": 3,
        "rank": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "gradient_clip": 1.0,
        "repair_penalty": 0.001,
        "smoke_only": args.smoke,
        "training_examples": len(rows),
        "training_origins": len({r["origin_id"] for r in rows}),
        "training_families": sorted({r["family_id"] for r in rows}),
        "source_weighting": "family/dataset balanced forecast loss; per-example relative repair-energy penalty",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial repair training definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    fields = (
        "episode_id",
        "episode_index",
        "origin_id",
        "dataset_id",
        "family_id",
        "item_id",
        "origin",
        "split",
        "dimensions",
        "path",
        "sha256",
    )
    _write_json(
        output / "population.json",
        {
            "training": [{k: r[k] for k in fields} for r in rows],
            "validation": [{k: r[k] for k in fields} for r in validation],
            "excluded_dimension_over_64": [{k: r[k] for k in fields} for r in excluded],
        },
    )
    weight = training_weights(rows)
    orders = [np.random.default_rng(5101 + epoch).permutation(len(rows)) for epoch in range(3)]
    total_steps = 4 if args.smoke else 3 * len(rows)
    cached, model_records, loads = {}, [], []
    torch.set_num_threads(1)
    for model_id in ("chronos2",):
        load_started = perf_counter()
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id,
            ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
            ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
        )
        loads.append({"model_id": model_id, "load_seconds": perf_counter() - load_started})
        width, patch = repair_dimensions(backbone, model_id)
        counter = {"forward_batches": 0}

        def count_forward(module, args, kwargs, counter=counter):
            counter["forward_batches"] += 1

        counter_handle = backbone.register_forward_pre_hook(count_forward, with_kwargs=True)
        for condition in ("fraction",):
            directory = output / model_id / condition
            marker = directory / "manifest.json"
            if marker.exists():
                done = read_json(marker)
                if (
                    done["identity_sha256"] != identity_sha
                    or file_sha256(output / done["checkpoint_path"]) != done["checkpoint_sha256"]
                ):
                    raise ValueError("an already completed repair model changed")
                model_records.append(done)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(5101)
            repair = PatchRepair(width, patch).to("cuda")
            initial = cpu_state(repair)
            initial_path = directory / "initial.pt"
            if initial_path.exists():
                previous = torch.load(initial_path, map_location="cpu", weights_only=True)
                for name in initial:
                    if not torch.equal(initial[name], previous[name]):
                        raise ValueError("the registered initialization changed")
            else:
                save_state(initial_path, initial)
            optimizer = torch.optim.AdamW(
                repair.parameters(), lr=0.001, weight_decay=0.001, foreach=False
            )
            progress_path = directory / "progress.pt"
            start, updates, previous_seconds = 0, 0, 0.0
            if progress_path.exists():
                saved = torch.load(progress_path, map_location="cpu", weights_only=True)
                if saved["identity_sha256"] != identity_sha:
                    raise ValueError("a resume checkpoint belongs to different definitions")
                repair.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                start, updates, previous_seconds = (
                    saved["next_step"],
                    saved["optimizer_updates"],
                    saved["elapsed_seconds"],
                )
            first_forward = counter["forward_batches"]
            started = perf_counter()
            with (directory / "training_steps.jsonl").open("a", encoding="utf-8") as log:
                for step in range(start, total_steps):
                    epoch, position = divmod(step, len(rows))
                    index = int(orders[epoch][position])
                    if index not in cached:
                        cached[index] = load_case(rows[index])
                    if step == total_steps - 1:
                        save_state(
                            directory / "last_step_before.pt",
                            {
                                "identity_sha256": identity_sha,
                                "model": cpu_state(repair),
                                "optimizer": optimizer.state_dict(),
                                "step": step,
                                "row_index": index,
                                "weight": float(weight[index]),
                            },
                        )
                    loss, gradient, updated = run_step(
                        model_id,
                        adapter,
                        backbone,
                        repair,
                        optimizer,
                        condition,
                        rows[index],
                        cached[index],
                        float(weight[index]),
                    )
                    updates += int(updated)
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "row_index": index,
                                "source_loss": loss,
                                "gradient_norm_before_clip": gradient,
                                "optimizer_updated": updated,
                            }
                        )
                        + "\n"
                    )
                    if (step + 1) % 100 == 0 or step + 1 == total_steps:
                        log.flush()
                        save_state(
                            progress_path,
                            {
                                "identity_sha256": identity_sha,
                                "model": cpu_state(repair),
                                "optimizer": optimizer.state_dict(),
                                "next_step": step + 1,
                                "optimizer_updates": updates,
                                "elapsed_seconds": previous_seconds + perf_counter() - started,
                            },
                        )
                        _write_json(
                            output / "progress.json",
                            {
                                "model_id": model_id,
                                "condition": condition,
                                "completed_steps": step + 1,
                                "total_steps": total_steps,
                                "completed_models": len(model_records),
                                "total_models": 1,
                                "elapsed_seconds_this_model": previous_seconds
                                + perf_counter()
                                - started,
                            },
                        )
                        print(f"{model_id}/{condition}: {step + 1}/{total_steps}", flush=True)
            repair.eval()
            first_index = int(orders[0][0])
            if first_index not in cached:
                cached[first_index] = load_case(rows[first_index])
            parity = verify_deployment(
                model_id,
                runner,
                adapter,
                backbone,
                repair,
                condition,
                cached[first_index],
                rows[first_index],
            )
            if parameter_digest(backbone) != digest or any(
                p.grad is not None for p in backbone.parameters()
            ):
                raise ValueError("the supposedly frozen backbone acquired an update or gradient")
            checkpoint = directory / "model.pt"
            save_state(checkpoint, cpu_state(repair))
            record = {
                "model_id": model_id,
                "condition": condition,
                "identity_sha256": identity_sha,
                "width": width,
                "patch_size": patch,
                "rank": 8,
                "parameter_count": sum(p.numel() for p in repair.parameters()),
                "backbone_sha256": digest,
                "steps": total_steps,
                "optimizer_updates": updates,
                "checkpoint_path": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": file_sha256(checkpoint),
                "initial_path": str(initial_path.relative_to(output)),
                "initial_sha256": file_sha256(initial_path),
                "last_step_before_path": str(
                    (directory / "last_step_before.pt").relative_to(output)
                ),
                "last_step_before_sha256": file_sha256(directory / "last_step_before.pt"),
                "deployment_parity": parity,
                "wall_seconds_accumulated": previous_seconds + perf_counter() - started,
                "forward_batches_this_invocation": counter["forward_batches"] - first_forward,
                "resumed_from_step": start,
                "cost_limit": "a failed attempt may include an unrecorded in-flight step; training_steps.jsonl retains completed attempts",
            }
            _write_json(marker, record)
            model_records.append(record)
            del optimizer, repair
            gc.collect()
            torch.cuda.empty_cache()
        counter_handle.remove()
        del runner, adapter, backbone
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": model_records,
            "loads": loads,
            "population_sha256": file_sha256(output / "population.json"),
            "evaluation_future_labels_read": False,
            "limits": "single-seed source-trained adapter pilot; all evaluation and independent audit pending",
        },
    )


if __name__ == "__main__":
    main()
