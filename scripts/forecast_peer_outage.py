"""Freeze all peer-outage predictions; this stage only reads prepared historical inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from conditioning_core import source_controls
from peer_outage_core import ROOT
from probe_differentiable_imputation import parameter_digest
from r6_runtime import make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    inputs, output = args.input_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed peer forecasts")
    prepared = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    if prepared["status"] != "completed":
        raise ValueError("finish all shared-information imputers first")
    identity = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            inputs / "manifest.json",
            ROOT / "scripts/peer_outage_core.py",
            ROOT / "docs/iclr2027/R30_PEER_OUTAGE_PROTOCOL.md",
        )
    }
    for path in (
        ROOT / "docs/iclr2027/R30_CONTROL_ADDENDUM.md",
        ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json",
        ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001/matched_controls.json",
    ):
        identity[str(path)] = file_sha256(path)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and json.loads(
        (output / "identity.json").read_text(encoding="utf-8")
    ) != identity:
        raise ValueError("partial peer forecasting definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    started = perf_counter()
    _, adapter, backbone, digest, _ = make_forecaster(
        "chronos2",
        ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    pipeline = adapter._ensure_backend()
    mid = pipeline.quantiles.index(0.5)
    counters = {
        "logical_requests": 0,
        "new_queries": 0,
        "cache_hits": 0,
        "model_forwards": 0,
        "variable_rows": 0,
    }
    keys, entries, checks = set(), [], []
    torch.cuda.reset_peak_memory_stats()
    for number, row in enumerate(prepared["cases"]):
        if file_sha256(inputs / row["path"]) != row["sha256"]:
            raise ValueError("a shared-information input changed")
        with np.load(inputs / row["path"], allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        x, mean, scale, keep = data["context"], data["mean"], data["scale"], data["keep"]
        horizon = row["horizon"]
        queries = []

        def query(name, raw, selection, *, h=horizon, records=queries, center=mean, units=scale):
            counters["logical_requests"] += 1
            selected = np.flatnonzero(selection)
            canonical = np.array(
                ((raw[:, selected] - center[selected]) / units[selected]).T,
                dtype=np.float32,
                order="C",
            )
            canonical[~np.isfinite(canonical)] = np.nan
            binding = json.dumps(
                {
                    "identity": identity_sha,
                    "parameters": digest,
                    "shape": list(canonical.shape),
                    "horizon": h,
                },
                sort_keys=True,
            )
            key = hashlib.sha256(binding.encode() + canonical.tobytes()).hexdigest()
            path = output / "queries" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    np.testing.assert_array_equal(old["context_z"], canonical)
                    q = old["quantiles"]
                counters["cache_hits"] += 1
            else:
                with torch.inference_mode():
                    q = (
                        backbone(
                            context=torch.tensor(canonical, device="cuda"),
                            group_ids=torch.zeros(len(canonical), device="cuda", dtype=torch.long),
                            num_output_patches=int(np.ceil(h / pipeline.model_output_patch_size)),
                        )
                        .quantile_preds.float()
                        .cpu()
                        .numpy()
                    )
                if not np.isfinite(q).all():
                    raise ValueError("diagnose nonfinite native forecasts before scoring")
                _save_npz(
                    path,
                    context_z=canonical,
                    horizon=np.asarray(h),
                    quantiles=q,
                    binding=np.asarray(binding),
                )
                counters["new_queries"] += 1
                counters["model_forwards"] += 1
                counters["variable_rows"] += len(canonical)
            keys.add(key)
            records.append(
                {
                    "name": name,
                    "key": key,
                    "sha256": file_sha256(path),
                    "columns": selected.tolist(),
                }
            )
            return q

        methods = {}
        local_keep = np.r_[np.ones(11, bool), np.zeros(6, bool)]
        methods["native_local"] = query("native_local", x, local_keep)[:2, mid, :horizon].T
        peer_q = query("native_peer", x, keep)
        methods["native_peer"] = peer_q[:2, mid, :horizon].T
        full_bank, target_bank = [methods["native_peer"]], [methods["native_peer"]]
        for index, action in enumerate(data["actions"].tolist()):
            full = data["candidates"][index]
            q = query("peer_" + action, full, keep)
            methods["peer_" + action] = q[:2, mid, :horizon].T
            target = x.copy()
            target[:, :2] = full[:, :2]
            q = query("target_" + action, target, keep)
            methods["target_" + action] = q[:2, mid, :horizon].T
            full_bank.append(methods["peer_" + action])
            target_bank.append(methods["target_" + action])
        for prefix, bank in (("peer", full_bank), ("target", target_bank)):
            values = np.stack(bank)
            methods[prefix + "_mean8"] = values.mean(0, dtype=np.float64)
            methods[prefix + "_median8"] = np.median(values, axis=0).astype(float)
            for source_name, control in zip(
                ("source", "matched_source"), source_controls(), strict=True
            ):
                ordered = np.stack(
                    [
                        methods["native_peer"]
                        if action == "guarded_direct"
                        else methods[prefix + "_" + action]
                        for action in control["actions"]
                    ]
                ).astype(float)
                methods[prefix + "_" + source_name + "_single_mae"] = ordered[
                    control["single_index"]
                ]
                for loss, weights in (
                    ("mae", control["fixed_mae"]["weights"]),
                    (
                        "joint",
                        control.get(
                            "fixed_joint_weights", control.get("fixed_joint", {}).get("weights")
                        ),
                    ),
                ):
                    methods[prefix + "_" + source_name + "_fixed_" + loss] = (
                        ordered * np.asarray(weights)[:, None, None]
                    ).sum(0)
        for name, targets in zip(data["stat_names"].tolist(), data["stat_targets"], strict=True):
            raw = x.copy()
            raw[:, :2] = targets
            methods[name] = query(name, raw, keep)[:2, mid, :horizon].T
        future_z = np.full((horizon, 17), np.nan)
        future_z[:, keep] = peer_q[:, mid, :horizon].T
        reconciled = methods["native_peer"].astype(float).copy()
        for target, model in enumerate(json.loads(str(data["future_models"]))):
            if model["features"]:
                beta = np.asarray(model["beta"])
                reconciled[:, target] = beta[0] + future_z[:, model["features"]] @ beta[1:]
        methods["peer_future_ridge"] = reconciled
        if prepared["smoke"]:
            tensor = torch.tensor(
                np.ascontiguousarray(((x[:, keep] - mean[keep]) / scale[keep]).T),
                dtype=torch.float32,
            )
            public, _ = pipeline.predict_quantiles(
                [{"target": tensor}],
                prediction_length=horizon,
                quantile_levels=[0.5],
                batch_size=8,
                predict_batches_jointly=False,
            )
            expected = public[0].numpy()[:2, :, 0].T
            np.testing.assert_array_equal(expected, methods["native_peer"])
            checks.append({"case_id": row["case_id"], "native_public_difference": 0})
            counters["model_forwards"] += 1
        names = sorted(methods)
        points = np.stack([methods[n] for n in names]).astype(float)
        if len(names) != 37 or points.shape != (37, horizon, 2) or not np.isfinite(points).all():
            raise ValueError("invalid registered 37-method forecast panel")
        path = output / "predictions" / f"{row['case_id']}.npz"
        _save_npz(
            path, methods=np.asarray(names), points=points, queries=np.asarray(json.dumps(queries))
        )
        entries.append(
            {
                "case_id": row["case_id"],
                "panel": row["panel"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json",
            {"completed": number + 1, "total": len(prepared["cases"]), "counters": counters},
        )
        if (number + 1) % 25 == 0:
            print(
                json.dumps({"predicted": number + 1, "total": len(prepared["cases"])}), flush=True
            )
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen forecasting model changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": prepared["smoke"],
            "identity": identity,
            "input_root": str(inputs),
            "parameter_sha256": digest,
            "quantiles": pipeline.quantiles,
            "patch_size": pipeline.model_output_patch_size,
            "cases": entries,
            "counters": counters,
            "unique_queries": len(keys),
            "public_checks": checks,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "future_values_read": False,
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
