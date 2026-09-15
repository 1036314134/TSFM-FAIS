"""Collect fixed predictor representations and same-call-style source forecasts."""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from latent_source_inputs import ROOT, combine_features, project_heads, projection_matrix, read_json
from probe_differentiable_imputation import parameter_digest
from r6_policy_inputs import decision_inputs
from r6_runtime import forecast_spec, make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def capture(runner, spec, backbone, values, model_id, matrix):
    heads, shapes = [], []
    module = (
        backbone.output_patch_embedding
        if model_id == "chronos2"
        else backbone.output_projection_point
    )

    def observe(_module, inputs):
        value = inputs[0]
        shapes.append(list(value.shape))
        selected = value[:2] if model_id == "chronos2" else value[:2, -1]
        heads.append(selected.detach().float().cpu().numpy().copy())

    handle = module.register_forward_pre_hook(observe)
    try:
        result = runner.predict_missing(values[None], spec)
    finally:
        handle.remove()
    projected = project_heads(model_id, heads, matrix)
    return result.point[0], result.quantiles[0], heads, projected, shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "legacy-bundle": "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        "previous-forecasts": "artifacts/iclr27-r5/native-confirmation-v001",
        "interface-probe": "artifacts/iclr27-r6/latent-interface-probe-v001",
        "protocol": "docs/iclr2027/R7_LATENT_SOURCE_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed source representation collection")
    if read_json(args.interface_probe / "manifest.json")["status"] != "completed":
        raise ValueError("complete the real-model interface checks first")
    source = read_json(args.source_root / "episodes_manifest.json")
    chosen = [
        (index, row) for index, row in enumerate(source["episodes"]) if row["mask_seed"] == 6101
    ]
    if len(chosen) != 3906 or len({row["origin_id"] for _, row in chosen}) != 217:
        raise ValueError("the single-seed full-mechanism source selection changed")
    standards = read_json(args.accuracy_root / "standardizers.json")
    scalers = {(row["dataset_id"], row["item_id"]): row for row in standards}
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_module_sha256": file_sha256(ROOT / "scripts/latent_source_inputs.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
        "source_manifest_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "interface_probe_sha256": file_sha256(args.interface_probe / "manifest.json"),
        "mask_seed": 6101,
        "input_tasks": 3906,
        "forecast_horizon": 96,
        "future_arrays_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial source representation definitions changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    models = []
    for model_id in ("chronos2", "timesfm2p5"):
        root = output / model_id
        root.mkdir(exist_ok=True)
        if (root / "manifest.json").exists():
            done = read_json(root / "manifest.json")
            if done["identity_sha256"] != identity_sha or done["status"] != "completed":
                raise ValueError("a completed predictor collection has different definitions")
            models.append(
                {
                    "model_id": model_id,
                    "path": str((root / "manifest.json").relative_to(output)),
                    "sha256": file_sha256(root / "manifest.json"),
                }
            )
            continue
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id, args.legacy_bundle, args.previous_forecasts
        )
        devices = sorted({str(value.device) for value in backbone.parameters()})
        if any(not device.startswith("cuda") for device in devices):
            raise ValueError(
                f"first-priority representation collection requires the available GPU; got {devices}"
            )
        print(f"{model_id}: loaded on {devices}", flush=True)
        spec, matrix = forecast_spec(model_id, 96, joint), projection_matrix(model_id)
        matrix_path = root / "projection.npy"
        if matrix_path.exists():
            np.testing.assert_array_equal(np.load(matrix_path, allow_pickle=False), matrix)
        else:
            np.save(matrix_path, matrix, allow_pickle=False)
        new_queries, repeat_checks = 0, 0

        def query(
            values,
            *,
            runner=runner,
            backbone=backbone,
            spec=spec,
            model_id=model_id,
            matrix=matrix,
            root=root,
            digest=digest,
            joint=joint,
        ):
            nonlocal new_queries, repeat_checks
            effective = np.asarray(values if joint else values[:, :2], dtype=np.float32).copy(
                order="C"
            )
            effective[np.isnan(effective)] = np.nan
            key = hashlib.sha256(str(effective.shape).encode() + effective.tobytes()).hexdigest()
            path = root / "queries" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["parameter_sha256"]) != digest
                    ):
                        raise ValueError("a cached forecast representation has another identity")
                    np.testing.assert_array_equal(saved["effective_input"], effective)
                    return key, saved["point"].copy(), saved["projected"].copy()
            attempt = root / "attempts" / f"{time.time_ns()}.json"
            _write_json(
                attempt, {"status": "started", "query": key, "identity_sha256": identity_sha}
            )
            point, quantiles, heads, projected, shapes = capture(
                runner, spec, backbone, effective, model_id, matrix
            )
            if point.shape != (96, 2) or not np.isfinite(point).all():
                raise ValueError("a source candidate forecast is incomplete")
            repeated = new_queries % 128 == 0
            if repeated:
                reference = runner.predict_missing(effective[None], spec).point[0]
                np.testing.assert_array_equal(point, reference)
                repeat_checks += 1
            _save_npz(
                path,
                point=point,
                quantiles=quantiles,
                projected=projected,
                effective_input=effective,
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(digest),
                head_count=np.asarray(len(heads)),
                **{f"head_{index}": value for index, value in enumerate(heads)},
            )
            _write_json(
                attempt,
                {
                    "status": "saved",
                    "query": key,
                    "identity_sha256": identity_sha,
                    "sha256": file_sha256(path),
                    "head_shapes": shapes,
                    "point_repeat_checked": repeated,
                },
            )
            new_queries += 1
            return key, point, projected

        records, action_order = [], None
        for position, (source_index, row) in enumerate(chosen):
            path = root / "episodes" / f"{source_index:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["source_sha256"]) != row["sha256"]
                    ):
                        raise ValueError("a partial source episode belongs to another study")
                    action_order = saved["actions"].tolist()
                records.append(
                    {
                        "source_index": source_index,
                        "episode_id": row["episode_id"],
                        "origin_id": row["origin_id"],
                        "path": str(path.relative_to(root)),
                        "sha256": file_sha256(path),
                    }
                )
                continue
            source_path = args.source_root / row["path"]
            if file_sha256(source_path) != row["sha256"]:
                raise ValueError("an original source input changed")
            with np.load(source_path, allow_pickle=False) as saved:
                context, clean, candidates = (
                    saved["context"],
                    saved["clean_context"],
                    saved["candidate_values"],
                )
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            points, projected, keys = [], [], []
            for action in [*actions, "guarded_direct"]:
                current = assemble_selected_context(
                    context,
                    candidates,
                    actions,
                    [action] if joint else [action, action],
                    [0, 1],
                    joint=joint,
                )
                key, point, latent = query(current)
                keys.append(key)
                points.append((point - mean[:2]) / scale[:2])
                projected.append(latent)
            teacher_key, teacher_point, _ = query(clean)
            teacher_z = (teacher_point - mean[:2]) / scale[:2]
            inputs = decision_inputs(
                context,
                candidates,
                actions,
                coverage,
                np.stack(points),
                mean,
                scale,
                joint=joint,
                period=row["period"],
                metadata={**row, "episode_index": source_index, "model_id": model_id},
            )
            action_order = list(inputs["actions"])
            native_order = [*actions, "guarded_direct"]
            latent = np.stack(projected)[
                [native_order.index(name) for name in action_order]
            ].transpose(1, 0, 2)
            features = combine_features(inputs["gate_features"], latent, action_order.index("locf"))
            decisions = inputs["decisions"].assign(
                source_episode_id=row["episode_id"], episode_index=source_index
            )
            teacher = np.stack(
                [
                    teacher_z.reshape(-1) if slot == -1 else teacher_z[:, slot]
                    for slot in decisions.target_slot
                ]
            )
            _save_npz(
                path,
                features=features,
                vectors=inputs["vectors"],
                teacher=teacher,
                decisions=np.asarray(decisions.to_json(orient="records")),
                actions=np.asarray(action_order),
                query_keys=np.asarray([keys[native_order.index(name)] for name in action_order]),
                teacher_key=np.asarray(teacher_key),
                identity_sha256=np.asarray(identity_sha),
                source_sha256=np.asarray(row["sha256"]),
            )
            records.append(
                {
                    "source_index": source_index,
                    "episode_id": row["episode_id"],
                    "origin_id": row["origin_id"],
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                }
            )
            if (position + 1) % 50 == 0:
                _write_json(
                    root / "progress.json",
                    {
                        "completed_inputs": position + 1,
                        "total_inputs": 3906,
                        "new_queries_this_execution": new_queries,
                    },
                )
                print(f"{model_id}: {position + 1}/3906 source inputs saved", flush=True)
        if parameter_digest(backbone) != digest:
            raise ValueError("forecasting parameters changed during collection")
        attempts = [read_json(path) for path in (root / "attempts").glob("*.json")]
        _write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "model_id": model_id,
                "actions": action_order,
                "parameter_sha256": digest,
                "devices": devices,
                "projection_sha256": file_sha256(matrix_path),
                "episodes": records,
                "unique_cached_queries": len(list((root / "queries").glob("*.npz"))),
                "request_attempts": len(attempts),
                "unfinished_attempt_records": sum(row["status"] != "saved" for row in attempts),
                "extra_repeat_checks_this_execution": repeat_checks,
            },
        )
        models.append(
            {
                "model_id": model_id,
                "path": str((root / "manifest.json").relative_to(output)),
                "sha256": file_sha256(root / "manifest.json"),
            }
        )
        del query, runner, adapter, backbone
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": models,
            "new_imputer_fits": 0,
            "limits": "single mask seed; source collection only; forecasting controls must be fitted and evaluated on this same bank",
        },
    )


if __name__ == "__main__":
    main()
