"""Query frozen predictors only for newly registered source training histories."""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from latent_source_inputs import ROOT, read_json
from probe_differentiable_imputation import parameter_digest
from r6_policy_inputs import decision_inputs
from r6_runtime import forecast_spec, make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r9/source-supplement-inputs-v001",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "legacy-bundle": "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        "previous-forecasts": "artifacts/iclr27-r5/native-confirmation-v001",
        "protocol": "docs/iclr2027/R9_SOURCE_EXPANSION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepared, base = (
        read_json(args.prepared_root / "manifest.json"),
        read_json(args.base_root / "manifest.json"),
    )
    if (
        prepared["status"] != "completed"
        or base["status"] != "completed"
        or len(prepared["episodes"]) != 2826
    ):
        raise ValueError("the registered source inputs are incomplete")
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed supplemental forecast collection")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "base_sha256": file_sha256(args.base_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
    }
    if identity["standardizers_sha256"] != base["identity"]["standardizers_sha256"]:
        raise ValueError("source standardizers changed")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial supplemental forecasts have different definitions")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    torch.set_num_threads(1)
    models = []
    for model_id in ("chronos2", "timesfm2p5"):
        root = output / model_id
        root.mkdir(exist_ok=True)
        if (root / "manifest.json").exists():
            done = read_json(root / "manifest.json")
            if done["status"] != "completed" or done["identity_sha256"] != identity_sha:
                raise ValueError("completed supplemental predictor definitions changed")
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
        devices = sorted({str(parameter.device) for parameter in backbone.parameters()})
        if any(not device.startswith("cuda") for device in devices):
            raise ValueError("supplemental forecasting requires the registered GPU runtime")
        spec = forecast_spec(model_id, 96, joint)
        new_queries, repeat_checks = 0, 0

        def query(
            values,
            runner=runner,
            spec=spec,
            model_id=model_id,
            root=root,
            digest=digest,
            joint=joint,
        ):
            nonlocal new_queries, repeat_checks
            effective = np.asarray(values if joint else values[:, :2], np.float32).copy(order="C")
            effective[np.isnan(effective)] = np.nan
            key = hashlib.sha256(str(effective.shape).encode() + effective.tobytes()).hexdigest()
            current, previous = (
                root / "queries" / f"{key}.npz",
                args.base_root / model_id / "queries" / f"{key}.npz",
            )
            path = current if current.exists() else previous
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["effective_input"], effective)
                    if str(saved["parameter_sha256"]) != digest:
                        raise ValueError("a reused forecast has changed parameters")
                    if path == current and str(saved["identity_sha256"]) != identity_sha:
                        raise ValueError("a resumed forecast belongs to another collection")
                    return key, saved["point"].copy(), path == previous
            attempt = root / "attempts" / f"{time.time_ns()}.json"
            _write_json(
                attempt, {"status": "started", "query": key, "identity_sha256": identity_sha}
            )
            point = runner.predict_missing(effective[None], spec).point[0]
            if point.shape != (96, 2) or not np.isfinite(point).all():
                raise ValueError("a supplemental point forecast is incomplete")
            repeated = new_queries % 128 == 0
            if repeated:
                np.testing.assert_array_equal(
                    point, runner.predict_missing(effective[None], spec).point[0]
                )
                repeat_checks += 1
            _save_npz(
                current,
                point=point,
                effective_input=effective,
                parameter_sha256=np.asarray(digest),
                identity_sha256=np.asarray(identity_sha),
            )
            _write_json(
                attempt,
                {
                    "status": "saved",
                    "query": key,
                    "sha256": file_sha256(current),
                    "repeated": repeated,
                    "identity_sha256": identity_sha,
                },
            )
            new_queries += 1
            return key, point, False

        records = []
        for row in prepared["episodes"]:
            source_path = args.prepared_root / row["path"]
            if file_sha256(source_path) != row["sha256"] or row["split"] != "train":
                raise ValueError("a supplemental source input changed or is not training")
            key = hashlib.sha256(row["episode_id"].encode()).hexdigest()[:24]
            path = root / "episodes" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["source_sha256"]) != row["sha256"]
                    ):
                        raise ValueError("a resumed supplemental decision changed")
            else:
                with np.load(source_path, allow_pickle=False) as saved:
                    context, clean, future, candidates = [
                        saved[name]
                        for name in ("context", "clean_context", "future", "candidate_values")
                    ]
                    actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
                scaler = scalers[(row["dataset_id"], row["item_id"])]
                mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
                points, keys, reused = [], [], []
                for action in [*actions, "guarded_direct"]:
                    values = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if joint else [action, action],
                        [0, 1],
                        joint=joint,
                    )
                    query_key, point, borrowed = query(values)
                    points.append((point - mean[:2]) / scale[:2])
                    keys.append(query_key)
                    reused.append(borrowed)
                teacher_key, teacher, teacher_reused = query(clean)
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
                    metadata={**row, "episode_index": -1, "model_id": model_id},
                )
                frame = inputs["decisions"].assign(
                    source_episode_id=row["episode_id"], episode_index=-1
                )
                teacher_z, truth_z = (
                    (teacher - mean[:2]) / scale[:2],
                    (future[:, :2] - mean[:2]) / scale[:2],
                )

                def target_vectors(values, slots=frame.target_slot):
                    return np.stack(
                        [values.reshape(-1) if slot == -1 else values[:, slot] for slot in slots]
                    )

                order = [[*actions, "guarded_direct"].index(action) for action in inputs["actions"]]
                _save_npz(
                    path,
                    features=inputs["gate_features"],
                    vectors=inputs["vectors"],
                    teacher=target_vectors(teacher_z),
                    truth=target_vectors(truth_z),
                    decisions=np.asarray(frame.to_json(orient="records")),
                    actions=np.asarray(inputs["actions"]),
                    query_keys=np.asarray(keys)[order],
                    query_reused=np.asarray(reused)[order],
                    teacher_key=np.asarray(teacher_key),
                    teacher_reused=np.asarray(teacher_reused),
                    identity_sha256=np.asarray(identity_sha),
                    source_sha256=np.asarray(row["sha256"]),
                )
            records.append(
                {
                    "episode_id": row["episode_id"],
                    "origin_id": row["origin_id"],
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                }
            )
            if len(records) % 180 == 0:
                print(f"{model_id}: {len(records)}/2826 supplemental inputs", flush=True)
        if parameter_digest(backbone) != digest:
            raise ValueError("frozen predictor parameters changed")
        _write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "model_id": model_id,
                "parameter_sha256": digest,
                "devices": devices,
                "episodes": records,
                "new_queries_this_process": new_queries,
                "repeat_checks_this_process": repeat_checks,
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
            "training_origins_added": 157,
            "input_tasks_added": 2826,
            "validation_origins_added": 0,
        },
    )


if __name__ == "__main__":
    main()
