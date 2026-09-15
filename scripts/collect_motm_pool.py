"""Add only MoTM forecast queries and rebuild matched seven/eight-candidate inputs."""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401 - initialize Arrow before Torch on Windows.
import torch
from latent_source_inputs import ROOT, load_source_inputs, read_json
from pool_gate_inputs import pool_inputs
from probe_differentiable_imputation import parameter_digest
from r6_runtime import forecast_spec, make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "motm-root": "artifacts/iclr27-r12/source-motm-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "legacy-bundle": "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        "previous-forecasts": "artifacts/iclr27-r5/native-confirmation-v001",
        "protocol": "docs/iclr2027/R12_MOTM_POOL_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed eight-candidate collection")
    source = read_json(args.source_root / "episodes_manifest.json")
    base = read_json(args.base_root / "manifest.json")
    motm = read_json(args.motm_root / "manifest.json")
    if (
        motm["status"] != "completed"
        or motm["identity"]["base_sha256"] != file_sha256(args.base_root / "manifest.json")
        or base["identity"]["standardizers_sha256"]
        != file_sha256(args.accuracy_root / "standardizers.json")
    ):
        raise ValueError("the source MoTM inputs or normalizers changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_module_sha256": file_sha256(ROOT / "scripts/pool_gate_inputs.py"),
        "source_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
        "base_sha256": file_sha256(args.base_root / "manifest.json"),
        "motm_sha256": file_sha256(args.motm_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial eight-candidate definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    motm_by_id = {row["episode_id"]: row for row in motm["episodes"]}
    torch.set_num_threads(1)
    models = []
    for model_id in ("chronos2", "timesfm2p5"):
        root = output / model_id
        root.mkdir(exist_ok=True)
        if (root / "manifest.json").exists():
            old = read_json(root / "manifest.json")
            if old["status"] != "completed" or old["identity_sha256"] != identity_sha:
                raise ValueError("a completed pool model has different provenance")
            models.append(
                {
                    "model_id": model_id,
                    "path": str((root / "manifest.json").relative_to(output)),
                    "sha256": file_sha256(root / "manifest.json"),
                }
            )
            continue
        base_model, frame, arrays = load_source_inputs(args.base_root, model_id)
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id, args.legacy_bundle, args.previous_forecasts
        )
        spec = forecast_spec(model_id, 96, joint)
        new_queries, repeats = 0, 0

        def query(
            values,
            root=root,
            model_id=model_id,
            runner=runner,
            spec=spec,
            digest=digest,
            joint=joint,
        ):
            nonlocal new_queries, repeats
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
                    if str(saved["parameter_sha256"]) != digest or (
                        path == current and str(saved["identity_sha256"]) != identity_sha
                    ):
                        raise ValueError("a reused forecast has different provenance")
                    return key, saved["point"].copy(), path == previous
            attempt = root / "attempts" / f"{time.time_ns()}.json"
            _write_json(
                attempt, {"status": "started", "query": key, "identity_sha256": identity_sha}
            )
            point = runner.predict_missing(effective[None], spec).point[0]
            if point.shape != (96, 2) or not np.isfinite(point).all():
                raise ValueError("a MoTM downstream forecast is incomplete")
            repeated = new_queries % 128 == 0
            if repeated:
                np.testing.assert_array_equal(
                    point, runner.predict_missing(effective[None], spec).point[0]
                )
                repeats += 1
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
                },
            )
            new_queries += 1
            return key, point, False

        records = []
        checked = 0
        for position, record in enumerate(base_model["episodes"]):
            row = source["episodes"][record["source_index"]]
            raw_path = args.source_root / row["path"]
            if file_sha256(raw_path) != row["sha256"] or row["episode_id"] != record["episode_id"]:
                raise ValueError("an original source context changed")
            motm_record = motm_by_id[row["episode_id"]]
            motm_path = args.motm_root / motm_record["path"]
            if file_sha256(motm_path) != motm_record["sha256"]:
                raise ValueError("a MoTM completion changed")
            indices = np.array([position]) if joint else np.array([2 * position, 2 * position + 1])
            old_vectors = arrays["vectors"][indices]
            sorted_points = (
                old_vectors[0].reshape(7, 96, 2) if joint else np.stack(old_vectors, axis=-1)
            )
            with (
                np.load(raw_path, allow_pickle=False) as raw,
                np.load(motm_path, allow_pickle=False) as extra,
            ):
                context, candidates, actions, coverage = (
                    raw["context"],
                    raw["candidate_values"],
                    raw["candidate_ids"].tolist(),
                    raw["native_coverage"],
                )
                completion, extra_coverage = extra["values"], float(extra["native_coverage"])
            old_names = sorted([*actions, "guarded_direct"])
            raw_points = sorted_points[
                [old_names.index(name) for name in [*actions, "guarded_direct"]]
            ]
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            metadata = {**row, "episode_index": record["source_index"], "model_id": model_id}
            old_frame, old_features, reconstructed, _ = pool_inputs(
                context,
                candidates,
                actions,
                coverage,
                raw_points,
                mean,
                scale,
                joint=joint,
                period=row["period"],
                metadata=metadata,
            )
            np.testing.assert_array_equal(old_features, arrays["features"][indices, :, :33])
            np.testing.assert_array_equal(reconstructed, old_vectors)
            np.testing.assert_array_equal(old_frame.episode_id, frame.iloc[indices].episode_id)
            checked += len(indices)
            path = root / "episodes" / raw_path.name
            if not path.exists():
                key, point, borrowed = query(completion)
                points = np.concatenate(
                    [raw_points[:-1], ((point - mean[:2]) / scale[:2])[None], raw_points[-1:]]
                )
                values = np.concatenate([candidates, completion[None]])
                decisions, features, vectors, names = pool_inputs(
                    context,
                    values,
                    [*actions, "motm_reference"],
                    np.r_[coverage, extra_coverage],
                    points,
                    mean,
                    scale,
                    joint=joint,
                    period=row["period"],
                    metadata=metadata,
                )
                _save_npz(
                    path,
                    features=features,
                    vectors=vectors,
                    decisions=np.asarray(decisions.to_json(orient="records")),
                    actions=np.asarray(names),
                    motm_query=np.asarray(key),
                    query_borrowed=np.asarray(borrowed),
                    source_sha256=np.asarray(row["sha256"]),
                    motm_sha256=np.asarray(motm_record["sha256"]),
                    identity_sha256=np.asarray(identity_sha),
                )
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["identity_sha256"]) != identity_sha
                    or str(saved["source_sha256"]) != row["sha256"]
                    or str(saved["motm_sha256"]) != motm_record["sha256"]
                ):
                    raise ValueError("a resumed pool input changed")
            records.append(
                {
                    "episode_id": row["episode_id"],
                    "source_index": record["source_index"],
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                }
            )
            if len(records) % 200 == 0:
                print(f"{model_id}: {len(records)}/3906 eight-candidate inputs", flush=True)
        if parameter_digest(backbone) != digest:
            raise ValueError("frozen forecasting parameters changed")
        _write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "model_id": model_id,
                "parameter_sha256": digest,
                "episodes": records,
                "old_decisions_exactly_replayed": checked,
                "new_queries_this_process": new_queries,
                "repeated_queries_this_process": repeats,
            },
        )
        models.append(
            {
                "model_id": model_id,
                "path": str((root / "manifest.json").relative_to(output)),
                "sha256": file_sha256(root / "manifest.json"),
            }
        )
        del runner, adapter, backbone, query
        gc.collect()
        torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": models,
            "input_tasks": 3906,
            "future_arrays_read": False,
        },
    )


if __name__ == "__main__":
    main()
