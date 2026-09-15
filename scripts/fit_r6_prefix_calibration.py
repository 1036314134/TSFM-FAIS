"""Fit frozen per-series forecast weights using historical complete-input teachers."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_differentiable_imputation import parameter_digest  # noqa: E402
from r6_runtime import forecast_spec, make_forecaster  # noqa: E402
from replay_preforecast_student import query_candidate_points  # noqa: E402

from tsfm_fais.routing.forecast_gate import compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-root",
        "prepared-root",
        "method-freeze",
        "legacy-bundle",
        "previous-forecasts",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"), required=True)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.model
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed historical calibration")
    input_path = args.input_root / "manifest.json"
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    binding = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    controls_path = Path(binding["controls_path"])
    if file_sha256(controls_path) != binding["controls_sha256"]:
        raise ValueError("the frozen source control changed")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))[args.model]
    scaler_path = args.prepared_root / "standardizers.json"
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    if inputs["status"] != "completed" or inputs["identity"]["protocol_sha256"] != file_sha256(
        args.protocol
    ):
        raise ValueError("calibration input preparation is incomplete or changed protocol")
    if file_sha256(scaler_path) != inputs["identity"]["standardizers_sha256"]:
        raise ValueError("the original prefix scaling changed")
    actions = inputs["identity"]["candidate_ids"]
    query_order = [*actions, "guarded_direct"]
    if set(query_order) != set(controls["actions"]):
        raise ValueError("source and local candidate identities disagree")
    order = [query_order.index(name) for name in controls["actions"]]
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "input_manifest_sha256": file_sha256(input_path),
        "protocol_sha256": file_sha256(args.protocol),
        "method_freeze_sha256": file_sha256(args.method_freeze),
        "controls_sha256": file_sha256(controls_path),
        "model_id": args.model,
        "context_length": 96,
        "calibration_horizon": 96,
        "source_fraction": 0.5,
        "candidate_ids": controls["actions"],
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial calibration identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    runner, adapter, backbone, digest, joint = make_forecaster(
        args.model, args.legacy_bundle, args.previous_forecasts
    )
    spec = forecast_spec(args.model, 96, joint)
    teachers, teacher_files = {}, []
    for history in inputs["histories"]:
        source = args.input_root / history["path"]
        if file_sha256(source) != history["sha256"]:
            raise ValueError("a historical teacher input changed")
        path = output / "teachers" / (history["history_id"] + ".npz")
        scaler = scalers[(history["dataset_id"], history["item_id"])]
        if not path.exists():
            with np.load(source, allow_pickle=False) as saved:
                clean = saved["clean"]
            point = (
                runner.predict(clean[None], spec).point[0] - np.asarray(scaler["mean"])[:2]
            ) / np.asarray(scaler["scale"])[:2]
            _save_npz(
                path,
                point_z=point,
                source_sha256=np.asarray(history["sha256"]),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(digest),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != history["sha256"]
                or str(saved["parameter_sha256"]) != digest
                or saved["point_z"].shape != (96, 2)
                or not np.isfinite(saved["point_z"]).all()
            ):
                raise ValueError("a cached teacher prediction changed")
            teachers[history["history_id"]] = saved["point_z"]
        teacher_files.append(
            {**history, "path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        )
    predictions, point_files = {}, []
    for index, record in enumerate(inputs["episodes"]):
        source = args.input_root / record["path"]
        if file_sha256(source) != record["sha256"]:
            raise ValueError("a calibration candidate input changed")
        path = output / "predictions" / (record["episode_id"] + ".npz")
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        if not path.exists():
            with np.load(source, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                if saved["candidate_ids"].tolist() != actions:
                    raise ValueError("candidate order changed")
            points, distinct = query_candidate_points(
                runner,
                spec,
                context,
                candidates,
                actions,
                [0, 1],
                np.asarray(scaler["mean"]),
                np.asarray(scaler["scale"]),
                joint=joint,
            )
            _save_npz(
                path,
                point_z=points[order],
                candidate_ids=np.asarray(controls["actions"]),
                distinct_contexts=np.asarray(distinct),
                source_sha256=np.asarray(record["sha256"]),
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(digest),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != record["sha256"]
                or str(saved["parameter_sha256"]) != digest
                or saved["point_z"].shape != (7, 96, 2)
                or not np.isfinite(saved["point_z"]).all()
                or saved["candidate_ids"].tolist() != controls["actions"]
            ):
                raise ValueError("a cached calibration forecast changed")
            predictions[record["episode_id"]] = saved["point_z"]
            distinct = int(saved["distinct_contexts"])
        point_files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "distinct_contexts": distinct,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(inputs["episodes"]):
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "completed_inputs": index + 1,
                        "total_inputs": len(inputs["episodes"]),
                    }
                ),
                flush=True,
            )
    if parameter_digest(backbone) != digest:
        raise ValueError("the frozen forecaster changed during calibration")
    source_weights = np.asarray(controls["convex_weights"], dtype=float)
    fitted = []
    for support in inputs["support"]:
        episodes = [
            row
            for row in inputs["episodes"]
            if (row["dataset_id"], row["item_id"]) == (support["dataset_id"], support["item_id"])
        ]
        if len(episodes) != 18 * len(support["origins"]):
            raise ValueError("history and mask multiplicities differ")
        for slot in [-1] if joint else [0, 1]:
            local, gap, metrics = source_weights.copy(), 0.0, {}
            if episodes:
                points = np.stack([predictions[row["episode_id"]] for row in episodes])
                teacher = np.stack([teachers[row["history_id"]] for row in episodes])
                vectors = points.reshape(len(points), 7, -1) if joint else points[:, :, :, slot]
                target = teacher.reshape(len(teacher), -1) if joint else teacher[:, :, slot]
                _, _, _, grams = forecast_geometry(vectors)
                alignments = projection_targets(vectors, target)["raw_projection"]
                optimum, certificate, _ = simplex_quadratic_weights(
                    grams.mean(0)[None], alignments.mean(0)[None]
                )
                local, gap = optimum[0], float(certificate[0])
                for name, weights in (
                    ("local", local),
                    ("half_local", 0.5 * (local + source_weights)),
                    ("source", source_weights),
                ):
                    point = compose_forecasts(
                        vectors, np.repeat(weights[None], len(vectors), axis=0)
                    )
                    metrics[name] = {
                        "mae": float(np.abs(point - target).mean()),
                        "mse": float(((point - target) ** 2).mean()),
                    }
                if metrics["local"]["mse"] > metrics["source"]["mse"] + 1e-9:
                    raise ValueError(
                        "local teacher minimization is worse than a feasible source mixture"
                    )
            fitted.append(
                {
                    **support,
                    "target_slot": slot,
                    "source_weights": source_weights.tolist(),
                    "local_weights": local.tolist(),
                    "primary_weights": (0.5 * (local + source_weights)).tolist(),
                    "optimality_gap": gap,
                    "calibration_teacher_metrics": metrics,
                }
            )
    if len(fitted) != (15 if joint else 30) or len(teachers) != 43 or len(predictions) != 774:
        raise ValueError("calibration fitting coverage changed")
    weights_path = output / "weights.json"
    _write_json(weights_path, fitted)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "parameter_sha256": digest,
            "parameters_unchanged": True,
            "teachers": teacher_files,
            "predictions": point_files,
            "weights_sha256": file_sha256(weights_path),
            "teacher_input_calls": len(teacher_files),
            "candidate_distinct_contexts": sum(row["distinct_contexts"] for row in point_files),
            "evaluation_future_arrays_read": False,
            "evaluation_accuracy_computed": False,
            "limits": "target-prefix teachers only; weights frozen before evaluation; R6 is already-used development data",
        },
    )


if __name__ == "__main__":
    main()
