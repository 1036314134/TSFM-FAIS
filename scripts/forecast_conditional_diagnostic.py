"""Freeze forecasts before drawing any conditional future outcomes."""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401 - initialize Arrow before Torch on Windows.
import torch
from latent_source_inputs import ROOT, read_json
from probe_differentiable_imputation import parameter_digest
from r6_runtime import forecast_spec, make_forecaster
from replay_preforecast_student import assemble_selected_context

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r14/conditional-inputs-v001",
        "legacy-bundle": "artifacts/iclr27-r5/confirmation-source-bundle-v001",
        "previous-forecasts": "artifacts/iclr27-r5/native-confirmation-v001",
        "protocol": "docs/iclr2027/R14_CONDITIONAL_RISK_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    prep = read_json(args.prepared_root / "manifest.json")
    if prep["status"] != "completed" or len(prep["episodes"]) != 180:
        raise ValueError("complete the prespecified synthetic inputs first")
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed controlled forecasts")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "future_noise_multipliers": [0.0, 0.5, 1.0, 2.0],
        "future_samples_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial controlled forecasting definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    models = []
    for model_id in ("chronos2", "timesfm2p5"):
        root = output / model_id
        root.mkdir(exist_ok=True)
        if (root / "manifest.json").exists():
            done = read_json(root / "manifest.json")
            if done["status"] != "completed" or done["identity_sha256"] != identity_sha:
                raise ValueError("a completed synthetic forecast bank changed")
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
        spec = forecast_spec(model_id, 96, joint)
        new_queries, repeats = 0, 0

        def query(values, runner=runner, spec=spec, root=root, digest=digest, joint=joint):
            nonlocal new_queries, repeats
            effective = np.asarray(values if joint else values[:, :2], np.float32).copy(order="C")
            effective[np.isnan(effective)] = np.nan
            key = hashlib.sha256(str(effective.shape).encode() + effective.tobytes()).hexdigest()
            path = root / "queries" / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["effective_input"], effective)
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["parameter_sha256"]) != digest
                    ):
                        raise ValueError("a controlled forecast cache changed")
                    return key, saved["point"].copy()
            attempt = root / "attempts" / f"{time.time_ns()}.json"
            _write_json(attempt, {"status": "started", "query": key})
            point = runner.predict_missing(effective[None], spec).point[0]
            if point.shape != (96, 2) or not np.isfinite(point).all():
                raise ValueError("a controlled forecast is incomplete")
            repeat = new_queries % 128 == 0
            if repeat:
                np.testing.assert_array_equal(
                    point, runner.predict_missing(effective[None], spec).point[0]
                )
                repeats += 1
            _save_npz(
                path,
                point=point,
                effective_input=effective,
                identity_sha256=np.asarray(identity_sha),
                parameter_sha256=np.asarray(digest),
            )
            _write_json(
                attempt,
                {"status": "saved", "query": key, "sha256": file_sha256(path), "repeated": repeat},
            )
            new_queries += 1
            return key, point

        records = []
        for row in prep["episodes"]:
            source = args.prepared_root / row["path"]
            if file_sha256(source) != row["sha256"]:
                raise ValueError("a controlled imputation input changed")
            path = root / "episodes" / source.name
            if not path.exists():
                with np.load(source, allow_pickle=False) as saved:
                    context = saved["context"]
                    candidates = np.concatenate(
                        [saved["candidate_values"], saved["motm_values"][None]]
                    )
                    actions = [*saved["candidate_ids"].tolist(), "motm_reference"]
                names = sorted([*actions, "guarded_direct"])
                points, keys = [], []
                for action in names:
                    values = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if joint else [action, action],
                        [0, 1],
                        joint=joint,
                    )
                    key, point = query(values)
                    points.append(point)
                    keys.append(key)
                scaler = prep["prefixes"][row["generator"]]
                mean, scale = np.asarray(scaler["mean"])[:2], np.asarray(scaler["scale"])[:2]
                points = (np.stack(points) - mean) / scale
                if row["mechanism"] == "complete":
                    np.testing.assert_array_equal(points, np.repeat(points[:1], 8, axis=0))
                _save_npz(
                    path,
                    point_z=points,
                    actions=np.asarray(names),
                    query_keys=np.asarray(keys),
                    identity_sha256=np.asarray(identity_sha),
                    source_sha256=np.asarray(row["sha256"]),
                )
            with np.load(path, allow_pickle=False) as saved:
                if (
                    str(saved["identity_sha256"]) != identity_sha
                    or str(saved["source_sha256"]) != row["sha256"]
                ):
                    raise ValueError("a resumed controlled forecast changed")
            records.append(
                {
                    "episode_id": row["episode_id"],
                    "path": str(path.relative_to(root)),
                    "sha256": file_sha256(path),
                }
            )
        if parameter_digest(backbone) != digest:
            raise ValueError("frozen forecast parameters changed")
        _write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "model_id": model_id,
                "parameter_sha256": digest,
                "episodes": records,
                "new_queries_this_process": new_queries,
                "repeats_this_process": repeats,
            },
        )
        models.append(
            {
                "model_id": model_id,
                "path": str((root / "manifest.json").relative_to(output)),
                "sha256": file_sha256(root / "manifest.json"),
            }
        )
        print(f"{model_id}: all 180 controlled forecast inputs frozen", flush=True)
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
            "noise_levels_share_identical_forecasts": True,
        },
    )


if __name__ == "__main__":
    main()
