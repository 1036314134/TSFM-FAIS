"""Check read-only forecast-head inputs under the existing frozen runtime."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from audit_source_quantiles import read_json
from probe_differentiable_imputation import parameter_digest
from r6_runtime import ROOT, forecast_spec, make_forecaster

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def cached_probe_call(output, key, identity_sha, parameter_sha, values, invoke):
    path = output / "calls" / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["parameter_sha256"]) != parameter_sha
                or str(saved["call_key"]) != key
            ):
                raise ValueError("a saved interface call belongs to another run")
            np.testing.assert_array_equal(saved["input_values"], values)
            point = saved["point"].copy()
            hidden = [
                saved[f"head_input_{index}"].copy() for index in range(int(saved["head_count"]))
            ]
        return point, hidden
    attempt_path = output / "attempts" / f"{time.time_ns()}.json"
    attempt = {"status": "started", "call_key": key, "identity_sha256": identity_sha}
    _write_json(attempt_path, attempt)
    point, hidden = invoke()
    if not np.isfinite(point).all() or any(not np.isfinite(value).all() for value in hidden):
        raise ValueError("an interface call returned nonfinite values")
    _save_npz(
        path,
        point=point,
        input_values=values,
        head_count=np.asarray(len(hidden)),
        identity_sha256=np.asarray(identity_sha),
        parameter_sha256=np.asarray(parameter_sha),
        call_key=np.asarray(key),
        **{f"head_input_{index}": value for index, value in enumerate(hidden)},
    )
    _write_json(
        attempt_path,
        {
            **attempt,
            "status": "saved",
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
        },
    )
    return point, hidden


def capture_prediction(runner, spec, backbone, values, model_id):
    captured = []
    head = (
        backbone.output_patch_embedding
        if model_id == "chronos2"
        else backbone.output_projection_point
    )

    def observe(_module, inputs):
        value = inputs[0]
        if value.ndim != 3 or not bool(torch.isfinite(value).all()):
            raise ValueError("forecast-head input is not a finite three-dimensional tensor")
        captured.append(value.detach().float().cpu().numpy().copy())

    handle = head.register_forward_pre_hook(observe)
    try:
        point = runner.predict(values[None], spec).point[0]
    finally:
        handle.remove()
    if not captured:
        raise ValueError("the public forecast did not expose the selected head input")
    if model_id == "chronos2":
        if len(captured) != 1 or captured[0].shape[0] != values.shape[1]:
            raise ValueError(
                "Chronos variable rows or prediction passes differ from the expected mapping"
            )
    elif len(captured) != 2 or any(value.shape[0] < 2 for value in captured):
        raise ValueError("TimesFM must expose both sign branches and both true target rows")
    return point, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "artifacts/iclr27-r3/development-expanded-v001"
    )
    parser.add_argument(
        "--legacy-bundle",
        type=Path,
        default=ROOT / "artifacts/iclr27-r5/confirmation-source-bundle-v001",
    )
    parser.add_argument(
        "--previous-forecasts",
        type=Path,
        default=ROOT / "artifacts/iclr27-r5/native-confirmation-v001",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R6_LATENT_INTERFACE_PROBE_PLAN.md"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed interface probe")
    source = read_json(args.source_root / "episodes_manifest.json")
    chosen = []
    for dataset in ("ETTh1", "Coastal_T_S_H"):
        row = next(
            row
            for row in source["episodes"]
            if row["dataset_id"] == dataset and row["split"] == "train"
        )
        path = args.source_root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("a selected original source input changed")
        with np.load(path, allow_pickle=False) as saved:
            ids, candidates = saved["candidate_ids"].tolist(), saved["candidate_values"]
            for action in ("locf", "linear_interp"):
                values = candidates[ids.index(action)].copy()
                if values.shape[0] != 96 or not np.isfinite(values).all():
                    raise ValueError(
                        "the interface probe requires the cached finite candidate history"
                    )
                chosen.append((row, action, values))
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "runtime_sha256": file_sha256(ROOT / "scripts/r6_runtime.py"),
        "source_manifest_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
        "legacy_bundle_sha256": file_sha256(args.legacy_bundle / "manifest.json"),
        "future_arrays_read": False,
        "selected_inputs": [
            {"episode_id": row["episode_id"], "candidate": action, "input_sha256": row["sha256"]}
            for row, action, _ in chosen
        ],
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("partial interface probe definitions changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    records, model_records, request_count = [], [], 0
    for model_id in ("chronos2", "timesfm2p5"):
        runner, adapter, backbone, digest, joint = make_forecaster(
            model_id, args.legacy_bundle, args.previous_forecasts
        )
        spec = forecast_spec(model_id, 96, joint)
        for index, (row, action, values) in enumerate(chosen):

            def request(
                kind,
                current_values,
                capture,
                *,
                runner=runner,
                spec=spec,
                backbone=backbone,
                model_id=model_id,
                index=index,
                digest=digest,
            ):
                return cached_probe_call(
                    output,
                    f"{model_id}_{index}_{kind}",
                    identity_sha,
                    digest,
                    current_values,
                    lambda: (
                        capture_prediction(runner, spec, backbone, current_values, model_id)
                        if capture
                        else (runner.predict(current_values[None], spec).point[0], [])
                    ),
                )

            reference, _ = request("reference", values, False)
            repeated, _ = request("repeat", values, False)
            request_count += 2
            np.testing.assert_array_equal(reference, repeated)
            point, hidden = request("capture", values, True)
            repeated_point, repeated_hidden = request("capture_repeat", values, True)
            request_count += 2
            np.testing.assert_array_equal(point, reference)
            np.testing.assert_array_equal(repeated_point, reference)
            if len(hidden) != len(repeated_hidden):
                raise ValueError("forecast-head call count is not repeatable")
            for first, second in zip(hidden, repeated_hidden, strict=True):
                np.testing.assert_array_equal(first, second)
            path = output / f"{model_id}_{index}.npz"
            _save_npz(
                path,
                point=point,
                input_values=values,
                **{f"head_input_{slot}": value for slot, value in enumerate(hidden)},
            )
            entry = {
                "model_id": model_id,
                "episode_id": row["episode_id"],
                "candidate": action,
                "input_shape": list(values.shape),
                "head_input_shapes": [list(value.shape) for value in hidden],
                "path": path.name,
                "sha256": file_sha256(path),
                "point_repeat_difference": 0,
                "point_capture_difference": 0,
                "representation_repeat_difference": 0,
            }
            if model_id == "timesfm2p5" and index == 0:
                changed = values.copy()
                changed[:, 1] += 0.25 * max(float(np.std(changed[:, 1])), 1e-3)
                changed_point, changed_hidden = request("perturb_target1", changed, True)
                request_count += 1
                np.testing.assert_array_equal(changed_point[:, 0], point[:, 0])
                for original, perturbed in zip(hidden, changed_hidden, strict=True):
                    np.testing.assert_array_equal(original[0], perturbed[0])
                entry["independent_first_target_unchanged"] = True
            records.append(entry)
            print(
                f"{model_id} {row['dataset_id']} {action}: point and representation replay passed",
                flush=True,
            )
        after = parameter_digest(backbone)
        if after != digest:
            raise ValueError("the interface probe changed forecasting parameters")
        model_records.append(
            {
                "model_id": model_id,
                "parameter_sha256_before": digest,
                "parameter_sha256_after": after,
            }
        )
        del request, runner, adapter, backbone
        gc.collect()
        torch.cuda.empty_cache()
    if request_count != 33:
        raise ValueError("the bounded interface request budget changed")
    attempts = [read_json(path) for path in sorted((output / "attempts").glob("*.json"))]
    call_files = sorted((output / "calls").glob("*.npz"))
    if len(call_files) != 33:
        raise ValueError("the completed interface call coverage is incomplete")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": model_records,
            "cases": records,
            "single_candidate_forecast_requests": request_count,
            "request_attempts_registered_before_call": len(attempts),
            "unfinished_attempt_records": sum(row["status"] != "saved" for row in attempts),
            "call_files_sha256": {
                str(path.relative_to(output)): file_sha256(path) for path in call_files
            },
            "new_fits": 0,
            "limits": "finite source candidates and H96 only; different head-input semantics across models; no accuracy, missing-input generalization or selection benefit is established",
        },
    )


if __name__ == "__main__":
    main()
