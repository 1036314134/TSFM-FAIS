"""Build visible selector inputs and separate exact-risk labels after forecast freeze."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from conditional_supervision import conditional_labels, factual_future, supervision_scenarios
from latent_source_inputs import ROOT, read_json
from pool_gate_inputs import motm_coverage, pool_inputs

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def visible_inputs(
    row,
    index,
    context,
    candidates,
    actions,
    coverage,
    motm,
    diagnostics,
    points,
    names,
    center,
    scale,
    model_id,
    period,
):
    actions = [*actions, "motm_reference"]
    candidates = np.concatenate([candidates, motm[None]])
    coverage = np.r_[coverage, motm_coverage(context, diagnostics["fallback_columns"])]
    ordered = points[[names.index(name) for name in [*actions, "guarded_direct"]]]
    decisions, features, vectors, order = pool_inputs(
        context,
        candidates,
        actions,
        coverage,
        ordered,
        center,
        scale,
        joint=model_id == "chronos2",
        period=period,
        metadata={**row, "episode_index": index, "model_id": model_id},
    )
    if order != names:
        raise ValueError("the simulated and original eight-candidate pools differ")
    return decisions, np.pad(features, ((0, 0), (0, 0), (0, 64))), vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r16/conditional-inputs-v001",
        "forecast-root": "artifacts/iclr27-r16/conditional-forecasts-v001",
        "protocol": "docs/iclr2027/R16_CONDITIONAL_SUPERVISION_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditional supervision")
    prep = read_json(args.prepared_root / "manifest.json")
    forecast = read_json(args.forecast_root / "manifest.json")
    if (
        forecast["status"] != "completed"
        or len(prep["episodes"]) != 1200
        or forecast["identity"]["prepared_sha256"]
        != file_sha256(args.prepared_root / "manifest.json")
    ):
        raise ValueError("freeze all forecasts before generating supervised targets")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "forecast_sha256": file_sha256(args.forecast_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "supervision_module_sha256": file_sha256(ROOT / "scripts/conditional_supervision.py"),
        "features_module_sha256": file_sha256(ROOT / "scripts/pool_gate_inputs.py"),
        "training_histories": 192,
        "validation_histories": 48,
        "process_parameters_in_features": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial conditional supervision definitions changed")
    _write_json(output / "identity.json", identity)
    models, files = supervision_scenarios(), []
    for entry in forecast["models"]:
        model_id = entry["model_id"]
        path = args.forecast_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen forecast manifest changed")
        bank = {item["episode_id"]: item for item in read_json(path)["episodes"]}
        frames, xs, ps = [], [], []
        labels = {name: [] for name in ("future", "conditional_mean", "conditional_variance")}
        for index, row in enumerate(prep["episodes"]):
            path = args.prepared_root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a simulated observed input changed")
            point_record = bank[row["episode_id"]]
            point_path = args.forecast_root / model_id / point_record["path"]
            if file_sha256(point_path) != point_record["sha256"]:
                raise ValueError("a frozen forecast changed")
            prefix = prep["prefixes"][row["generator"]]
            center, scale = np.asarray(prefix["mean"]), np.asarray(prefix["scale"])
            model = models[row["generator"]]
            with (
                np.load(path, allow_pickle=False) as saved,
                np.load(point_path, allow_pickle=False) as prediction,
            ):
                action_names = prediction["actions"].tolist()
                decisions, x, points = visible_inputs(
                    row,
                    index,
                    saved["context"],
                    saved["candidate_values"],
                    saved["candidate_ids"].tolist(),
                    saved["native_coverage"],
                    saved["motm_values"],
                    json.loads(str(saved["motm_diagnostics"])),
                    prediction["point_z"],
                    action_names,
                    center,
                    scale,
                    model_id,
                    model["period"],
                )
                raw_future = factual_future(
                    model, saved["clean_context"], row["phase"], row["future_seed"]
                )
                mean, variance = conditional_labels(
                    model,
                    saved["posterior_mean"],
                    saved["posterior_covariance"],
                    row["phase"],
                    center,
                    scale,
                )
            future = (raw_future[:, :2] - center[:2]) / scale[:2]
            for name, value in (
                ("future", future),
                ("conditional_mean", mean),
                ("conditional_variance", variance),
            ):
                labels[name].append(
                    np.stack(
                        [
                            value.reshape(-1) if slot == -1 else value[:, slot]
                            for slot in decisions.target_slot
                        ]
                    )
                )
            frames.append(decisions)
            xs.append(x)
            ps.append(points)
            if (index + 1) % 200 == 0:
                print(f"{model_id}: {index + 1}/1200 supervised source inputs", flush=True)
        frame = pd.concat(frames, ignore_index=True)
        if frame.episode_id.duplicated().any() or len(frame) != (
            1200 if model_id == "chronos2" else 2400
        ):
            raise ValueError("source decisions must retain their original target units")
        for split, count in (("train", 192), ("validation", 48)):
            if frame[frame.split == split].origin_id.nunique() != count:
                raise ValueError("source train/validation history counts changed")
        root = output / model_id
        root.mkdir(exist_ok=True)
        frame.to_parquet(root / "decisions.parquet", index=False)
        _save_npz(
            root / "inputs.npz",
            features=np.concatenate(xs),
            vectors=np.concatenate(ps),
            actions=np.asarray(action_names),
        )
        _save_npz(
            root / "labels.npz", **{name: np.concatenate(values) for name, values in labels.items()}
        )
        files.extend(
            {"path": str((root / name).relative_to(output)), "sha256": file_sha256(root / name)}
            for name in ("decisions.parquet", "inputs.npz", "labels.npz")
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "files": files,
            "observed_inputs": 1200,
            "training_histories": 192,
            "validation_histories": 48,
            "forecasts_frozen_before_labels": True,
            "new_forecaster_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
