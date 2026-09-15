"""Compare broadcast and target-local inputs with identical Chronos training rows."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aligned_portfolio_io import decision_truth
from latent_source_inputs import ROOT, read_json
from metric_source_gate import fit_fixed_metric
from target_local_inputs import load_target_local
from train_calibrated_source_gates import probability_from_state
from train_latent_source_gates import aggregate
from train_metric_source_gates import fit_metric_gate

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name, path in {
        "prepared-root": "artifacts/iclr27-r11/target-local-inputs-v001",
        "metric-root": "artifacts/iclr27-r10/metric-source-v002",
        "metric-audit": "artifacts/iclr27-r10/metric-source-audit-v002",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "protocol": "docs/iclr2027/R11_TARGET_LOCAL_PROTOCOL.md",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def checked_reference(args, prepared):
    if prepared["module_sha256"] != file_sha256(
        ROOT / "scripts/target_local_inputs.py"
    ) or prepared["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("prepared target-feature definitions changed")
    metric = read_json(args.metric_root / "manifest.json")
    audit = read_json(args.metric_audit / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.metric_root / "manifest.json")
        or metric["identity"]["source_sha256"] != prepared["base_sha256"]
    ):
        raise ValueError("the metric study and prepared source do not match")
    reference_path = ROOT / "artifacts/iclr27-r7/latent-source-gates-v001/manifest.json"
    reference = read_json(reference_path)
    accuracy = read_json(args.accuracy_root / "manifest.json")
    if (
        file_sha256(reference_path) != metric["identity"]["reference_sha256"]
        or file_sha256(args.accuracy_root / "manifest.json")
        != reference["identity"]["accuracy_sha256"]
        or file_sha256(args.accuracy_root / "truth_z.npy")
        != accuracy["prediction_arrays"]["truth_z.npy"]
    ):
        raise ValueError("source outcome definitions changed")
    return metric, reference


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed target-local study")
    prep, frame, arrays = load_target_local(args.prepared_root)
    metric, reference = checked_reference(args, prep)
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/target_local_inputs.py"),
        "fit_module_sha256": file_sha256(ROOT / "scripts/train_metric_source_gates.py"),
        "loss_module_sha256": file_sha256(ROOT / "scripts/metric_source_gate.py"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "metric_sha256": file_sha256(args.metric_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "primary": "target",
        "loss": "joint_future",
        "timesfm_model": "reuse audited R10 joint_future",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial target-local definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    points = arrays["vectors"]
    _, _, _, gram = forecast_geometry(points)
    training = np.flatnonzero(frame.split.to_numpy() == "train")
    truth = np.full((len(frame), 96), np.nan)
    truth[training] = decision_truth(
        frame.iloc[training], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    )
    alignment = np.full(points.shape[:2], np.nan)
    alignment[training] = projection_targets(points[training], truth[training])["raw_projection"]
    checkpoints, folds = [], []
    for family in [None, *sorted(frame.family_id.unique())]:
        indices = (
            training
            if family is None
            else np.flatnonzero(
                (frame.split.to_numpy() == "train") & (frame.family_id.to_numpy() != family)
            )
        )
        validation = (
            np.array([], np.int64)
            if family is None
            else np.flatnonzero(
                (frame.split.to_numpy() == "validation") & (frame.family_id.to_numpy() == family)
            )
        )
        if family in set(frame.iloc[indices].family_id) or set(frame.iloc[indices].origin_id) & set(
            frame.iloc[validation].origin_id
        ):
            raise ValueError("a validation family or history entered fitting")
        directory = output / (family or "full_source")
        directory.mkdir(parents=True, exist_ok=True)
        predictions = {}
        if len(validation):
            old = next(
                row
                for row in metric["folds"]
                if row["model_id"] == "chronos2" and row["held_family"] == family
            )
            path = args.metric_root / old["prediction_path"]
            if file_sha256(path) != old["prediction_sha256"]:
                raise ValueError("an original metric prediction changed")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    frame.iloc[validation].base_position, np.repeat(saved["validation_indices"], 2)
                )
                for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                    predictions[method] = (
                        point.reshape(len(point), 96, 2).transpose(0, 2, 1).reshape(-1, 96)
                    )
        for mode in ("broadcast", "target"):
            features = np.ascontiguousarray(
                np.pad(arrays[mode + "_features"], ((0, 0), (0, 0), (0, 64)))
            )
            seeds = []
            for seed in (5101, 5102, 5103):
                metadata = {
                    "identity_sha256": identity_sha,
                    "held_family": family,
                    "mode": mode,
                    "seed": seed,
                }
                path = directory / f"{mode}_{seed}.pt"
                if not path.exists():
                    saved = {
                        **fit_metric_gate(
                            frame, features, points, truth, gram, alignment, indices, seed, "joint"
                        ),
                        "metadata": metadata,
                    }
                    temporary = path.with_suffix(".tmp")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    saved["metadata"] != metadata
                    or saved["initial_parameter_sha256"]
                    != reference["initial_parameters"][str(seed)]
                ):
                    raise ValueError("a target-local checkpoint or initialization changed")
                checkpoints.append(
                    {"path": str(path.relative_to(output)), "sha256": file_sha256(path), **metadata}
                )
                if len(validation):
                    probability = probability_from_state(saved["state_dict"], features[validation])
                    if mode == "broadcast":
                        np.testing.assert_array_equal(probability[::2], probability[1::2])
                    seeds.append(probability)
                    predictions[f"scope_{mode}_seed{seed}"] = compose_forecasts(
                        points[validation], probability
                    )
            if len(validation):
                predictions[f"scope_{mode}"] = compose_forecasts(
                    points[validation], np.mean(seeds, axis=0)
                )
        controls = {}
        fixed = np.zeros((len(validation), 96))
        single = fixed.copy()
        for slot in (0, 1):
            selected = indices[frame.iloc[indices].target_slot.to_numpy() == slot]
            path = directory / f"fixed_slot{slot}.json"
            if not path.exists():
                control = fit_fixed_metric(
                    points[selected],
                    truth[selected],
                    _family_weights(frame.iloc[selected]),
                    "joint",
                )
                _write_json(
                    path,
                    {
                        **control,
                        "identity_sha256": identity_sha,
                        "train_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                    },
                )
            control = read_json(path)
            if (
                control["identity_sha256"] != identity_sha
                or control["train_indices_sha256"] != hashlib.sha256(selected.tobytes()).hexdigest()
            ):
                raise ValueError("a target-fixed control changed")
            controls[str(slot)] = {
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
            if len(validation):
                mask = frame.iloc[validation].target_slot.to_numpy() == slot
                fixed[mask] = compose_forecasts(
                    points[validation[mask]],
                    np.broadcast_to(control["weights"], (int(mask.sum()), 7)),
                )
                single[mask] = points[validation[mask], control["single_index"]]
        fold = {"held_family": family, "train_indices": indices.tolist(), "controls": controls}
        if len(validation):
            predictions["scope_fixed_by_target"] = fixed
            predictions["scope_single_by_target"] = single
            path = directory / "predictions.npz"
            _save_npz(
                path,
                validation_indices=validation,
                methods=np.asarray(list(predictions)),
                point=np.stack(list(predictions.values())),
            )
            fold.update(
                prediction_path=str(path.relative_to(output)), prediction_sha256=file_sha256(path)
            )
        folds.append(fold)
        print(f"{family or 'full_source'}: matched target-local source fits saved", flush=True)
    if len(checkpoints) != 96 or len(folds) != 16:
        raise ValueError("the matched target-local source study is incomplete")
    _write_json(
        output / "prediction_freeze.json",
        {
            "checkpoints": checkpoints,
            "folds": folds,
            "identity_sha256": identity_sha,
            "validation_future_arrays_read": False,
        },
    )
    rows = []
    for fold in (row for row in folds if row["held_family"] is not None):
        with np.load(output / fold["prediction_path"], allow_pickle=False) as saved:
            evaluation = frame.iloc[saved["validation_indices"]]
            future = decision_truth(
                evaluation, np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
            )
            for method, point in zip(saved["methods"].tolist(), saved["point"], strict=True):
                rows.append(
                    evaluation.assign(
                        method=method,
                        mae=abs(point - future).mean(1),
                        mse=((point - future) ** 2).mean(1),
                    )
                )
    scores = pd.concat(rows, ignore_index=True)
    if len(scores) != 104832 or scores.method.nunique() != 56:
        raise ValueError("target-local source evaluation coverage changed")
    episodes, families, summary = aggregate(scores)
    old = pd.read_csv(args.metric_audit / "summary.csv", float_precision="round_trip")
    old = old[old.model_id == "chronos2"]
    expected = (
        summary[summary.method.isin(old.method)]
        .set_index(["model_id", "method"])
        .loc[old.set_index(["model_id", "method"]).index]
    )
    baseline_delta = float(
        abs(expected[["mae", "mse"]].to_numpy() - old[["mae", "mse"]].to_numpy()).max()
    )
    np.testing.assert_allclose(
        expected[["mae", "mse"]], old[["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    scores.to_parquet(output / "decision_scores.parquet", index=False)
    episodes.to_parquet(output / "episode_scores.parquet", index=False)
    families.to_csv(output / "family_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "checkpoints": checkpoints,
            "folds": folds,
            "score_rows": len(scores),
            "maximum_baseline_aggregation_difference": baseline_delta,
            "new_forecaster_calls": 0,
            "limits": "Chronos target-feature development; TimesFM unchanged; independent audit required",
        },
    )


if __name__ == "__main__":
    main()
