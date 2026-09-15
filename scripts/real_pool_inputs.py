"""Prepare original source training and disjoint groups of real-missing histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from aligned_portfolio_io import decision_truth
from evaluate_motm_pool_transfer import motm_forecasts
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs
from latent_source_inputs import ROOT, read_json
from masked_pool_gate import observed_geometry
from native_source_transfer_io import (
    holdout_group,
    input_arguments,
    observed_errors,
    summarize_scores,
)
from pool_gate_inputs import load_pool_inputs, motm_coverage, pool_inputs

from tsfm_fais.routing.forecast_projection import forecast_geometry, projection_targets
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    input_arguments(parser)
    parser.set_defaults(comparison_root=ROOT / "artifacts/iclr27-r12/motm-pool-transfer-v001")
    for name, path in {
        "pool-root": "artifacts/iclr27-r12/motm-pool-inputs-v001",
        "pool-study": "artifacts/iclr27-r12/motm-pool-source-v001",
        "pool-audit": "artifacts/iclr27-r12/motm-pool-audit-v001",
        "legacy-motm": "artifacts/iclr27-r5/native-confirmation-v001/motm",
        "protocol": "docs/iclr2027/R13_REAL_POOL_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def build_model_inputs(args, model_id):
    manifest, original, base = load_pool_inputs(args.pool_root, model_id)
    selected = np.flatnonzero(original.split.to_numpy() == "train")
    frame = original.iloc[selected].copy().reset_index(drop=True)
    frame["source_position"] = selected
    frame["cohort"] = "source"
    frame["holdout_group"] = "source"
    truth = decision_truth(
        original.iloc[selected], np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    )
    points = base["vectors"][selected]
    data = {
        "features": base["features"][selected],
        "points": points,
        "truth": truth,
        "observed": np.ones_like(truth, bool),
        "gram": forecast_geometry(base["vectors"])[3][selected],
        "alignment": projection_targets(points, truth)["raw_projection"],
        "coordinates": np.full_like(truth, 1 / truth.shape[1]),
    }
    with np.load(
        args.pool_root / model_id / manifest["episodes"][0]["path"], allow_pickle=False
    ) as saved:
        actions = saved["actions"].tolist()
    old_actions = [name for name in actions if name != "motm_reference"]
    comparison = read_json(args.comparison_root / "manifest.json")
    if comparison["status"] != "completed":
        raise ValueError("finish the audited source-only target comparison first")
    legacy_motm = read_json(args.legacy_motm / "manifest.json")
    motm_rows = {row["episode_id"]: row for row in legacy_motm["episodes"]}
    frames = [frame]
    blocks = {name: [value] for name, value in data.items()}
    reference_parts = {}
    for cohort, root in (
        ("legacy_native", args.legacy_input / "prepared"),
        ("r6", args.r6_prepared),
    ):
        prep = read_json(root / "manifest.json")
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("native prefix standardizers changed")
        scalers = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
        _, _, _, bank, _ = (
            r6_inputs(args, model_id, 96, old_actions, prep)
            if cohort == "r6"
            else legacy_inputs(args, model_id, old_actions, prep, scalers)
        )
        extra_points, _ = motm_forecasts(args, model_id, 96, cohort, prep)
        entry = next(
            row
            for row in comparison["prediction_banks"]
            if row["cohort"] == cohort and row["model_id"] == model_id and row["horizon"] == 96
        )
        reference_path = args.comparison_root / entry["path"]
        if file_sha256(reference_path) != entry["sha256"]:
            raise ValueError("source-only native predictions changed")
        with np.load(reference_path, allow_pickle=False) as saved:
            reference_names = saved["methods"].tolist()
            reference_bank = saved["point_z"]
            if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
                raise ValueError("reference native episode order changed")
        current_frames, features, vectors, targets, masks = [], [], [], [], []
        references = {name: [] for name in reference_names}
        for index, row in enumerate(prep["episodes"]):
            if not row["window"]["context_has_missing"] or row.get("panel") == "new_synthetic":
                continue
            path = root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an original native history changed")
            with np.load(path, allow_pickle=False) as raw:
                context, candidates, names, coverage = (
                    raw["context"],
                    raw["candidate_values"],
                    raw["candidate_ids"].tolist(),
                    raw["native_coverage"],
                )
                future, observed = raw["future"][:96], raw["future_observed"][:96].astype(bool)
                if cohort == "r6":
                    extra = raw["motm_values"]
                    diagnostics = json.loads(str(raw["motm_diagnostics"]))
            if cohort == "legacy_native":
                record = motm_rows[row["episode_id"]]
                path = args.legacy_motm / record["path"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("a native MoTM completion changed")
                with np.load(path, allow_pickle=False) as saved:
                    extra = saved["values"]
                    diagnostics = json.loads(str(saved["diagnostics"]))
            np.testing.assert_array_equal(observed, np.isfinite(future))
            if (observed.sum(0) < 48).any():
                raise ValueError("a native future violates the original observation threshold")
            scaler = scalers[(row["dataset_id"], row["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            raw_points = np.concatenate(
                [
                    bank[index, [old_actions.index(name) for name in names]],
                    extra_points[index : index + 1],
                    bank[index, old_actions.index("guarded_direct")][None],
                ]
            )
            decision, x, p, order = pool_inputs(
                context,
                np.concatenate([candidates, extra[None]]),
                [*names, "motm_reference"],
                np.r_[coverage, motm_coverage(context, diagnostics["fallback_columns"])],
                raw_points,
                mean,
                scale,
                joint=model_id == "chronos2",
                period=row["period"],
                metadata={
                    **row,
                    "episode_index": index,
                    "model_id": model_id,
                    "split": "native_development",
                },
            )
            if order != actions:
                raise ValueError("source and native pool identities differ")
            z = np.where(observed, (future - mean[:2]) / scale[:2], np.nan)

            def reshape(value, slots=decision.target_slot):
                return np.stack(
                    [value.reshape(-1) if slot == -1 else value[:, slot] for slot in slots]
                )

            targets.append(reshape(z))
            masks.append(reshape(observed))
            features.append(np.pad(x, ((0, 0), (0, 0), (0, 64))))
            vectors.append(p)
            for method_position, method in enumerate(reference_names):
                references[method].append(reshape(reference_bank[index, method_position]))
            decision["cohort"] = cohort
            decision["holdout_group"] = holdout_group(row["family_id"])
            decision["source_position"] = -1
            decision["episode_id"] = cohort + "::" + decision.episode_id
            current_frames.append(decision)
        native = pd.concat(current_frames, ignore_index=True)
        values = {
            "features": np.concatenate(features),
            "points": np.concatenate(vectors),
            "truth": np.concatenate(targets),
            "observed": np.concatenate(masks),
        }
        values["gram"], values["alignment"], values["coordinates"] = observed_geometry(
            values["points"], values["truth"], values["observed"], model_id == "chronos2"
        )
        frames.append(native)
        for name, value in values.items():
            blocks[name].append(value)
        for name, parts in references.items():
            reference_parts.setdefault(name, []).append(np.concatenate(parts))
    frame = pd.concat(frames, ignore_index=True)
    data = {name: np.concatenate(parts) for name, parts in blocks.items()}
    native = frame[frame.cohort != "source"]
    if (
        native.origin_id.nunique() != 358
        or native.family_id.nunique() != 9
        or native.holdout_group.nunique() != 8
        or set(native.origin_id) & set(frame[frame.cohort == "source"].origin_id)
        or set(native.family_id) & set(frame[frame.cohort == "source"].family_id)
    ):
        raise ValueError("the registered real-missing population or group separation changed")
    references = {
        name: np.concatenate([np.full((len(selected), points.shape[-1]), np.nan), *parts])
        for name, parts in reference_parts.items()
    }
    return frame, data, references, actions


def reference_check(args, frame, data, references, model_id):
    old = pd.read_csv(args.comparison_root / "comparison_summary.csv", float_precision="round_trip")
    maximum = 0.0
    for cohort, panel, keep in (
        ("legacy_native", "naturally_missing", frame.cohort == "legacy_native"),
        ("r6", "new_native_missing", frame.family_id == "beijing_multisite"),
        ("r6", "time_grid_gap_missing", frame.family_id == "bike_sharing"),
    ):
        indices = np.flatnonzero(keep.to_numpy())
        rows = []
        for method, point in references.items():
            mae, mse = observed_errors(
                point[indices],
                data["truth"][indices],
                data["observed"][indices],
                joint=model_id == "chronos2",
            )
            rows.append(frame.iloc[indices].assign(method=method, mae=mae, mse=mse))
        summary = summarize_scores(pd.concat(rows, ignore_index=True))[2].set_index("method")
        expected = (
            old[
                (old.model_id == model_id)
                & (old.cohort == cohort)
                & (old.horizon == 96)
                & (old.panel == panel)
            ]
            .set_index("method")
            .loc[summary.index]
        )
        difference = float(
            abs(summary[["mae", "mse"]].to_numpy() - expected[["mae", "mse"]].to_numpy()).max()
        )
        maximum = max(maximum, difference)
        np.testing.assert_allclose(
            summary[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-12, atol=1e-12
        )
    return maximum


def load_real_inputs(root, model_id):
    parent = read_json(root / "manifest.json")
    entry = next(row for row in parent["models"] if row["model_id"] == model_id)
    if (
        parent["status"] != "completed"
        or file_sha256(root / entry["frame_path"]) != entry["frame_sha256"]
        or file_sha256(root / entry["data_path"]) != entry["data_sha256"]
    ):
        raise ValueError("prepared real-source inputs changed")
    frame = pd.read_parquet(root / entry["frame_path"])
    with np.load(root / entry["data_path"], allow_pickle=False) as saved:
        data = {
            name: saved[name]
            for name in (
                "features",
                "points",
                "truth",
                "observed",
                "gram",
                "alignment",
                "coordinates",
            )
        }
        references = dict(
            zip(saved["reference_methods"].tolist(), saved["reference_points"], strict=True)
        )
    return entry, frame, data, references


def main():
    args = arguments(__doc__).parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed real-source preparation")
    source_study = read_json(args.pool_study / "manifest.json")
    audit = read_json(args.pool_audit / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["study_sha256"] != file_sha256(args.pool_study / "manifest.json")
        or source_study["identity"]["pool_sha256"] != file_sha256(args.pool_root / "manifest.json")
    ):
        raise ValueError("the original source study and collection do not match")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "loss_module_sha256": file_sha256(ROOT / "scripts/masked_pool_gate.py"),
        "pool_sha256": file_sha256(args.pool_root / "manifest.json"),
        "source_study_sha256": file_sha256(args.pool_study / "manifest.json"),
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "native_outcomes_read_as_sources_for_other_groups": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for model_id in ("chronos2", "timesfm2p5"):
        frame, data, references, actions = build_model_inputs(args, model_id)
        difference = reference_check(args, frame, data, references, model_id)
        directory = output / model_id
        directory.mkdir(exist_ok=True)
        frame_path = directory / "frame.parquet"
        data_path = directory / "data.npz"
        frame.to_parquet(frame_path, index=False)
        _save_npz(
            data_path,
            **data,
            reference_methods=np.asarray(list(references)),
            reference_points=np.stack(list(references.values())),
        )
        entries.append(
            {
                "model_id": model_id,
                "frame_path": str(frame_path.relative_to(output)),
                "frame_sha256": file_sha256(frame_path),
                "data_path": str(data_path.relative_to(output)),
                "data_sha256": file_sha256(data_path),
                "actions": actions,
                "reference_maximum_difference": difference,
            }
        )
        print(f"{model_id}: real-missing source inputs and references verified", flush=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": entries,
            "native_origins": 358,
            "native_families": 9,
            "native_groups": 8,
            "new_forecaster_calls": 0,
            "new_imputer_fits": 0,
        },
    )


if __name__ == "__main__":
    main()
