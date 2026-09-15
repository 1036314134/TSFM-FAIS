"""Evaluate a matched eight-candidate portfolio using audited target caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs
from evaluate_r6_policies import restore
from latent_source_inputs import ROOT, read_json
from native_source_transfer_io import input_arguments
from pool_gate_inputs import motm_coverage, pool_inputs
from pool_gate_model import pool_probability
from score_fixed_transfer import score_transfer_banks

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def motm_forecasts(args, model_id, horizon, cohort, prep):
    if cohort == "r6":
        root = args.r6_policy / model_id
        model = read_json(root / "manifest.json")
        entry = next(row for row in model["horizons"] if row["horizon"] == horizon)
        directory = root / entry["directory"]
        marker = read_json(directory / "predictions_frozen.json")
        path = directory / "policy_predictions.npz"
        expected = marker["prediction_sha256"]
    else:
        frozen = read_json(args.legacy_results / "prediction_freeze.json")
        entry = next(row for row in frozen["banks"] if row["model_id"] == model_id)
        path = args.legacy_results / entry["path"]
        expected = entry["sha256"]
    if file_sha256(path) != expected:
        raise ValueError("the audited target MoTM prediction bank changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["episode_ids"].tolist() != [row["episode_id"] for row in prep["episodes"]]:
            raise ValueError("the target MoTM episode order changed")
        methods = saved["methods"].tolist()
        points = saved["point_z"]
        return points[:, methods.index("motm_reference")].copy(), points[
            :, methods.index("forecast_median_with_motm")
        ].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.set_defaults(comparison_root=ROOT / "artifacts/iclr27-r12/motm-pool-transfer-v001")
    for name, path in {
        "study-root": "artifacts/iclr27-r16/conditional-training-v001",
        "reference-study": "artifacts/iclr27-r12/motm-pool-source-v001",
        "study-audit": "artifacts/iclr27-r16/conditional-training-audit-v001",
        "source-motm": "artifacts/iclr27-r12/source-motm-v001",
        "legacy-motm": "artifacts/iclr27-r5/native-confirmation-v001/motm",
        "protocol": "docs/iclr2027/R16_TRANSFER_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed eight-candidate transfer")
    study = read_json(args.study_root / "manifest.json")
    audit = read_json(args.study_audit / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["verified_models"] != 18
        or audit["study_sha256"] != file_sha256(args.study_root / "manifest.json")
    ):
        raise ValueError("audit the complete pool source study before transfer")
    reference = read_json(args.reference_study / "manifest.json")
    if (
        study["identity"]["reference_sha256"] != file_sha256(args.reference_study / "manifest.json")
        or study["identity"]["original_model_sha256"]
        != file_sha256(ROOT / "scripts/pool_gate_model.py")
        or reference["identity"]["input_module_sha256"]
        != file_sha256(ROOT / "scripts/pool_gate_inputs.py")
    ):
        raise ValueError("the fixed source or pool definitions changed")
    source_motm = read_json(args.source_motm / "manifest.json")
    legacy_motm = read_json(args.legacy_motm / "manifest.json")
    cohorts = {
        "r6": (args.r6_prepared, read_json(args.r6_prepared / "manifest.json"), (96, 192)),
        "legacy_native": (
            args.legacy_input / "prepared",
            read_json(args.legacy_input / "prepared/manifest.json"),
            (96,),
        ),
    }
    for key, target_key, legacy_key in (
        ("reference_sha256", "motm_reference_sha256", "reference_manifest_sha256"),
        ("runtime_sha256", "motm_runtime_sha256", "runtime_manifest_sha256"),
    ):
        if (
            source_motm["identity"][key] != cohorts["r6"][1]["identity"][target_key]
            or source_motm["identity"][key] != legacy_motm["identity"][legacy_key]
        ):
            raise ValueError("source and target MoTM references differ")
    legacy_motm_rows = {row["episode_id"]: row for row in legacy_motm["episodes"]}
    scalers = {}
    for name, (root, prep, _) in cohorts.items():
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("target prefix standardizers changed")
        scalers[name] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "study_audit_sha256": file_sha256(args.study_audit / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "scorer_sha256": file_sha256(ROOT / "scripts/score_fixed_transfer.py"),
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "primary": "risk16_conditional_expected",
        "source_and_target_motm_reference_match": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    base_actions = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    banks, checked, maximum_delta = [], 0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        joint = model_id == "chronos2"
        fold = next(row for row in study["folds"] if row["model_id"] == model_id)
        actions = fold["actions"]
        groups = {}
        for condition in (
            "conditional_expected",
            "conditional_factual",
            "source_steps_matched",
            "pool8_joint_future",
        ):
            original = condition == "pool8_joint_future"
            source = reference if original else study
            directory = args.reference_study if original else args.study_root
            entries = sorted(
                [
                    row
                    for row in source["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_family"] is None
                    and (original or row["condition"] == condition)
                ],
                key=lambda row: row["seed"],
            )
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("a matched transfer group lost a source seed")
            group = []
            for entry in entries:
                path = directory / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a fixed source model changed")
                group.append(torch.load(path, map_location="cpu", weights_only=True))
            groups[condition if original else "risk16_" + condition] = group
        source_fold = next(
            row
            for row in reference["folds"]
            if row["model_id"] == model_id and row["held_family"] is None
        )
        controls = {}
        for entry in [
            {
                "condition": "pool8_joint_future",
                "path": source_fold["control_path"],
                "sha256": source_fold["control_sha256"],
            },
            *fold["controls"],
        ]:
            original = entry["condition"] == "pool8_joint_future"
            path = (args.reference_study if original else args.study_root) / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a fixed source control changed")
            controls[entry["condition"] if original else "risk16_" + entry["condition"]] = (
                read_json(path)
            )
        models = [model for group in groups.values() for model in group]
        for cohort, (root, prep, horizons) in cohorts.items():
            for model in models:
                if set(model["training_origins"]) & {
                    row["origin_id"] for row in prep["episodes"]
                } or set(model["training_families"]) & {
                    row["family_id"] for row in prep["episodes"]
                }:
                    raise ValueError("a target family or history entered source training")
            for horizon in horizons:
                _, _, _, base_bank, median7 = (
                    r6_inputs(args, model_id, horizon, base_actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, base_actions, prep, scalers[cohort])
                )
                extra_points, median8 = motm_forecasts(args, model_id, horizon, cohort, prep)
                frames, features, vectors = [], [], []
                for index, row in enumerate(prep["episodes"]):
                    path = root / row["path"]
                    if file_sha256(path) != row["sha256"]:
                        raise ValueError("a target context changed")
                    with np.load(path, allow_pickle=False) as raw:
                        context, candidates, names, coverage = (
                            raw["context"],
                            raw["candidate_values"],
                            raw["candidate_ids"].tolist(),
                            raw["native_coverage"],
                        )
                        if cohort == "r6":
                            extra = raw["motm_values"]
                            diagnostics = json.loads(str(raw["motm_diagnostics"]))
                    if cohort != "r6":
                        record = legacy_motm_rows[row["episode_id"]]
                        path = args.legacy_motm / record["path"]
                        if file_sha256(path) != record["sha256"]:
                            raise ValueError("a legacy MoTM completion changed")
                        with np.load(path, allow_pickle=False) as saved:
                            extra = saved["values"]
                            diagnostics = json.loads(str(saved["diagnostics"]))
                    np.testing.assert_array_equal(
                        extra[np.isfinite(context)], context[np.isfinite(context)]
                    )
                    if not np.isfinite(extra).all():
                        raise ValueError("an audited target MoTM completion is nonfinite")
                    all_names = [*names, "motm_reference"]
                    point = np.concatenate(
                        [
                            base_bank[index, [base_actions.index(name) for name in names]],
                            extra_points[index : index + 1],
                            base_bank[index, base_actions.index("guarded_direct")][None],
                        ]
                    )
                    scaler = scalers[cohort][(row["dataset_id"], row["item_id"])]
                    decisions, x, p, order = pool_inputs(
                        context,
                        np.concatenate([candidates, extra[None]]),
                        all_names,
                        np.r_[coverage, motm_coverage(context, diagnostics["fallback_columns"])],
                        point,
                        np.asarray(scaler["mean"]),
                        np.asarray(scaler["scale"]),
                        joint=joint,
                        period=row["period"],
                        metadata={
                            **row,
                            "episode_index": index,
                            "model_id": model_id,
                            "split": "known_pool_transfer",
                        },
                    )
                    if order != actions:
                        raise ValueError("source and target pool order differs")
                    frames.append(decisions)
                    features.append(np.pad(x, ((0, 0), (0, 0), (0, 64))))
                    vectors.append(p)
                frame = pd.concat(frames, ignore_index=True)
                x = np.ascontiguousarray(np.concatenate(features))
                p = np.concatenate(vectors)
                predictions = {}
                for method_group, group in groups.items():
                    probabilities = [pool_probability(model["state_dict"], x) for model in group]
                    for method, probability in [
                        (method_group, np.mean(probabilities, axis=0)),
                        *[
                            (f"{method_group}_seed{seed}", probability)
                            for seed, probability in zip(
                                (5101, 5102, 5103), probabilities, strict=True
                            )
                        ],
                    ]:
                        point = compose_forecasts(p, probability)
                        probability = probability / probability.sum(1, keepdims=True)
                        direct = (p * probability[:, :, None]).sum(1)
                        maximum_delta = max(maximum_delta, float(abs(point - direct).max()))
                        np.testing.assert_allclose(point, direct, rtol=1e-12, atol=1e-12)
                        predictions[method] = restore(
                            point, frame, len(prep["episodes"]), horizon, joint
                        )
                        checked += len(frame)
                for method_group, control in controls.items():
                    original = method_group == "pool8_joint_future"
                    fixed_name = "pool8_fixed_joint_future" if original else method_group + "_fixed"
                    single_name = (
                        "pool8_single_joint_future" if original else method_group + "_single"
                    )
                    fixed = compose_forecasts(
                        p, np.broadcast_to(control["weights"], (len(frame), 8))
                    )
                    predictions[fixed_name] = restore(
                        fixed, frame, len(prep["episodes"]), horizon, joint
                    )
                    predictions[single_name] = restore(
                        p[:, control["single_index"]], frame, len(prep["episodes"]), horizon, joint
                    )
                predictions["pool8_mean"] = restore(
                    p.mean(1), frame, len(prep["episodes"]), horizon, joint
                )
                predictions["pool8_median"] = restore(
                    np.median(p, axis=1), frame, len(prep["episodes"]), horizon, joint
                )
                predictions["pool8_motm_reference"] = restore(
                    p[:, actions.index("motm_reference")],
                    frame,
                    len(prep["episodes"]),
                    horizon,
                    joint,
                )
                predictions["forecast_median_guarded"] = median7
                np.testing.assert_array_equal(predictions["pool8_median"], median8)
                np.testing.assert_array_equal(predictions["pool8_motm_reference"], extra_points)
                previous = read_json(args.comparison_root / "manifest.json")
                entry = next(
                    row
                    for row in previous["prediction_banks"]
                    if row["cohort"] == cohort
                    and row["model_id"] == model_id
                    and row["horizon"] == horizon
                )
                path = args.comparison_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an audited R12 transfer bank changed")
                with np.load(path, allow_pickle=False) as old:
                    for method, point in zip(
                        old["methods"].tolist(), old["point_z"].transpose(1, 0, 2, 3), strict=True
                    ):
                        np.testing.assert_array_equal(predictions[method], point)
                if len(predictions) != 26:
                    raise ValueError("the 26-method transfer panel changed")
                path = output / cohort / model_id / f"h{horizon}_predictions.npz"
                _save_npz(
                    path,
                    point_z=np.stack(list(predictions.values()), axis=1),
                    methods=np.asarray(list(predictions)),
                    episode_ids=np.asarray([row["episode_id"] for row in prep["episodes"]]),
                )
                banks.append(
                    {
                        "cohort": cohort,
                        "model_id": model_id,
                        "horizon": horizon,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "method_count": len(predictions),
                    }
                )
        print(f"{model_id}: matched-pool target predictions saved", flush=True)
    _write_json(
        output / "prediction_freeze.json",
        {"identity": identity, "banks": banks, "target_future_arrays_read": False},
    )
    aliases = [
        (model_id, name, original)
        for model_id in ("chronos2", "timesfm2p5")
        for name, original in (
            ("pool8_median", "forecast_median_with_motm"),
            ("pool8_motm_reference", "motm_reference"),
        )
    ]
    score_transfer_banks(
        output,
        banks,
        cohorts,
        scalers,
        args.comparison_root,
        identity,
        checked,
        maximum_delta,
        prefix="risk16_",
        aliases=aliases,
    )


if __name__ == "__main__":
    main()
