"""Replay frozen positional objectives on the already audited target populations."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs
from latent_source_inputs import ROOT, read_json
from native_source_transfer_io import input_arguments
from positional_forecast_portfolio import PositionalPortfolio, position_inputs, predict_position
from positional_portfolio_io import restore_positions, target_nodes
from score_fixed_transfer import score_transfer_banks

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.set_defaults(comparison_root=ROOT / "artifacts/iclr27-r16/conditional-transfer-v001")
    for name, path in {
        "study-root": "artifacts/iclr27-r17/position-objectives-v001",
        "study-audit": "artifacts/iclr27-r17/position-objectives-audit-v002",
        "reference-study": "artifacts/iclr27-r6/positional-source-v001",
        "reference-transfer": "artifacts/iclr27-r6/positional-transfer-v001",
        "protocol": "docs/iclr2027/R17_TRANSFER_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed positional objective transfers")
    study, audit = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.study_audit / "manifest.json"),
    )
    reference = read_json(args.reference_study / "manifest.json")
    old_transfer = read_json(args.reference_transfer / "manifest.json")
    if (
        audit["status"] != "completed"
        or audit["verified_models"] != 24
        or audit["study_sha256"] != file_sha256(args.study_root / "manifest.json")
    ):
        raise ValueError("complete the corrected source audit first")
    if study["identity"]["reference_sha256"] != file_sha256(args.reference_study / "manifest.json"):
        raise ValueError("the original teacher reference changed")
    for key, module in (
        ("model_module_sha256", "positional_forecast_portfolio.py"),
        ("io_module_sha256", "positional_portfolio_io.py"),
    ):
        if reference["identity"][key] != file_sha256(ROOT / "scripts" / module):
            raise ValueError("the frozen positional inference changed")
    cohorts = {
        "r6": (args.r6_prepared, read_json(args.r6_prepared / "manifest.json"), (96, 192)),
        "legacy_native": (
            args.legacy_input / "prepared",
            read_json(args.legacy_input / "prepared/manifest.json"),
            (96,),
        ),
    }
    scalers = {}
    for name, (root, prep, _) in cohorts.items():
        if file_sha256(root / "standardizers.json") != prep["standardizers_sha256"]:
            raise ValueError("an original prefix standardizer changed")
        scalers[name] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "study_audit_sha256": file_sha256(args.study_audit / "manifest.json"),
        "reference_transfer_sha256": file_sha256(args.reference_transfer / "manifest.json"),
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "primary": "position17_local_joint_future",
        "scorer_sha256": file_sha256(ROOT / "scripts/score_fixed_transfer.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "identity.json", identity)
    torch.set_num_threads(1)
    banks, checked, maximum_delta = [], 0, 0.0
    for model_id in ("chronos2", "timesfm2p5"):
        fold = next(row for row in study["folds"] if row["model_id"] == model_id)
        actions = fold["actions"]
        groups = {}
        for mode in ("local", "pooled"):
            for objective in ("teacher", "mse", "joint"):
                original = objective == "teacher"
                entries = sorted(
                    [
                        row
                        for row in (
                            reference["source_models"] if original else study["checkpoints"]
                        )
                        if row["model_id"] == model_id
                        and row["mode"] == mode
                        and (original or row["objective"] == objective)
                    ],
                    key=lambda row: row["seed"],
                )
                if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                    raise ValueError("a positional source seed is missing")
                loaded = []
                for entry in entries:
                    path = (args.reference_study if original else args.study_root) / entry["path"]
                    if file_sha256(path) != entry["sha256"]:
                        raise ValueError("a frozen source checkpoint changed")
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    model = PositionalPortfolio(mode).eval().requires_grad_(False)
                    model.load_state_dict(saved["state_dict"])
                    origins = (
                        saved["training_origins"]
                        if original
                        else saved["metadata"]["training_origins"]
                    )
                    loaded.append((model, origins))
                groups[(mode, objective)] = loaded
        controls = {}
        for entry in fold["controls"]:
            path = args.study_root / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("a matched positional fixed control changed")
            controls[entry["objective"]] = read_json(path)
        for cohort, (_, prep, horizons) in cohorts.items():
            for horizon in horizons:
                original, base, vectors, _, median = (
                    r6_inputs(args, model_id, horizon, actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, actions, prep, scalers[cohort])
                )
                decisions, base, points = target_nodes(
                    original, base, vectors, joint=model_id == "chronos2"
                )
                inputs = position_inputs(base, points)
                predictions = {
                    "forecast_median_guarded": restore_positions(
                        inputs["median"], decisions, len(prep["episodes"]), horizon
                    )
                }
                np.testing.assert_array_equal(predictions["forecast_median_guarded"], median)
                for (mode, objective), group in groups.items():
                    outputs = []
                    for seed, (model, origins) in zip((5101, 5102, 5103), group, strict=True):
                        if set(origins) & {row["origin_id"] for row in prep["episodes"]}:
                            raise ValueError("a target history entered source fitting")
                        point = predict_position(model, inputs)
                        if np.any(point < inputs["lower"]) or np.any(point > inputs["upper"]):
                            raise ValueError("a positional prediction left its bounds")
                        outputs.append(point)
                        checked += len(point)
                        name = (
                            f"position17_{mode}_teacher_seed{seed}"
                            if objective == "teacher"
                            else f"position17_{mode}_{objective}_future_seed{seed}"
                        )
                        predictions[name] = restore_positions(
                            point, decisions, len(prep["episodes"]), horizon
                        )
                    anchor = inputs["median"]
                    average = np.clip(
                        anchor + np.mean(np.stack(outputs) - anchor[None], axis=0),
                        inputs["lower"],
                        inputs["upper"],
                    )
                    name = (
                        f"position_{mode}"
                        if objective == "teacher"
                        else f"position17_{mode}_{objective}_future"
                    )
                    predictions[name] = restore_positions(
                        average, decisions, len(prep["episodes"]), horizon
                    )
                for objective, control in controls.items():
                    probability = np.asarray(control["weights"])
                    point = (points * probability[None, :, None]).sum(1)
                    anchored = points[:, 0] + (
                        (points - points[:, :1]) * probability[None, :, None]
                    ).sum(1)
                    maximum_delta = max(maximum_delta, float(abs(point - anchored).max()))
                    np.testing.assert_allclose(point, anchored, rtol=1e-12, atol=1e-12)
                    predictions[f"position17_fixed_{objective}"] = restore_positions(
                        point, decisions, len(prep["episodes"]), horizon
                    )
                    predictions[f"position17_single_{objective}"] = restore_positions(
                        points[:, control["single_index"]],
                        decisions,
                        len(prep["episodes"]),
                        horizon,
                    )
                entry = next(
                    row
                    for row in old_transfer["prediction_banks"]
                    if row["model_id"] == model_id
                    and row["cohort"] == cohort
                    and row["horizon"] == horizon
                )
                path = args.reference_transfer / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an original teacher target bank changed")
                with np.load(path, allow_pickle=False) as saved:
                    for method, point in zip(
                        saved["methods"].tolist(),
                        saved["point_z"].transpose(1, 0, 2, 3),
                        strict=True,
                    ):
                        np.testing.assert_array_equal(predictions[method], point)
                if len(predictions) != 29:
                    raise ValueError("the registered positional transfer panel changed")
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
        print(f"{model_id}: all positional target predictions frozen", flush=True)
    if checked != 246024:
        raise ValueError("the positional seed trajectory coverage changed")
    _write_json(
        output / "prediction_freeze.json",
        {"identity": identity, "banks": banks, "target_future_arrays_read": False},
    )
    score_transfer_banks(
        output,
        banks,
        cohorts,
        scalers,
        args.comparison_root,
        identity,
        checked,
        maximum_delta,
        prefix="position17_",
    )


if __name__ == "__main__":
    main()
