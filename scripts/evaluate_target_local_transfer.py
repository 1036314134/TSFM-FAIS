"""Replay target-local Chronos gates and the unchanged TimesFM procedure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_r6_geometry_gates import legacy_inputs, r6_inputs
from evaluate_r6_policies import restore
from latent_source_inputs import ROOT, read_json
from native_source_transfer_io import input_arguments
from score_fixed_transfer import score_transfer_banks
from shared_context_inference import broadcast_context_weights
from target_local_inputs import target_features
from train_calibrated_source_gates import probability_from_state

from tsfm_fais.routing.forecast_gate import compose_forecasts
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_arguments(parser)
    parser.set_defaults(comparison_root=ROOT / "artifacts/iclr27-r10/metric-transfer-v001")
    for name, path in {
        "study-root": "artifacts/iclr27-r11/target-local-source-v001",
        "study-audit": "artifacts/iclr27-r11/target-local-audit-v001",
        "metric-root": "artifacts/iclr27-r10/metric-source-v002",
        "metric-audit": "artifacts/iclr27-r10/metric-source-audit-v002",
        "protocol": "docs/iclr2027/R11_TRANSFER_PROTOCOL_V002.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed target-local transfer evaluation")
    study, metric = (
        read_json(args.study_root / "manifest.json"),
        read_json(args.metric_root / "manifest.json"),
    )
    for root, audit_root, count in (
        (args.study_root, args.study_audit, 96),
        (args.metric_root, args.metric_audit, 384),
    ):
        audit = read_json(audit_root / "manifest.json")
        if (
            audit["status"] != "completed"
            or audit["study_sha256"] != file_sha256(root / "manifest.json")
            or audit["verified_checkpoints"] != count
        ):
            raise ValueError("complete and audit the source studies before transfer")
    if study["identity"]["module_sha256"] != file_sha256(ROOT / "scripts/target_local_inputs.py"):
        raise ValueError("target-feature semantics changed after training")
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
            raise ValueError("target prefix standardizers changed")
        scalers[name] = {
            (row["dataset_id"], row["item_id"]): row
            for row in read_json(root / "standardizers.json")
        }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "scripts/target_local_inputs.py"),
        "scorer_sha256": file_sha256(ROOT / "scripts/score_fixed_transfer.py"),
        "study_sha256": file_sha256(args.study_root / "manifest.json"),
        "metric_sha256": file_sha256(args.metric_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "comparison_sha256": file_sha256(args.comparison_root / "manifest.json"),
        "primary": "scope_target",
        "timesfm_reuses": "metric_joint_future",
        "broadcast_inference": "compute one context decision and reuse it for both targets",
        "shared_inference_sha256": file_sha256(ROOT / "scripts/shared_context_inference.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    actions = [
        "guarded_direct",
        "knn_multivariate",
        "linear_interp",
        "locf",
        "saits",
        "seasonal_lag",
        "timemixerpp",
    ]
    banks, checked, maximum_delta, verified_timesfm = [], 0, 0.0, 0
    for model_id in ("chronos2", "timesfm2p5"):
        joint = model_id == "chronos2"
        modes = ("broadcast", "target") if joint else ("target",)
        models = {}
        for mode in modes:
            entries = (
                [
                    row
                    for row in study["checkpoints"]
                    if row["held_family"] is None and row["mode"] == mode
                ]
                if joint
                else [
                    row
                    for row in metric["checkpoints"]
                    if row["model_id"] == model_id
                    and row["held_family"] is None
                    and row["condition"] == "joint_future"
                ]
            )
            if [row["seed"] for row in entries] != [5101, 5102, 5103]:
                raise ValueError("a fixed source procedure lost a seed")
            models[mode] = []
            for entry in entries:
                path = (args.study_root if joint else args.metric_root) / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a frozen source model changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if len(set(saved["training_origins"])) != 165:
                    raise ValueError("source training population changed")
                models[mode].append(saved)
        for cohort, (root, prep, horizons) in cohorts.items():
            for group in models.values():
                for saved in group:
                    if set(saved["training_origins"]) & {
                        row["origin_id"] for row in prep["episodes"]
                    } or set(saved["training_families"]) & {
                        row["family_id"] for row in prep["episodes"]
                    }:
                        raise ValueError("a target history or family entered source training")
            for horizon in horizons:
                old_decisions, old_base, old_vectors, candidate_bank, median = (
                    r6_inputs(args, model_id, horizon, actions, prep)
                    if cohort == "r6"
                    else legacy_inputs(args, model_id, actions, prep, scalers[cohort])
                )
                decisions, local = [], []
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
                    if sorted([*names, "guarded_direct"]) != actions:
                        raise ValueError("source and target candidate pools differ")
                    order = [actions.index(name) for name in [*names, "guarded_direct"]]
                    scaler = scalers[cohort][(row["dataset_id"], row["item_id"])]
                    current, features = target_features(
                        context,
                        candidates,
                        names,
                        coverage,
                        candidate_bank[index, order],
                        np.asarray(scaler["mean"]),
                        np.asarray(scaler["scale"]),
                        backbone_joint=joint,
                        period=row["period"],
                        metadata={
                            **row,
                            "episode_index": index,
                            "model_id": model_id,
                            "split": "known_target_transfer",
                        },
                    )
                    decisions.append(current)
                    local.append(features)
                decisions = pd.concat(decisions, ignore_index=True)
                local = np.concatenate(local)
                if joint:
                    np.testing.assert_array_equal(
                        np.sort(old_decisions.episode_index), np.arange(len(prep["episodes"]))
                    )
                    global_features = np.empty_like(old_base)
                    global_features[old_decisions.episode_index.to_numpy(int)] = old_base
                    feature_views = {
                        "broadcast": np.repeat(global_features, 2, axis=0),
                        "target": local,
                    }
                    vectors = candidate_bank.transpose(0, 3, 1, 2).reshape(
                        2 * len(prep["episodes"]), 7, horizon
                    )
                else:
                    index = pd.MultiIndex.from_frame(decisions[["episode_index", "target_slot"]])
                    positions = index.get_indexer(
                        pd.MultiIndex.from_frame(old_decisions[["episode_index", "target_slot"]])
                    )
                    if (positions < 0).any():
                        raise ValueError("TimesFM target correspondence changed")
                    np.testing.assert_array_equal(local[positions], old_base)
                    verified_timesfm += len(old_base)
                    decisions, vectors = old_decisions, old_vectors
                    feature_views = {"target": old_base}
                predictions = {}
                for mode in modes:
                    features = np.ascontiguousarray(
                        np.pad(feature_views[mode], ((0, 0), (0, 0), (0, 64)))
                    )
                    seeds = []
                    for saved in models[mode]:
                        probability = (
                            broadcast_context_weights(saved["state_dict"], features)
                            if mode == "broadcast"
                            else probability_from_state(saved["state_dict"], features)
                        )
                        if mode == "broadcast":
                            np.testing.assert_array_equal(probability[::2], probability[1::2])
                        seeds.append(probability)
                    for name, probability in [
                        (f"scope_{mode}", np.mean(seeds, axis=0)),
                        *[
                            (f"scope_{mode}_seed{seed}", probability)
                            for seed, probability in zip((5101, 5102, 5103), seeds, strict=True)
                        ],
                    ]:
                        point = compose_forecasts(vectors, probability)
                        probability = probability / probability.sum(1, keepdims=True)
                        direct = (vectors * probability[:, :, None]).sum(1)
                        maximum_delta = max(maximum_delta, float(abs(point - direct).max()))
                        np.testing.assert_allclose(point, direct, rtol=1e-12, atol=1e-12)
                        predictions[name] = restore(
                            point, decisions, len(prep["episodes"]), horizon, False
                        )
                        checked += len(decisions)
                if joint:
                    fold = next(row for row in study["folds"] if row["held_family"] is None)
                    fixed = np.zeros((len(decisions), horizon))
                    single = fixed.copy()
                    for slot in (0, 1):
                        entry = fold["controls"][str(slot)]
                        path = args.study_root / entry["path"]
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError("a target-specific fixed control changed")
                        control = read_json(path)
                        selected = decisions.target_slot.to_numpy() == slot
                        fixed[selected] = compose_forecasts(
                            vectors[selected],
                            np.broadcast_to(control["weights"], (int(selected.sum()), 7)),
                        )
                        single[selected] = vectors[selected, control["single_index"]]
                    predictions["scope_fixed_by_target"] = restore(
                        fixed, decisions, len(prep["episodes"]), horizon, False
                    )
                    predictions["scope_single_by_target"] = restore(
                        single, decisions, len(prep["episodes"]), horizon, False
                    )
                predictions["forecast_median_guarded"] = restore(
                    np.median(vectors, axis=1), decisions, len(prep["episodes"]), horizon, False
                )
                np.testing.assert_array_equal(predictions["forecast_median_guarded"], median)
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
        print(f"{model_id}: target-local transfer predictions frozen", flush=True)
    identity["timesfm_target_decisions_exactly_replayed"] = verified_timesfm
    _write_json(
        output / "prediction_freeze.json",
        {"identity": identity, "banks": banks, "target_future_arrays_read": False},
    )
    aliases = [
        ("timesfm2p5", f"scope_target{suffix}", f"metric_joint_future{suffix}")
        for suffix in ("", "_seed5101", "_seed5102", "_seed5103")
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
        aliases=aliases,
    )


if __name__ == "__main__":
    main()
