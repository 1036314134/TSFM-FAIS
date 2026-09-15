"""Build target-local Chronos features and prove unchanged TimesFM feature semantics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from latent_source_inputs import ROOT, load_source_inputs, read_json
from target_local_inputs import split_joint_vectors, target_features

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "accuracy-root": "artifacts/iclr27-r4/accuracy-development-v002",
        "protocol": "docs/iclr2027/R11_TARGET_LOCAL_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed target-local preparation")
    parent = read_json(args.base_root / "manifest.json")
    source = read_json(args.source_root / "episodes_manifest.json")
    if parent["identity"]["source_manifest_sha256"] != file_sha256(
        args.source_root / "episodes_manifest.json"
    ) or parent["identity"]["standardizers_sha256"] != file_sha256(
        args.accuracy_root / "standardizers.json"
    ):
        raise ValueError("source inputs or prefix standardizers changed")
    _, c_frame, c_arrays = load_source_inputs(args.base_root, "chronos2")
    _, t_frame, t_arrays = load_source_inputs(args.base_root, "timesfm2p5")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.accuracy_root / "standardizers.json")
    }
    decisions, features = [], []
    checked_timesfm = 0
    for index, base in enumerate(c_frame.itertuples(index=False)):
        record = source["episodes"][base.episode_index]
        if record["episode_id"] != base.source_episode_id or record["mask_seed"] != 6101:
            raise ValueError("source episode correspondence changed")
        path = args.source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a source context changed")
        with np.load(path, allow_pickle=False) as saved:
            context, candidates, actions, coverage = (
                saved["context"],
                saved["candidate_values"],
                saved["candidate_ids"].tolist(),
                saved["native_coverage"],
            )
        order = sorted([*actions, "guarded_direct"])
        reorder = [order.index(name) for name in [*actions, "guarded_direct"]]
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        points = c_arrays["vectors"][index].reshape(7, 96, 2)[reorder]
        current, local = target_features(
            context,
            candidates,
            actions,
            coverage,
            points,
            mean,
            scale,
            backbone_joint=True,
            period=record["period"],
            metadata={**record, "model_id": "chronos2", "episode_index": base.episode_index},
        )
        current["base_position"] = index
        decisions.append(current)
        features.append(local)
        selected = t_frame.iloc[2 * index : 2 * index + 2]
        if selected.source_episode_id.tolist() != [
            record["episode_id"]
        ] * 2 or selected.target_slot.tolist() != [0, 1]:
            raise ValueError("the original TimesFM target order changed")
        t_points = np.stack(
            [t_arrays["vectors"][2 * index], t_arrays["vectors"][2 * index + 1]], axis=-1
        )[reorder]
        _, t_features = target_features(
            context,
            candidates,
            actions,
            coverage,
            t_points,
            mean,
            scale,
            backbone_joint=False,
            period=record["period"],
            metadata={**record, "model_id": "timesfm2p5", "episode_index": base.episode_index},
        )
        np.testing.assert_array_equal(
            t_features, t_arrays["features"][2 * index : 2 * index + 2, :, :33]
        )
        checked_timesfm += 2
        if (index + 1) % 500 == 0:
            print(f"{index + 1}/3906 source contexts rebuilt", flush=True)
    frame = pd.concat(decisions, ignore_index=True)
    if (
        len(frame) != 7812
        or frame[frame.split == "train"].origin_id.nunique() != 165
        or frame[frame.split == "validation"].origin_id.nunique() != 52
    ):
        raise ValueError("the target-local source population changed")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "decisions.parquet", index=False)
    _save_npz(
        output / "arrays.npz",
        broadcast_features=np.repeat(c_arrays["features"][:, :, :33], 2, axis=0),
        target_features=np.concatenate(features),
        vectors=split_joint_vectors(c_arrays["vectors"]),
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "module_sha256": file_sha256(ROOT / "scripts/target_local_inputs.py"),
            "protocol_sha256": file_sha256(args.protocol),
            "base_sha256": file_sha256(args.base_root / "manifest.json"),
            "source_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
            "decisions_sha256": file_sha256(output / "decisions.parquet"),
            "arrays_sha256": file_sha256(output / "arrays.npz"),
            "timesfm_decisions_exactly_replayed": checked_timesfm,
            "new_forecaster_calls": 0,
            "future_arrays_read": False,
        },
    )


if __name__ == "__main__":
    main()
