"""Build dependency-aware student features from cached histories and imputations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.preforecast import METADATA, STATIC_FEATURES  # noqa: E402
from tsfm_fais.routing.preforecast_replay import assemble_selected_context  # noqa: E402
from tsfm_fais.routing.structured_preforecast import (  # noqa: E402
    ALL_FEATURES,
    EXTRA_FEATURES,
    input_change_features,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed structured features")
    output.mkdir(parents=True, exist_ok=True)
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(accuracy["source_root"])
    source_path = source_root / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("the input episode source changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "source_manifest_sha256": file_sha256(source_path),
        "base_feature_sha256": file_sha256(root / "candidate_accuracy.parquet"),
        "standardizers_sha256": file_sha256(root / "standardizers.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/structured_preforecast.py",
                "src/tsfm_fais/routing/preforecast_replay.py",
            )
        },
        "bins": 6,
        "minimum_correlation_pairs": 8,
        "features": list(ALL_FEATURES),
        "dependency_scope": {
            "chronos2": "all inputs, conditioned on forecast targets",
            "timesfm2p5": "each forecast target independently",
        },
        "information": "original context, cached completed candidates, input mask and prefix statistics only; no forecasts or outcomes read",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("structured feature identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name in identity["source_sha256"]:
        (output / (Path(name).stem + "_snapshot.py")).write_bytes((ROOT / name).read_bytes())
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((root / "standardizers.json").read_text(encoding="utf-8"))
    }
    targets = source["identity"]["config"]["target_indices"]
    slots = [
        ("chronos2", -1, targets, True),
        *[("timesfm2p5", slot, [target], False) for slot, target in enumerate(targets)],
    ]
    extras, files = [], []
    for index, record in enumerate(source["episodes"]):
        source_file = source_root / record["path"]
        if file_sha256(source_file) != record["sha256"]:
            raise ValueError("an imputation input changed")
        cache = (
            output
            / "cases"
            / (hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24] + ".npz")
        )
        actions = source["identity"]["config"]["candidate_ids"]
        all_actions = [*actions, "guarded_direct"]
        if not cache.exists():
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            with np.load(source_file, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                if saved["candidate_ids"].tolist() != actions:
                    raise ValueError("candidate ordering changed")
            reference = (candidates[actions.index("locf")] - mean) / scale
            context_z = (context - mean) / scale
            values = np.empty((len(slots), len(all_actions), len(EXTRA_FEATURES)))
            for slot_index, (_, _, selected_targets, joint) in enumerate(slots):
                for action_index, action in enumerate(all_actions):
                    effective = assemble_selected_context(
                        context,
                        candidates,
                        actions,
                        [action] if joint else [action] * len(selected_targets),
                        selected_targets,
                        joint=joint,
                    )
                    features = input_change_features(
                        context_z,
                        (effective - mean) / scale,
                        reference,
                        selected_targets,
                        joint=joint,
                    )
                    values[slot_index, action_index] = [features[name] for name in EXTRA_FEATURES]
            _save_npz(
                cache,
                features=values,
                identity_sha256=np.asarray(identity_sha),
                source_sha256=np.asarray(record["sha256"]),
            )
        with np.load(cache, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != record["sha256"]
            ):
                raise ValueError("a structured feature cache has different provenance")
            values = saved["features"]
        if (
            values.shape != (len(slots), len(all_actions), len(EXTRA_FEATURES))
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid structured feature array")
        for slot_index, (model, slot, _, _) in enumerate(slots):
            for action_index, action in enumerate(all_actions):
                extras.append(
                    {
                        "model_id": model,
                        "episode_id": record["episode_id"],
                        "target_slot": slot,
                        "candidate_id": action,
                        **dict(zip(EXTRA_FEATURES, values[slot_index, action_index], strict=True)),
                    }
                )
        files.append(
            {
                "episode_id": record["episode_id"],
                "path": str(cache.relative_to(output)),
                "sha256": file_sha256(cache),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(source["episodes"]):
            _write_json(
                output / "progress.json",
                {
                    "status": "preparing",
                    "completed_episodes": index + 1,
                    "total_episodes": len(source["episodes"]),
                },
            )
            print(
                json.dumps(
                    {"completed_episodes": index + 1, "total_episodes": len(source["episodes"])}
                ),
                flush=True,
            )
    base_columns = [name for name in METADATA if name != "source_episode_id"] + list(
        STATIC_FEATURES
    )
    base = pd.read_parquet(root / "candidate_accuracy.parquet", columns=base_columns)
    base = base[
        ((base.model_id == "chronos2") & (base.target_slot == -1))
        | ((base.model_id == "timesfm2p5") & base.target_slot.isin([0, 1]))
    ]
    base = base[base.candidate_id.isin(all_actions)]
    frame = base.merge(
        pd.DataFrame(extras),
        on=["model_id", "episode_id", "target_slot", "candidate_id"],
        validate="one_to_one",
    )
    if len(frame) != len(base) or len(frame) != len(source["episodes"]) * len(all_actions) * len(
        slots
    ):
        raise ValueError("structured features lost source decisions")
    feature_path = output / "features.parquet"
    frame.to_parquet(feature_path, index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "feature_file": feature_path.name,
            "feature_sha256": file_sha256(feature_path),
            "rows": len(frame),
            "episodes": len(files),
            "feature_count": len(ALL_FEATURES),
            "cases": files,
            "forecaster_calls": 0,
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_episodes": len(files), "total_episodes": len(files)},
    )


if __name__ == "__main__":
    main()
