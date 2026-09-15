"""Check structured feature provenance, scope, coverage and selected raw-input replays."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.preforecast import STATIC_FEATURES  # noqa: E402
from tsfm_fais.routing.preforecast_replay import assemble_selected_context  # noqa: E402
from tsfm_fais.routing.structured_preforecast import (  # noqa: E402
    ALL_FEATURES,
    COVARIATE_FEATURES,
    EXTRA_FEATURES,
    input_change_features,
)
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "input-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed feature audits")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    feature_path = root / manifest["feature_file"]
    if manifest["status"] != "completed" or file_sha256(feature_path) != manifest["feature_sha256"]:
        raise ValueError("structured features are incomplete or changed")
    frame = pd.read_parquet(feature_path)
    if (
        set(name for name in frame if name.startswith("static.")) != set(ALL_FEATURES)
        or not np.isfinite(frame[list(ALL_FEATURES)]).all().all()
    ):
        raise ValueError("the structured feature schema or values are invalid")
    if any(
        name.startswith("response.") or name in {"mae", "mse", "loss", "teacher_mse"}
        for name in frame
    ):
        raise ValueError("forecasts or outcome costs entered the input feature table")
    keys = ["model_id", "episode_id", "candidate_id", "target_slot"]
    if frame.duplicated(keys).any():
        raise ValueError("structured feature rows are not unique")
    base = pd.read_parquet(
        args.accuracy_root / "candidate_accuracy.parquet", columns=[*keys, *STATIC_FEATURES]
    )
    paired = frame[keys + list(STATIC_FEATURES)].merge(
        base, on=keys, suffixes=("", "_original"), validate="one_to_one"
    )
    np.testing.assert_array_equal(
        paired[list(STATIC_FEATURES)].to_numpy(),
        paired[[name + "_original" for name in STATIC_FEATURES]].to_numpy(),
    )
    np.testing.assert_array_equal(
        frame[frame.model_id == "timesfm2p5"][list(COVARIATE_FEATURES)].to_numpy(), 0.0
    )
    fractions = [
        name for name in EXTRA_FEATURES if name.endswith(("missing", "valid_fraction", "enabled"))
    ]
    if ((frame[fractions] < 0) | (frame[fractions] > 1)).any().any():
        raise ValueError("feature proportions are outside [0,1]")
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(accuracy["source_root"])
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    if len(frame) != len(source["episodes"]) * 21:
        raise ValueError("the feature table does not cover every registered source decision")
    for record in manifest["cases"]:
        if file_sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("a per-episode feature cache changed")
    metadata = frame[["episode_id", "episode_index", "family_id", "split"]].drop_duplicates()
    sampled = []
    for _, group in metadata.groupby(["family_id", "split"]):
        ordered = group.sort_values("episode_index")
        sampled.extend(ordered.iloc[[0, -1]].episode_index.tolist())
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    target_ids = source["identity"]["config"]["target_indices"]
    replay_rows = 0
    largest = 0.0
    indexed = frame.set_index(keys)
    for index in sorted(set(sampled)):
        record = source["episodes"][index]
        path = source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("a raw imputation source changed")
        with np.load(path, allow_pickle=False) as saved:
            context, candidates, actions = (
                saved["context"],
                saved["candidate_values"],
                saved["candidate_ids"].tolist(),
            )
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        reference = (candidates[actions.index("locf")] - mean) / scale
        for row in frame[frame.episode_index == index].itertuples(index=False):
            joint = row.model_id == "chronos2"
            targets = target_ids if joint else [target_ids[row.target_slot]]
            effective = assemble_selected_context(
                context, candidates, actions, [row.candidate_id], targets, joint=joint
            )
            rebuilt = input_change_features(
                (context - mean) / scale,
                (effective - mean) / scale,
                reference,
                targets,
                joint=joint,
            )
            actual = indexed.loc[
                (row.model_id, row.episode_id, row.candidate_id, row.target_slot),
                list(EXTRA_FEATURES),
            ].to_numpy()
            expected = np.array([rebuilt[name] for name in EXTRA_FEATURES])
            largest = max(largest, float(np.abs(actual - expected).max()))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            replay_rows += 1
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_rows": len(frame),
            "verified_case_hashes": len(manifest["cases"]),
            "original_static_features_unchanged": True,
            "independent_covariate_features_zero": True,
            "raw_input_replay_episodes": len(set(sampled)),
            "raw_input_replay_rows": replay_rows,
            "maximum_raw_input_replay_difference": largest,
            "forecaster_calls": 0,
            "scope": "all source identities and original columns checked; raw-input feature correspondence replayed on first/last cases in every family and split",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "status": "completed",
                "verified_rows": len(frame),
                "replay_episodes": len(set(sampled)),
                "maximum_difference": largest,
            }
        )
    )


if __name__ == "__main__":
    main()
