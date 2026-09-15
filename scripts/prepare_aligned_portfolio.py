"""Prepare observable portfolio features and separate complete-history teacher labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.aligned_portfolio import build_option_features, option_catalog  # noqa: E402
from tsfm_fais.routing.forecast_projection import forecast_offsets, projection_targets  # noqa: E402
from tsfm_fais.routing.forecast_response import FORECAST_FEATURES  # noqa: E402
from tsfm_fais.routing.preforecast import METADATA, STATIC_FEATURES  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "teacher-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed portfolio preparations")
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    teachers = json.loads((args.teacher_root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(accuracy["source_root"])
    source_path = source_root / "episodes_manifest.json"
    if (
        file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]
        or teachers["status"] != "completed"
        or teachers["identity"]["accuracy_manifest_sha256"]
        != file_sha256(args.accuracy_root / "manifest.json")
    ):
        raise ValueError("teacher, candidate forecasts, and source episodes must agree")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = pd.DataFrame(source["episodes"]).assign(
        episode_index=np.arange(len(source["episodes"]))
    )
    if (
        len(metadata) != 7812
        or metadata[metadata.split == "train"].origin_id.nunique() != 165
        or metadata[metadata.split == "validation"].origin_id.nunique() != 52
    ):
        raise ValueError("the fixed source-history split changed")
    table_path = args.accuracy_root / "candidate_accuracy.parquet"
    schema = pq.read_schema(table_path).names
    if {name for name in schema if name.startswith(("static.", "response."))} != set(
        FORECAST_FEATURES
    ):
        raise ValueError("the source feature whitelist changed")
    columns = [name for name in (*METADATA, *FORECAST_FEATURES) if name in schema]
    source_features = pd.read_parquet(table_path, columns=columns)
    identity = {
        "accuracy_manifest_sha256": file_sha256(args.accuracy_root / "manifest.json"),
        "teacher_manifest_sha256": file_sha256(args.teacher_root / "manifest.json"),
        "source_manifest_sha256": file_sha256(source_path),
        "source_root": str(source_root),
        "candidate_table_sha256": file_sha256(table_path),
        "standardizers_sha256": file_sha256(args.accuracy_root / "standardizers.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/routing/aligned_portfolio.py",
                "src/tsfm_fais/routing/forecast_projection.py",
                "src/tsfm_fais/routing/forecast_response.py",
                "src/tsfm_fais/routing/preforecast.py",
                "src/tsfm_fais/routing/utility.py",
            )
        },
        "chunk_size": 256,
        "target_kinds": ["unit_projection", "direct_risk"],
        "primary_menu": "full",
        "menus": ["single", "triple", "mixed", "full"],
        "source_supervision": "complete source-history forecast; actual source futures are not read",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial feature preparation changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    last_z = np.empty((len(metadata), 2))
    for start in range(0, len(metadata), identity["chunk_size"]):
        stop = min(start + identity["chunk_size"], len(metadata))
        path = output / "last_locf" / f"{start:06}.npz"
        if not path.exists():
            values = []
            for record in source["episodes"][start:stop]:
                original = source_root / record["path"]
                if file_sha256(original) != record["sha256"]:
                    raise ValueError("a current-history input changed")
                with np.load(original, allow_pickle=False) as saved:
                    actions = saved["candidate_ids"].tolist()
                    last = saved["candidate_values"][actions.index("locf"), -1, :2]
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                values.append(
                    (last - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
                )
            _save_npz(path, last_z=np.asarray(values), identity_sha256=np.asarray(identity_sha))
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["identity_sha256"]) != identity_sha or saved["last_z"].shape != (
                stop - start,
                2,
            ):
                raise ValueError("a last-LOCF cache changed provenance")
            last_z[start:stop] = saved["last_z"]
    model_records = []
    metadata_columns = [
        "episode_id",
        "episode_index",
        "origin_id",
        "dataset_id",
        "family_id",
        "item_id",
        "split",
    ]
    for model in ("chronos2", "timesfm2p5"):
        directory = output / model
        directory.mkdir(exist_ok=True)
        marker = directory / "manifest.json"
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record["identity_sha256"] != identity_sha:
                raise ValueError("a prepared model changed identity")
            model_records.append(
                {
                    "model_id": model,
                    "path": str(marker.relative_to(output)),
                    "sha256": file_sha256(marker),
                }
            )
            continue
        actions = sorted(
            set(accuracy["action_orders"][model]) - {"native_missing", "vendor_missing"}
        )
        names, members = option_catalog(actions)
        point_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("the candidate forecast bank changed")
        bank = np.load(point_path, mmap_mode="r")[
            :, [accuracy["action_orders"][model].index(name) for name in actions]
        ]
        teacher_record = next(row for row in teachers["models"] if row["model_id"] == model)
        teacher_path = args.teacher_root / teacher_record["teacher_file"]
        if file_sha256(teacher_path) != teacher_record["teacher_sha256"]:
            raise ValueError("the complete-history teacher bank changed")
        teacher = np.load(teacher_path, mmap_mode="r")
        slots = [-1] if model == "chronos2" else [0, 1]
        decisions, chunks, max_feature_difference = [], [], 0.0
        for slot_number, slot in enumerate(slots):
            decision = metadata[metadata_columns].copy()
            decision["source_episode_id"] = decision.episode_id
            decision["target_slot"] = slot
            if slot >= 0:
                decision["episode_id"] = decision.episode_id + "|target=" + str(slot)
            decisions.append(decision)
            view = source_features[
                (source_features.model_id == model) & (source_features.target_slot == slot)
            ]
            original_features = []
            for action in actions:
                rows = view[view.candidate_id == action].set_index("episode_index")
                if len(rows) != len(metadata) or not rows.index.is_unique:
                    raise ValueError("source candidate features have incomplete episode coverage")
                rows = rows.loc[np.arange(len(metadata))]
                pd.testing.assert_frame_equal(
                    rows.reset_index()[metadata_columns],
                    metadata[metadata_columns],
                    check_dtype=False,
                )
                original_features.append(rows[list(FORECAST_FEATURES)].to_numpy(float))
            original_features = np.stack(original_features, axis=1)
            for start in range(0, len(metadata), identity["chunk_size"]):
                stop = min(start + identity["chunk_size"], len(metadata))
                path = directory / "chunks" / f"slot{slot}_{start:06}.npz"
                if not path.exists():
                    candidates = (
                        bank[start:stop].reshape(stop - start, 7, -1)
                        if slot == -1
                        else bank[start:stop, :, :, slot]
                    )
                    targets = (
                        teacher[start:stop].reshape(stop - start, -1)
                        if slot == -1
                        else teacher[start:stop, :, slot]
                    )
                    features, vectors, _ = build_option_features(
                        candidates,
                        original_features[start:stop, :, : len(STATIC_FEATURES)],
                        last_z[start:stop] if slot == -1 else last_z[start:stop, slot : slot + 1],
                        actions,
                        horizon=96,
                        targets=2 if slot == -1 else 1,
                    )
                    labels = projection_targets(vectors, targets, anchor=vectors[:, -1])
                    _, _, energy = forecast_offsets(vectors, anchor=vectors[:, -1])
                    _save_npz(
                        path,
                        features=features,
                        norm=np.sqrt(energy),
                        unit_projection=labels["unit_projection"],
                        direct_risk=labels["direct_risk"],
                        identity_sha256=np.asarray(identity_sha),
                    )
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["identity_sha256"]) != identity_sha or saved["features"].shape != (
                        stop - start,
                        43,
                        40,
                    ):
                        raise ValueError("an option-feature cache changed provenance or shape")
                    np.testing.assert_allclose(
                        saved["features"][:, :7, : len(FORECAST_FEATURES)],
                        original_features[start:stop],
                        rtol=1e-6,
                        atol=1e-6,
                    )
                    max_feature_difference = max(
                        max_feature_difference,
                        float(
                            np.abs(
                                saved["features"][:, :7, : len(FORECAST_FEATURES)]
                                - original_features[start:stop]
                            ).max()
                        ),
                    )
                chunks.append(
                    {
                        "start": slot_number * len(metadata) + start,
                        "stop": slot_number * len(metadata) + stop,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                _write_json(
                    output / "progress.json",
                    {
                        "status": "preparing",
                        "model_id": model,
                        "slot": slot,
                        "completed_episode_rows_in_slot": stop,
                        "episode_rows_per_slot": len(metadata),
                    },
                )
        decisions = pd.concat(decisions, ignore_index=True)
        decisions.to_parquet(directory / "decisions.parquet", index=False)
        _write_json(
            marker,
            {
                "status": "completed",
                "identity_sha256": identity_sha,
                "model_id": model,
                "actions": actions,
                "option_names": names,
                "members": members,
                "feature_names": [*FORECAST_FEATURES, *["member." + name for name in actions]],
                "decisions_path": str((directory / "decisions.parquet").relative_to(output)),
                "decisions_sha256": file_sha256(directory / "decisions.parquet"),
                "decision_count": len(decisions),
                "chunks": chunks,
                "single_candidate_feature_maximum_difference": max_feature_difference,
                "single_candidate_feature_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            },
        )
        model_records.append(
            {
                "model_id": model,
                "path": str(marker.relative_to(output)),
                "sha256": file_sha256(marker),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "models": model_records,
            "new_forecaster_calls": 0,
            "new_selector_fits": 0,
            "information_boundary": "Features use current imputed history and seven forecasts; teacher labels are separate; actual source futures are not read.",
        },
    )
    print(json.dumps({"status": "completed", "models": 2, "options": 43}), flush=True)


if __name__ == "__main__":
    main()
