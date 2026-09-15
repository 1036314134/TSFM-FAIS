"""Replay the new deployment feature path against frozen development inputs."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.followup_portfolio import portfolio_feature_frame  # noqa: E402
from tsfm_fais.routing.preforecast_replay import candidate_feature_frame  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "aligned-root", "source-bundle", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed runtime replay")
    output.mkdir(parents=True, exist_ok=True)
    test_result = pytest.main(
        [
            "-q",
            str(ROOT / "tests/unit/test_followup_portfolio.py"),
            "--basetemp=" + str(output / "pytest"),
        ]
    )
    if test_result != 0:
        raise ValueError("deployment feature boundary tests failed")
    source = json.loads((args.source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((args.accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    aligned = json.loads((args.aligned_root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    if accuracy["source_episode_manifest_sha256"] != file_sha256(
        args.source_root / "episodes_manifest.json"
    ):
        raise ValueError("the development forecast bank uses different source episodes")
    if aligned["identity"]["accuracy_manifest_sha256"] != file_sha256(
        args.accuracy_root / "manifest.json"
    ):
        raise ValueError("the portfolio feature bank uses different forecasts")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(
            (args.accuracy_root / "standardizers.json").read_text(encoding="utf-8")
        )
    }
    selected, families = [], set()
    for index, row in enumerate(source["episodes"]):
        if row["split"] == "validation" and row["family_id"] not in families:
            selected.append((index, row))
            families.add(row["family_id"])
    if len(families) != 15:
        raise ValueError("runtime replay must cover all 15 source development families")
    comparisons, maximum = [], 0.0
    for model in ("chronos2", "timesfm2p5"):
        entry = next(row for row in aligned["models"] if row["model_id"] == model)
        info_path = args.aligned_root / entry["path"]
        if file_sha256(info_path) != entry["sha256"]:
            raise ValueError("aligned feature metadata changed")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        decisions = pd.read_parquet(args.aligned_root / info["decisions_path"])
        bank_path = args.accuracy_root / f"{model}_point_z.npy"
        if file_sha256(bank_path) != accuracy["prediction_arrays"][bank_path.name]:
            raise ValueError("development candidate forecasts changed")
        bank = np.load(bank_path, mmap_mode="r")
        generated_frames, expected_frames = [], []
        for index, record in selected:
            path = args.source_root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError("a deployment replay source changed")
            with np.load(path, allow_pickle=False) as saved:
                context, candidates = saved["context"], saved["candidate_values"]
                actions, coverage = saved["candidate_ids"].tolist(), saved["native_coverage"]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
            frame = candidate_feature_frame(
                context,
                candidates,
                actions,
                coverage,
                mean,
                scale,
                [0, 1],
                joint=model == "chronos2",
                period=record["period"],
                metadata={**record, "episode_index": index, "model_id": model},
            )
            order = [*actions, "guarded_direct"]
            points = bank[index, [accuracy["action_orders"][model].index(name) for name in order]]
            last = (candidates[actions.index("locf"), -1, :2] - mean[:2]) / scale[:2]
            generated, _, _, _ = portfolio_feature_frame(
                frame, points, order, last, joint=model == "chronos2"
            )
            for slot in [-1] if model == "chronos2" else [0, 1]:
                locations = np.flatnonzero(
                    (decisions.episode_index == index).to_numpy()
                    & (decisions.target_slot == slot).to_numpy()
                )
                if len(locations) != 1:
                    raise ValueError("an original source decision is missing or ambiguous")
                location = int(locations[0])
                chunk = next(
                    row for row in info["chunks"] if row["start"] <= location < row["stop"]
                )
                chunk_path = args.aligned_root / chunk["path"]
                if file_sha256(chunk_path) != chunk["sha256"]:
                    raise ValueError("the original feature chunk changed")
                with np.load(chunk_path, allow_pickle=False) as saved:
                    expected = pd.DataFrame(
                        saved["features"][location - chunk["start"], 7:42],
                        columns=info["feature_names"],
                    )
                expected["candidate_id"] = info["option_names"][7:42]
                expected["episode_id"] = decisions.iloc[location].episode_id
                actual = (
                    generated[generated.target_slot == slot]
                    .set_index("candidate_id")
                    .loc[expected.candidate_id]
                )
                difference = float(
                    np.abs(
                        actual[info["feature_names"]].to_numpy()
                        - expected[info["feature_names"]].to_numpy()
                    ).max()
                )
                maximum = max(maximum, difference)
                np.testing.assert_allclose(
                    actual[info["feature_names"]],
                    expected[info["feature_names"]],
                    rtol=1e-6,
                    atol=1e-6,
                )
                expected_frames.append(expected)
            generated_frames.append(generated)
        generated = pd.concat(generated_frames, ignore_index=True)
        expected = pd.concat(expected_frames, ignore_index=True)
        for entry in bundle["models"]:
            if entry["model_id"] != model:
                continue
            marker = args.source_bundle / entry["path"]
            if file_sha256(marker) != entry["sha256"]:
                raise ValueError("a frozen source-model record changed")
            record = json.loads(marker.read_text(encoding="utf-8"))
            path = args.source_bundle / record["model_path"]
            if file_sha256(path) != record["model_sha256"]:
                raise ValueError("a frozen source selector changed")
            learner = joblib.load(path)
            left = learner.select(generated).set_index("episode_id").candidate_id.sort_index()
            right = learner.select(expected).set_index("episode_id").candidate_id.sort_index()
            pd.testing.assert_series_equal(left, right)
            comparisons.append(
                {
                    "model_id": model,
                    "label_kind": record["label_kind"],
                    "verified_decisions": len(left),
                }
            )
            del learner
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "runtime_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/followup_portfolio.py"),
            "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
            "comparisons": comparisons,
            "maximum_feature_difference": maximum,
            "source_histories": len(selected),
            "unit_tests_passed": 2,
            "new_forecaster_calls": 0,
            "followup_future_arrays_read": False,
        },
    )


if __name__ == "__main__":
    main()
