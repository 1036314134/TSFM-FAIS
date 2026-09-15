from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_fais.utility_experiment import file_sha256


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


def test_current_future_values_do_not_change_probe_decisions(tmp_path, monkeypatch):
    script = Path(__file__).parents[2] / "scripts/analyze_recent_forecast_probes.py"
    spec = importlib.util.spec_from_file_location("recent_probe_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_root, root, accuracy_root = (
        tmp_path / "source",
        tmp_path / "probes",
        tmp_path / "accuracy",
    )
    accuracy_root.mkdir()
    current = {
        "episode_id": "target_current",
        "origin_id": "target_20",
        "origin": 20,
        "family_id": "target",
        "dataset_id": "target",
        "item_id": "i",
        "split": "validation",
        "mechanism": "random_point",
        "missing_rate": 0.1,
        "mask_seed": 7,
    }
    source = {"episodes": [current]}
    write_json(source_root / "episodes_manifest.json", source)
    source_sha = file_sha256(source_root / "episodes_manifest.json")
    actions = ["locf", "linear_interp", "guarded_direct"]
    predictions = np.stack([np.zeros((4, 2)), np.ones((4, 2)), np.full((4, 2), 0.5)])
    np.save(accuracy_root / "timesfm2p5_point_z.npy", predictions[None])
    np.save(accuracy_root / "truth_z.npy", np.zeros((1, 4, 2)))
    reference = pd.DataFrame(
        [
            {
                "model_id": "timesfm2p5",
                "split": "train",
                "target_slot": -1,
                "candidate_id": "locf",
                "family_id": "reference",
                "dataset_id": "reference",
                "mae": 1.0,
                "mse": 1.0,
            }
        ]
    )
    reference.to_parquet(accuracy_root / "candidate_accuracy.parquet", index=False)
    accuracy = {
        "source_episode_manifest_sha256": source_sha,
        "action_orders": {"timesfm2p5": actions},
    }
    write_json(accuracy_root / "manifest.json", accuracy)
    plans, links, prepared, forecasts = [], [], [], []
    for offset in (1, 2):
        probe_id = f"probe{offset}"
        origin = 20 - 4 * offset
        plans.append(current | {"probe_id": probe_id, "origin": origin, "horizon": 4})
        links.append(
            {
                "episode_id": current["episode_id"],
                "probe_id": probe_id,
                "probe_horizon": 4,
                "offset": offset,
            }
        )
        prepared_path = root / f"{probe_id}.npz"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        observed = np.ones((4, 2))
        observed[0, 0] = np.nan
        np.savez(prepared_path, observed_future_z=observed)
        digest = file_sha256(prepared_path)
        prepared.append({"probe_id": probe_id, "path": prepared_path.name, "sha256": digest})
        forecast_path = root / f"{probe_id}-forecast.npz"
        np.savez(
            forecast_path,
            point_z=predictions,
            action_ids=np.asarray(actions),
            probe_sha256=np.asarray(digest),
        )
        forecasts.append(
            {"probe_id": probe_id, "path": forecast_path.name, "sha256": file_sha256(forecast_path)}
        )
    write_json(
        root / "plan.json",
        {
            "source_manifest_sha256": source_sha,
            "decision_episode_ids": [current["episode_id"]],
            "probes": plans,
            "links": links,
            "horizons": [4],
            "offsets": [1, 2],
        },
    )
    write_json(root / "prepared_manifest.json", {"probes": prepared})
    forecast_manifest = {
        "identity": {"accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json")},
        "probes": forecasts,
        "action_ids": actions,
    }
    write_json(root / "timesfm2p5/manifest.json", forecast_manifest)
    argv = [
        str(script),
        "--source-root",
        str(source_root),
        "--probe-root",
        str(root),
        "--accuracy-root",
        str(accuracy_root),
        "--models",
        "timesfm2p5",
    ]
    monkeypatch.setattr("sys.argv", argv)
    module.main()
    first = pd.read_parquet(root / "analysis-v001/input_mixture_weights.parquet")
    first_results = pd.read_parquet(root / "analysis-v001/episode_results.parquet")
    first_risks = pd.read_parquet(root / "analysis-v001/historical_risks.parquet")
    # The forecasting decisions keep exactly the same observed historical
    # inputs; only the held-out outcome being scored changes.
    np.save(accuracy_root / "truth_z.npy", np.full((1, 4, 2), 100.0))
    monkeypatch.setattr("sys.argv", argv + ["--output-name", "analysis-future-changed"])
    module.main()
    second = pd.read_parquet(root / "analysis-future-changed/input_mixture_weights.parquet")
    second_results = pd.read_parquet(root / "analysis-future-changed/episode_results.parquet")
    pd.testing.assert_frame_equal(first, second)
    trace_columns = ["selected_action_ids", "forecast_weights"]
    pd.testing.assert_frame_equal(first_results[trace_columns], second_results[trace_columns])
    pd.testing.assert_frame_equal(
        first_risks,
        pd.read_parquet(root / "analysis-future-changed/historical_risks.parquet"),
    )
    assert not np.allclose(first_results.mae, second_results.mae)
    assert len(first_results) == len(second_results) > 0
