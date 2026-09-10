"""Evaluate observed-history selection with delayed full candidate feedback."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.data import (  # noqa: E402
    MaskingSpec,
    load_dataset,
    load_manifest,
    mask_time_series,
    stable_seed,
)
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256, load_utility_config  # noqa: E402


def historical_choice(
    history: list[np.ndarray], candidate_ids: list[str], anchor: str, mode: str
) -> str:
    if not history:
        return anchor
    losses = history[-1] if mode == "last" else np.mean(history, axis=0)
    best = np.min(losses)
    anchor_index = candidate_ids.index(anchor)
    if losses[anchor_index] == best:
        return anchor
    return candidate_ids[int(np.argmin(losses))]


def observed_probe_losses(
    point: np.ndarray, truth: np.ndarray, scales: np.ndarray, minimum: int
) -> np.ndarray | None:
    if (
        point.ndim != 3
        or truth.shape != point.shape[1:]
        or scales.shape != (point.shape[2],)
        or minimum < 1
    ):
        raise ValueError("history feedback requires [A,H,K], [H,K], positive [K] scales")
    if not np.isfinite(point).all() or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("history forecasts and scales must be finite; scales must be positive")
    if np.isinf(truth).any():
        raise ValueError("unobserved history values must use NaN, not infinity")
    counts = np.isfinite(truth).sum(axis=0)
    if np.any(counts < minimum):
        return None
    errors = np.abs(point - truth[None]) / scales[None, None, :]
    return np.mean(np.nansum(errors, axis=1) / counts[None, :], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    args = parser.parse_args()
    config = load_utility_config(args.config)
    root = config.output_root
    episode_manifest = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    data_manifest = load_manifest(config.data_manifest)
    groups = defaultdict(list)
    for record in episode_manifest["episodes"]:
        if record["split"] == "validation":
            key = tuple(
                record[name]
                for name in ("dataset_id", "item_id", "mechanism", "missing_rate", "mask_seed")
            )
            groups[key].append(record)
    results = []
    cached_dataset, cached_items = None, {}
    models = args.models.split(",")
    source_rows = {
        model: pd.read_parquet(root / model / "utility_rows.parquet").set_index(
            ["episode_id", "candidate_id"]
        )
        for model in models
    }
    forecast_hashes = {
        model: {
            entry["episode_id"]: entry["sha256"]
            for entry in json.loads(
                (root / model / "forecast_manifest.json").read_text(encoding="utf-8")
            )["episodes"]
        }
        for model in models
    }
    for (dataset_id, item_id, mechanism, rate, seed), records in groups.items():
        if cached_dataset != dataset_id:
            dataset_record = next(
                entry for entry in episode_manifest["datasets"] if entry["dataset_id"] == dataset_id
            )
            for path, expected_hash in dataset_record["sources"].items():
                if file_sha256(Path(path)) != expected_hash:
                    raise ValueError("historical source changed after episode preparation")
            cached_items = {
                item.item_id: item for item in load_dataset(data_manifest.get(dataset_id))
            }
            cached_dataset = dataset_id
        item = cached_items[item_id]
        prefix_end = next(
            entry["prefix_end"]
            for dataset in episode_manifest["datasets"]
            if dataset["dataset_id"] == dataset_id
            for entry in dataset["items"]
            if entry["item_id"] == item_id
        )
        realization = mask_time_series(
            item.values,
            MaskingSpec(mechanism, rate, config.block_lengths),
            stable_seed(
                config.protocol_id, dataset_id, item_id, "validation", mechanism, rate, seed
            ),
            calibration_values=item.values[:prefix_end],
        )
        for model_id in models:
            action_pool = sorted(
                source_rows[model_id].index.get_level_values("candidate_id").unique()
            )
            anchors = ["locf", "native_missing"] if "native_missing" in action_pool else ["locf"]
            history: list[np.ndarray] = []
            previous_end = -1
            for record in sorted(records, key=lambda entry: entry["origin"]):
                if previous_end > record["origin"]:
                    raise ValueError("a historical outcome has not arrived before the decision")
                chosen = {
                    (anchor, mode): historical_choice(history, action_pool, anchor, mode)
                    for anchor in anchors
                    for mode in ("last", "mean")
                }
                # Only past feedback is used above. Current future truth below
                # scores the decision and is released for subsequent origins.
                for (anchor, mode), candidate in chosen.items():
                    row = source_rows[model_id].loc[(record["episode_id"], candidate)]
                    results.append(
                        {
                            "episode_id": record["episode_id"],
                            "origin_id": record["origin_id"],
                            "model_id": model_id,
                            "family_id": record["family_id"],
                            "dataset_id": dataset_id,
                            "anchor_id": anchor,
                            "method": f"observed_history_{mode}",
                            "selected_candidate": candidate,
                            "loss": float(row.loss),
                            "available_probe_count": len(history),
                        }
                    )
                episode_path = root / record["path"]
                forecast_path = root / model_id / "predictions" / episode_path.name
                if file_sha256(episode_path) != record["sha256"]:
                    raise ValueError("historical episode changed after preparation")
                if file_sha256(forecast_path) != forecast_hashes[model_id][record["episode_id"]]:
                    raise ValueError("historical forecast changed after generation")
                with (
                    np.load(episode_path, allow_pickle=False) as episode,
                    np.load(forecast_path, allow_pickle=False) as forecast,
                ):
                    ids = episode["candidate_ids"].tolist()
                    if len(forecast["point"]) == len(ids) + 1:
                        ids.append("native_missing")
                    point = forecast["point"][[ids.index(candidate) for candidate in action_pool]]
                    truth = realization.values[
                        record["origin"] : record["origin"] + config.horizon, config.target_indices
                    ]
                    losses = observed_probe_losses(
                        point,
                        truth,
                        episode["mase_scales"][list(config.target_indices)],
                        max(1, config.horizon // 8),
                    )
                    if losses is not None:
                        history.append(losses)
                previous_end = record["origin"] + config.horizon
    frame = pd.DataFrame(results)
    summary = [
        {
            "model_id": model_id,
            "anchor_id": anchor,
            "method": method,
            "family_macro_mase": family_macro(group),
            "episode_count": int(group.episode_id.nunique()),
            "history_available_fraction": float((group.available_probe_count > 0).mean()),
        }
        for (model_id, anchor, method), group in frame.groupby(["model_id", "anchor_id", "method"])
    ]
    output = root / "analysis-history-v001"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "episode_results.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "evidence_role": "development",
                "script_sha256": file_sha256(Path(__file__)),
                "source_episode_manifest_sha256": file_sha256(root / "episodes_manifest.json"),
                "source_forecast_manifest_sha256": {
                    model: file_sha256(root / model / "forecast_manifest.json") for model in models
                },
                "feedback": "only originally observed cells of completed past horizons in the same missingness realization; at least H/8 observations per target",
                "cost": "all candidate forecasts must be generated at every served origin for later feedback; cached offline execution does not make those queries free",
                "cold_start": "fixed declared anchor; no labels from held-family training episodes",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
