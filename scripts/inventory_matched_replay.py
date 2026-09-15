"""Inventory causal exact-mask replay support and declared prior data usage."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import anchor_support, native_sources

from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed preflight inventories")
    output.mkdir(parents=True, exist_ok=True)
    sources = native_sources()
    rows, series = [], []
    for source in sources:
        values = source["values"]
        scaler = PrefixStandardizer.fit(values[: source["prefix_end"]])
        expected = next(
            row
            for row in read_json(source["root"] / "standardizers.json")
            if row["dataset_id"] == source["dataset_id"] and row["item_id"] == source["item_id"]
        )
        np.testing.assert_array_equal(scaler.mean, expected["mean"])
        np.testing.assert_array_equal(scaler.scale, expected["scale"])
        known = []
        for row in source["episodes"]:
            path = source["root"] / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("an existing native observation changed")
            origin = int(row["window"]["origin"])
            context = np.asarray(values[origin - 96 : origin])
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(context, saved["context"])
            mask = ~np.isfinite(context)
            known.append(
                {
                    "episode_id": row["episode_id"],
                    "origin": origin,
                    "input_interval": [origin - 96, origin],
                    "scored_future_interval": [
                        origin,
                        origin + (192 if source["cohort"] == "r6_native" else 96),
                    ],
                    "prediction_generated": True,
                    "scores_read_for_development": True,
                    "used_in_r13_other_group_training": bool(row["window"]["context_has_missing"]),
                    "observation_artifact": str(path),
                    "observation_artifact_sha256": row["sha256"],
                }
            )
            if not mask[:, :2].any():
                continue
            for budget in (1024, 4096):
                support = anchor_support(values, mask, origin, source["prefix_end"], budget)
                rows.append(
                    {
                        "cohort": source["cohort"],
                        "family_id": source["family_id"],
                        "dataset_id": source["dataset_id"],
                        "item_id": source["item_id"],
                        "episode_id": row["episode_id"],
                        "origin_id": row["origin_id"],
                        "origin": origin,
                        "prefix_end": source["prefix_end"],
                        "history_budget": budget,
                        "target_missing_fraction": float(mask[:, :2].mean()),
                        "context_missing_fraction": float(mask.mean()),
                        "complete_anchors": len(support["complete_context_origins"]),
                        "mask_compatible_anchors": len(support["mask_compatible_origins"]),
                        "future_observed_anchors": len(support["future_observed_origins"]),
                        "supported": bool(support["selected"]),
                        **support,
                    }
                )
        series.append(
            {
                key: source[key]
                for key in (
                    "cohort",
                    "dataset_id",
                    "family_id",
                    "item_id",
                    "start",
                    "frequency",
                    "prefix_end",
                    "period",
                    "source_files",
                    "unused_series_in_loaded_dataset",
                )
            }
        )
        series[-1].update(
            shape=list(values.shape),
            known_evaluation_windows=known,
            fit_population=source["dataset"]["fit_items"],
            independent_confirmation_status="not established; listed unused series require cross-artifact and time-overlap audit",
        )
    _write_json(output / "support.json", rows)
    _write_json(
        output / "data_usage.json",
        {
            "scope": "existing R5 native and R6 native populations; prior score and training reuse declared",
            "series": series,
            "unseen_confirmation_selected": False,
        },
    )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["history_budget", "family_id"])
        .agg(
            tasks=("episode_id", "size"),
            supported=("supported", "sum"),
            complete_anchors=("complete_anchors", "median"),
            mask_compatible_anchors=("mask_compatible_anchors", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "support_summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "io_module_sha256": file_sha256(ROOT / "scripts/matched_replay_sources.py"),
            "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R19_PREFLIGHT_PROTOCOL.md"),
            "series": len(series),
            "eligible_current_tasks": frame.episode_id.nunique(),
            "support_sha256": file_sha256(output / "support.json"),
            "usage_sha256": file_sha256(output / "data_usage.json"),
            "new_forecaster_calls": 0,
            "new_imputer_fits": 0,
            "current_future_values_scored": False,
            "limits": "support inventory only; no new independent confirmation is certified",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
