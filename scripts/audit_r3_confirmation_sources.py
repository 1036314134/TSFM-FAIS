"""Inspect source observability without evaluating or choosing a predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.data.catalog import DatasetSpec  # noqa: E402
from tsfm_fais.data.episodes import fit_prefix_end  # noqa: E402
from tsfm_fais.data.loaders import load_dataset  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def confirmation_windows(observed, context_length=96, horizon=96, minimum_future=48):
    """Describe later evaluation windows using observation masks alone."""
    observed = np.asarray(observed)
    if observed.ndim != 2 or observed.dtype != np.bool_ or observed.shape[1] < 2:
        raise ValueError("confirmation audit requires a boolean [T,D] mask with two targets")
    if not 1 <= minimum_future <= horizon:
        raise ValueError("minimum future observations must lie within the forecast horizon")
    prefix = fit_prefix_end(len(observed), context_length, horizon, fraction=0.2)
    boundary = int(np.floor(0.6 * len(observed)))
    prefix_counts = observed[:prefix].sum(axis=0)
    prefix_valid = bool((prefix_counts >= 2).all())
    # Match the development time split. Both completed probe horizons must
    # follow imputer fitting; they may precede the current evaluation interval.
    first = max(boundary + context_length, prefix + context_length + 2 * horizon)
    windows = []
    for origin in range(first, len(observed) - horizon + 1, context_length + horizon):
        context = observed[origin - context_length : origin]
        future_counts = observed[origin : origin + horizon, :2].sum(axis=0)
        future_valid = bool((future_counts >= minimum_future).all())
        windows.append(
            {
                "origin": origin,
                "eligible": prefix_valid and future_valid,
                "future_observed_by_target": future_counts.tolist(),
                "complete_target_future": bool((future_counts == horizon).all()),
                "context_has_missing": bool((~context).any()),
                "context_target_has_missing": bool((~context[:, :2]).any()),
                "context_missing_by_target": (~context[:, :2]).sum(axis=0).tolist(),
                "context_empty_channel_count": int((~context.any(axis=0)).sum()),
                "context_empty_target_count": int((~context[:, :2].any(axis=0)).sum()),
                "post_fit_history_length": origin - prefix,
                "recent_probe_observed_by_target": [
                    observed[origin - offset * horizon : origin - (offset - 1) * horizon, :2]
                    .sum(axis=0)
                    .tolist()
                    for offset in (1, 2)
                ],
                "exclusion_reason": (
                    "insufficient_prefix_observations"
                    if not prefix_valid
                    else "insufficient_future_observations"
                    if not future_valid
                    else None
                ),
            }
        )
    eligible = [window for window in windows if window["eligible"]]
    return {
        "prefix_end": prefix,
        "temporal_boundary": boundary,
        "prefix_observed_counts": prefix_counts.tolist(),
        "prefix_eligible": prefix_valid,
        "candidate_window_count": len(windows),
        "eligible_window_count": len(eligible),
        "eligible_missing_context_count": sum(window["context_has_missing"] for window in eligible),
        "eligible_missing_target_context_count": sum(
            window["context_target_has_missing"] for window in eligible
        ),
        "eligible_complete_future_count": sum(
            window["complete_target_future"] for window in eligible
        ),
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirmation-windows", action="store_true")
    parser.add_argument("--max-items", type=int, default=4)
    args = parser.parse_args()
    if args.max_items < 1:
        parser.error("max-items must be positive")
    if args.output.exists():
        parser.error("preserve the existing audit; select a new output path")
    properties_path = args.data_root / "dataset_properties.json"
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    families = [
        "Australia_Solar",
        "Water_Quality_Darwin",
        "Crypto",
        "MetroPT-3",
        "SG_PM25",
        "SG_Weather",
        "Smart_Manufacturing",
        "Oil_Price",
        "US_Term_Structure",
    ]
    output = []
    for dataset_id, meta in properties.items():
        family, frequency = dataset_id.rsplit("_", 1)
        if family not in families:
            continue
        directory = args.data_root / family / frequency
        record = {
            "dataset_id": dataset_id,
            "family": family,
            "metadata": meta,
            "path": str(directory),
        }
        try:
            spec = DatasetSpec(
                dataset_id=dataset_id,
                family_id=family.lower(),
                format="arrow",
                path=directory,
                frequency=frequency,
                period=meta["period"],
                expected_num_variates=meta["num_variates"],
                missingness="native",
                provenance="source_native_missing",
                allow_implicit_regular_time=True,
            )
            items = load_dataset(spec)
            record["item_count"] = len(items)
            record["source_sha256"] = {
                str(path): file_sha256(path) for path in sorted(directory.glob("*.arrow"))
            }
            record["items"] = []
            for item in items[: args.max_items]:
                observed = np.isfinite(item.values)
                prefix = max(96, int(0.2 * len(observed)))
                starts = range(prefix + 96, len(observed) - 96 + 1, 192)
                future_complete = sum(
                    bool(observed[origin : origin + 96, :2].all()) for origin in starts
                )
                full_complete = sum(
                    bool(observed[origin - 96 : origin + 96].all()) for origin in starts
                )
                future_half_observed = sum(
                    bool((observed[origin : origin + 96, :2].sum(axis=0) >= 48).all())
                    for origin in starts
                )
                record["items"].append(
                    {
                        "item_id": item.item_id,
                        "length": len(observed),
                        "dimensions": observed.shape[1],
                        "observed_fraction": float(observed.mean()),
                        "prefix_observed_by_variate": observed[:prefix].mean(axis=0).tolist(),
                        "candidate_origin_count": len(starts),
                        "fully_observed_target_future_count": future_complete,
                        "fully_observed_context_and_future_count": full_complete,
                        "at_least_half_observed_target_future_count": future_half_observed,
                    }
                )
                if args.confirmation_windows:
                    record["items"][-1]["confirmation"] = confirmation_windows(observed)
            record["status"] = "structurally_readable"
        except Exception as error:
            record["status"] = "requires_source_review"
            record["reason"] = f"{type(error).__name__}: {error}"
        output.append(record)
        print(
            json.dumps(
                {
                    "dataset": dataset_id,
                    "status": record["status"],
                    "items": record.get("item_count"),
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "completed",
                "purpose": "structural audit only; no forecasting outcomes inspected",
                "script_sha256": file_sha256(Path(__file__)),
                "properties_sha256": file_sha256(properties_path),
                "max_items_per_dataset": args.max_items,
                "confirmation_protocol": (
                    {
                        "context_length": 96,
                        "horizon": 96,
                        "target_indices": [0, 1],
                        "fit_prefix_fraction": 0.2,
                        "temporal_boundary": 0.6,
                        "stride": 192,
                        "minimum_future_observations_per_target": 48,
                        "minimum_prefix_observations_per_channel": 2,
                        "completed_probe_count": 2,
                        "sampling": "first items in source order; all eligible later origins; no selection using forecast accuracy",
                        "future_metric": "score only originally observed future values; macro-average targets, items, datasets and families",
                        "missingness_reporting": "retain all eligible contexts and report the naturally missing subset separately",
                    }
                    if args.confirmation_windows
                    else None
                ),
                "datasets": output,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
