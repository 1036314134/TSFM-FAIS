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
from tsfm_fais.data.loaders import load_dataset  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
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
            for item in items[:4]:
                observed = np.isfinite(item.values)
                prefix = max(96, int(0.2 * len(observed)))
                starts = range(prefix + 96, len(observed) - 96 + 1, 192)
                future_complete = sum(
                    bool(observed[origin : origin + 96, :2].all()) for origin in starts
                )
                full_complete = sum(
                    bool(observed[origin - 96 : origin + 96].all()) for origin in starts
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
                    }
                )
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
                "purpose": "structural audit only; no forecasting outcomes inspected",
                "properties_sha256": file_sha256(properties_path),
                "datasets": output,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
