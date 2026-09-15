"""Inspect unused series and local new-family sources without forecasting outcomes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_r3_confirmation_sources import confirmation_windows  # noqa: E402

from tsfm_fais.data.catalog import DatasetSpec  # noqa: E402
from tsfm_fais.data.loaders import load_dataset  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed eligibility inventories")
    previous = json.loads(args.previous_cohort.read_text(encoding="utf-8"))
    excluded = {}
    for task in previous["tasks"]:
        excluded.setdefault(task["dataset_id"], set()).add(task["item_id"])
    properties_path = args.data_root / "arrow/dataset_properties.json"
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    candidates = [
        ("Water_Quality_Darwin_15T", "unused_item_in_previous_family"),
        ("SG_Weather_D", "unused_item_in_previous_family"),
        ("Smart_Manufacturing_H", "unused_item_in_previous_family"),
        ("Global_Influenza_W", "local_family_outside_previous_catalog_and_confirmation"),
        ("Global_Price_Q", "local_family_outside_previous_catalog_and_confirmation"),
        ("WUI_Global_Q", "local_family_outside_previous_catalog_and_confirmation"),
    ]
    specs = []
    for dataset, role in candidates:
        family, frequency = dataset.rsplit("_", 1)
        prop = properties[dataset]
        specs.append(
            (
                DatasetSpec(
                    dataset_id=dataset,
                    family_id=family.lower(),
                    format="arrow",
                    path=args.data_root / "arrow" / family / frequency,
                    frequency=frequency,
                    period=prop["period"],
                    expected_num_variates=prop["num_variates"],
                    missingness="native",
                    provenance="source_native_missing",
                    allow_implicit_regular_time=True,
                ),
                role,
            )
        )
    weather = args.data_root / "csv/weather.csv"
    if weather.exists():
        csv_properties = json.loads(
            (args.data_root / "csv/dataset_properties.json").read_text(encoding="utf-8")
        )
        prop = csv_properties["weather"]
        specs.append(
            (
                DatasetSpec(
                    dataset_id="weather",
                    family_id="weather",
                    format="csv",
                    path=weather,
                    frequency=prop["frequency"],
                    period=prop["period"],
                    timestamp_column="date",
                    expected_num_variates=prop["num_variates"],
                ),
                "new_local_family_for_separate_synthetic_confirmation",
            )
        )
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for spec, role in specs:
        files = sorted(spec.path.glob("*.arrow")) if spec.path.is_dir() else [spec.path]
        record = {
            "dataset_id": spec.dataset_id,
            "family_id": spec.family_id,
            "role": role,
            "path": str(spec.path),
            "frequency": spec.frequency,
            "period": spec.period,
            "expected_dimensions": spec.expected_num_variates,
            "source_sha256": {str(path): file_sha256(path) for path in files},
            "items": [],
        }
        try:
            items = load_dataset(spec)
            record["source_item_count"] = len(items)
            for item in items:
                if item.item_id in excluded.get(spec.dataset_id, set()):
                    continue
                item_record = {
                    "item_id": item.item_id,
                    "series_length": len(item.values),
                    "dimensions": item.values.shape[1],
                }
                try:
                    item_record["observability"] = confirmation_windows(np.isfinite(item.values))
                    item_record["status"] = "inspected"
                except ValueError as error:
                    item_record.update(status="ineligible", reason=str(error))
                record["items"].append(item_record)
            record["status"] = "inspected"
        except Exception as error:
            record.update(
                status="requires_source_review", reason=f"{type(error).__name__}: {error}"
            )
        record["eligible_windows"] = sum(
            item.get("observability", {}).get("eligible_window_count", 0)
            for item in record["items"]
        )
        record["eligible_missing_contexts"] = sum(
            item.get("observability", {}).get("eligible_missing_context_count", 0)
            for item in record["items"]
        )
        records.append(record)
        _write_json(output / (spec.dataset_id + ".json"), record)
        print(
            json.dumps(
                {
                    name: record[name]
                    for name in (
                        "dataset_id",
                        "status",
                        "eligible_windows",
                        "eligible_missing_contexts",
                    )
                }
            ),
            flush=True,
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "previous_cohort_sha256": file_sha256(args.previous_cohort),
            "properties_sha256": file_sha256(properties_path),
            "protocol": previous["protocol"],
            "datasets": records,
            "information_boundary": "source metadata and original observation masks only; no forecasts or predictive errors",
            "limits": "eligibility inventory only; an unused series in a known family is not a new family; no new confirmation run has started",
        },
    )


if __name__ == "__main__":
    main()
