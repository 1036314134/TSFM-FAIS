"""Register the fixed new-cohort tasks before any R6 forecasting."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freeze_followup_cohort import evenly_spaced, reference_windows, target_window_sha  # noqa: E402
from prepare_native_confirmation import MECHANISMS  # noqa: E402

from tsfm_fais.data.catalog import load_manifest  # noqa: E402
from tsfm_fais.data.masking import stable_seed  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "public-root",
        "method-freeze",
        "source-control-root",
        "source-root",
        "old-prepared-root",
        "used-cohort-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the complete independent cohort freeze")
    method = json.loads(args.method_freeze.read_text(encoding="utf-8"))
    control = json.loads((args.source_control_root / "manifest.json").read_text(encoding="utf-8"))
    if control["status"] != "completed" or len(control["models"]) != 6:
        raise ValueError("freeze the matched source-future control before the cohort")
    confirmation = Path(method["identity"]["confirmation_root"])
    if confirmation.exists() and any(confirmation.rglob("*predictions*.npz")):
        raise ValueError("R6 forecasts already exist")
    inventory = json.loads((args.public_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        inventory["status"] != "completed"
        or inventory["method_freeze_sha256"] != file_sha256(args.method_freeze)
        or inventory["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("source parsing does not match the fixed method and protocol")
    if inventory["issues"]:
        raise ValueError("resolve the recorded source issues before fixing cohort membership")
    old_datasets = {
        spec.dataset_id for spec in load_manifest(ROOT / "configs/data/datasets.yaml").datasets
    }
    used = json.loads((args.used_cohort_root / "manifest.json").read_text(encoding="utf-8"))
    old_datasets |= {row["dataset_id"] for row in used["sources"]}
    original = json.loads((args.old_prepared_root / "manifest.json").read_text(encoding="utf-8"))
    old_datasets |= {row["dataset_id"] for row in original["episodes"]}
    if {row["dataset_id"] for row in inventory["sources"]} & old_datasets:
        raise ValueError("a new source ID occurs in an earlier catalog or evaluation")
    references = reference_windows(args.source_root, args.old_prepared_root)
    used_items = {(row["dataset_id"], row["item_id"]): row for row in used["sources"]}
    seen_origins = set()
    used_arrays = {}
    for task in used["tasks"]:
        origin_id = task["origin_id"]
        if origin_id in seen_origins:
            continue
        seen_origins.add(origin_id)
        item_key = (task["dataset_id"], task["item_id"])
        source = used_items[item_key]
        path = args.used_cohort_root / source["path"]
        if item_key not in used_arrays:
            if file_sha256(path) != source["sha256"]:
                raise ValueError("a previously used raw trajectory changed")
            used_arrays[item_key] = np.load(path, mmap_mode="r")
        values = used_arrays[item_key]
        origin = task["window"]["origin"]
        references.setdefault(target_window_sha(values[origin - 96 : origin + 96, :2]), []).append(
            origin_id
        )
    first_stations = sorted(
        row["item_id"] for row in inventory["sources"] if row["dataset_id"] == "beijing_multisite"
    )[:4]
    sources, tasks, exclusions, selected = [], [], [], []
    content_groups = {}
    for record in inventory["sources"]:
        path = args.public_root / record["path"]
        availability_path = args.public_root / record["eligibility_path"]
        if (
            file_sha256(path) != record["sha256"]
            or file_sha256(availability_path) != record["eligibility_sha256"]
        ):
            raise ValueError("a new trajectory or its eligibility inventory changed")
        values = np.load(path, mmap_mode="r")
        available = json.loads(availability_path.read_text(encoding="utf-8"))
        windows = [row for row in available["windows"] if row["eligible"]]
        natural = []
        for missing in (False, True):
            natural.extend(
                evenly_spaced([row for row in windows if row["context_has_missing"] == missing], 16)
            )
        dataset = record["dataset_id"]
        panel = (
            "new_native"
            if dataset == "beijing_multisite"
            else "time_grid_gap"
            if record["parsing"]["inserted_unobserved_bins"]
            else "complete_release"
        )
        candidates = [(panel, row) for row in sorted(natural, key=lambda row: row["origin"])]
        if dataset != "beijing_multisite" or record["item_id"] in first_stations:
            candidates.extend(
                ("new_synthetic", row)
                for row in evenly_spaced([row for row in windows if row["synthetic_eligible"]], 8)
            )
        accepted = []
        for group, window in candidates:
            origin = window["origin"]
            origin_id = f"{dataset}|{record['item_id']}|{origin}"
            digest = target_window_sha(values[origin - 96 : origin + 96, :2])
            if digest in references:
                exclusions.append(
                    {
                        "origin_id": origin_id,
                        "panel": group,
                        "reason": "exact_previous_target_history_and_96_future",
                        "matches": references[digest],
                    }
                )
                continue
            content_groups.setdefault(digest, set()).add(origin_id)
            conditions = (
                [("native", 0.0, 0)]
                if group != "new_synthetic"
                else [
                    (mechanism, rate, seed)
                    for mechanism in MECHANISMS
                    for rate in (0.1, 0.3, 0.5)
                    for seed in (8101, 8102)
                ]
            )
            for mechanism, rate, seed in conditions:
                tasks.append(
                    {
                        "dataset_id": dataset,
                        "family_id": record["family_id"],
                        "item_id": record["item_id"],
                        "panel": group,
                        "origin_id": origin_id,
                        "prefix_end": available["prefix_end"],
                        "series_length": len(values),
                        "dimensions": values.shape[1],
                        "period": record["period"],
                        "window": window,
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "mask_seed": seed,
                        "realization_seed": stable_seed(
                            dataset, record["item_id"], "r6_confirmation", mechanism, rate, seed
                        ),
                        "episode_id": f"{origin_id}|r6|{group}|{mechanism}|{rate}|{seed}",
                    }
                )
            accepted.append({"panel": group, "origin": origin})
        selected.append(
            {
                "dataset_id": dataset,
                "item_id": record["item_id"],
                "selected_origins_by_panel": accepted,
            }
        )
        if accepted:
            sources.append(
                {**record, "path": str(path.resolve()), "prefix_end": available["prefix_end"]}
            )
    if not tasks or len({row["episode_id"] for row in tasks}) != len(tasks):
        raise ValueError("the independent cohort is empty or has duplicate task IDs")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "schema_version": 1,
            "identity_sha256": {
                "method_freeze": file_sha256(args.method_freeze),
                "source_control": file_sha256(args.source_control_root / "manifest.json"),
                "public_inventory": file_sha256(args.public_root / "manifest.json"),
                "protocol": file_sha256(args.protocol),
                "script": file_sha256(Path(__file__)),
                "masking": file_sha256(ROOT / "src/tsfm_fais/data/masking.py"),
            },
            "sources": sources,
            "tasks": tasks,
            "task_count": len(tasks),
            "source_count": len(sources),
            "origin_count": len({row["origin_id"] for row in tasks}),
            "horizons": [96, 192],
            "context_length": 96,
            "selected_items": selected,
            "beijing_synthetic_station_rule": first_stations,
            "duplicate_exclusions": exclusions,
            "within_cohort_exact_target_duplicates": [
                sorted(ids) for ids in content_groups.values() if len(ids) > 1
            ],
            "historical_reference_window_count": len(references),
            "forecast_status_at_freeze": "not_started",
            "information_boundary": "source identity, time grids and observation masks determine inclusion; no R6 forecast errors are available",
            "limits": "one original-NA source family; nominal time-bin gaps are separate and do not imply zero-valued labels or identified sensor failure",
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "input_tasks": len(tasks),
                "items": len(sources),
                "origins": len({row["origin_id"] for row in tasks}),
                "exclusions": len(exclusions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
