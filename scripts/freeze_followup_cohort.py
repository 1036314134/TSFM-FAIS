"""Freeze later follow-up windows using source identity and observation masks."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_r3_confirmation_sources import confirmation_windows  # noqa: E402
from freeze_followup_portfolios import assert_unstarted_confirmation  # noqa: E402
from prepare_native_confirmation import MECHANISMS  # noqa: E402

from tsfm_fais.data.catalog import DatasetSpec, load_manifest  # noqa: E402
from tsfm_fais.data.loaders import load_dataset  # noqa: E402
from tsfm_fais.data.masking import stable_seed  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def evenly_spaced(rows, maximum):
    if maximum < 1:
        raise ValueError("positive sampling budget required")
    positions = np.linspace(0, len(rows) - 1, min(len(rows), maximum), dtype=int)
    return [rows[index] for index in positions]


def select_windows(observability, panel, observed):
    eligible = [row for row in observability["windows"] if row["eligible"]]
    if panel == "held_items":
        return eligible
    if panel == "new_native":
        rows = []
        for missing in (False, True):
            rows.extend(
                evenly_spaced(
                    [row for row in eligible if row["context_has_missing"] == missing], 16
                )
            )
        return sorted(rows, key=lambda row: row["origin"])
    if panel == "new_synthetic":
        eligible = [
            row
            for row in eligible
            if observed[row["origin"] - 96 : row["origin"]].all() and row["complete_target_future"]
        ]
        return evenly_spaced(eligible, 8)
    raise ValueError("unknown registered panel")


def target_window_sha(values):
    values = np.asarray(values, dtype="<f8").copy()
    if values.shape != (192, 2) or np.isinf(values).any():
        raise ValueError("duplicate comparison requires two 192-step target trajectories")
    observed = np.isfinite(values)
    values[~observed] = 0
    values[values == 0] = 0  # Canonicalize signed zero.
    return hashlib.sha256(observed.tobytes() + values.tobytes()).hexdigest()


def reference_windows(source_root, old_root):
    references = {}
    for root, manifest_name, context_name in (
        (source_root, "episodes_manifest.json", "clean_context"),
        (old_root, "manifest.json", "context"),
    ):
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        seen = set()
        for row in manifest["episodes"]:
            if row["origin_id"] in seen:
                continue
            seen.add(row["origin_id"])
            path = root / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError("a historical reference window changed")
            with np.load(path, allow_pickle=False) as saved:
                values = np.concatenate([saved[context_name][:, :2], saved["future"][:, :2]])
            references.setdefault(target_window_sha(values), []).append(row["origin_id"])
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inventory-root",
        "public-root",
        "source-root",
        "old-prepared-root",
        "previous-cohort",
        "source-bundle",
        "confirmation-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the registered follow-up cohort")
    assert_unstarted_confirmation(args.confirmation_root)
    inventory = json.loads((args.inventory_root / "manifest.json").read_text(encoding="utf-8"))
    public = json.loads((args.public_root / "manifest.json").read_text(encoding="utf-8"))
    previous = json.loads(args.previous_cohort.read_text(encoding="utf-8"))
    bundle = json.loads((args.source_bundle / "manifest.json").read_text(encoding="utf-8"))
    if any(item["status"] != "completed" for item in (inventory, public, bundle)):
        raise ValueError("complete the source model freeze and raw-data inventories first")
    if Path(bundle["identity"]["confirmation_root"]).resolve() != args.confirmation_root.resolve():
        raise ValueError("use the confirmation directory fixed with the source models")
    previous_items = {(row["dataset_id"], row["item_id"]) for row in previous["tasks"]}
    references = reference_windows(args.source_root, args.old_prepared_root)
    catalog = load_manifest(ROOT / "configs/data/datasets.yaml")
    prior_datasets = {spec.dataset_id for spec in catalog.datasets} | {
        row["dataset_id"] for row in previous["tasks"]
    }
    output.mkdir(parents=True, exist_ok=True)
    sources, tasks, exclusions, inventories = [], [], [], []
    current_hashes = {}

    def register(metadata, values, panel, provenance):
        dataset, item = metadata["dataset_id"], metadata["item_id"]
        if (dataset, item) in previous_items:
            raise ValueError("a previous confirmation item re-entered the follow-up")
        if panel.startswith("new_") and dataset in prior_datasets:
            raise ValueError("a declared new source occurs in an existing evaluation catalog")
        observed = np.isfinite(values)
        eligibility = confirmation_windows(observed)
        windows = select_windows(eligibility, panel, observed)
        valid = []
        for window in windows:
            origin = window["origin"]
            digest = target_window_sha(values[origin - 96 : origin + 96, :2])
            identifier = f"{dataset}|{item}|{origin}"
            if digest in references:
                exclusions.append(
                    {
                        "origin_id": identifier,
                        "reason": "exact_previous_target_window",
                        "matches": references[digest],
                    }
                )
                continue
            current_hashes.setdefault(digest, []).append(identifier)
            valid.append(window)
        inventories.append(
            {
                "dataset_id": dataset,
                "item_id": item,
                "panel": panel,
                "eligible_windows": eligibility["eligible_window_count"],
                "eligible_missing_contexts": eligibility["eligible_missing_context_count"],
                "sampled_before_duplicate_check": len(windows),
                "registered_origins": len(valid),
            }
        )
        if not valid:
            return
        key = hashlib.sha256(f"{dataset}|{item}".encode()).hexdigest()[:24]
        path = output / "series" / f"{key}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, values, allow_pickle=False)
        sources.append(
            {
                **metadata,
                "panel": panel,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "shape": list(values.shape),
                "prefix_end": eligibility["prefix_end"],
                "provenance": provenance,
            }
        )
        conditions = (
            [("native", 0.0, 0)]
            if panel != "new_synthetic"
            else [
                (mechanism, rate, seed)
                for mechanism in MECHANISMS
                for rate in (0.1, 0.3, 0.5)
                for seed in (7101, 7102)
            ]
        )
        for window in valid:
            origin = window["origin"]
            for mechanism, rate, seed in conditions:
                tasks.append(
                    {
                        "dataset_id": dataset,
                        "family_id": metadata["family_id"],
                        "item_id": item,
                        "panel": panel,
                        "series_length": len(values),
                        "dimensions": values.shape[1],
                        "prefix_end": eligibility["prefix_end"],
                        "period": metadata["period"],
                        "window": window,
                        "origin_id": f"{dataset}|{item}|{origin}",
                        "episode_id": f"{dataset}|{item}|{origin}|followup|{mechanism}|{rate}|{seed}",
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "mask_seed": seed,
                        "realization_seed": stable_seed(
                            dataset, item, "followup", mechanism, rate, seed
                        ),
                    }
                )

    for row in inventory["datasets"]:
        if row["dataset_id"] not in {
            "Water_Quality_Darwin_15T",
            "SG_Weather_D",
            "Smart_Manufacturing_H",
            "weather",
        }:
            continue
        for name, sha in row["source_sha256"].items():
            if file_sha256(Path(name)) != sha:
                raise ValueError("a local raw source changed after inventory")
        spec = DatasetSpec(
            dataset_id=row["dataset_id"],
            family_id=row["family_id"],
            format="csv" if row["dataset_id"] == "weather" else "arrow",
            path=Path(row["path"]),
            frequency=row["frequency"],
            period=row["period"],
            expected_num_variates=row["expected_dimensions"],
            timestamp_column="date" if row["dataset_id"] == "weather" else None,
            allow_implicit_regular_time=True,
        )
        permitted = {item["item_id"] for item in row["items"]}
        for item in load_dataset(spec):
            if item.item_id not in permitted:
                continue
            register(
                {
                    "dataset_id": spec.dataset_id,
                    "family_id": spec.family_id,
                    "item_id": item.item_id,
                    "frequency": item.freq,
                    "period": spec.period,
                    "columns": list(item.variate_names),
                    "start": None if str(item.start) == "NaT" else item.start.isoformat(),
                },
                item.values,
                "new_synthetic" if spec.dataset_id == "weather" else "held_items",
                row["source_sha256"],
            )
    for row in public["datasets"]:
        path = args.public_root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("a parsed public trajectory changed")
        register(
            {
                key: row[key]
                for key in (
                    "dataset_id",
                    "family_id",
                    "item_id",
                    "frequency",
                    "period",
                    "columns",
                    "start",
                )
            },
            np.load(path, mmap_mode="r"),
            "new_synthetic" if row["dataset_id"] == "solar_alabama" else "new_native",
            row["source"],
        )
    if not tasks or len({row["episode_id"] for row in tasks}) != len(tasks):
        raise ValueError("the cohort is empty or contains repeated task IDs")
    assert_unstarted_confirmation(args.confirmation_root)
    identity_paths = {
        "inventory": args.inventory_root / "manifest.json",
        "public": args.public_root / "manifest.json",
        "source": args.source_root / "episodes_manifest.json",
        "previous": args.previous_cohort,
        "source_bundle": args.source_bundle / "manifest.json",
        "protocol": args.protocol,
        "script": Path(__file__),
        "masking": ROOT / "src/tsfm_fais/data/masking.py",
    }
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity_sha256": {key: file_sha256(path) for key, path in identity_paths.items()},
            "sources": sources,
            "tasks": tasks,
            "task_count": len(tasks),
            "source_count": len(sources),
            "origin_count": len({row["origin_id"] for row in tasks}),
            "inventories": inventories,
            "duplicate_exclusions": exclusions,
            "within_followup_exact_target_duplicates": [
                rows for rows in current_hashes.values() if len(rows) > 1
            ],
            "reference_target_window_count": len(references),
            "forecast_status_at_freeze": "not_started",
            "information_boundary": "sampling uses observation masks and exact-data identity only; no predictive errors",
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "tasks": len(tasks),
                "sources": len(sources),
                "duplicate_exclusions": len(exclusions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
