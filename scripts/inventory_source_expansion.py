"""Choose additional nonoverlapping source histories before evaluating any errors."""

import argparse
from pathlib import Path

import numpy as np
from latent_source_inputs import ROOT, read_json

from tsfm_fais.data import load_dataset, load_manifest
from tsfm_fais.utility_experiment import (
    _write_json,
    file_sha256,
    load_utility_config,
    purged_origins,
)


def nested_training_origins(pool, original, maximum=64):
    pool, original = tuple(sorted(set(pool))), tuple(sorted(set(original)))
    if not set(original).issubset(pool):
        raise ValueError("original source origins are outside the unchanged time grid")
    available = [origin for origin in pool if origin not in original]
    count = max(0, min(maximum, len(pool)) - len(original))
    indices = np.linspace(0, len(available) - 1, count, dtype=int) if count else []
    return tuple(sorted([*original, *(available[index] for index in indices)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/iclr27-r3/development_expanded.yaml"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve a completed source inventory")
    config = load_utility_config(args.config)
    source = read_json(config.output_root / "episodes_manifest.json")
    data = load_manifest(config.data_manifest)
    active = {row["dataset_id"] for row in source["episodes"]}
    records = []
    for dataset_id in config.dataset_ids:
        if dataset_id not in active:
            continue
        spec = data.get(dataset_id)
        original_dataset = next(
            row for row in source["datasets"] if row["dataset_id"] == dataset_id
        )
        sources = sorted(spec.path.rglob("*")) if spec.path.is_dir() else [spec.path]
        hashes = {str(path): file_sha256(path) for path in sources if path.is_file()}
        if sorted((Path(path).name, digest) for path, digest in hashes.items()) != sorted(
            (Path(path).name, digest) for path, digest in original_dataset["sources"].items()
        ):
            raise ValueError("the raw source files changed")
        for item in load_dataset(spec)[: config.max_items]:
            previous = [
                row
                for row in source["episodes"]
                if row["dataset_id"] == dataset_id
                and row["item_id"] == item.item_id
                and row["mask_seed"] == 6101
            ]
            old_train = sorted({row["origin"] for row in previous if row["split"] == "train"})
            old_validation = sorted(
                {row["origin"] for row in previous if row["split"] == "validation"}
            )
            if not old_train:
                continue
            prefix, limits = purged_origins(len(item.values), config)
            if limits["train"] != tuple(old_train) or limits["validation"] != tuple(old_validation):
                raise ValueError("source length or original temporal definitions changed")
            pool = range(
                prefix + 96,
                int(np.floor(len(item.values) * config.temporal_boundary)) - 96 + 1,
                192,
            )
            expanded = nested_training_origins(pool, old_train)
            added = sorted(set(expanded) - set(old_train))
            for origin in added:
                if (
                    not np.isfinite(item.values[origin - 96 : origin]).all()
                    or not np.isfinite(item.values[origin : origin + 96, :2]).all()
                ):
                    raise ValueError("a registered added training history is not complete")
            records.append(
                {
                    "dataset_id": dataset_id,
                    "family_id": spec.family_id,
                    "item_id": item.item_id,
                    "length": len(item.values),
                    "prefix_end": prefix,
                    "old_train": old_train,
                    "validation": old_validation,
                    "expanded_train": list(expanded),
                    "added_train": added,
                    "sources": hashes,
                }
            )
            print(f"{dataset_id}: {len(old_train)} -> {len(expanded)} source histories", flush=True)
    counts = {
        name: sum(len(row[key]) for row in records)
        for name, key in (
            ("original_training_origins", "old_train"),
            ("expanded_training_origins", "expanded_train"),
            ("added_training_origins", "added_train"),
            ("validation_origins", "validation"),
        )
    }
    if counts["original_training_origins"] != 165 or counts["validation_origins"] != 52:
        raise ValueError("source population coverage changed")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "config_sha256": file_sha256(args.config),
            "source_sha256": file_sha256(config.output_root / "episodes_manifest.json"),
            "selection_uses_forecast_errors": False,
            "maximum_origins_per_item": 64,
            "counts": counts,
            "items": records,
        },
    )


if __name__ == "__main__":
    main()
