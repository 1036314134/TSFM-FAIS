"""Read verified native trajectories and their existing per-prefix imputer provenance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401 - initialize Arrow before importing Torch helpers.
from latent_source_inputs import ROOT, read_json

from tsfm_fais.data.catalog import DatasetSpec
from tsfm_fais.data.loaders import load_dataset
from tsfm_fais.utility_experiment import file_sha256


def relocated_source(path):
    path = Path(path)
    if path.exists():
        return path
    changed = Path(
        str(path)
        .replace("data\\Origin\\", "data\\original\\")
        .replace("data/Origin/", "data/original/")
    )
    if not changed.exists():
        raise FileNotFoundError(path)
    return changed


def native_sources():
    result = []
    legacy_root = ROOT / "artifacts/iclr27-r5/native-confirmation-v001/prepared"
    legacy = read_json(legacy_root / "manifest.json")
    for dataset in legacy["datasets"]:
        rows = [row for row in legacy["episodes"] if row["dataset_id"] == dataset["dataset_id"]]
        first = rows[0]
        paths = []
        provenance = []
        for name, digest in dataset["source_sha256"].items():
            path = relocated_source(name)
            if file_sha256(path) != digest:
                raise ValueError("a relocated native source changed")
            paths.append(path)
            provenance.append(
                {"registered_path": name, "resolved_path": str(path), "sha256": digest}
            )
        spec = DatasetSpec(
            dataset_id=dataset["dataset_id"],
            family_id=first["family_id"],
            format="arrow",
            path=paths[0].parent,
            frequency=dataset["dataset_id"].rsplit("_", 1)[1],
            period=first["period"],
            expected_num_variates=first["dimensions"],
            missingness="native",
            provenance="source_native_missing",
            allow_implicit_regular_time=True,
        )
        items = {item.item_id: item for item in load_dataset(spec)}
        for item_id in sorted({row["item_id"] for row in rows}):
            item = items[item_id]
            selected = [row for row in rows if row["item_id"] == item_id]
            result.append(
                {
                    "cohort": "legacy_native",
                    "root": legacy_root,
                    "dataset": dataset,
                    "dataset_id": dataset["dataset_id"],
                    "family_id": first["family_id"],
                    "item_id": item_id,
                    "values": item.values,
                    "start": str(item.start),
                    "frequency": item.frequency,
                    "columns": list(item.columns),
                    "prefix_end": selected[0]["prefix_end"],
                    "period": first["period"],
                    "episodes": selected,
                    "source_files": provenance,
                    "unused_series_in_loaded_dataset": sorted(
                        set(items) - set(dataset["fit_items"])
                    ),
                }
            )
    r6_root = ROOT / "artifacts/iclr27-r6/confirmation-v001/prepared"
    r6 = read_json(r6_root / "manifest.json")
    cohort = read_json(ROOT / "artifacts/iclr27-r6/cohort-v001/manifest.json")
    for source in cohort["sources"]:
        rows = [
            row
            for row in r6["episodes"]
            if row["dataset_id"] == source["dataset_id"]
            and row["item_id"] == source["item_id"]
            and row["mechanism"] == "native"
        ]
        if not rows:
            continue
        path = Path(source["path"])
        if file_sha256(path) != source["sha256"]:
            raise ValueError("an R6 normalized trajectory changed")
        dataset = next(row for row in r6["datasets"] if row["dataset_id"] == source["dataset_id"])
        result.append(
            {
                "cohort": "r6_native",
                "root": r6_root,
                "dataset": dataset,
                "dataset_id": source["dataset_id"],
                "family_id": source["family_id"],
                "item_id": source["item_id"],
                "values": np.load(path, mmap_mode="r"),
                "start": source["start"],
                "frequency": source["frequency"],
                "columns": source["columns"],
                "prefix_end": source["prefix_end"],
                "period": source["period"],
                "episodes": rows,
                "source_files": [
                    {
                        "registered_path": str(path),
                        "resolved_path": str(path),
                        "sha256": source["sha256"],
                    }
                ],
                "unused_series_in_loaded_dataset": [],
            }
        )
    return result


def timestamp(source, position):
    return pd.Timestamp(source["start"]) + position * pd.tseries.frequencies.to_offset(
        source["frequency"]
    )


def anchor_support(values, current_mask, origin, prefix_end, history_budget):
    complete, mask_compatible, eligible = [], [], []
    for past in range(origin - 96, max(prefix_end + 95, origin - history_budget + 95), -96):
        context = values[past - 96 : past]
        observed_future = np.isfinite(values[past : past + 96, :2])
        if (observed_future.sum(0) < 48).any():
            continue
        eligible.append(past)
        missing = ~np.isfinite(context)
        if not (missing & ~current_mask).any():
            mask_compatible.append(past)
        if not missing.any():
            complete.append(past)
    return {
        "complete_context_origins": complete,
        "mask_compatible_origins": mask_compatible,
        "future_observed_origins": eligible,
        "selected": complete[:8] if len(complete) >= 8 else [],
    }
