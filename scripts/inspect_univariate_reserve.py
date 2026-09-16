"""Inspect univariate native-missing source availability and recorded prior-use mentions."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json

from tsfm_fais.data.loaders import _read_arrow_table
from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed reserve inspections")
    source_root = ROOT.parent / "TSFM-SPImpute/data/original/arrow"
    properties = read_json(source_root / "dataset_properties.json")
    names = ("CPHL_30T", "ECDC_COVID_D", "SG_Carpark_15T")
    families = ("CPHL", "ECDC_COVID", "SG_Carpark")
    checked, mentions = [], []
    skip = {
        "queries",
        "cases",
        "predictions",
        "checkpoints",
        "models",
        "__pycache__",
        "imputer_artifacts",
        "latent",
        "features",
    }
    for directory, children, files in os.walk(ROOT / "artifacts"):
        children[:] = [name for name in children if name not in skip]
        parent = Path(directory)
        if parent == ROOT / "artifacts/iclr27-r22":
            children[:] = []
            continue
        for filename in files:
            if filename not in (
                "manifest.json",
                "episodes_manifest.json",
                "cohort.json",
                "identity.json",
            ):
                continue
            path = parent / filename
            text = path.read_text(encoding="utf-8-sig")
            record = {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            checked.append(record)
            found = [name for name in families if name.lower() in text.lower()]
            if found:
                mentions.append({**record, "families_mentioned": found})
    datasets = []
    for name in names:
        family, frequency = name.rsplit("_", 1)
        directory = source_root / family / frequency
        if properties[name]["num_variates"] != 1:
            raise ValueError("the declared single-variate source changed")
        trajectories = []
        for path in sorted(directory.glob("*.arrow")):
            for raw in _read_arrow_table(path).to_pylist():
                target = np.asarray(raw["target"], dtype=float)
                if target.ndim not in (1, 2) or (target.ndim == 2 and min(target.shape) != 1):
                    raise ValueError(
                        "reserve inventory requires an actual single-variate trajectory"
                    )
                trajectories.append(
                    SimpleNamespace(
                        item_id=str(raw.get("item_id", len(trajectories))),
                        values=target.reshape(-1, 1),
                        start=raw["start"],
                        freq=raw.get("freq") or frequency,
                    )
                )
        items = []
        for item in trajectories:
            observed = np.isfinite(item.values[:, 0])
            length = len(observed)
            prefix = max(96, int(length * 0.2))
            start = max(prefix + 96, int(length * 0.6) + 96)
            windows = []
            for origin in range(start, length - 96 + 1, 192):
                context = observed[origin - 96 : origin]
                future_count = int(observed[origin : origin + 96].sum())
                valid = int(observed[:prefix].sum()) >= 2 and future_count >= 48
                windows.append(
                    {
                        "origin": origin,
                        "eligible": valid,
                        "history_missing_count": int((~context).sum()),
                        "future_observed_count": future_count,
                        "original_time_grid_preserved": True,
                    }
                )
            items.append(
                {
                    "item_id": item.item_id,
                    "length": length,
                    "dimensions": 1,
                    "start": str(item.start),
                    "frequency": item.freq,
                    "prefix_end": prefix,
                    "prefix_observed": int(observed[:prefix].sum()),
                    "eligible_windows": sum(w["eligible"] for w in windows),
                    "eligible_native_missing_windows": sum(
                        w["eligible"] and w["history_missing_count"] > 0 for w in windows
                    ),
                    "windows": windows,
                }
            )
        datasets.append(
            {
                "dataset_id": name,
                "proposed_dependency_group": "singapore"
                if family == "SG_Carpark"
                else "source_lineage_not_yet_verified",
                "path": str(directory),
                "source_files": [
                    {"path": str(p), "sha256": file_sha256(p)}
                    for p in sorted(directory.glob("*.arrow"))
                ],
                "items": items,
                "prior_metadata_mentions": [
                    r for r in mentions if family in r["families_mentioned"]
                ],
                "status": "availability_checked_reserve_candidate_only",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "property_source_sha256": file_sha256(source_root / "dataset_properties.json"),
            "scanned_metadata": checked,
            "metadata_mentions": mentions,
            "datasets": datasets,
            "forecast_calls": 0,
            "imputer_fits": 0,
            "future_error_values_read": False,
            "independent_confirmation_certified": False,
            "limits": "Repository metadata scan and raw observation-mask eligibility only. Source lineage, unregistered prior usage and model pretraining overlap remain unverified. Univariate evaluation requires a one-target contract.",
        },
    )
    print(
        json.dumps(
            {
                "metadata_files": len(checked),
                "prior_mentions": len(mentions),
                "datasets": [
                    {
                        "dataset_id": d["dataset_id"],
                        "items": len(d["items"]),
                        "eligible_windows": sum(i["eligible_windows"] for i in d["items"]),
                        "eligible_native_missing_windows": sum(
                            i["eligible_native_missing_windows"] for i in d["items"]
                        ),
                    }
                    for d in datasets
                ],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
