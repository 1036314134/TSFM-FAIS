"""Freeze all previously used native target-missing episodes at context length 192."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources

from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-pool plans")
    sources = native_sources()
    output.mkdir(parents=True, exist_ok=True)
    cases, datasets = [], {}
    for source in sources:
        selected = []
        for episode in source["episodes"]:
            origin = int(episode["window"]["origin"])
            if np.isfinite(source["values"][origin - 96 : origin, :2]).all():
                continue
            if origin - 192 < source["prefix_end"]:
                raise ValueError("the longer context overlaps its historical fitting prefix")
            context = source["values"][origin - 192 : origin]
            case_id = hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:16]
            path = output / "contexts" / f"{case_id}.npz"
            scaler = PrefixStandardizer.fit(source["values"][: source["prefix_end"]])
            _save_npz(
                path,
                context=context,
                mean=scaler.mean,
                scale=scaler.scale,
                defaults=np.nanmedian(source["values"][: source["prefix_end"]], axis=0),
            )
            row = {
                name: source[name]
                for name in ("cohort", "dataset_id", "family_id", "item_id", "prefix_end", "period")
            }
            row.update(
                case_id=case_id,
                origin=origin,
                episode_id=episode["episode_id"],
                group_id="singapore"
                if source["family_id"] in ("sg_pm25", "sg_weather")
                else source["family_id"],
                original_path=str(source["root"] / episode["path"]),
                original_sha256=episode["sha256"],
                context_path=str(path.relative_to(output)),
                context_sha256=file_sha256(path),
                prior_use="existing R5/R6 development",
            )
            cases.append(row)
            selected.append(case_id)
        if selected:
            key = source["cohort"] + "|" + source["dataset_id"]
            if key not in datasets:
                old = read_json(source["root"] / "manifest.json")
                datasets[key] = {
                    "cohort": source["cohort"],
                    "dataset_id": source["dataset_id"],
                    "fit_items": source["dataset"]["fit_items"],
                    "old_prepared_root": str(source["root"]),
                    "old_prepared_sha256": file_sha256(source["root"] / "manifest.json"),
                    "training_missing_rates": old["identity"]["training_missing_rates"],
                    "training_mask_seeds": old["identity"]["training_mask_seeds"],
                }
    if len(cases) != 301 or len({r["group_id"] for r in cases}) != 8:
        raise ValueError("the registered existing-data expansion population changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R25_LONG_POOL_PROTOCOL.md"),
            "cases": cases,
            "datasets": list(datasets.values()),
            "current_future_scores_read": False,
            "independent_confirmation": False,
        },
    )
    print({"cases": len(cases), "datasets": len(datasets), "groups": 8}, flush=True)


if __name__ == "__main__":
    main()
