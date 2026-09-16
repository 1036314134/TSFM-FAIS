"""Freeze the older prior context and equal-information direct context."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning inputs")
    old_root = ROOT / "artifacts/iclr27-r21/provenance-inputs-v001"
    old = read_json(old_root / "manifest.json")
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in old["cases"]:
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        if row["origin"] - 192 < source["prefix_end"]:
            raise ValueError("the prefix statistics extend into the earlier prior context")
        if file_sha256(old_root / row["path"]) != row["sha256"]:
            raise ValueError("the previous fixed current context changed")
        full = source["values"][row["origin"] - 192 : row["origin"]]
        with np.load(old_root / row["path"], allow_pickle=False) as previous:
            np.testing.assert_array_equal(full[96:], previous["context"])
            target = output / "cases" / f"{row['case_id']}.npz"
            _save_npz(
                target,
                prior_context=full[:96],
                current_context=full[96:],
                direct_context=full,
                defaults=np.nanmedian(source["values"][: source["prefix_end"]], axis=0),
                mean=previous["mean"],
                scale=previous["scale"],
            )
        cases.append(
            {
                **row,
                "old_input_path": str(old_root / row["path"]),
                "old_input_sha256": row["sha256"],
                "path": str(target.relative_to(output)),
                "sha256": file_sha256(target),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "old_inputs_sha256": file_sha256(old_root / "manifest.json"),
            "protocol_sha256": file_sha256(
                ROOT / "docs/iclr2027/R23_OBSERVATION_CONDITIONING_PROTOCOL.md"
            ),
            "cases": cases,
            "current_future_read": False,
            "imputer_fits": 0,
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
