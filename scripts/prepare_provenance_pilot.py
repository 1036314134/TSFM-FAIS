"""Freeze a source-balanced, previously-used native development panel and cached imputations."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
from audit_tail_bridge import fit_cutoff
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources, timestamp
from plan_matched_replay import interleave

from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed pilot preparation")
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    eligible = {}
    for source in sources:
        group = (
            "singapore" if source["family_id"] in ("sg_weather", "sg_pm25") else source["family_id"]
        )
        for episode in source["episodes"]:
            origin = int(episode["window"]["origin"])
            context = source["values"][origin - 96 : origin]
            if np.isfinite(context[:, :2]).all():
                continue
            row = {
                name: source[name]
                for name in ("cohort", "dataset_id", "family_id", "item_id", "prefix_end")
            }
            row.update(
                origin=origin,
                group_id=group,
                episode_id=episode["episode_id"],
                original_path=str(source["root"] / episode["path"]),
                original_sha256=episode["sha256"],
            )
            eligible.setdefault(group, []).append(row)
    selected = []
    for group in sorted(eligible):
        queue = interleave(eligible[group])
        selected.extend(queue.popleft() for _ in range(min(3, len(queue))))
    if not 4 <= len(eligible) <= 8 or len(selected) > 24:
        raise ValueError("the bounded source-balanced pilot population changed")
    motm_root = ROOT / "artifacts/iclr27-r5/native-confirmation-v001/motm"
    motm = {r["episode_id"]: r for r in read_json(motm_root / "manifest.json")["episodes"]}
    output.mkdir(parents=True, exist_ok=True)
    cases, fits = [], {}
    for row in selected:
        key = row["cohort"], row["dataset_id"], row["item_id"]
        source = source_map[key]
        records = []
        for fit in source["dataset"]["imputers"]:
            path = Path(fit["path"])
            if not path.is_absolute():
                path = source["root"] / path
            records.append(
                {"candidate_id": fit["candidate_id"], "path": str(path), "sha256": fit["sha256"]}
            )
        entry = {"frozen_neural_imputers": records}
        if key[:2] not in fits:
            fits[key[:2]] = fit_cutoff(source, sources, entry)
        if timestamp(source, row["origin"] - 96) < fits[key[:2]]:
            raise ValueError("a current context precedes the frozen fit cutoff")
        path = Path(row["original_path"])
        if file_sha256(path) != row["original_sha256"]:
            raise ValueError("an original prepared episode changed")
        with np.load(path, allow_pickle=False) as old:
            context = old["context"]
            np.testing.assert_array_equal(
                context, source["values"][row["origin"] - 96 : row["origin"]]
            )
            candidates, ids = old["candidate_values"], old["candidate_ids"].tolist()
            extra = old["motm_values"] if row["cohort"] == "r6_native" else None
        extra_record = None
        if extra is None:
            extra_record = motm[row["episode_id"]]
            path = motm_root / extra_record["path"]
            if file_sha256(path) != extra_record["sha256"]:
                raise ValueError("a frozen MoTM imputation changed")
            with np.load(path, allow_pickle=False) as old:
                extra = old["values"]
        values = np.concatenate([candidates, extra[None]])
        if not np.isfinite(values).all():
            raise ValueError("a cached imputation is incomplete")
        for candidate in values:
            np.testing.assert_array_equal(
                candidate[np.isfinite(context)], context[np.isfinite(context)]
            )
        scaler = PrefixStandardizer.fit(source["values"][: source["prefix_end"]])
        case_id = hashlib.sha256(row["episode_id"].encode()).hexdigest()[:16]
        path = output / "cases" / f"{case_id}.npz"
        _save_npz(
            path,
            context=context,
            candidate_values=values,
            candidate_ids=np.asarray([*ids, "motm_reference"]),
            mean=scaler.mean,
            scale=scaler.scale,
        )
        cases.append(
            {
                **row,
                **entry,
                "case_id": case_id,
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "latest_neural_training_boundary": str(fits[key[:2]]),
                "motm_record": extra_record,
                "prior_use": "R5/R6 development inputs and outcomes previously used",
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "plan_sha256": file_sha256(ROOT / "docs/iclr2027/R21_PROVENANCE_INTERFACE_PLAN.md"),
            "cases": cases,
            "eligible_counts": {g: len(rows) for g, rows in eligible.items()},
            "selection": "sorted source groups; interleave series and ascending origins; at most three per group",
            "imputer_fits": 0,
            "new_motm_context_optimizations": 0,
            "new_future_errors_read": False,
            "independent_confirmation": False,
        },
    )
    print(
        {"cases": len(cases), "eligible_counts": {g: len(rows) for g, rows in eligible.items()}},
        flush=True,
    )


if __name__ == "__main__":
    main()
