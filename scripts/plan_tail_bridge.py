"""Freeze all eligible natural tails and twelve previously-used complete histories."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources
from plan_matched_replay import interleave
from tail_bridge_core import mask_tail, supported_gaps, trailing_gap

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed tail-bridge plans")
    sources = native_sources()
    source_map = {(row["cohort"], row["dataset_id"], row["item_id"]): row for row in sources}
    r19 = read_json(ROOT / "artifacts/iclr27-r19/pilot-plan-v001/manifest.json")
    groups, seen = {}, set()
    for row in r19["cases"]:
        for origin in row["selected_anchors"]:
            key = row["cohort"], row["dataset_id"], row["item_id"], origin
            if key in seen:
                continue
            seen.add(key)
            groups.setdefault(row["group_id"], []).append(
                {
                    **row,
                    "origin": origin,
                    "episode_id": f"{row['dataset_id']}|{row['item_id']}|{origin}|r20_base",
                }
            )
    bases = []
    for group in sorted(groups):
        queue = interleave(groups[group])
        bases.extend([queue.popleft() for _ in range(3)])
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, base in enumerate(bases):
        source = source_map[(base["cohort"], base["dataset_id"], base["item_id"])]
        clean = source["values"][base["origin"] - 96 : base["origin"]]
        for gap in (8, 24, 48):
            context = mask_tail(clean, gap)
            case = {
                name: base[name]
                for name in (
                    "cohort",
                    "dataset_id",
                    "family_id",
                    "item_id",
                    "origin",
                    "prefix_end",
                    "group_id",
                )
            }
            case.update(
                case_id=f"synthetic_{index:02d}_g{gap}",
                base_id=f"synthetic_{index:02d}",
                panel="synthetic",
                gap=gap,
                models=["chronos2", "timesfm2p5"],
                gaps={"chronos2": [gap, gap], "timesfm2p5": [gap, gap]},
                prior_use="R19 historical forecast evaluation",
                current_artifact=None,
            )
            path = output / "contexts" / f"{case['case_id']}.npz"
            _save_npz(path, context=context)
            case.update(
                context_path=str(path.relative_to(output)), context_sha256=file_sha256(path)
            )
            cases.append(case)
    for source in sources:
        for row in source["episodes"]:
            if not row["window"]["context_has_missing"]:
                continue
            origin = int(row["window"]["origin"])
            context = np.asarray(source["values"][origin - 96 : origin])
            cg = supported_gaps(context, True)
            tg = supported_gaps(context, False)
            if not any(tg):
                continue
            case_id = "native_" + hashlib.sha256(row["episode_id"].encode()).hexdigest()[:16]
            group = (
                "singapore"
                if source["family_id"] in ("sg_pm25", "sg_weather")
                else source["family_id"]
            )
            case = {
                name: source[name]
                for name in ("cohort", "dataset_id", "family_id", "item_id", "prefix_end")
            }
            case.update(
                case_id=case_id,
                base_id=case_id,
                origin=origin,
                group_id=group,
                panel="native_common" if any(cg) else "native_target_only",
                gap=trailing_gap(context),
                models=["chronos2", "timesfm2p5"] if any(cg) else ["timesfm2p5"],
                gaps={"chronos2": cg, "timesfm2p5": tg},
                current_artifact=str(source["root"] / row["path"]),
                current_artifact_sha256=row["sha256"],
                original_episode_id=row["episode_id"],
                prior_use="R5/R6 development scores already read",
            )
            path = output / "contexts" / f"{case_id}.npz"
            _save_npz(path, context=context)
            case.update(
                context_path=str(path.relative_to(output)), context_sha256=file_sha256(path)
            )
            cases.append(case)
    if (
        len(bases) != 12
        or sum(row["panel"] == "native_common" for row in cases) != 11
        or len(cases) != 53
    ):
        raise ValueError("the prespecified synthetic or natural tail population changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "core_module_sha256": file_sha256(ROOT / "scripts/tail_bridge_core.py"),
            "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_PROTOCOL.md"),
            "r19_plan_sha256": file_sha256(
                ROOT / "artifacts/iclr27-r19/pilot-plan-v001/manifest.json"
            ),
            "cases": cases,
            "synthetic_histories": 12,
            "native_common": 11,
            "native_timesfm_total": 17,
            "forecaster_case_pairs": 100,
            "current_future_scores_read": False,
            "limits": "previously-used development cases, including reused historical forecast targets",
        },
    )
    print(
        {
            "synthetic_histories": 12,
            "synthetic_cases": 36,
            "native_common": 11,
            "native_target_only": 6,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
