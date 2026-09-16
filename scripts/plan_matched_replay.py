"""Select an outcome-independent, balanced pilot from exact replay support."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from latent_source_inputs import ROOT, read_json
from matched_replay_core import mask_rules
from matched_replay_sources import native_sources

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def interleave(rows):
    items = defaultdict(list)
    for row in rows:
        items[(row["dataset_id"], row["item_id"])].append(row)
    queues = [
        deque(sorted(items[key], key=lambda row: (row["origin"], row["episode_id"])))
        for key in sorted(items)
    ]
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return deque(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed pilot plans")
    preflight = ROOT / "artifacts/iclr27-r19/preflight-v001"
    manifest = read_json(preflight / "manifest.json")
    if manifest["status"] != "completed" or manifest["support_sha256"] != file_sha256(
        preflight / "support.json"
    ):
        raise ValueError("the observed support inventory changed")
    sources = native_sources()
    source_map = {(row["cohort"], row["dataset_id"], row["item_id"]): row for row in sources}
    eligible, exclusions = [], []
    for row in read_json(preflight / "support.json"):
        if row["history_budget"] != 4096:
            continue
        if not row["supported"]:
            exclusions.append(
                {"episode_id": row["episode_id"], "reason": "fewer_than_eight_exact_anchors"}
            )
            continue
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        context = np.asarray(source["values"][row["origin"] - 96 : row["origin"]])
        masks, shift = mask_rules(~np.isfinite(context), row["episode_id"])
        if masks is None:
            exclusions.append({"episode_id": row["episode_id"], **shift})
            continue
        group = "singapore" if row["family_id"] in ("sg_pm25", "sg_weather") else row["family_id"]
        eligible.append({**row, "group_id": group, "mask_definition": shift})
    groups = {
        name: interleave([row for row in eligible if row["group_id"] == name])
        for name in sorted({row["group_id"] for row in eligible})
    }
    selected = []
    while any(groups.values()) and len(selected) < 24:
        for queue in groups.values():
            if queue and len(selected) < 24:
                selected.append(queue.popleft())
    if len(selected) != 24 or len({row["group_id"] for row in selected}) < 4:
        raise ValueError("insufficient exact support for the registered four-group pilot")
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        source = source_map[(row["cohort"], row["dataset_id"], row["item_id"])]
        current = next(
            entry for entry in source["episodes"] if entry["episode_id"] == row["episode_id"]
        )
        context = np.asarray(source["values"][row["origin"] - 96 : row["origin"]])
        masks, shift = mask_rules(~np.isfinite(context), row["episode_id"])
        path = output / "masks" / f"case_{index:02d}.npz"
        _save_npz(path, context=context, masks=masks)
        row.update(
            case_id=f"case_{index:02d}",
            mask_path=str(path.relative_to(output)),
            mask_sha256=file_sha256(path),
            current_artifact=str(source["root"] / current["path"]),
            current_artifact_sha256=current["sha256"],
            input_root=str(source["root"]),
            selected_anchors=row["selected"],
            start=source["start"],
            frequency=source["frequency"],
            period=source["period"],
            long_start=max(source["prefix_end"], row["origin"] - 4096),
            current_future_reused_development_data=True,
        )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "core_module_sha256": file_sha256(ROOT / "scripts/matched_replay_core.py"),
            "preflight_sha256": file_sha256(preflight / "manifest.json"),
            "cases": selected,
            "excluded": exclusions,
            "supported_with_distinct_position_control": len(eligible),
            "source_groups": sorted(groups),
            "history_budget": 4096,
            "anchors": 8,
            "new_forecaster_calls": 0,
            "current_future_values_read": False,
            "limits": "purposive previously-used development pilot; no independent confirmation",
        },
    )
    print(
        {group: sum(row["group_id"] == group for row in selected) for group in groups}, flush=True
    )


if __name__ == "__main__":
    main()
