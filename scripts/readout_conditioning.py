"""Score all R28 forecasts only after the complete registered panel is frozen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from conditioning_core import BASE, INPUTS, ROOT, load_inputs, population
from evaluate_matched_replay import aggregate_groups, score_points

from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning scores")
    forecasts = BASE / "conditioning-forecasts-v001"
    manifest = json.loads((forecasts / "manifest.json").read_text(encoding="utf-8"))
    metadata = {r["case_id"]: r for r in population()}
    if (
        manifest["status"] != "completed"
        or manifest["smoke"]
        or {r["case_id"] for r in manifest["cases"]} != set(metadata)
    ):
        raise ValueError("freeze all 301 registered predictions before reading outcomes")
    scores, targets = [], []
    for item in manifest["cases"]:
        row = metadata[item["case_id"]]
        data = load_inputs(row)
        for path, digest in (
            (forecasts / item["path"], item["sha256"]),
            (Path(row["original_path"]), row["original_sha256"]),
        ):
            if file_sha256(path) != digest:
                raise ValueError("frozen prediction or original future changed")
        with np.load(row["original_path"], allow_pickle=False) as original:
            truth = original["future"][:96, :2]
            np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96, :2])
        with np.load(forecasts / item["path"], allow_pickle=False) as prediction:
            points, names = prediction["points"], prediction["methods"].tolist()
        mae, mse = score_points(points, (truth - data["mean"][:2]) / data["scale"][:2])
        recent_observed = np.isfinite(data["context"][96:])
        missing_times = (~recent_observed[:, :2]).any(1)
        internal_gap = any(
            (
                (~recent_observed[:, slot])
                & (np.arange(96) < np.flatnonzero(recent_observed[:, slot]).max(initial=-1))
            ).any()
            for slot in (0, 1)
        )
        aux_rate = (
            float(recent_observed[missing_times, 2:].mean())
            if recent_observed.shape[1] > 2
            else None
        )
        for i, name in enumerate(names):
            record = {
                k: row[k]
                for k in (
                    "case_id",
                    "episode_id",
                    "group_id",
                    "family_id",
                    "dataset_id",
                    "item_id",
                    "origin",
                )
            }
            rate = item["recent_target_missing_rate"]
            record.update(
                model_id="chronos2",
                method=name,
                mae=float(mae[i].mean()),
                mse=float(mse[i].mean()),
                missing_rate_bin="(0,.1]"
                if rate <= 0.1
                else "(.1,.3]"
                if rate <= 0.3
                else "(.3,1]",
                target_tail_gap=item["target_tail_gap"],
                target_internal_gap=bool(internal_gap),
                concurrent_aux_observed_rate=aux_rate,
                fallback=name in item["fallback"],
            )
            scores.append(record)
            for slot in (0, 1):
                targets.append(
                    {
                        **record,
                        "target_slot": slot,
                        "observed_count": int(np.isfinite(truth[:, slot]).sum()),
                        "mae": float(mae[i, slot]),
                        "mse": float(mse[i, slot]),
                    }
                )
    if len(scores) != 12341 or len(targets) != 24682:
        raise ValueError("registered score counts differ")
    frame = pd.DataFrame(scores)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    tables = aggregate_groups(frame)
    for name, table in zip(("series", "datasets", "groups", "summary"), tables, strict=True):
        table.to_csv(output / f"{name}.csv", index=False)
    leave_out = []
    for group in sorted(frame.group_id.unique()):
        table = (
            tables[2]
            .query("group_id != @group")
            .groupby(["model_id", "method"])[["mae", "mse"]]
            .mean()
            .reset_index()
        )
        table["omitted_group"] = group
        leave_out.append(table)
    pd.concat(leave_out, ignore_index=True).to_csv(output / "leave_one_group_out.csv", index=False)
    strata = []
    for column in ("missing_rate_bin", "target_tail_gap", "target_internal_gap", "fallback"):
        for value, part in frame.groupby(column):
            table = aggregate_groups(part)[-1]
            table["stratum"], table["value"] = column, str(value)
            strata.append(table)
    pd.concat(strata, ignore_index=True).to_csv(output / "input_strata.csv", index=False)
    frame.drop_duplicates("case_id")[
        [
            "case_id",
            "group_id",
            "missing_rate_bin",
            "target_tail_gap",
            "target_internal_gap",
            "concurrent_aux_observed_rate",
        ]
    ].to_csv(output / "input_profiles.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": "conditioned_repair",
            "primary_metric": "mae",
            "score_rows": len(scores),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecasts / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R28_CONDITIONING_PROTOCOL.md"),
            "input_sha256": file_sha256(INPUTS / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
