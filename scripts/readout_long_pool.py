"""Score the frozen L192 expanded development comparison on original observed targets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from evaluate_matched_replay import aggregate_groups, score_points
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed long-pool results")
    base = ROOT / "artifacts/iclr27-r25"
    input_root, forecast_root = base / "long-inputs-v001", base / "long-forecasts-v001"
    prepared, forecasts = [read_json(p / "manifest.json") for p in (input_root, forecast_root)]
    if (
        prepared["status"] != "completed"
        or forecasts["status"] != "completed"
        or len(forecasts["cases"]) != 301
    ):
        raise ValueError("complete the entire registered forecast panel first")
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    scores, targets = [], []
    for entry in forecasts["cases"]:
        row = metadata[entry["case_id"]]
        for path, digest in (
            (forecast_root / entry["path"], entry["sha256"]),
            (input_root / row["path"], row["sha256"]),
            (Path(row["original_path"]), row["original_sha256"]),
        ):
            if file_sha256(path) != digest:
                raise ValueError("a frozen input, forecast or target source changed")
        with (
            np.load(forecast_root / entry["path"], allow_pickle=False) as predicted,
            np.load(input_root / row["path"], allow_pickle=False) as data,
            np.load(row["original_path"], allow_pickle=False) as original,
        ):
            truth = original["future"][:96, :2]
            np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96, :2])
            normalized = (truth - data["mean"][:2]) / data["scale"][:2]
            mae, mse = score_points(predicted["points"], normalized)
            for index, method in enumerate(predicted["methods"].tolist()):
                record = {
                    name: row[name]
                    for name in (
                        "case_id",
                        "episode_id",
                        "group_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "origin",
                    )
                }
                record.update(
                    model_id="chronos2",
                    method=method,
                    mae=float(mae[index].mean()),
                    mse=float(mse[index].mean()),
                )
                scores.append(record)
                for slot in (0, 1):
                    targets.append(
                        {
                            **record,
                            "target_slot": slot,
                            "observed_count": int(np.isfinite(truth[:, slot]).sum()),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if len(frame) != 5418 or len(targets) != 10836:
        raise ValueError("the registered long-pool score population changed")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    for name, table in zip(
        ("series", "datasets", "groups", "summary"), aggregate_groups(frame), strict=True
    ):
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "files": {
                str(p.relative_to(ROOT)): file_sha256(p)
                for p in (
                    Path(__file__),
                    input_root / "manifest.json",
                    forecast_root / "manifest.json",
                    ROOT / "docs/iclr2027/R25_LONG_POOL_PROTOCOL.md",
                )
            },
            "score_rows": len(frame),
            "target_score_rows": len(targets),
            "primary": "fraction_repair",
            "primary_model": "chronos2",
            "primary_metric": "mae",
            "context_length": 192,
            "horizon": 96,
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
