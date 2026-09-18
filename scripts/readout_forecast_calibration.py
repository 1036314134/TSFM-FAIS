"""Score frozen R31 predictions using observed-prefix standardized downstream errors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from forecast_calibration_core import BASE, PARENT, ROOT, load_npz, read_json
from peer_outage_core import sources
from readout_peer_outage import aggregate

from tsfm_fais.utility_experiment import _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete R31 scores")
    forecast = BASE / "calibrated-forecasts-v001"
    fm = read_json(forecast / "manifest.json")
    trained = read_json(Path(fm["training_root"]) / "manifest.json")
    metadata = {r["case_id"]: r for r in trained["evaluation"]}
    if fm["smoke"] or len(metadata) != 93 or {r["case_id"] for r in fm["cases"]} != set(metadata):
        raise ValueError("freeze all 93 formal evaluations before reading outcomes")
    records, _ = sources()
    rows, targets, auxiliary = [], [], []
    for entry in fm["cases"]:
        row = metadata[entry["case_id"]]
        d = load_npz(PARENT / "peer-inputs-v001" / row["path"], row["sha256"])
        f = load_npz(forecast / entry["path"], entry["sha256"])
        t = row["origin"]
        truth = records[row["station"]]["values"][t : t + 24, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < 12).any():
            raise ValueError("registered evaluation support changed")
        truth = (truth - d["mean"][:2]) / d["scale"][:2]
        error = np.where(valid[None], f["points"] - truth[None], 0.0)
        mae, mse = abs(error).sum(1) / valid.sum(0), (error * error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for index, name in enumerate(f["methods"].tolist()):
            rows.append(
                {
                    **info,
                    "method": name,
                    "mae": float(mae[index].mean()),
                    "mse": float(mse[index].mean()),
                }
            )
            for slot in (0, 1):
                targets.append(
                    {
                        **info,
                        "method": name,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[index, slot]),
                        "mse": float(mse[index, slot]),
                    }
                )
        if row["panel"] == "synthetic_outage_h24":
            hidden = (records[row["station"]]["values"][t - 24 : t, :2] - d["mean"][:2]) / d[
                "scale"
            ][:2]
            for name, context in zip(f["input_methods"].tolist(), f["contexts_z"], strict=True):
                e = context[-24:, :2] - hidden
                auxiliary.append(
                    {
                        **info,
                        "method": name,
                        "mae": float(abs(e).mean()),
                        "mse": float((e * e).mean()),
                    }
                )
    if len(rows) != 4092 or len(targets) != 8184:
        raise ValueError("registered score counts changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    pd.DataFrame(auxiliary).to_csv(output / "imputation_auxiliary.csv", index=False)
    stations, summary = aggregate(frame)
    stations.to_csv(output / "stations.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    leave = []
    for (panel, station), _ in stations.groupby(["panel", "station"]):
        part = stations.loc[(stations["panel"] == panel) & (stations["station"] != station)]
        part = part.groupby(["panel", "method"])[["mae", "mse"]].mean().reset_index()
        part["omitted_station"] = station
        leave.append(part)
    pd.concat(leave, ignore_index=True).to_csv(output / "leave_one_station_out.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": "forecast_gate",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecast / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": file_sha256(
                ROOT / "docs/iclr2027/R31_FORECAST_CALIBRATION_PROTOCOL.md"
            ),
        },
    )


if __name__ == "__main__":
    main()
