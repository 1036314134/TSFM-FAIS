"""Read outcome values only after all peer-information forecasts have been frozen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from peer_outage_core import BASE, ROOT, sources

from tsfm_fais.utility_experiment import _write_json, file_sha256


def aggregate(frame):
    stations = frame.groupby(["panel", "method", "station"])[["mae", "mse"]].mean().reset_index()
    summary = stations.groupby(["panel", "method"])[["mae", "mse"]].mean().reset_index()
    return stations, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve complete peer readout")
    forecast_root = BASE / "peer-forecasts-v001"
    manifest = json.loads((forecast_root / "manifest.json").read_text(encoding="utf-8"))
    inputs = Path(manifest["input_root"])
    prepared = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    metadata = {r["case_id"]: r for r in prepared["cases"]}
    if (
        manifest["smoke"]
        or len(manifest["cases"]) != 231
        or {r["case_id"] for r in manifest["cases"]} != set(metadata)
    ):
        raise ValueError("finish all prespecified forecasts before scoring")
    records, _ = sources()
    rows, targets, imputation = [], [], []
    for entry in manifest["cases"]:
        row = metadata[entry["case_id"]]
        for p, sha in (
            (inputs / row["path"], row["sha256"]),
            (forecast_root / entry["path"], entry["sha256"]),
        ):
            if file_sha256(p) != sha:
                raise ValueError("a frozen input or forecast changed")
        with np.load(inputs / row["path"], allow_pickle=False) as saved:
            mean, scale = saved["mean"][:2], saved["scale"][:2]
            stat_names, stat_targets = saved["stat_names"].tolist(), saved["stat_targets"]
            candidate_names, candidates = saved["actions"].tolist(), saved["candidates"][:, :, :2]
        with np.load(forecast_root / entry["path"], allow_pickle=False) as saved:
            names, points = saved["methods"].tolist(), saved["points"]
        t, horizon = row["origin"], row["horizon"]
        truth = records[row["station"]]["values"][t : t + horizon, :2]
        valid = np.isfinite(truth)
        if (valid.sum(0) < horizon // 2).any():
            raise ValueError("registered outcome support changed")
        normalized = (truth - mean) / scale
        error = np.where(valid[None], points - normalized[None], 0.0)
        mae, mse = np.abs(error).sum(1) / valid.sum(0), np.square(error).sum(1) / valid.sum(0)
        info = {k: row[k] for k in ("case_id", "panel", "station", "origin", "horizon")}
        for i, method in enumerate(names):
            rows.append(
                {**info, "method": method, "mae": float(mae[i].mean()), "mse": float(mse[i].mean())}
            )
            for slot in (0, 1):
                targets.append(
                    {
                        **info,
                        "method": method,
                        "slot": slot,
                        "observed_count": int(valid[:, slot].sum()),
                        "mae": float(mae[i, slot]),
                        "mse": float(mse[i, slot]),
                    }
                )
        if row["panel"] == "synthetic_outage_h24":
            hidden = records[row["station"]]["values"][t - 24 : t, :2]
            for method, values in zip(
                [*candidate_names, *stat_names],
                np.concatenate([candidates, stat_targets]),
                strict=True,
            ):
                e = (values[-24:] - hidden) / scale
                imputation.append(
                    {
                        **info,
                        "method": method,
                        "mae": float(abs(e).mean()),
                        "mse": float((e * e).mean()),
                    }
                )
    if len(rows) != 8547 or len(targets) != 17094:
        raise ValueError("registered score counts changed")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    pd.DataFrame(imputation).to_csv(output / "imputation_auxiliary.csv", index=False)
    stations, summary = aggregate(frame)
    stations.to_csv(output / "stations.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    leave = []
    for (panel, excluded), _ in stations.groupby(["panel", "station"]):
        part = (
            stations.loc[(stations["panel"] == panel) & (stations["station"] != excluded)]
            .groupby(["panel", "method"])[["mae", "mse"]]
            .mean()
            .reset_index()
        )
        part["omitted_station"] = excluded
        leave.append(part)
    pd.concat(leave, ignore_index=True).to_csv(output / "leave_one_station_out.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "primary": "peer_ar_bridge",
            "primary_panel": "natural_outage_h24",
            "primary_metric": "mae",
            "score_rows": len(rows),
            "target_score_rows": len(targets),
            "independent_confirmation": False,
            "forecast_sha256": file_sha256(forecast_root / "manifest.json"),
            "input_sha256": file_sha256(inputs / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R30_PEER_OUTAGE_PROTOCOL.md"),
        },
    )


if __name__ == "__main__":
    main()
