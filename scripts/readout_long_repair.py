"""Summarize the audited R47 fixed-baseline comparison without model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/iclr27-r47"
RESULTS = RUN / "long-repair-results-v001"
OUTPUT = RUN / "long-repair-readout-v001"
METRICS = ["mae", "mse"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit_path = RUN / "long-repair-audit-v001/manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forecast_path = RUN / "long-repair-forecasts-v001/manifest.json"
    assert audit["status"] == "completed"
    assert audit["prediction_difference"] == 0
    assert audit["forecast_sha256"] == sha256(forecast_path)
    summary = pd.read_csv(RESULTS / "summary.csv")
    stations = pd.read_csv(RESULTS / "stations.csv")
    leave_one = pd.read_csv(RESULTS / "leave_one_station_out.csv")
    cases = pd.read_parquet(RESULTS / "case_scores.parquet")
    indexed = summary.set_index(["panel", "method"])
    comparisons: list[dict] = []
    best: list[dict] = []

    def compare(panel: str, method: str, baseline: str, comparison: str) -> None:
        current = indexed.loc[(panel, method), METRICS]
        reference = indexed.loc[(panel, baseline), METRICS]
        row = {
            "panel": panel,
            "method": method,
            "baseline": baseline,
            "comparison": comparison,
        }
        for metric in METRICS:
            row[metric] = float(current[metric])
            row[f"baseline_{metric}"] = float(reference[metric])
            row[f"{metric}_change_percent"] = float(100 * (current[metric] / reference[metric] - 1))
        for label, frame, key in [
            ("stations", stations, "station"),
            ("cases", cases, "case_id"),
            ("leave_one_station_out", leave_one, "omitted_station"),
        ]:
            a = frame[(frame.panel == panel) & (frame.method == method)].set_index(key)[METRICS]
            b = frame[(frame.panel == panel) & (frame.method == baseline)].set_index(key)[METRICS]
            assert a.index.is_unique and b.index.is_unique
            assert set(a.index) == set(b.index)
            b = b.loc[a.index]
            row[f"{label}_count"] = len(a)
            for metric in METRICS:
                difference = a[metric] - b[metric]
                row[f"{label}_{metric}_wins"] = int((difference < 0).sum())
                row[f"{label}_{metric}_ties"] = int((difference == 0).sum())
                if label == "leave_one_station_out":
                    change = 100 * (a[metric] / b[metric] - 1)
                    row[f"{label}_{metric}_min_percent"] = float(change.min())
                    row[f"{label}_{metric}_max_percent"] = float(change.max())
        comparisons.append(row)

    bj_short = {
        "knn_multivariate": "target_knn_multivariate",
        "gaussian": "target_static_point",
        "local_ridge": "local_ridge",
        "peer_ridge": "peer_ridge",
    }
    for panel, frame in summary.groupby("panel", sort=True):
        # Raw-unit/prefix-unit native duplicates differ only at floating precision.
        canonical = frame[~frame.method.str.contains("native_long_raw_")]
        is_new = canonical.method.str.contains("long_repair_|short_repair_")
        old = canonical[~is_new]
        new = canonical[is_new]
        previous_best = {metric: str(old.loc[old[metric].idxmin(), "method"]) for metric in METRICS}
        for population, values in [("previous", old), ("new", new), ("all", canonical)]:
            for metric in METRICS:
                selected = values.loc[values[metric].idxmin()]
                best.append(
                    {
                        "panel": panel,
                        "population": population,
                        "selection_metric": metric,
                        "method": selected["method"],
                        "mae": float(selected.mae),
                        "mse": float(selected.mse),
                    }
                )
        for method in new.method:
            name = method
            prefix = ""
            if name.startswith("half_var_"):
                prefix = "half_var_"
                name = name.removeprefix(prefix)
            is_long = name.startswith("long_repair_")
            repair, scope = name.removeprefix(
                "long_repair_" if is_long else "short_repair_"
            ).rsplit("_", 1)
            native = f"native_long_prefix_{scope}" if is_long else "native_peer192"
            compare(panel, method, prefix + native, "same_length_native")
            if is_long:
                if scope == "targets":
                    short = f"isolated_{repair}"
                elif panel.startswith("beijing/"):
                    short = bj_short[repair]
                else:
                    short = f"short_repair_{repair}_peer"
                compare(panel, method, prefix + short, "same_repair_short")
            for metric, baseline in previous_best.items():
                compare(
                    panel,
                    method,
                    baseline,
                    f"previous_best_{metric}_development_reference",
                )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(best).to_csv(OUTPUT / "development_minima.csv", index=False)
    sources = [audit_path, forecast_path] + [
        RESULTS / name
        for name in [
            "summary.csv",
            "stations.csv",
            "leave_one_station_out.csv",
            "case_scores.parquet",
        ]
    ]
    manifest = {
        "status": "completed",
        "diagnostic": True,
        "new_forecasts": 0,
        "new_training": False,
        "heldout_value_analysis": False,
        "independent_confirmation": False,
        "comparison_rows": len(comparisons),
        "minima_rows": len(best),
        "minima_are_posthoc_development_descriptions": True,
        "station_omissions_are_sensitivity_not_confidence_intervals": True,
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "comparison_rows": len(comparisons)}))


if __name__ == "__main__":
    main()
