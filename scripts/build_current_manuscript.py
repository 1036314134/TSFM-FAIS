"""Build manuscript tables and figures from the frozen current score snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/iclr27-r47/long-repair-results-v001"
PAPER = ROOT / "docs/iclr2027"
OUT = PAPER / "current_results"
QA = ROOT / "tmp/paper_revision_20260918"
PRIMARY = "half_var_long_repair_peer_ridge_peer"
BJ = ["beijing/natural_outage_h24", "beijing/synthetic_outage_h24", "beijing/legacy_native_h96"]
HDB = ["hdb/natural_outage_h24", "hdb/synthetic_outage_h24", "hdb/native_grid_h24"]
PANELS = BJ + HDB

MAIN = [
    ("native_peer", "Native, 192 h"),
    ("native_long_prefix_peer", "Native, long"),
    ("native_long_prefix_targets", "Native, long, targets only"),
    ("linear_var_direct", "VAR"),
    ("half_var_native_long_prefix_peer", "Native, long + VAR"),
    ("half_var_native_long_prefix_targets", "Native, long, targets + VAR"),
    ("half_var_target_knn_multivariate", "Target KNN, 192 h + VAR"),
    ("long_repair_gaussian_peer", "Gaussian repair, long"),
    ("half_var_long_repair_gaussian_peer", "Gaussian repair, long + VAR"),
    ("half_var_long_repair_knn_multivariate_peer", "KNN repair, long + VAR"),
    ("long_repair_local_ridge_targets", "Local ridge, long, targets only"),
    ("corruption_lora", "Corruption-trained LoRA, 192 h"),
    (PRIMARY, r"\method{}"),
]
BASELINES = [
    ("target_locf", "LOCF"),
    ("target_linear_interp", "Linear interpolation"),
    ("target_seasonal_lag", "Seasonal lag"),
    ("target_knn_multivariate", "Multivariate KNN"),
    ("target_saits", "SAITS"),
    ("target_timemixerpp", "TimeMixer++"),
    ("target_motm_reference", "MoTM"),
    ("target_mean8", "Eight-candidate mean"),
    ("target_median8", "Eight-candidate median"),
    ("peer_ridge", "Peer ridge"),
    ("peer_ar_bridge", "Peer ridge + AR residual"),
    ("peer_residual_linear", "Peer ridge + linear residual"),
    ("natural_lora", "Natural-history LoRA"),
    ("corruption_lora", "Corruption-trained LoRA"),
]
HDB_ROWS = [
    ("native_peer192", "Native, 192 h"),
    ("native_long_prefix_peer", "Native, long"),
    ("native_long_prefix_targets", "Native, long, target only"),
    ("linear_var_direct", "VAR"),
    ("half_var_native_long_prefix_peer", "Native, long + VAR"),
    ("long_repair_gaussian_peer", "Gaussian repair, long"),
    ("half_var_long_repair_gaussian_peer", "Gaussian repair, long + VAR"),
    ("long_repair_median_targets", "Median repair, long, target only"),
    ("long_repair_median_peer", "Median repair, long, with peers"),
    ("query_aux_time_native_long", "Auxiliary-query mask, time"),
    ("query_aux_both_native_long", "Auxiliary-query mask, time/group"),
    ("median9", "Nine-candidate median, 192 h"),
    ("corruption_lora", "Corruption-trained LoRA, 192 h"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SOURCE / "summary.csv", float_precision="round_trip")
    stations = pd.read_csv(SOURCE / "stations.csv", float_precision="round_trip")
    cases = pd.read_parquet(SOURCE / "case_scores.parquet")
    scores = summary.set_index(["panel", "method"])[["mae", "mse"]]
    assert not scores.index.duplicated().any()
    reconstructed = cases.groupby(["panel", "method", "station"])[["mae", "mse"]].mean()
    macro = reconstructed.groupby(["panel", "method"]).mean().loc[scores.index]
    difference = float(np.max(np.abs(macro.to_numpy() - scores.to_numpy())))
    if difference > 1e-12:
        raise ValueError(f"Station-macro reconstruction differs: {difference}")
    assert len(summary) == 1275 and len(cases) == 74643
    counts = cases.drop_duplicates("case_id").groupby("panel").size().to_dict()
    assert [counts[p] for p in PANELS] == [45, 48, 138, 19, 32, 32]
    summary.to_csv(OUT / "all_panel_results.csv", index=False, float_format="%.17g")
    stations.to_csv(OUT / "all_station_results.csv", index=False, float_format="%.17g")

    def values(panel, method):
        return scores.loc[(panel, method)].to_numpy(dtype=float)

    def rows_to_tex(rows, panels, filename):
        all_values = np.array([[values(p, m) for p in panels] for m, _ in rows])
        best = all_values.min(axis=0)
        lines = []
        for (method, label), row in zip(rows, all_values, strict=True):
            if method == PRIMARY:
                lines.append(r"\midrule")
            cells = []
            for numbers, minima in zip(row, best, strict=True):
                for number, minimum in zip(numbers, minima, strict=True):
                    text = f"{number:.4f}"
                    cells.append(r"\textbf{" + text + "}" if number == minimum else text)
            lines.append(label + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows_to_tex(MAIN, BJ, "main_rows.tex")
    rows_to_tex(BASELINES, BJ, "baseline_rows.tex")
    rows_to_tex(HDB_ROWS, HDB, "hdb_rows.tex")

    station_index = stations.set_index(["panel", "method", "station"])
    names = sorted(stations.loc[stations.panel == BJ[0], "station"].unique())
    lines = []
    deltas = {}
    for reference in ["half_var_native_long_prefix_peer", "half_var_target_knn_multivariate"]:
        paired = []
        for station in names:
            a = station_index.loc[(BJ[0], PRIMARY, station), ["mae", "mse"]].to_numpy(float)
            b = station_index.loc[(BJ[0], reference, station), ["mae", "mse"]].to_numpy(float)
            paired.append(100 * (a / b - 1))
        deltas[reference] = np.array(paired)
    for index, station in enumerate(names):
        a = station_index.loc[(BJ[0], PRIMARY, station), ["mae", "mse"]].to_numpy(float)
        row = [f"{v:.4f}" for v in a] + [f"{v:+.2f}" for v in deltas["half_var_native_long_prefix_peer"][index]]
        lines.append(station + " & " + " & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    (OUT / "station_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparisons = []
    reference_ids = [m for m, _ in MAIN if m != PRIMARY] + ["peer_ridge", "half_var_peer_ridge", "local_ridge"]
    for panel in BJ:
        a = values(panel, PRIMARY)
        for reference in reference_ids:
            b = values(panel, reference)
            pair = reconstructed.loc[(panel, PRIMARY)] - reconstructed.loc[(panel, reference)]
            # The same sites are omitted from both methods; this is a sensitivity range.
            loso = []
            station_a, station_b = reconstructed.loc[(panel, PRIMARY)], reconstructed.loc[(panel, reference)]
            for station in station_a.index:
                aa, bb = station_a.drop(station).mean(), station_b.drop(station).mean()
                loso.append(100 * (aa.to_numpy() / bb.to_numpy() - 1))
            loso = np.asarray(loso)
            comparisons.append({"panel": panel, "reference": reference,
                "mae_change_percent": 100 * (a[0] / b[0] - 1),
                "mse_change_percent": 100 * (a[1] / b[1] - 1),
                "mae_station_wins": int((pair.mae < 0).sum()),
                "mse_station_wins": int((pair.mse < 0).sum()),
                "mae_loso_min": loso[:, 0].min(), "mae_loso_max": loso[:, 0].max(),
                "mse_loso_min": loso[:, 1].min(), "mse_loso_max": loso[:, 1].max()})
    pd.DataFrame(comparisons).to_csv(OUT / "primary_comparisons.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7))
    curves = [
        ("native_peer", "native_long_prefix_peer", "Native", "#777777", "o", "-"),
        ("peer_ridge", "long_repair_peer_ridge_peer", "Peer repair", "#2F5597", "s", "-"),
        ("half_var_native_peer", "half_var_native_long_prefix_peer", "Native + VAR", "#A96A22", "^", "--"),
        ("half_var_peer_ridge", PRIMARY, "Peer repair + VAR", "#257657", "D", "--"),
    ]
    for metric, ax in enumerate(axes):
        for short, long, label, color, marker, linestyle in curves:
            y = [values(BJ[0], short)[metric], values(BJ[0], long)[metric]]
            ax.plot([0, 1], y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.5, markersize=5)
        ax.set_xticks([0, 1], ["192 h", "8192 h"])
        ax.set_xlim(-0.15, 1.15)
        ax.set_xlabel("Forecaster history")
        ax.set_ylabel(["Standardized MAE", "Standardized MSE"][metric])
        ax.set_title(["(a) Absolute error", "(b) Squared error"][metric], loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.29, wspace=0.36)
    fig.savefig(OUT / "history_repair_fusion.pdf")
    fig.savefig(OUT / "history_repair_fusion.png", dpi=200)
    plt.close(fig)

    manifest = {"primary": PRIMARY, "panels": PANELS, "counts": counts,
        "source_summary_sha256": hashlib.sha256((SOURCE / "summary.csv").read_bytes()).hexdigest(),
        "macro_reconstruction_max_difference": difference,
        "panel_method_rows": len(summary), "case_score_rows": len(cases),
        "method_labels": dict(MAIN + BASELINES + HDB_ROWS),
        "primary_beijing_only": True, "new_forecasting_or_training": False}
    (QA / "evidence_check.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": 4, "figures": 1, "summary_rows": len(summary),
                      "score_reconstruction_difference": difference}))


if __name__ == "__main__":
    main()
