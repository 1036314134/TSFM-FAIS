"""Plot both primary errors for every registered synthetic follow-up method."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402

GROUPS = {
    "median_risk": ("Portfolio-risk selector", "#0072B2", "*", 115),
    "member_risk": ("Member-risk control", "#56B4E9", "o", 38),
    "old_teacher_rank3": ("Original teacher top three", "#009E73", "s", 35),
    "source_fixed_median_risk": ("Source-fixed triples", "#666666", "^", 36),
    "old_source_fixed3": ("Source-fixed triples", "#666666", "^", 36),
    "locf": ("LOCF", "#E69F00", "v", 36),
    "linear_interp": ("Linear interpolation", "#D55E00", "P", 36),
    "motm_reference": ("MoTM", "#CC79A7", "D", 34),
    "forecast_median_with_motm": ("Eight-member median", "#882255", "X", 42),
    "forecast_median_guarded": ("Seven-member median", "#111111", "+", 70),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("readout-root", "supplement-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed figure artifacts")
    for directory in (args.readout_root, args.supplement_root):
        if (
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"]
            != "completed"
        ):
            raise ValueError("both audited result readouts are required")
    main_table = pd.read_csv(args.readout_root / "summary.csv")
    extra = pd.read_csv(args.supplement_root / "summary.csv")
    panel = "new_synthetic_all"
    main_table = main_table[main_table.panel == panel]
    extra = extra[extra.panel == panel]
    for model in ("chronos2", "timesfm2p5"):
        for method in ("median_risk", "forecast_median_guarded"):
            left = main_table[(main_table.model_id == model) & (main_table.method == method)][
                ["mae", "mse"]
            ]
            right = extra[(extra.model_id == model) & (extra.method == method)][["mae", "mse"]]
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    points = pd.concat(
        [main_table, extra[extra.method.isin(["motm_reference", "forecast_median_with_motm"])]],
        ignore_index=True,
    )
    if len(points) != 36 or points.duplicated(["model_id", "method"]).any():
        raise ValueError("every one of the 18 methods per model must be plotted exactly once")
    for model in ("chronos2", "timesfm2p5"):
        mask = points.model_id == model
        base = points[mask & (points.method == "forecast_median_guarded")].iloc[0]
        points.loc[mask, "relative_mae"] = 100 * points.loc[mask, "mae"] / base.mae
        points.loc[mask, "relative_mse"] = 100 * points.loc[mask, "mse"] / base.mse
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), sharex=True, sharey=True)
    legend = {}
    for axis, model, title in zip(
        axes, ("chronos2", "timesfm2p5"), ("Chronos-2", "TimesFM 2.5"), strict=True
    ):
        axis.axvline(100, color="0.65", linewidth=0.8, linestyle="--", zorder=0)
        axis.axhline(100, color="0.65", linewidth=0.8, linestyle="--", zorder=0)
        for row in points[points.model_id == model].itertuples(index=False):
            label, color, marker, size = GROUPS.get(
                row.method, ("Other registered outputs", "#BBBBBB", "o", 15)
            )
            handle = axis.scatter(
                row.relative_mae,
                row.relative_mse,
                color=color,
                marker=marker,
                s=size,
                linewidths=0.9,
                zorder=4 if row.method == "median_risk" else 2,
            )
            legend.setdefault(label, handle)
        axis.set_title(title)
        axis.set_xlabel("Relative MAE (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel("Relative MSE (%)")
    axes[0].set_xlim(points.relative_mae.min() - 3, points.relative_mae.max() + 3)
    axes[0].set_ylim(points.relative_mse.min() - 5, points.relative_mse.max() + 5)
    fig.suptitle("New-source synthetic tasks: MAE and MSE trade-offs", fontsize=11, y=0.98)
    order = list(dict.fromkeys(item[0] for item in GROUPS.values())) + ["Other registered outputs"]
    fig.legend(
        [legend[name] for name in order],
        order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.30, wspace=0.16)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "followup_tradeoffs.pdf", bbox_inches="tight")
    fig.savefig(output / "followup_tradeoffs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    points.to_csv(output / "plotted_values.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "readout_sha256": file_sha256(args.readout_root / "manifest.json"),
            "supplement_sha256": file_sha256(args.supplement_root / "manifest.json"),
            "plotted_methods_per_model": 18,
            "panel": panel,
            "files_sha256": {
                name: file_sha256(output / name)
                for name in (
                    "followup_tradeoffs.pdf",
                    "followup_tradeoffs.png",
                    "plotted_values.csv",
                )
            },
            "visual_review_status": "pending",
            "caption": "Family-macro errors over weather and Alabama Solar, eight histories each and 36 masks per history. Values are relative to the seven-member median, and lower is better. MoTM is an additional comparator; its reference includes a Solar-trained component. The eight-member median can require one extra candidate forecast. Source-fixed definitions coincide for Chronos.",
        },
    )


if __name__ == "__main__":
    main()
