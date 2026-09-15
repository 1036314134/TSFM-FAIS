"""Plot audited teacher-oracle tradeoffs and primary-gate loss discrepancies."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402

METHODS = {
    "gate_source_fixed_convex": ("Source-fixed mixture", "#444444", "s", True),
    "ensemble_gate": ("Primary shared gate", "#0072B2", "*", True),
    "complete_history_teacher": ("Complete-history forecast", "#009E73", "v", False),
    "unavailable_teacher_item_convex": ("Teacher oracle: per item", "#E69F00", "^", False),
    "unavailable_teacher_window_convex": ("Teacher oracle: per input", "#D55E00", "o", False),
    "unavailable_future_item_convex": ("Future oracle: per item", "#CC79A7", "D", False),
}
FAMILIES = {
    "appliances": ("Appliances", "#0072B2", "o"),
    "beijing_multisite": ("Beijing", "#E69F00", "s"),
    "bike_sharing": ("Bike sharing", "#009E73", "^"),
    "occupancy": ("Occupancy", "#CC79A7", "D"),
}


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("readout-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed figures")
    manifest = json.loads((args.readout_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "completed"
        or file_sha256(args.readout_root / "summary.csv") != manifest["summary_sha256"]
    ):
        raise ValueError("finish and retain the audited teacher diagnostic")
    summary = pd.read_csv(args.readout_root / "summary.csv", float_precision="round_trip")
    family = pd.read_csv(args.readout_root / "family_metrics.csv", float_precision="round_trip")
    points = summary[(summary.horizon == 96) & summary.method.isin(METHODS)].copy()
    discrepancies = family[(family.horizon == 96) & (family.method == "ensemble_gate")].copy()
    if (
        len(points) != 12
        or len(discrepancies) != 8
        or set(discrepancies.family_id) != set(FAMILIES)
    ):
        raise ValueError("plot coverage differs from the declared comparison")
    for model in ("chronos2", "timesfm2p5"):
        selected = points.model_id == model
        base = points[selected & (points.method == "gate_source_fixed_convex")].iloc[0]
        points.loc[selected, "relative_mae"] = 100 * points.loc[selected, "mae"] / base.mae
        points.loc[selected, "relative_mse"] = 100 * points.loc[selected, "mse"] / base.mse
        actual = discrepancies[discrepancies.model_id == model][
            ["real_mse_delta", "teacher_mse_delta", "empirical_cross_term"]
        ].mean()
        expected = points[selected & (points.method == "ensemble_gate")].iloc[0]
        np.testing.assert_allclose(
            actual.to_numpy(), expected[actual.index].to_numpy(float), rtol=1e-12, atol=1e-12
        )
    output.mkdir(parents=True, exist_ok=True)
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), sharex=True, sharey=True)
    for axis, model, title in zip(
        axes, ("chronos2", "timesfm2p5"), ("Chronos-2", "TimesFM 2.5"), strict=True
    ):
        for row in points[points.model_id == model].itertuples(index=False):
            _, color, marker, available = METHODS[row.method]
            axis.scatter(
                row.relative_mae,
                row.relative_mse,
                s=75 if marker == "*" else 40,
                marker=marker,
                edgecolors=color,
                facecolors=color if available else "none",
                linewidths=1.0,
                zorder=3,
            )
        axis.axvline(100, color="0.65", linewidth=0.7, linestyle="--")
        axis.axhline(100, color="0.65", linewidth=0.7, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Relative downstream MAE (%)")
        style_axis(axis)
    axes[0].set_ylabel("Relative downstream MSE (%)")
    axes[0].set_xlim(points.relative_mae.min() - 3, points.relative_mae.max() + 3)
    axes[0].set_ylim(points.relative_mse.min() - 4, points.relative_mse.max() + 5)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markeredgecolor=color,
            markerfacecolor=color if available else "none",
            label=label,
            markersize=7,
        )
        for label, color, marker, available in METHODS.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=8,
    )
    fig.text(
        0.5,
        0.005,
        "H = 96; four-source macro average. Hollow markers use unavailable information.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.28, top=0.90, wspace=0.14)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"teacher_oracle_tradeoffs.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharex=True, sharey=True)
    low = min(0.0, discrepancies.teacher_mse_delta.min(), discrepancies.real_mse_delta.min())
    high = max(0.0, discrepancies.teacher_mse_delta.max(), discrepancies.real_mse_delta.max())
    padding = 0.1 * (high - low)
    for axis, model, title in zip(
        axes, ("chronos2", "timesfm2p5"), ("Chronos-2", "TimesFM 2.5"), strict=True
    ):
        axis.axvline(0, color="0.65", linewidth=0.7)
        axis.axhline(0, color="0.65", linewidth=0.7)
        axis.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            color="0.65",
            linewidth=0.7,
            linestyle="--",
        )
        for row in discrepancies[discrepancies.model_id == model].itertuples(index=False):
            _, color, marker = FAMILIES[row.family_id]
            axis.scatter(
                row.teacher_mse_delta,
                row.real_mse_delta,
                color=color,
                marker=marker,
                s=42,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Change in teacher MSE")
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        style_axis(axis)
    axes[0].set_ylabel("Change in downstream MSE")
    handles = [
        Line2D([], [], linestyle="none", marker=marker, color=color, label=label, markersize=6)
        for label, color, marker in FAMILIES.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=8,
    )
    fig.text(
        0.5,
        0.005,
        "Primary gate minus source-fixed mixture; H = 96. Diagonal: zero empirical cross-term.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.23, top=0.90, wspace=0.14)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"teacher_loss_discrepancy.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    points.to_csv(output / "oracle_plotted_values.csv", index=False)
    discrepancies.to_csv(output / "discrepancy_plotted_values.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "readout_sha256": file_sha256(args.readout_root / "manifest.json"),
            "plotted_oracle_points": len(points),
            "plotted_source_discrepancies": len(discrepancies),
            "files_sha256": {
                path.name: file_sha256(path) for path in sorted(output.iterdir()) if path.is_file()
            },
            "visual_review_status": "pending",
            "limits": "H96 diagnostic; unavailable-information markers are not deployable improvements; four selected sources do not establish population effects",
        },
    )


if __name__ == "__main__":
    main()
