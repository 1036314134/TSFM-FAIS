"""Plot matched recent-feedback strategies with paired family uncertainty."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def method_rows(frame, prefix, objective, count, shrinkage):
    def matches(name):
        if not name.startswith(prefix + "|"):
            return False
        settings = dict(part.split("=", 1) for part in name.split("|")[1:])
        return (
            settings["objective"] == objective
            and float(settings["H"]) == 96
            and float(settings["P"]) == count
            and float(settings["shrink"]) == shrinkage
        )

    return frame[frame.method.map(matches)].set_index("family_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("figure evidence already exists; select another output directory")
    if args.bootstrap_draws < 1000:
        parser.error("bootstrap-draws must be at least 1000")
    manifest = json.loads((args.readout_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("evidence_role") != "development":
        raise ValueError("figures require a completed development readout")
    frame = pd.read_csv(args.readout_root / "family_results.csv")
    records, pairs, model_configurations = [], [], {}
    generator = np.random.default_rng(6101)
    for model in ("chronos2", "timesfm2p5"):
        view = frame[frame.model_id == model]
        reference = view[view.method == "forecast_median_guarded"].set_index("family_id")
        families = sorted(reference.index)
        if len(families) < 2 or not reference.index.is_unique:
            raise ValueError("one baseline value per family is required")
        draw = generator.integers(0, len(families), size=(args.bootstrap_draws, len(families)))
        granularity = "sequence" if model == "chronos2" else "target"
        configurations = [
            ("Historical selection (7)", "recent_" + granularity, "joint", 0.0),
            ("Input mixture (6)", "recent_imputation_" + granularity, "mse", 0.5),
            ("Output mixture (6)", "recent_forecast_finite_" + granularity, "mse", 0.5),
        ]
        model_configurations[model] = configurations
        for label, method, objective, shrinkage in configurations:
            for count in (1, 2):
                selected = method_rows(view, method, objective, count, shrinkage)
                if not selected.index.is_unique or set(selected.index) != set(families):
                    raise ValueError(f"incomplete or duplicated curve: {model} {method} {count}")
                for metric in ("mae", "mse"):
                    before = reference.loc[families, metric].to_numpy()
                    after = selected.loc[families, metric].to_numpy()
                    delta = after - before
                    if not np.isfinite(delta).all():
                        raise ValueError("every plotted comparison must have finite paired errors")
                    low, high = np.quantile(delta[draw].mean(axis=1), [0.025, 0.975])
                    records.append(
                        {
                            "model_id": model,
                            "metric": metric,
                            "strategy": label,
                            "probe_count": count,
                            "mean_error": float(after.mean()),
                            "reference_error": float(before.mean()),
                            "mean_difference": float(delta.mean()),
                            "lower": float(low),
                            "upper": float(high),
                            "family_count": len(families),
                        }
                    )
                    pairs.extend(
                        {
                            "model_id": model,
                            "metric": metric,
                            "strategy": label,
                            "probe_count": count,
                            "family_id": family,
                            "reference_error": float(first),
                            "method_error": float(second),
                        }
                        for family, first, second in zip(families, before, after, strict=True)
                    )
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-config")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = Path("C:/Users/MDC/.codex/skills/group-paper-style/assets/group-paper.mplstyle")
    summary = pd.DataFrame(records)
    summary.to_csv(output / "plotted_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_family_errors.csv", index=False)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers, lines = ["o", "s", "^"], ["-", "--", "-."]
    labels = [entry[0] for entry in configurations]
    limits = {}
    for metric in ("mae", "mse"):
        data = summary[summary.metric == metric]
        low, high = min(0.0, data.lower.min()), max(0.0, data.upper.max())
        padding = max(0.01, (high - low) * 0.10)
        limits[metric] = (low - padding, high + padding)
    for grayscale in (False, True):
        with plt.style.context(style):
            fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
            for row, (model, title) in enumerate(
                (("chronos2", "Chronos-2"), ("timesfm2p5", "TimesFM 2.5"))
            ):
                for column, metric in enumerate(("mae", "mse")):
                    ax = axes[row, column]
                    ax.axhline(0, color="#555555", linewidth=0.8, linestyle=":")
                    for index, label in enumerate(labels):
                        data = summary[
                            (summary.model_id == model)
                            & (summary.metric == metric)
                            & (summary.strategy == label)
                        ].sort_values("probe_count")
                        color = (
                            ["#222222", "#777777", "#aaaaaa"][index] if grayscale else colors[index]
                        )
                        x = data.probe_count.to_numpy() + (index - 1) * 0.035
                        center = data.mean_difference.to_numpy()
                        ax.errorbar(
                            x,
                            center,
                            yerr=np.stack([center - data.lower, data.upper - center]),
                            color=color,
                            marker=markers[index],
                            linestyle=lines[index],
                            capsize=2.5,
                            linewidth=1.15,
                            markersize=4,
                            label=label,
                        )
                    ax.set_title(f"({chr(97 + row * 2 + column)}) {title}", loc="left")
                    ax.set_ylabel(r"$\Delta$ " + metric.upper() + " (standardized)")
                    ax.set_xticks([1, 2])
                    ax.set_xlim(0.82, 2.18)
                    ax.set_ylim(*limits[metric])
                    if row == 1:
                        ax.set_xlabel("Completed historical windows")
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0)
            )
            fig.subplots_adjust(
                left=0.11, right=0.98, bottom=0.11, top=0.88, wspace=0.36, hspace=0.32
            )
            name = "recent_feedback_grayscale" if grayscale else "recent_feedback"
            fig.savefig(output / (name + ".png"), dpi=300)
            if not grayscale:
                fig.savefig(output / (name + ".pdf"))
                fig.savefig(output / (name + ".svg"))
            plt.close(fig)
    caption = (
        "Recent-feedback strategies on the predeclared development screening set. "
        "Values are paired differences from the seven-action forecast-median reference; lower is better. "
        "Each model is evaluated on the same 90 tasks from 15 families, with one prediction origin and six masks per family. "
        "Errors use common training-prefix standardization. Each completed historical window spans 96 steps, and only originally observed labels enter feedback. "
        "Selection uses the joint MAE/MSE objective; input and output mixtures use identical MSE-derived weights over six finite imputers, shrunk halfway toward uniform weights. "
        "Chronos-2 uses sequence decisions, and independent TimesFM predictions use target decisions. "
        "Error bars are descriptive 95% percentile intervals from 2,000 paired family resamples; they are not independent confirmation or simultaneous hypothesis tests. "
        "Small horizontal offsets separate overlapping markers."
    )
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "readout_manifest_sha256": file_sha256(args.readout_root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "style_sha256": file_sha256(style),
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": 6101,
            "visual_review": "pending",
            "strategy_configuration": model_configurations,
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
