"""Audit and plot the independent-history learning curves."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed learning-curve readouts")
    output.mkdir(parents=True, exist_ok=True)
    root = args.input_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["fitted_models"] != 210:
        raise ValueError("complete all 210 fits before summarizing")
    groups = {}
    for item in manifest["folds"]:
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a saved fold changed")
        fold = json.loads(path.read_text(encoding="utf-8"))
        if fold["identity_sha256"] != manifest["identity_sha256"]:
            raise ValueError("a fold belongs to another experiment")
        for kind in ("model", "predictions"):
            if file_sha256(root / fold[kind + "_path"]) != fold[kind + "_sha256"]:
                raise ValueError("a saved model or selection changed")
        metadata = fold["metadata"]
        key = tuple(
            metadata[name] for name in ("model_id", "held_family", "sampling_seed", "fraction")
        )
        if key in groups:
            raise ValueError("duplicate fit identity")
        groups[key] = set(fold["training_origins"])
    nested_checks = 0
    for (model, family, seed, fraction), origins in groups.items():
        if fraction == 0.25:
            if (
                not origins
                <= groups[(model, family, seed, 0.5)]
                <= groups[(model, family, 6101, 1.0)]
            ):
                raise ValueError("the sampled histories are not nested")
            nested_checks += 1
    frame = pd.read_csv(root / "fold_metrics.csv")
    expected_rows = 210 * 4
    if len(frame) != expected_rows:
        raise ValueError("the fold-level scores have incomplete coverage")
    curve = pd.read_csv(root / "learning_curve.csv")
    keys = ["model_id", "scope", "method", "fraction", "sampling_seed"]
    rebuilt = frame.groupby(keys)[["mae", "mse", "fit_origins"]].mean().sort_index()
    np.testing.assert_allclose(
        rebuilt, curve.set_index(keys).sort_index()[rebuilt.columns], rtol=0, atol=1e-10
    )
    summary = (
        curve.groupby(["model_id", "scope", "method", "fraction"])
        .agg(
            mae=("mae", "mean"),
            mae_low=("mae", "min"),
            mae_high=("mae", "max"),
            mse=("mse", "mean"),
            mse_low=("mse", "min"),
            mse_high=("mse", "max"),
            fit_origins=("fit_origins", "mean"),
            sampling_seeds=("sampling_seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(output / "curve_summary.csv", index=False)
    join = ["model_id", "scope", "method", "held_family"]
    full = frame[frame.fraction == 1][join + ["mae", "mse"]]
    pairs = frame[frame.fraction < 1].merge(
        full, on=join, suffixes=("_smaller", "_full"), validate="many_to_one"
    )
    pairs["delta_mae_full_minus_smaller"] = pairs.mae_full - pairs.mae_smaller
    pairs["delta_mse_full_minus_smaller"] = pairs.mse_full - pairs.mse_smaller
    pairs.to_csv(output / "paired_family_changes.csv", index=False)
    effects = []
    for labels, group in pairs.groupby(["model_id", "scope", "method", "fraction"]):
        family = group.groupby("held_family")[
            ["delta_mae_full_minus_smaller", "delta_mse_full_minus_smaller"]
        ].mean()
        effects.append(
            {
                **dict(
                    zip(["model_id", "scope", "method", "smaller_fraction"], labels, strict=True)
                ),
                "delta_mae": float(family.iloc[:, 0].mean()),
                "delta_mse": float(family.iloc[:, 1].mean()),
                "both_metrics_improve_units": int((family < -1e-12).all(axis=1).sum()),
                "units": len(family),
                "unit": "held-out family"
                if labels[1] == "held_family"
                else "overlapping source-validation fold",
            }
        )
    pd.DataFrame(effects).to_csv(output / "growth_effects.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    styles = [
        ("response_selector", "Utility selector", "#1A6FDF", "o", "-"),
        ("fit_best_fixed", "Best fixed from training", "#F14040", "s", "--"),
    ]
    baselines = {row["model_id"]: row for row in manifest["strong_baselines"]}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    held = summary[summary.scope == "held_family"]
    for row, (model, title) in enumerate(
        (("chronos2", "Chronos-2"), ("timesfm2p5", "TimesFM 2.5"))
    ):
        for column, metric in enumerate(("mae", "mse")):
            ax = axes[row, column]
            for method, label, color, marker, linestyle in styles:
                part = held[(held.model_id == model) & (held.method == method)].sort_values(
                    "fraction"
                )
                y = part[metric].to_numpy()
                error = np.stack([y - part[metric + "_low"], part[metric + "_high"] - y])
                ax.errorbar(
                    part.fit_origins,
                    y,
                    yerr=error,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=4,
                    capsize=2,
                    elinewidth=0.8,
                )
            ax.axhline(
                baselines[model][metric],
                color="#444444",
                linestyle=":",
                linewidth=1.2,
                label="Forecast median",
            )
            ax.set_title(f"({chr(97 + row * 2 + column)}) {title}", loc="left")
            ax.set_ylabel(metric.upper())
            ax.set_xlabel("Training histories per fold")
            ax.set_xticks(part.fit_origins, [f"{value:.1f}" for value in part.fit_origins])
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.015)
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.11, top=0.90, hspace=0.56, wspace=0.28)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"learning_curve.{extension}"
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        paths.append(path)
    # Grayscale export uses the same lines, ranges and data.
    for line in fig.findobj(matplotlib.lines.Line2D):
        line.set_color("#333333")
        line.set_markeredgecolor("#333333")
        line.set_markerfacecolor("#333333")
    for collection in fig.findobj(matplotlib.collections.LineCollection):
        collection.set_color("#555555")
    gray = output / "learning_curve_grayscale.png"
    fig.savefig(gray, dpi=300, facecolor="white", bbox_inches="tight")
    paths.append(gray)
    plt.close(fig)
    caption = (
        "Forecast accuracy on held-out families as the number of independent source training histories increases. "
        "The horizontal axis reports the mean number of fitting origins per held-family fold at 25%, 50%, and 100% source coverage. "
        "All mask and target variants of an origin stay together. Points at 25% and 50% average three fixed sampling seeds; whiskers show their full range, not confidence intervals. "
        "The 100% model is fitted once because its sample is identical across seeds. Errors use historical-prefix standardization after the original R4 raw-input forecasting procedure. "
        "The predictor and selector settings are fixed, and the forecast median uses the same supported candidate pool."
    )
    (output / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "source_manifest_sha256": file_sha256(root / "manifest.json"),
            "script_sha256": file_sha256(Path(__file__)),
            "verified_fits": len(groups),
            "verified_nested_subsets": nested_checks,
            "growth_effects": effects,
            "figure_files": [{"path": path.name, "sha256": file_sha256(path)} for path in paths],
            "interpretation": "fixed-model development diagnostic; ranges describe sampling choices and do not establish population uncertainty or extrapolation gains",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(summary.to_string(index=False))
    print(json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
