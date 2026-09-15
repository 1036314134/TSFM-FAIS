"""Validate the R6 draft's evidence tables, compile it, and render every page."""

import argparse
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from build_r5_manuscript import sha, validate_source_inventory

ROOT = Path(__file__).resolve().parents[1]
STEM = "tsfm_fais_r6_draft"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_rows(tex, label):
    table = tex.split("\\label{" + label + "}", 1)[1].split("\\end{table}", 1)[0]
    return [
        [
            float(value)
            for value in re.findall(r"(?<![A-Za-z0-9])(?:\d+)?\.\d+", line.split("&", 1)[1])
        ]
        for line in table.splitlines()
        if re.search(r"&\s+(?:\\textbf\{)?(?:\d+)?\.\d+", line)
    ]


def selected_values(rows, **conditions):
    selected = [
        row for row in rows if all(str(row[key]) == str(value) for key, value in conditions.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"ambiguous evidence row: {conditions}")
    return [float(f"{float(selected[0][metric]):.6f}") for metric in ("mae", "mse")]


def verify_table(tex, label, expected):
    if numeric_rows(tex, label) != expected:
        raise ValueError(f"manuscript table differs from the recorded evidence: {label}")
    return sum(map(len, expected))


def validate_evidence(tex, source):
    r6 = ROOT / "artifacts/iclr27-r6"
    r5 = ROOT / "artifacts/iclr27-r5"
    dependencies = [
        r6 / "policy-audit-v002/manifest.json",
        r6 / "readout-v002/manifest.json",
        r5 / "shared-gate-audit-v001/manifest.json",
        r5 / "shared-gate-readout-v001/manifest.json",
        r5 / "imputer-budget-origin-audit-v001/manifest.json",
        r5 / "imputer-budget-origin-readout-v001/manifest.json",
    ]
    for path in dependencies:
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "completed":
            raise ValueError("complete the empirical checks before rendering")
    results = read_rows(r6 / "readout-v002/summary.csv")
    development = read_rows(r5 / "shared-gate-readout-v001/summary.csv")
    budget = read_rows(r5 / "imputer-budget-origin-readout-v001/summary.csv")
    r6_main = (source / "sections/r6_results.tex").read_text(encoding="utf-8")
    expected = []
    for method in (
        "ensemble_gate",
        "member_gate",
        "source_future_gate",
        "gate_source_fixed_convex",
        "forecast_median_guarded",
        "forecast_median_with_motm",
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(
                selected_values(
                    results, method=method, model_id=model, horizon=96, panel="new_synthetic_all"
                )
            )
        expected.append(values)
    count = verify_table(r6_main, "tab:r6_main", expected)
    expected = []
    for method in (
        "ensemble_gate",
        "member_gate",
        "source_fixed_convex",
        "forecast_median_guarded",
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(selected_values(development, method=method, model_id=model))
        expected.append(values)
    count += verify_table(tex, "tab:r6_source", expected)
    expected = []
    for model in ("chronos2", "timesfm2p5", "tirex"):
        values = []
        for setting in ("epochs10_windows64", "epochs50_windows512"):
            values.extend(
                selected_values(
                    budget,
                    panel="combined",
                    dataset_id="current_velocity_H",
                    model_id=model,
                    method="timemixerpp",
                    budget=setting,
                )
            )
        expected.append(values)
    count += verify_table(tex, "tab:budget-origins", expected)
    methods = (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "knn_multivariate",
        "saits",
        "timemixerpp",
        "guarded_direct",
        "motm_reference",
        "forecast_mean_finite",
        "forecast_median_finite",
        "forecast_mean_guarded",
        "forecast_median_guarded",
        "forecast_median_with_motm",
        "old_teacher_rank3",
        "old_source_fixed3",
        "median_risk",
        "member_risk",
        "source_fixed_median_risk",
        "ensemble_gate",
        "member_gate",
        "source_future_gate",
        "gate_source_fixed_convex",
        "gate_source_fixed_single",
    )
    full_results = (source / "sections/r6_full_results.tex").read_text(encoding="utf-8-sig")
    for panel, name in (("new_synthetic_all", "synthetic"), ("new_native_missing", "native")):
        for horizon in (96, 192):
            expected = []
            for method in methods:
                values = []
                for model in ("chronos2", "timesfm2p5"):
                    values.extend(
                        selected_values(
                            results, method=method, model_id=model, horizon=horizon, panel=panel
                        )
                    )
                expected.append(values)
            count += verify_table(full_results, f"tab:r6_all_{name}_h{horizon}", expected)
    inventory = validate_source_inventory(
        (source / "sections/r6_source_inventory.tex").read_text(encoding="utf-8-sig")
    )
    count += inventory["source_inventory_numeric_cells"]
    prefix_root = r6 / "prefix-label-control-v001"
    prefix_manifest = prefix_root / "manifest.json"
    if json.loads(prefix_manifest.read_text(encoding="utf-8"))["status"] != "completed":
        raise ValueError("complete the matched historical-label check")
    dependencies.append(prefix_manifest)
    prefix_rows = read_rows(prefix_root / "comparison_summary.csv")
    prefix_tex = (source / "sections/r6_prefix_results.tex").read_text(encoding="utf-8-sig")
    prefix_methods = (
        "gate_source_fixed_convex",
        "prefix_teacher_half",
        "prefix_teacher_local",
        "prefix_teacher_half_matched",
        "prefix_future_half_matched",
    )
    for horizon in (96, 192):
        expected = []
        for panel in ("new_synthetic_all", "new_native_missing"):
            for method in prefix_methods:
                values = []
                for model in ("chronos2", "timesfm2p5"):
                    values.extend(
                        selected_values(
                            prefix_rows, model_id=model, horizon=horizon, panel=panel, method=method
                        )
                    )
                expected.append(values)
        count += verify_table(prefix_tex, f"tab:r6_prefix_h{horizon}", expected)
    source_controls = (source / "sections/r6_source_controls.tex").read_text(encoding="utf-8")
    for name in (
        "source-future-cv-audit-v001",
        "native-source-audit-v002",
        "interval-gate-audit-v001",
    ):
        path = r6 / name / "manifest.json"
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "completed":
            raise ValueError("complete the additional source-training audits")
        dependencies.append(path)
    future_rows = read_rows(r6 / "source-future-cv-audit-v001/summary.csv")
    expected = []
    for rows, method in (
        (development, "ensemble_gate"),
        (future_rows, "source_future_gate"),
        (future_rows, "future_source_fixed_convex"),
        (future_rows, "forecast_median_guarded"),
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(selected_values(rows, model_id=model, method=method))
        expected.append(values)
    count += verify_table(source_controls, "tab:r6_source_labels", expected)
    native_rows = read_rows(r6 / "native-source-audit-v002/summary.csv")
    expected = []
    for method in (
        "augmented_native_gate",
        "augmented_native_fixed",
        "source_future_gate",
        "future_source_fixed_convex",
        "forecast_median_guarded",
        "forecast_median_with_motm",
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(selected_values(native_rows, model_id=model, method=method))
        expected.append(values)
    count += verify_table(source_controls, "tab:r6_native_source_extension", expected)
    interval_rows = read_rows(r6 / "interval-gate-audit-v001/summary.csv")
    expected = []
    for method in (
        "interval_gate",
        "ensemble_gate",
        "source_fixed_convex",
        "forecast_median_guarded",
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(selected_values(interval_rows, model_id=model, method=method))
        expected.append(values)
    count += verify_table(source_controls, "tab:r6_interval_inputs", expected)
    cohort_path = r6 / "cohort-v001/manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    observed = {}
    for dataset, name in (
        ("beijing_multisite", "Beijing Multi-Site"),
        ("appliances", "Appliances"),
        ("bike_sharing", "Bike Sharing"),
        ("occupancy", "Occupancy measurements"),
    ):
        items = [row for row in cohort["sources"] if row["dataset_id"] == dataset]
        tasks = [row for row in cohort["tasks"] if row["dataset_id"] == dataset]
        observed[name] = [
            len(items),
            items[0]["shape"][1],
            len({row["origin_id"] for row in tasks}),
            len(tasks),
        ]
    actual = {}
    setup = (source / "sections/r6_evaluation_setup.tex").read_text(encoding="utf-8")
    table = setup.split("\\label{tab:r6_coverage}", 1)[1].split("\\end{table}", 1)[0]
    for line in table.splitlines():
        cells = [part.strip().rstrip("\\").strip() for part in line.split("&")]
        if cells[0] in observed:
            actual[cells[0]] = [int(value.replace(",", "")) for value in cells[1:]]
    if actual != observed:
        raise ValueError("confirmation coverage table differs from the frozen cohort")
    prepared_path = r6 / "confirmation-v001/prepared/manifest.json"
    panels = {
        row["episode_id"]: row["panel"]
        for row in json.loads(prepared_path.read_text(encoding="utf-8"))["episodes"]
    }
    context_budgets = []
    for model in ("chronos2", "timesfm2p5"):
        directory = r6 / "confirmation-v001/forecasts" / model
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["horizons"]:
            path = directory / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError("forecast query-count records changed")
            records = [
                row
                for row in json.loads(path.read_text(encoding="utf-8"))["predictions"]
                if panels[row["episode_id"]] == "new_synthetic"
            ]
            if len(records) != 1152:
                raise ValueError("synthetic query-count coverage changed")
            primary = [row["primary_distinct_contexts"] for row in records]
            total = [row["total_distinct_contexts"] for row in records]
            row = {
                "model_id": model,
                "horizon": entry["horizon"],
                "primary_context_mean": sum(primary) / len(primary),
                "with_motm_context_mean": sum(total) / len(total),
                "primary_context_max": max(primary),
                "with_motm_context_max": max(total),
            }
            for field in ("primary_context_mean", "with_motm_context_mean"):
                if f"{row[field]:.3f}" not in tex:
                    raise ValueError("a stated candidate-context count differs from the record")
            context_budgets.append(row)
    return {
        "checked_numeric_cells": count + sum(map(len, observed.values())),
        "synthetic_context_budgets": context_budgets,
        "dependency_sha256": {str(path.relative_to(ROOT)): sha(path) for path in dependencies},
        "cohort_sha256": sha(cohort_path),
        "source_inventory": inventory,
        "limits": "table and coverage checks; prose claims and visual layout require separate review",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed manuscript renders")
    source = ROOT / "docs/iclr2027"
    tex = (source / f"{STEM}.tex").read_text(encoding="utf-8")
    evidence = validate_evidence(tex, source)
    files = [
        f"{STEM}.tex",
        "tsfm_fais_r5_draft.bib",
        "tsfm_fais_iclr2027.bib",
        "tsfm_fais_r6_additions.bib",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
        "figures/r6_forecast_portfolio.tex",
        *[
            f"sections/{name}.tex"
            for name in (
                "r6_method",
                "r6_evaluation_setup",
                "r6_results",
                "r6_full_results",
                "r6_source_inventory",
                "r6_prefix_results",
                "r6_source_controls",
            )
        ],
    ]
    identity = {
        "sources": {name: sha(source / name) for name in files},
        "script_sha256": sha(Path(__file__)),
        "evidence": evidence,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("use a new output directory for a changed draft")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    build = output / "build"
    build.mkdir(exist_ok=True)
    for name in files:
        (build / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, build / name)
    tex_bin = Path("D:/Programme/texlive/2023/bin/windows")
    pdf_bin = Path(
        "C:/Users/MDC/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin"
    )
    executables = {
        name: str(directory / (name + ".exe"))
        for directory, names in (
            (tex_bin, ("pdflatex", "bibtex")),
            (pdf_bin, ("pdfinfo", "pdftoppm")),
        )
        for name in names
    }
    for name, path in executables.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"existing {name} runtime is unavailable")
    commands = [
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        [executables["bibtex"], STEM],
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
    ]
    for index, command in enumerate(commands):
        with (output / f"compile-{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=build,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
    log = (build / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    if any("undefined" in line.lower() for line in log.splitlines()):
        raise ValueError("unresolved manuscript references")
    pdf = output / f"{STEM}.pdf"
    shutil.copyfile(build / pdf.name, pdf)
    info = subprocess.run(
        [executables["pdfinfo"], str(pdf)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout
    (output / "pdfinfo.txt").write_text(info, encoding="utf-8")
    pages = int(re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)[1])
    images = output / "pages"
    images.mkdir(exist_ok=True)
    subprocess.run(
        [executables["pdftoppm"], "-r", "110", "-png", str(pdf), str(images / "page")],
        check=True,
        capture_output=True,
        timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    rendered = sorted(images.glob("page-*.png"))
    if len(rendered) != pages:
        raise ValueError("not every page was rendered")
    manifest = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "evidence": evidence,
        "pdf_sha256": sha(pdf),
        "pages": pages,
        "rendered_pages": [
            {"path": str(path.relative_to(output)), "sha256": sha(path)} for path in rendered
        ],
        "layout_warnings": [
            line for line in log.splitlines() if "Overfull" in line or "Underfull" in line
        ],
        "visual_review_status": "pending",
        "paper_goal_achieved": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "compiled_and_rendered", "pages": pages, "visual_review": "pending"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
