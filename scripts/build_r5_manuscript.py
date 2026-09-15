"""Check the two numerical tables, compile the current draft, and render its pages."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "tsfm_fais_r5_draft"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def table_numbers(tex, label):
    table = tex.split("\\label{" + label + "}", 1)[1].split("\\end{table}", 1)[0]
    return [
        [
            float(value)
            for value in re.findall(r"(?<![A-Za-z0-9])(?:\d+)?\.\d+", line.split("&", 1)[1])
        ]
        for line in table.splitlines()
        if re.search(r"&\s+(?:\d+)?\.\d+", line)
    ]


def validate_tables(tex):
    confirmation = ROOT / "artifacts/iclr27-r5/native-confirmation-readout-v001"
    scope = ROOT / "artifacts/iclr27-r5/imputer-input-scope-v001/chronos2"
    for manifest in (
        confirmation / "manifest.json",
        ROOT / "artifacts/iclr27-r5/native-confirmation-audit-v001/manifest.json",
        ROOT / "artifacts/iclr27-r5/imputer-input-scope-audit-v001/manifest.json",
    ):
        if json.loads(manifest.read_text(encoding="utf-8-sig"))["status"] != "completed":
            raise ValueError("complete the result checks before manuscript rendering")
    methods = (
        "guarded_direct",
        "input_median_finite",
        "forecast_median_guarded",
        "source_fixed3_clean_forecast_mse",
        "future_supervised_rank3",
        "teacher_rank3",
        "forecast_median_with_motm",
    )
    records = csv_rows(confirmation / "summary.csv")
    expected = []
    for panel in ("all_registered", "naturally_missing"):
        for method in methods:
            values = []
            for model in ("chronos2", "timesfm2p5"):
                matches = [
                    row
                    for row in records
                    if (row["panel"], row["method"], row["model_id"]) == (panel, method, model)
                ]
                if len(matches) != 1:
                    raise ValueError("ambiguous confirmation table source")
                values.extend(
                    float(f"{float(matches[0][metric]):.6f}") for metric in ("mae", "mse")
                )
            expected.append(values)
    if table_numbers(tex, "tab:confirmation") != expected:
        raise ValueError("the manuscript confirmation table differs from the observed results")
    records = csv_rows(scope / "summary.csv")
    expected_scope = []
    for variant in ("base", "targets_only", "covariates_only", "both"):
        matches = [
            row
            for row in records
            if row["dataset_id"] == "electricity"
            and row["imputer"] == "timemixerpp"
            and row["budget"] == "epochs50_windows512"
            and row["variant"] == variant
        ]
        if len(matches) != 1:
            raise ValueError("ambiguous crossed-input table source")
        expected_scope.append(
            [float(f"{float(matches[0][metric]):.6f}") for metric in ("mae", "mse")]
        )
    if table_numbers(tex, "tab:crossed") != expected_scope:
        raise ValueError("the manuscript crossed-input table differs from the observed results")
    return {
        "confirmation_summary_sha256": sha(confirmation / "summary.csv"),
        "crossed_input_summary_sha256": sha(scope / "summary.csv"),
        "checked_numerical_cells": sum(map(len, expected)) + sum(map(len, expected_scope)),
    }


def validate_followup_tables(tex):
    diagnostic = ROOT / "artifacts/iclr27-r5/triple-objective-diagnostic-v001"
    origins = ROOT / "artifacts/iclr27-r5/imputer-budget-origin-readout-v001"
    for directory in (diagnostic, origins):
        if (
            json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"]
            != "completed"
        ):
            raise ValueError("complete the follow-up results before citing their tables")

    def row_values(records, **conditions):
        matches = [
            row for row in records if all(row[key] == value for key, value in conditions.items())
        ]
        if len(matches) != 1:
            raise ValueError("ambiguous follow-up result")
        return [float(f"{float(matches[0][metric]):.6f}") for metric in ("mae", "mse")]

    records = csv_rows(diagnostic / "summary.csv")
    objective_values = []
    for method in (
        "forecast_median_seven",
        "learned_teacher_rank3",
        "unavailable_individual_teacher_top3",
        "unavailable_joint_teacher_best3",
        "future_oracle_single_mse",
        "future_oracle_triple_mse",
    ):
        values = []
        for model in ("chronos2", "timesfm2p5"):
            values.extend(row_values(records, method=method, model_id=model))
        objective_values.append(values)
    if table_numbers(tex, "tab:objective-gap") != objective_values:
        raise ValueError("the objective-gap table differs from its recorded diagnostic")
    records = csv_rows(origins / "summary.csv")
    origin_values = []
    for model in ("chronos2", "timesfm2p5", "tirex"):
        values = []
        for budget in ("epochs10_windows64", "epochs50_windows512"):
            values.extend(
                row_values(
                    records,
                    panel="combined",
                    dataset_id="current_velocity_H",
                    model_id=model,
                    method="timemixerpp",
                    budget=budget,
                )
            )
        origin_values.append(values)
    if table_numbers(tex, "tab:budget-origins") != origin_values:
        raise ValueError("the four-history budget table differs from its recorded results")
    return {
        "objective_summary_sha256": sha(diagnostic / "summary.csv"),
        "origin_summary_sha256": sha(origins / "summary.csv"),
        "checked_numerical_cells": 36,
    }


def validate_source_inventory(tex):
    accuracy_root = ROOT / "artifacts/iclr27-r4/accuracy-development-v002"
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    path = Path(accuracy["source_root"]) / "episodes_manifest.json"
    if sha(path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("the source inventory metadata changed")
    source = json.loads(path.read_text(encoding="utf-8"))
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((accuracy_root / "standardizers.json").read_text(encoding="utf-8"))
    }
    groups = {}
    for row in source["episodes"]:
        groups.setdefault(row["dataset_id"], []).append(row)
    expected = {}
    for dataset, rows in groups.items():
        expected[dataset] = [
            len(scalers[(dataset, rows[0]["item_id"])]["mean"]),
            rows[0]["period"],
            len({row["origin_id"] for row in rows if row["split"] == "train"}),
            len({row["origin_id"] for row in rows if row["split"] == "validation"}),
        ]
    table = tex.split("\\label{tab:source-inventory}", 1)[1].split("\\end{table}", 1)[0]
    actual = {}
    for line in table.splitlines():
        cells = [cell.strip().rstrip("\\").strip() for cell in line.split("&")]
        if len(cells) == 5 and all(re.fullmatch(r"\d+", value) for value in cells[1:]):
            dataset = cells[0].replace("\\_", "_")
            if dataset in actual:
                raise ValueError("a source dataset appears twice in the manuscript table")
            actual[dataset] = [int(value) for value in cells[1:]]
    if actual != expected:
        raise ValueError("the source inventory table differs from the accepted episodes")
    return {
        "source_inventory_sha256": sha(path),
        "source_inventory_numeric_cells": len(expected) * 4,
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
    evidence = validate_tables(tex)
    followup = validate_followup_tables(tex)
    evidence["checked_numerical_cells"] += followup.pop("checked_numerical_cells")
    evidence.update(followup)
    inventory = validate_source_inventory(tex)
    evidence["checked_numerical_cells"] += inventory["source_inventory_numeric_cells"]
    evidence.update(inventory)
    files = [
        f"{STEM}.tex",
        f"{STEM}.bib",
        "tsfm_fais_iclr2027.bib",
        "iclr2027_conference.sty",
        "iclr2027_conference.bst",
        "natbib.sty",
        "fancyhdr.sty",
        "figures/r5_vector_forecast_geometry.tex",
    ]
    identity = {
        "sources": {name: sha(source / name) for name in files},
        "script_sha256": sha(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("the manuscript changed during a partial render; use a new output version")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    build = output / "build"
    build.mkdir(exist_ok=True)
    for name in files:
        (build / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, build / name)
    executables = {}
    for name in ("pdflatex", "bibtex", "pdftoppm", "pdfinfo"):
        executables[name] = shutil.which(name)
        if executables[name] is None:
            raise FileNotFoundError(f"the existing {name} runtime was not found")
    commands = [
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        [executables["bibtex"], STEM],
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
        [executables["pdflatex"], "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
    ]
    for index, command in enumerate(commands):
        with (output / f"compile-{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600
            )
    log = (build / f"{STEM}.log").read_text(encoding="utf-8", errors="replace")
    unresolved = [line for line in log.splitlines() if "undefined" in line.lower()]
    if unresolved:
        raise ValueError(f"unresolved LaTeX references: {unresolved}")
    pdf = output / f"{STEM}.pdf"
    shutil.copyfile(build / pdf.name, pdf)
    information = subprocess.run(
        [executables["pdfinfo"], str(pdf)], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    (output / "pdfinfo.txt").write_text(information, encoding="utf-8")
    pages = int(re.search(r"^Pages:\s+(\d+)", information, flags=re.MULTILINE)[1])
    images = output / "pages"
    images.mkdir(exist_ok=True)
    subprocess.run(
        [executables["pdftoppm"], "-r", "110", "-png", str(pdf), str(images / "page")],
        check=True,
        capture_output=True,
        timeout=600,
    )
    rendered = sorted(images.glob("page-*.png"))
    if len(rendered) != pages:
        raise ValueError("not every PDF page was rendered")
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
        "paper_readiness": "not established by compilation or numerical table checks",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "compiled_and_rendered", "pages": pages, "visual_review": "pending"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
