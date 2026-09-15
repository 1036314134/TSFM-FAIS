"""Run the cached MAE/MSE development stages using one low-priority CPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--objectives", default="joint,mae,mse")
    args = parser.parse_args()
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[-1]])
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": os.getpid(),
        "status": "running",
        "device": "cpu",
        "priority": "Idle",
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        temporary = output / "workflow_state.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(output / "workflow_state.json")

    commands = []
    if not (output / "manifest.json").exists():
        commands.append(
            (
                "export",
                [
                    "export_downstream_accuracy.py",
                    "--source-root",
                    str(args.source_root.resolve()),
                    "--output-root",
                    str(output),
                ],
            )
        )
    analysis = output / ("analysis-" + args.objectives.replace(",", "-") + "-v001")
    if not (analysis / "manifest.json").exists():
        commands.append(
            (
                "selection",
                [
                    "analyze_downstream_accuracy.py",
                    "--run-root",
                    str(output),
                    "--objectives",
                    args.objectives,
                ],
            )
        )
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"
    )
    try:
        for name, command in commands:
            state["stage"] = name
            save()
            subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts" / command[0]), *command[1:]],
                cwd=ROOT,
                env=environment,
                check=True,
            )
        state.update(status="completed", ended_at=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
