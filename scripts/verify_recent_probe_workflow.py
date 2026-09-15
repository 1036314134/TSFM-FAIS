"""Run focused CPU-only contract checks before expensive probe experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="probe-check-", dir=args.output.parent.resolve())
    )
    checks = [
        "tests/unit/test_recent_feedback.py",
        "tests/unit/test_recent_probe_analysis.py",
        "tests/unit/test_confirmation_observability.py",
        "tests/unit/test_observed_forecast_accuracy.py",
        "tests/unit/test_utility_idle_resume.py",
        "tests/unit/test_differentiable_imputation_probe.py",
    ]
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *checks, "-q", "--basetemp", str(temporary_directory)],
        cwd=ROOT,
        env=environment,
        timeout=900,
    )
    payload = {
        "status": "completed" if result.returncode == 0 else "failed",
        "checks": checks,
        "test_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in checks
        },
        "exit_code": result.returncode,
        "temporary_directory": str(temporary_directory),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
