"""Run the remaining target-assembly checks before R6 policy evaluation."""

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_root / "manifest.json").exists():
        raise ValueError("preserve the completed policy runtime check")
    args.output_root.mkdir(parents=True, exist_ok=True)
    code = pytest.main(
        [
            "-q",
            str(ROOT / "tests/unit/test_r6_policy_outputs.py"),
            "--basetemp=" + str(args.output_root / "pytest"),
        ]
    )
    if code != 0:
        raise ValueError("target assembly or frozen portfolio interpretation failed")
    _write_json(
        args.output_root / "manifest.json",
        {
            "status": "completed",
            "tests_passed": 4,
            "runtime_sha256": {
                name: file_sha256(ROOT / name)
                for name in (
                    "scripts/evaluate_r6_policies.py",
                    "scripts/r6_policy_inputs.py",
                    "scripts/r6_runtime.py",
                    "src/tsfm_fais/routing/forecast_gate.py",
                    "src/tsfm_fais/forecasting/observed_accuracy.py",
                )
            },
            "new_forecaster_calls": 0,
            "limits": "input shape and target/portfolio assembly tests; actual policy forecasts and metrics require separate replay",
        },
    )
    print(json.dumps({"status": "completed", "tests_passed": 4}), flush=True)


if __name__ == "__main__":
    main()
