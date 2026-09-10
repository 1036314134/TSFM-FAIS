"""Run a versioned R3 development stage without altering R2 artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.utility_experiment import (  # noqa: E402
    analyze_utility_experiment,
    forecast_utility_episodes,
    load_utility_config,
    prepare_utility_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "forecast", "analyze", "run", "followups"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", default="chronos2,timesfm2p5")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    config = load_utility_config(args.config)
    models = tuple(args.models.split(","))
    if args.stage == "followups":
        if models != ("chronos2", "timesfm2p5"):
            parser.error("development followups currently compare Chronos-2 and TimesFM 2.5")
        deadline = monotonic() + args.wait_seconds
        completion = config.output_root / "analysis" / "analysis_manifest.json"
        if not completion.exists():
            print("Waiting for the registered forecast and primary-analysis stages.", flush=True)
        while not completion.exists():
            if monotonic() >= deadline:
                raise TimeoutError("primary analysis has not completed")
            sleep(min(10, max(0.1, deadline - monotonic())))
        commands = [
            [
                "analyze_utility_anchor_ablation.py",
                "--run-root",
                str(config.output_root),
                "--objective",
                objective,
            ]
            for objective in ("regression_l1", "regression", "bounded_regression")
        ]
        commands.extend(
            [
                ["analyze_utility_history_baseline.py", "--config", str(args.config.resolve())],
                ["analyze_utility_forecaster_transfer.py", "--run-root", str(config.output_root)],
                ["evaluate_timesfm_vendor_missing.py", "--config", str(args.config.resolve())],
            ]
        )
        for script, *arguments in commands:
            print(
                json.dumps({"stage": "followup", "script": script, "arguments": arguments}),
                flush=True,
            )
            subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts" / script), *arguments], check=True
            )
        print(
            json.dumps(
                {
                    "stage": "followups",
                    "status": "completed",
                    "output_root": str(config.output_root),
                }
            ),
            flush=True,
        )
        return
    if args.stage == "run":
        stages = [
            ("prepare", None),
            *(("forecast", model) for model in models),
            ("analyze", args.models),
        ]
        for stage, stage_models in stages:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                stage,
                "--config",
                str(args.config.resolve()),
            ]
            if stage_models is not None:
                command.extend(["--models", stage_models])
            subprocess.run(command, check=True)
        return
    if args.stage == "analyze":
        result = analyze_utility_experiment(config, models)
    else:
        import torch

        torch.set_num_threads(4)
        if args.stage == "prepare":
            result = prepare_utility_episodes(config)
        else:
            if len(models) != 1:
                parser.error("forecast accepts one model per process to release GPU memory")
            result = forecast_utility_episodes(config, models[0])
    print(
        json.dumps(
            {
                "status": "completed",
                "stage": args.stage,
                "output_root": str(config.output_root),
                "evidence_role": config.evidence_role,
                "records": result.get("row_count", len(result.get("episodes", []))),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
