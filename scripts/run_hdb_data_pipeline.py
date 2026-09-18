"""Run public raw-data collection and its audit with a compact background status file."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.output_root.resolve()
    root = Path(__file__).resolve().parents[1]
    base.mkdir(parents=True, exist_ok=True)
    registration = json.loads((base / "registration.json").read_text(encoding="utf-8"))
    for row in registration["files"]:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("a registered acquisition runtime changed")
    if (base / "audit_manifest.json").exists():
        raise ValueError("preserve completed acquisition pipelines")
    state = {"worker_pid": os.getpid(), "started_unix": time.time(), "status": "starting"}

    def save():
        temporary = base / "pipeline_state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(base / "pipeline_state.json")

    save()
    try:
        for stage, script, marker in (
            ("collecting", "collect_hdb_hourly.py", "manifest.json"),
            ("auditing", "audit_hdb_hourly.py", "audit_manifest.json"),
        ):
            if (base / marker).exists():
                if json.loads((base / marker).read_text(encoding="utf-8"))["status"] != "completed":
                    raise ValueError("an acquisition stage marker is incomplete")
                continue
            state.update(status=stage, stage_started_unix=time.time())
            argv = [
                sys.executable,
                "-u",
                str(root / "scripts" / script),
                "--output-root",
                str(base),
            ]
            with (base / f"{stage}-{time.time_ns()}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise
            state["child_pid"] = None
            if code != 0 or not (base / marker).exists():
                raise RuntimeError(f"{stage} failed with return code {code}")
        state.update(status="completed", ended_unix=time.time())
    except BaseException as error:
        state.update(
            status="failed", error=f"{type(error).__name__}: {error}", ended_unix=time.time()
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
