"""Resume unfinished R3 followups while yielding to other Python/GPU work."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
from run_when_gpu_idle import (
    _sample_gpu,
    _terminate_process_tree,
    _track_launched_process,
    _utc_now,
    _write_state,
)

ROOT = Path(__file__).resolve().parents[1]


def completed_marker(marker: Path) -> bool:
    if not marker.is_file():
        return False
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("evidence_role") != "development":
        return False
    companions = (
        ("summary.csv", "episode_results.csv", "folds.json")
        if marker.parent.name.startswith("analysis-")
        else ("rows.parquet",)
    )
    return all(
        (marker.parent / name).is_file() and (marker.parent / name).stat().st_size > 0
        for name in companions
    )


def unfinished_steps(root: Path, config: Path) -> list[tuple[str, list[str]]]:
    candidates = [
        (
            "forecaster_transfer",
            "analysis-forecaster-transfer-v001/manifest.json",
            ["analyze_utility_forecaster_transfer.py", "--run-root", str(root)],
        ),
        (
            "timesfm_vendor_missing",
            "timesfm-vendor-missing-v001/manifest.json",
            [
                "evaluate_timesfm_vendor_missing.py",
                "--config",
                str(config),
                "--threads",
                "1",
                "--batch-size",
                "8",
            ],
        ),
    ]
    return [
        (name, [sys.executable, "-u", str(ROOT / "scripts" / command[0]), *command[1:]])
        for name, marker, command in candidates
        if not completed_marker(root / marker)
    ]


def registered_steps(jobs: list[dict]) -> list[tuple[str, list[str]]]:
    remaining = []
    for job in jobs:
        marker = Path(job["completion_marker"])
        complete = False
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                complete = payload.get(job.get("completion_field", "status")) == job.get(
                    "completion_value", "completed"
                )
            except (OSError, ValueError):
                pass
        if not complete:
            remaining.append((job["name"], job["argv"]))
    return remaining


def belongs_to_worker(pid: int, owned: set[int]) -> bool:
    if pid in owned:
        return True
    try:
        return any(parent.pid in owned for parent in psutil.Process(pid).parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def foreign_python_pids(owned: set[int], processes=None) -> list[int]:
    processes = psutil.process_iter(["pid", "name"]) if processes is None else processes
    found = []
    for process in processes:
        try:
            info = process.info
            if (info["name"] or "").lower().startswith(
                ("python", "pypy", "torchrun")
            ) and not belongs_to_worker(info["pid"], owned):
                found.append(int(info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(found)


def owned_pids(child=None) -> set[int]:
    owned = {os.getpid()}
    if child is not None:
        try:
            if child.is_running():
                descendants = child.children(recursive=True)
                owned.add(child.pid)
                owned.update(process.pid for process in descendants)
        except psutil.NoSuchProcess:
            # Exit may race with enumeration; the Popen handle still supplies its exit code.
            pass
    return owned


def inspect_resources(owned: set[int]) -> dict:
    record = {"checked_at": _utc_now(), "foreign_python_pids": foreign_python_pids(owned)}
    try:
        gpu = _sample_gpu()
        record.update(
            foreign_gpu_pids=sorted(
                pid for pid in gpu.active_compute_pids if not belongs_to_worker(pid, owned)
            ),
            gpu_free_mib=gpu.free_mib,
            gpu_utilization=gpu.utilization_percent,
            available_ram_gib=psutil.virtual_memory().available / 1024**3,
            sample_error=None,
        )
    except Exception as error:
        record.update(sample_error=f"{type(error).__name__}: {error}")
    return record


def has_priority_work(sample: dict) -> bool:
    if sample.get("first_priority"):
        return bool(sample.get("sample_error") or sample.get("priority_hold"))
    return bool(
        sample.get("sample_error")
        or sample.get("priority_hold")
        or sample["foreign_python_pids"]
        or sample.get("foreign_gpu_pids")
    )


def priority_hold(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("hold") is not False
    except (OSError, ValueError):
        return True


def quiet_capacity(sample: dict) -> bool:
    return (
        not has_priority_work(sample)
        and sample["gpu_free_mib"] >= 8192
        and sample["available_ram_gib"] >= 8
    )


def can_start(sample: dict, *, max_gpu_utilization=25.0) -> bool:
    return quiet_capacity(sample) and (
        sample.get("first_priority") or sample["gpu_utilization"] <= max_gpu_utilization
    )


def wait_for_capacity(
    state,
    state_path,
    stage,
    quiet_seconds,
    *,
    inspect,
    sleeper=time.sleep,
    clock=time.monotonic,
    max_gpu_utilization=25.0,
):
    quiet_since = None
    while True:
        sample = inspect(owned_pids())
        state.update(status="waiting_for_other_work", last_resource_check=sample, stage=stage)
        if quiet_capacity(sample):
            quiet_since = clock() if quiet_since is None else quiet_since
        else:
            quiet_since = None
        elapsed = 0.0 if quiet_since is None else clock() - quiet_since
        state["quiet_elapsed_seconds"] = elapsed
        _write_state(state_path, state)
        if elapsed >= quiet_seconds and can_start(sample, max_gpu_utilization=max_gpu_utilization):
            # A desktop utilization spike may delay launch, but only priority
            # work or insufficient capacity restarts the compute-idle period.
            final = inspect(owned_pids())
            state["last_resource_check"] = final
            if can_start(final, max_gpu_utilization=max_gpu_utilization):
                return
            if not quiet_capacity(final):
                quiet_since = None
                state["quiet_elapsed_seconds"] = 0.0
            _write_state(state_path, state)
        sleeper(10)


def watch_child(
    process,
    tracked,
    state,
    state_path,
    *,
    inspect=inspect_resources,
    sleeper=time.sleep,
    stop=_terminate_process_tree,
    timeout_seconds=14400.0,
    clock=time.monotonic,
) -> str:
    if timeout_seconds <= 0:
        raise ValueError("child timeout must be positive")
    started = clock()
    while process.poll() is None:
        elapsed = clock() - started
        state["child_elapsed_seconds"] = elapsed
        if elapsed >= timeout_seconds:
            cleanup = stop(process, tracked, utc_now=_utc_now, termination_grace_seconds=3.0)
            state["last_cleanup"] = cleanup
            state["child_pid"] = None
            state["status"] = "timed_out"
            _write_state(state_path, state)
            if not cleanup["verified_complete"]:
                raise RuntimeError("could not verify cleanup after child timeout")
            raise TimeoutError(f"{state['stage']} exceeded {timeout_seconds} seconds")
        sample = inspect(owned_pids(tracked))
        state["last_resource_check"] = sample
        if has_priority_work(sample):
            state["last_yield_reason"] = sample
            cleanup = stop(process, tracked, utc_now=_utc_now, termination_grace_seconds=3.0)
            state["last_cleanup"] = cleanup
            if not cleanup["verified_complete"]:
                raise RuntimeError("could not verify cleanup of this experiment's child")
            state["status"] = "yielded_to_other_work"
            state["yield_count"] += 1
            state["child_pid"] = None
            _write_state(state_path, state)
            return "yielded"
        _write_state(state_path, state)
        sleeper(2.0)
    if process.returncode:
        raise RuntimeError(
            f"{state['stage']} exited with code {process.returncode}; inspect its log"
        )
    state["child_pid"] = None
    return "completed"


def acquire_singleton(path: Path):
    handle = path.open("a+b")
    handle.seek(0)
    handle.write(b"0")
    handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("another R3 idle worker already owns this run") from None
    return handle


def start_exit_notifier(queue: Path, state_path: Path) -> dict:
    """Attach a Windows completion notification without polling the experiment."""
    config = ROOT / "configs/iclr27-r3/queue_notifications.json"
    if os.name != "nt" or not config.is_file():
        return {"status": "disabled"}
    try:
        if not json.loads(config.read_text(encoding="utf-8"))["enabled"]:
            return {"status": "disabled"}
        powershell = shutil.which("powershell.exe")
        if powershell is None:
            raise FileNotFoundError("Windows PowerShell is unavailable")
        pid = os.getpid()
        start_ticks = 621355968000000000 + round(psutil.Process(pid).create_time() * 10_000_000)
        record = queue / f"notification-{pid}.json"
        command = [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-File",
            str(ROOT / "scripts/notify_queue_exit.ps1"),
            "-ConfigPath",
            str(config),
            "-RecordPath",
            str(record),
            "-WorkerProcessId",
            str(pid),
            "-WorkerStartTicks",
            str(start_ticks),
            "-QueueStatePath",
            str(state_path),
        ]
        with (queue / f"notification-{pid}.log").open("ab", buffering=0) as log:
            watcher = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS,
            )
        return {"status": "launched", "watcher_pid": watcher.pid, "record": str(record)}
    except Exception as error:
        # Notification setup must not change the experiment's completion status.
        return {"status": "setup_failed", "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quiet-seconds", type=float, default=180.0)
    parser.add_argument("--max-gpu-utilization", type=float, default=25.0)
    parser.add_argument("--jobs-json", type=Path)
    parser.add_argument("--queue-name", default="idle-resume-v001")
    parser.add_argument("--priority-hold-file", type=Path)
    parser.add_argument("--priority", choices=("third", "first"), default="third")
    args = parser.parse_args()
    minimum_quiet = 0 if args.priority == "first" else 30
    if args.quiet_seconds < minimum_quiet:
        parser.error(f"quiet-seconds must be at least {minimum_quiet} for this priority")
    if not 0 <= args.max_gpu_utilization <= 100:
        parser.error("max-gpu-utilization must be between zero and 100")
    root, config = args.run_root.resolve(), args.config.resolve()
    if not config.is_file() or (
        args.jobs_json is None and not (root / "analysis/analysis_manifest.json").is_file()
    ):
        parser.error("completed primary analysis and an existing config are required")
    if not args.queue_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in args.queue_name
    ):
        parser.error("queue-name must contain only letters, digits, hyphens and underscores")
    jobs = (
        json.loads(args.jobs_json.read_text(encoding="utf-8"))["jobs"] if args.jobs_json else None
    )
    queue = root / args.queue_name
    queue.mkdir(parents=True, exist_ok=True)
    singleton = acquire_singleton(queue / "worker.lock")
    state_path = queue / "state.json"
    state = {
        "worker_pid": os.getpid(),
        "started_at": _utc_now(),
        "status": "waiting_for_other_work",
        "priority": args.priority,
        "quiet_seconds": args.quiet_seconds,
        "max_gpu_utilization": args.max_gpu_utilization,
        "running_check_seconds": 2,
        "yield_count": 0,
        "stage": None,
        "child_pid": None,
        "completed_stages": [],
    }
    process, tracked = None, None

    def inspect_with_priority(owned):
        sample = inspect_resources(owned)
        sample["priority_hold"] = priority_hold(args.priority_hold_file)
        sample["first_priority"] = args.priority == "first"
        return sample

    if os.name == "nt":
        psutil.Process().nice(
            psutil.NORMAL_PRIORITY_CLASS if args.priority == "first" else psutil.IDLE_PRIORITY_CLASS
        )
    try:
        _write_state(state_path, state)
        state["exit_notification"] = start_exit_notifier(queue, state_path)
        _write_state(state_path, state)
        while steps := (
            registered_steps(jobs) if jobs is not None else unfinished_steps(root, config)
        ):
            state["pending_stages"] = [name for name, _ in steps]
            stage, command = steps[0]
            timeout = (
                next(
                    (
                        float(job.get("hard_timeout_seconds", 14400))
                        for job in jobs
                        if job["name"] == stage
                    ),
                    14400.0,
                )
                if jobs is not None
                else 14400.0
            )
            wait_for_capacity(
                state,
                state_path,
                stage,
                args.quiet_seconds,
                inspect=inspect_with_priority,
                max_gpu_utilization=args.max_gpu_utilization,
            )
            flags = (
                subprocess.CREATE_NO_WINDOW
                | (
                    subprocess.NORMAL_PRIORITY_CLASS
                    if args.priority == "first"
                    else subprocess.IDLE_PRIORITY_CLASS
                )
                if os.name == "nt"
                else 0
            )
            environment = dict(os.environ)
            environment.update(
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
            )
            with (queue / f"{stage}.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=flags,
                    env=environment,
                )
                tracked = _track_launched_process(process.pid)
                if tracked is None:
                    raise RuntimeError("could not track the launched worker")
                tracked.cpu_affinity([psutil.Process().cpu_affinity()[-1]])
                state.update(
                    status="running",
                    stage=stage,
                    child_pid=process.pid,
                    child_started_at=_utc_now(),
                    command=command,
                    hard_timeout_seconds=timeout,
                )
                _write_state(state_path, state)
                result = watch_child(
                    process,
                    tracked,
                    state,
                    state_path,
                    inspect=inspect_with_priority,
                    timeout_seconds=timeout,
                )
                process, tracked = None, None
                if result == "completed":
                    state["completed_stages"].append(stage)
        state.update(status="completed", ended_at=_utc_now(), pending_stages=[], child_pid=None)
        _write_state(state_path, state)
    except BaseException as error:
        if process is not None and process.poll() is None:
            state["last_cleanup"] = _terminate_process_tree(
                process, tracked, utc_now=_utc_now, termination_grace_seconds=3.0
            )
        state.update(
            status="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "timed_out"
            if isinstance(error, TimeoutError)
            else "failed",
            error=f"{type(error).__name__}: {error}",
            ended_at=_utc_now(),
        )
        _write_state(state_path, state)
        raise
    finally:
        singleton.close()


if __name__ == "__main__":
    main()
