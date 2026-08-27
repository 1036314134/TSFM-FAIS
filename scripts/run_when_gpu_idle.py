"""Launch one argv command after blockers exit and GPU capacity stays available."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import psutil

STATE_SCHEMA_VERSION = 1
RUN_MONITOR_INTERVAL_SECONDS = 30.0
TERMINATION_GRACE_SECONDS = 10.0
STALLED_LOG_WARNING_AFTER_CHECKS = 3
GATE_SAMPLE_HISTORY_LIMIT = 360
RUN_CHECK_HISTORY_LIMIT = 360
NVIDIA_SMI_TIMEOUT_SECONDS = 10.0
# WDDM reports desktop clients as C+G; C and M identify dedicated compute workloads.
COMPUTE_ONLY_PROCESS_TYPES = frozenset({"C", "M"})


@dataclass(frozen=True)
class GpuSample:
    free_mib: int
    utilization_percent: float
    active_compute_pids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GateConfig:
    blocker_pids: tuple[int, ...]
    min_free_mib: int
    max_utilization_percent: float
    consecutive_samples: int
    sample_interval_seconds: float
    hard_timeout_seconds: float
    state_json: Path
    log_file: Path

    def __post_init__(self) -> None:
        if any(pid <= 0 for pid in self.blocker_pids):
            raise ValueError("blocker PIDs must be positive")
        if self.min_free_mib < 0:
            raise ValueError("min-free-mib must be non-negative")
        if not 0 <= self.max_utilization_percent <= 100:
            raise ValueError("max-util must lie between 0 and 100")
        if self.consecutive_samples < 1:
            raise ValueError("consecutive-samples must be positive")
        if self.sample_interval_seconds <= 0:
            raise ValueError("interval must be positive")
        if self.hard_timeout_seconds <= 0:
            raise ValueError("hard-timeout must be positive")
        state_path = os.path.normcase(str(self.state_json.resolve()))
        log_path = os.path.normcase(str(self.log_file.resolve()))
        if state_path == log_path:
            raise ValueError("state-json and log-file must resolve to different paths")


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_bounded_history(
    state: dict[str, Any],
    key: str,
    record: dict[str, Any],
    *,
    limit: int,
) -> None:
    records = state[key]
    stats = state[f"{key}_stats"]
    records.append(record)
    stats["total_count"] += 1
    overflow = len(records) - limit
    if overflow > 0:
        del records[:overflow]
        stats["dropped_count"] += overflow


def _parse_nvidia_quantity(value: str | None, suffix: str, field: str) -> float:
    if value is None:
        raise RuntimeError(f"nvidia-smi XML omitted {field}")
    normalized = value.strip()
    if not normalized.endswith(suffix):
        raise RuntimeError(f"unexpected nvidia-smi {field}: {normalized!r}")
    try:
        return float(normalized[: -len(suffix)].strip())
    except ValueError as error:
        raise RuntimeError(f"invalid nvidia-smi {field}: {normalized!r}") from error


def _sample_gpu() -> GpuSample:
    completed = subprocess.run(
        ["nvidia-smi", "-q", "-x"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    try:
        document = ET.fromstring(completed.stdout)
    except ET.ParseError as error:
        raise RuntimeError("nvidia-smi returned invalid XML") from error
    gpu_nodes = document.findall("gpu")
    if len(gpu_nodes) != 1:
        raise RuntimeError(f"expected exactly one GPU, found {len(gpu_nodes)}")
    gpu = gpu_nodes[0]
    free_mib = _parse_nvidia_quantity(
        gpu.findtext("./fb_memory_usage/free"),
        "MiB",
        "free memory",
    )
    utilization_percent = _parse_nvidia_quantity(
        gpu.findtext("./utilization/gpu_util"),
        "%",
        "GPU utilization",
    )
    active_compute_pids: set[int] = set()
    for process in gpu.findall("./processes/process_info"):
        process_type = (process.findtext("type") or "").strip().upper()
        if process_type not in COMPUTE_ONLY_PROCESS_TYPES:
            continue
        pid_text = (process.findtext("pid") or "").strip()
        try:
            pid = int(pid_text)
        except ValueError as error:
            raise RuntimeError(f"invalid nvidia-smi process PID: {pid_text!r}") from error
        if pid <= 0:
            raise RuntimeError(f"invalid nvidia-smi process PID: {pid_text!r}")
        active_compute_pids.add(pid)
    return GpuSample(
        free_mib=int(free_mib),
        utilization_percent=utilization_percent,
        active_compute_pids=tuple(sorted(active_compute_pids)),
    )


def _process_exists(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _initial_state(config: GateConfig, command: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "waiting_for_gpu",
        "command": list(command),
        "configuration": {
            "blocker_pids": list(config.blocker_pids),
            "min_free_mib": config.min_free_mib,
            "max_utilization_percent": config.max_utilization_percent,
            "consecutive_samples": config.consecutive_samples,
            "sample_interval_seconds": config.sample_interval_seconds,
            "hard_timeout_seconds": config.hard_timeout_seconds,
            "run_monitor_interval_seconds": RUN_MONITOR_INTERVAL_SECONDS,
            "log_stall_warning_after_checks": STALLED_LOG_WARNING_AFTER_CHECKS,
            "gate_sample_history_limit": GATE_SAMPLE_HISTORY_LIMIT,
            "run_check_history_limit": RUN_CHECK_HISTORY_LIMIT,
        },
        "created_at": _utc_now(),
        "gate_samples": [],
        "gate_samples_stats": {"total_count": 0, "dropped_count": 0},
        "run_checks": [],
        "run_checks_stats": {"total_count": 0, "dropped_count": 0},
        "warnings": [],
        "child": None,
        "log_file": str(config.log_file.resolve()),
    }


def _track_launched_process(pid: int) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return None
        process.create_time()
        return process
    except psutil.Error:
        return None


def _is_tracked_process_alive(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _terminate_process_tree(
    process: ChildProcess,
    tracked_root: psutil.Process | None,
    *,
    utc_now: Callable[[], str],
    termination_grace_seconds: float,
) -> dict[str, Any]:
    """Stop only the launched process and descendants tied to its identity."""

    cleanup: dict[str, Any] = {
        "started_at": utc_now(),
        "completed_at": None,
        "root_pid": int(process.pid),
        "tree_tracking_available": tracked_root is not None,
        "tree_discovery_succeeded": False,
        "descendant_pids": [],
        "terminate_sent_pids": [],
        "kill_sent_pids": [],
        "remaining_pids": [],
        "errors": [],
        "root_exit_code": None,
        "verified_complete": False,
    }
    descendants: list[psutil.Process] = []
    if tracked_root is not None and _is_tracked_process_alive(tracked_root):
        try:
            descendants = [
                candidate
                for candidate in tracked_root.children(recursive=True)
                if _is_tracked_process_alive(candidate)
            ]
            cleanup["tree_discovery_succeeded"] = True
        except psutil.Error as error:
            cleanup["errors"].append(f"tree discovery failed: {type(error).__name__}: {error}")
    descendants_by_pid = {candidate.pid: candidate for candidate in descendants}
    descendants = [descendants_by_pid[pid] for pid in sorted(descendants_by_pid)]
    cleanup["descendant_pids"] = [candidate.pid for candidate in descendants]

    if process.poll() is None:
        try:
            process.terminate()
            cleanup["terminate_sent_pids"].append(int(process.pid))
        except Exception as error:
            cleanup["errors"].append(f"root terminate failed: {type(error).__name__}: {error}")
    for descendant in descendants:
        if not _is_tracked_process_alive(descendant):
            continue
        try:
            descendant.terminate()
            cleanup["terminate_sent_pids"].append(descendant.pid)
        except psutil.Error as error:
            cleanup["errors"].append(
                f"PID {descendant.pid} terminate failed: {type(error).__name__}: {error}"
            )

    tracked_processes = ([tracked_root] if tracked_root is not None else []) + descendants
    if tracked_processes:
        try:
            _, alive = psutil.wait_procs(
                tracked_processes,
                timeout=termination_grace_seconds,
            )
        except psutil.Error as error:
            cleanup["errors"].append(f"terminate wait failed: {type(error).__name__}: {error}")
            alive = [
                candidate for candidate in tracked_processes if _is_tracked_process_alive(candidate)
            ]
    else:
        try:
            cleanup["root_exit_code"] = int(process.wait(timeout=termination_grace_seconds))
            alive = []
        except subprocess.TimeoutExpired:
            alive = []
        except Exception as error:
            cleanup["errors"].append(f"root wait failed: {type(error).__name__}: {error}")
            alive = []

    if process.poll() is None:
        try:
            process.kill()
            cleanup["kill_sent_pids"].append(int(process.pid))
        except Exception as error:
            cleanup["errors"].append(f"root kill failed: {type(error).__name__}: {error}")
    for candidate in alive:
        if candidate.pid == process.pid or not _is_tracked_process_alive(candidate):
            continue
        try:
            candidate.kill()
            cleanup["kill_sent_pids"].append(candidate.pid)
        except psutil.Error as error:
            cleanup["errors"].append(
                f"PID {candidate.pid} kill failed: {type(error).__name__}: {error}"
            )

    try:
        cleanup["root_exit_code"] = int(process.wait(timeout=termination_grace_seconds))
    except subprocess.TimeoutExpired:
        cleanup["errors"].append("root remained alive after kill grace period")
    except Exception as error:
        cleanup["errors"].append(f"root final wait failed: {type(error).__name__}: {error}")
    if tracked_processes:
        try:
            _, alive = psutil.wait_procs(
                tracked_processes,
                timeout=termination_grace_seconds,
            )
        except psutil.Error as error:
            cleanup["errors"].append(f"kill wait failed: {type(error).__name__}: {error}")
            alive = [
                candidate for candidate in tracked_processes if _is_tracked_process_alive(candidate)
            ]
    cleanup["remaining_pids"] = sorted(
        {candidate.pid for candidate in alive if _is_tracked_process_alive(candidate)}
    )
    root_stopped = cleanup["root_exit_code"] is not None or process.poll() is not None
    cleanup["verified_complete"] = (
        cleanup["tree_discovery_succeeded"] and root_stopped and not cleanup["remaining_pids"]
    )
    cleanup["completed_at"] = utc_now()
    return cleanup


def _stop_for_timeout(
    process: ChildProcess,
    tracked_root: psutil.Process | None,
    state: dict[str, Any],
    config: GateConfig,
    *,
    utc_now: Callable[[], str],
    termination_grace_seconds: float,
) -> int:
    child = state["child"]
    child["timeout_reached_at"] = utc_now()
    cleanup = _terminate_process_tree(
        process,
        tracked_root,
        utc_now=utc_now,
        termination_grace_seconds=termination_grace_seconds,
    )
    child["cleanup"] = cleanup
    child["terminate_sent"] = int(process.pid) in cleanup["terminate_sent_pids"]
    child["kill_sent"] = int(process.pid) in cleanup["kill_sent_pids"]
    child["ended_at"] = cleanup["completed_at"]
    child["exit_code"] = cleanup["root_exit_code"]
    state["status"] = "timed_out" if cleanup["verified_complete"] else "cleanup_failed"
    _write_state(config.state_json, state)
    return 124


def run_when_gpu_idle(
    config: GateConfig,
    command: Sequence[str],
    *,
    sample_gpu: Callable[[], GpuSample] = _sample_gpu,
    process_exists: Callable[[int], bool] = _process_exists,
    popen_factory: Callable[..., ChildProcess] = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], str] = _utc_now,
    monitor_interval_seconds: float = RUN_MONITOR_INTERVAL_SECONDS,
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
) -> int:
    """Wait for the gate, launch with ``shell=False``, and monitor the child."""

    if not command:
        raise ValueError("the argv after -- must not be empty")
    if monitor_interval_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError("monitor and termination intervals must be positive")
    state = _initial_state(config, command)
    state["created_at"] = utc_now()
    state["configuration"]["run_monitor_interval_seconds"] = monitor_interval_seconds
    _write_state(config.state_json, state)

    consecutive = 0
    while consecutive < config.consecutive_samples:
        alive = [pid for pid in config.blocker_pids if process_exists(pid)]
        sample_record: dict[str, Any] = {
            "sampled_at": utc_now(),
            "alive_blocker_pids": alive,
        }
        try:
            gpu = sample_gpu()
            sample_record.update(
                {
                    "free_mib": gpu.free_mib,
                    "utilization_percent": gpu.utilization_percent,
                    "active_compute_pids": list(gpu.active_compute_pids),
                    "sample_error": None,
                }
            )
            eligible = (
                not alive
                and not gpu.active_compute_pids
                and gpu.free_mib >= config.min_free_mib
                and gpu.utilization_percent <= config.max_utilization_percent
            )
        except Exception as error:  # nvidia-smi can fail transiently while drivers reset.
            sample_record.update(
                {
                    "free_mib": None,
                    "utilization_percent": None,
                    "active_compute_pids": None,
                    "sample_error": f"{type(error).__name__}: {error}",
                }
            )
            eligible = False
        consecutive = consecutive + 1 if eligible else 0
        sample_record["eligible"] = eligible
        sample_record["consecutive_eligible_samples"] = consecutive
        _append_bounded_history(
            state,
            "gate_samples",
            sample_record,
            limit=GATE_SAMPLE_HISTORY_LIMIT,
        )
        _write_state(config.state_json, state)
        if consecutive < config.consecutive_samples:
            sleep(config.sample_interval_seconds)

    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_initial_size = config.log_file.stat().st_size if config.log_file.exists() else 0
    launch_time = utc_now()
    state["status"] = "launching"
    state["gate_satisfied_at"] = launch_time
    _write_state(config.state_json, state)
    process: ChildProcess | None = None
    tracked_root: psutil.Process | None = None
    try:
        with config.log_file.open("ab", buffering=0) as log_handle:
            process = popen_factory(
                list(command),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            state["child"] = {
                "pid": int(process.pid),
                "root_create_time_epoch": None,
                "started_at": launch_time,
                "ended_at": None,
                "exit_code": None,
                "log_initial_size_bytes": log_initial_size,
                "terminate_sent": False,
                "kill_sent": False,
            }
            tracked_root = _track_launched_process(process.pid)
            if tracked_root is not None:
                try:
                    state["child"]["root_create_time_epoch"] = tracked_root.create_time()
                except psutil.Error:
                    tracked_root = None
            started = monotonic()
            state["status"] = "running"
            _write_state(config.state_json, state)
            last_log_size = config.log_file.stat().st_size
            consecutive_stalled_checks = 0

            while True:
                exit_code = process.poll()
                elapsed = monotonic() - started
                if exit_code is not None:
                    state["child"]["ended_at"] = utc_now()
                    state["child"]["exit_code"] = int(exit_code)
                    state["status"] = "completed" if exit_code == 0 else "failed"
                    _write_state(config.state_json, state)
                    return int(exit_code)
                if elapsed >= config.hard_timeout_seconds:
                    return _stop_for_timeout(
                        process,
                        tracked_root,
                        state,
                        config,
                        utc_now=utc_now,
                        termination_grace_seconds=termination_grace_seconds,
                    )

                sleep(
                    min(
                        monitor_interval_seconds,
                        max(0.0, config.hard_timeout_seconds - elapsed),
                    )
                )
                current_log_size = config.log_file.stat().st_size
                grew = current_log_size > last_log_size
                alive_after_sleep = process.poll() is None
                if alive_after_sleep and not grew:
                    consecutive_stalled_checks += 1
                else:
                    consecutive_stalled_checks = 0
                check = {
                    "checked_at": utc_now(),
                    "elapsed_seconds": float(monotonic() - started),
                    "alive": alive_after_sleep,
                    "log_size_bytes": current_log_size,
                    "log_grew": grew,
                    "consecutive_stalled_checks": consecutive_stalled_checks,
                }
                _append_bounded_history(
                    state,
                    "run_checks",
                    check,
                    limit=RUN_CHECK_HISTORY_LIMIT,
                )
                if consecutive_stalled_checks == STALLED_LOG_WARNING_AFTER_CHECKS:
                    state["warnings"].append(
                        {
                            "recorded_at": check["checked_at"],
                            "kind": "log_stalled",
                            "consecutive_checks": consecutive_stalled_checks,
                            "message": "child is alive but the log did not grow for three check intervals",
                        }
                    )
                last_log_size = current_log_size
                _write_state(config.state_json, state)
    except BaseException as error:
        if process is None:
            state["status"] = "launch_failed"
            state["launch_error"] = f"{type(error).__name__}: {error}"
            state["ended_at"] = utc_now()
        else:
            cleanup = _terminate_process_tree(
                process,
                tracked_root,
                utc_now=utc_now,
                termination_grace_seconds=termination_grace_seconds,
            )
            state["child"]["cleanup"] = cleanup
            state["child"]["terminate_sent"] = int(process.pid) in cleanup["terminate_sent_pids"]
            state["child"]["kill_sent"] = int(process.pid) in cleanup["kill_sent_pids"]
            state["child"]["ended_at"] = cleanup["completed_at"]
            state["child"]["exit_code"] = cleanup["root_exit_code"]
            state["status"] = (
                "monitor_failed" if cleanup["verified_complete"] else "monitor_cleanup_failed"
            )
            state["monitor_error"] = f"{type(error).__name__}: {error}"
            state["ended_at"] = cleanup["completed_at"]
        _write_state(config.state_json, state)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker-pids", nargs="*", type=int, default=[])
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--max-util", type=float, required=True)
    parser.add_argument("--consecutive-samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--hard-timeout", type=float, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> tuple[GateConfig, list[str]]:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if "--" not in raw:
        parser.error("a -- separator followed by the child argv is required")
    separator = raw.index("--")
    args = parser.parse_args(raw[:separator])
    command = raw[separator + 1 :]
    if not command:
        parser.error("the child argv after -- must not be empty")
    try:
        config = GateConfig(
            blocker_pids=tuple(args.blocker_pids),
            min_free_mib=args.min_free_mib,
            max_utilization_percent=args.max_util,
            consecutive_samples=args.consecutive_samples,
            sample_interval_seconds=args.interval,
            hard_timeout_seconds=args.hard_timeout,
            state_json=args.state_json,
            log_file=args.log_file,
        )
    except ValueError as error:
        parser.error(str(error))
    return config, command


def main(argv: Sequence[str] | None = None) -> int:
    config, command = _parse_args(argv)
    return run_when_gpu_idle(config, command)


if __name__ == "__main__":
    raise SystemExit(main())
