from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_when_gpu_idle.py"
    spec = importlib.util.spec_from_file_location("run_when_gpu_idle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def utc_now(self) -> str:
        return f"2026-08-06T00:00:{int(self.value):02d}Z"


class CompletingProcess:
    pid = 4321

    def __init__(self, log: BinaryIO) -> None:
        self.log = log
        self.poll_count = 0

    def poll(self) -> int | None:
        self.poll_count += 1
        if self.poll_count == 1:
            self.log.write(b"progress\n")
        return 0 if self.poll_count >= 3 else None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        raise AssertionError("completed process must not be terminated")

    def kill(self) -> None:
        raise AssertionError("completed process must not be killed")


class HangingProcess:
    pid = 9876

    def __init__(self) -> None:
        self.terminate_called = False
        self.kill_called = False

    def poll(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        if not self.kill_called:
            raise subprocess.TimeoutExpired("child", timeout)
        return -9

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True


class GrowthResetProcess:
    pid = 2468

    def __init__(self, log: BinaryIO) -> None:
        self.log = log
        self.poll_count = 0

    def poll(self) -> int | None:
        self.poll_count += 1
        if self.poll_count == 3:
            self.log.write(b"progress\n")
        return 0 if self.poll_count >= 11 else None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        raise AssertionError("completed process must not be terminated")

    def kill(self) -> None:
        raise AssertionError("completed process must not be killed")


def _config(tmp_path: Path, *, consecutive: int = 2, timeout: float = 120.0) -> Any:
    return MODULE.GateConfig(
        blocker_pids=(101,),
        min_free_mib=9000,
        max_utilization_percent=10.0,
        consecutive_samples=consecutive,
        sample_interval_seconds=5.0,
        hard_timeout_seconds=timeout,
        state_json=tmp_path / "state.json",
        log_file=tmp_path / "child.log",
    )


def test_sample_gpu_parses_single_card_xml_and_compute_only_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = """\
<nvidia_smi_log>
  <gpu>
    <fb_memory_usage><free>10001 MiB</free></fb_memory_usage>
    <utilization><gpu_util>3 %</gpu_util></utilization>
    <processes>
      <process_info><pid>42</pid><type>C</type></process_info>
      <process_info><pid>43</pid><type>C+G</type></process_info>
      <process_info><pid>44</pid><type>G</type></process_info>
      <process_info><pid>45</pid><type>M</type></process_info>
    </processes>
  </gpu>
</nvidia_smi_log>
"""

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv == ["nvidia-smi", "-q", "-x"]
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == MODULE.NVIDIA_SMI_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(argv, 0, stdout=xml, stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", run)

    assert MODULE._sample_gpu() == MODULE.GpuSample(10001, 3.0, (42, 45))


def test_parser_preserves_child_argv_after_separator(tmp_path: Path) -> None:
    config, command = MODULE._parse_args(
        [
            "--blocker-pids",
            "10",
            "20",
            "--min-free-mib",
            "9000",
            "--max-util",
            "10",
            "--consecutive-samples",
            "3",
            "--interval",
            "2.5",
            "--hard-timeout",
            "600",
            "--state-json",
            str(tmp_path / "state.json"),
            "--log-file",
            str(tmp_path / "run.log"),
            "--",
            "python.exe",
            "worker.py",
            "--flag=value with spaces",
        ]
    )
    assert config.blocker_pids == (10, 20)
    assert command == ["python.exe", "worker.py", "--flag=value with spaces"]


def test_config_rejects_state_and_log_resolving_to_same_path(tmp_path: Path) -> None:
    shared = tmp_path / "output.json"
    with pytest.raises(
        ValueError,
        match="state-json and log-file must resolve to different paths",
    ):
        MODULE.GateConfig(
            blocker_pids=(),
            min_free_mib=9000,
            max_utilization_percent=10.0,
            consecutive_samples=1,
            sample_interval_seconds=5.0,
            hard_timeout_seconds=60.0,
            state_json=tmp_path / "nested" / ".." / shared.name,
            log_file=shared,
        )


@pytest.mark.parametrize(
    ("key", "limit"),
    [
        ("gate_samples", MODULE.GATE_SAMPLE_HISTORY_LIMIT),
        ("run_checks", MODULE.RUN_CHECK_HISTORY_LIMIT),
    ],
)
def test_bounded_history_keeps_latest_records_and_counts_drops(
    tmp_path: Path,
    key: str,
    limit: int,
) -> None:
    state = MODULE._initial_state(_config(tmp_path), ["python.exe", "job.py"])

    for index in range(limit + 5):
        MODULE._append_bounded_history(state, key, {"index": index}, limit=limit)

    assert len(state[key]) == limit
    assert state[key][0]["index"] == 5
    assert state[key][-1]["index"] == limit + 4
    assert state[f"{key}_stats"] == {
        "total_count": limit + 5,
        "dropped_count": 5,
    }


def test_waits_for_blockers_and_consecutive_gpu_samples(tmp_path: Path) -> None:
    clock = FakeClock()
    gpu_samples = iter(
        [
            MODULE.GpuSample(9500, 5.0),
            MODULE.GpuSample(9500, 5.0),
            MODULE.GpuSample(9500, 5.0, (555,)),
            MODULE.GpuSample(8000, 5.0),
            MODULE.GpuSample(9500, 5.0),
            MODULE.GpuSample(9600, 4.0),
        ]
    )
    blocker_states = iter([True, False, False, False, False, False])
    launch: dict[str, object] = {}

    def popen(argv: list[str], **kwargs: object) -> CompletingProcess:
        launch.update({"argv": argv, **kwargs})
        return CompletingProcess(kwargs["stdout"])  # type: ignore[arg-type]

    exit_code = MODULE.run_when_gpu_idle(
        _config(tmp_path),
        ["python.exe", "job.py", "--x", "a b"],
        sample_gpu=lambda: next(gpu_samples),
        process_exists=lambda pid: next(blocker_states),
        popen_factory=popen,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        utc_now=clock.utc_now,
        monitor_interval_seconds=30.0,
    )

    assert exit_code == 0
    assert launch["argv"] == ["python.exe", "job.py", "--x", "a b"]
    assert launch["shell"] is False
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert len(state["gate_samples"]) == 6
    assert state["gate_samples_stats"] == {"total_count": 6, "dropped_count": 0}
    assert [sample["consecutive_eligible_samples"] for sample in state["gate_samples"]] == [
        0,
        1,
        0,
        0,
        1,
        2,
    ]
    assert state["gate_samples"][2]["active_compute_pids"] == [555]
    assert state["child"]["pid"] == 4321
    assert state["child"]["exit_code"] == 0
    assert len(state["run_checks"]) == 1
    assert state["run_checks_stats"] == {"total_count": 1, "dropped_count": 0}
    assert state["run_checks"][0]["log_grew"] is True
    assert state["warnings"] == []


def test_gpu_query_failure_breaks_consecutive_eligible_samples(tmp_path: Path) -> None:
    clock = FakeClock()
    samples: Any = iter(
        [
            MODULE.GpuSample(9500, 0.0),
            subprocess.TimeoutExpired("nvidia-smi", 10.0),
            MODULE.GpuSample(9500, 0.0),
            MODULE.GpuSample(9500, 0.0),
        ]
    )

    def sample_gpu() -> Any:
        sample = next(samples)
        if isinstance(sample, BaseException):
            raise sample
        return sample

    def popen(argv: list[str], **kwargs: object) -> CompletingProcess:
        return CompletingProcess(kwargs["stdout"])  # type: ignore[arg-type]

    exit_code = MODULE.run_when_gpu_idle(
        _config(tmp_path),
        ["python.exe", "job.py"],
        sample_gpu=sample_gpu,
        process_exists=lambda pid: False,
        popen_factory=popen,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        utc_now=clock.utc_now,
        monitor_interval_seconds=30.0,
    )

    assert exit_code == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert [sample["consecutive_eligible_samples"] for sample in state["gate_samples"]] == [
        1,
        0,
        1,
        2,
    ]
    assert state["gate_samples"][1]["eligible"] is False
    assert state["gate_samples"][1]["active_compute_pids"] is None
    assert state["gate_samples"][1]["sample_error"].startswith("TimeoutExpired:")


def test_hard_timeout_terminates_then_kills_without_gpu_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    process = HangingProcess()
    monkeypatch.setattr(MODULE, "_track_launched_process", lambda _pid: None)

    exit_code = MODULE.run_when_gpu_idle(
        _config(tmp_path, consecutive=1, timeout=50.0),
        ["python.exe", "hung.py"],
        sample_gpu=lambda: MODULE.GpuSample(9500, 0.0),
        process_exists=lambda pid: False,
        popen_factory=lambda *args, **kwargs: process,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        utc_now=clock.utc_now,
        monitor_interval_seconds=30.0,
        termination_grace_seconds=3.0,
    )

    assert exit_code == 124
    assert process.terminate_called is True
    assert process.kill_called is True
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "cleanup_failed"
    assert state["child"]["terminate_sent"] is True
    assert state["child"]["kill_sent"] is True
    assert state["child"]["exit_code"] == -9
    assert state["child"]["cleanup"]["verified_complete"] is False
    assert state["child"]["cleanup"]["tree_tracking_available"] is False
    assert state["warnings"] == []


def test_log_growth_resets_stalled_check_count_before_warning(tmp_path: Path) -> None:
    clock = FakeClock()

    def popen(argv: list[str], **kwargs: object) -> GrowthResetProcess:
        return GrowthResetProcess(kwargs["stdout"])  # type: ignore[arg-type]

    exit_code = MODULE.run_when_gpu_idle(
        _config(tmp_path, consecutive=1, timeout=300.0),
        ["python.exe", "job.py"],
        sample_gpu=lambda: MODULE.GpuSample(9500, 0.0),
        process_exists=lambda pid: False,
        popen_factory=popen,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        utc_now=clock.utc_now,
        monitor_interval_seconds=30.0,
    )

    assert exit_code == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert [check["consecutive_stalled_checks"] for check in state["run_checks"]] == [
        1,
        0,
        1,
        2,
        3,
    ]
    assert len(state["warnings"]) == 1
    assert state["warnings"][0]["kind"] == "log_stalled"
    assert state["warnings"][0]["consecutive_checks"] == 3


def test_monitor_exception_stops_launched_process(tmp_path: Path) -> None:
    launched: subprocess.Popen[bytes] | None = None

    def popen(argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal launched
        launched = subprocess.Popen(argv, **kwargs)  # type: ignore[arg-type]
        return launched

    def fail_monitor_sleep(seconds: float) -> None:
        raise RuntimeError(f"monitor failed before sleeping {seconds} seconds")

    with pytest.raises(RuntimeError, match="monitor failed"):
        MODULE.run_when_gpu_idle(
            _config(tmp_path, consecutive=1),
            [sys.executable, "-c", "import time; time.sleep(60)"],
            sample_gpu=lambda: MODULE.GpuSample(9500, 0.0),
            process_exists=lambda pid: False,
            popen_factory=popen,
            sleep=fail_monitor_sleep,
            termination_grace_seconds=3.0,
        )

    assert launched is not None
    try:
        assert launched.poll() is not None
        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "monitor_failed"
        assert state["child"]["cleanup"]["verified_complete"] is True
        assert state["child"]["cleanup"]["remaining_pids"] == []
    finally:
        if launched.poll() is None:
            launched.kill()
            launched.wait(timeout=3.0)


def test_state_write_failure_after_launch_stops_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: subprocess.Popen[bytes] | None = None
    original_write_state = MODULE._write_state
    failed_once = False

    def popen(argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal launched
        launched = subprocess.Popen(argv, **kwargs)  # type: ignore[arg-type]
        return launched

    def write_state(path: Path, state: dict[str, Any]) -> None:
        nonlocal failed_once
        if state["status"] == "running" and not failed_once:
            failed_once = True
            raise OSError("transient state write failure")
        original_write_state(path, state)

    monkeypatch.setattr(MODULE, "_write_state", write_state)
    with pytest.raises(OSError, match="transient state write failure"):
        MODULE.run_when_gpu_idle(
            _config(tmp_path, consecutive=1),
            [sys.executable, "-c", "import time; time.sleep(60)"],
            sample_gpu=lambda: MODULE.GpuSample(9500, 0.0),
            process_exists=lambda pid: False,
            popen_factory=popen,
            termination_grace_seconds=3.0,
        )

    assert failed_once is True
    assert launched is not None
    try:
        assert launched.poll() is not None
        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "monitor_failed"
        assert state["child"]["cleanup"]["verified_complete"] is True
        assert state["monitor_error"].startswith("OSError:")
    finally:
        if launched.poll() is None:
            launched.kill()
            launched.wait(timeout=3.0)


def test_tree_cleanup_stops_descendant_and_leaves_blocker_running() -> None:
    blocker: subprocess.Popen[bytes] | None = None
    root: subprocess.Popen[str] | None = None
    descendant: Any = None
    sleeper = "import time; time.sleep(60)"
    tree_launcher = (
        "import subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {sleeper!r}]); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    try:
        blocker = subprocess.Popen(
            [sys.executable, "-c", sleeper],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        root = subprocess.Popen(
            [sys.executable, "-c", tree_launcher],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert root.stdout is not None
        descendant_pid = int(root.stdout.readline().strip())
        descendant = MODULE.psutil.Process(descendant_pid)
        assert descendant.is_running()
        tracked_root = MODULE._track_launched_process(root.pid)
        assert tracked_root is not None

        cleanup = MODULE._terminate_process_tree(
            root,
            tracked_root,
            utc_now=lambda: "2026-08-06T00:00:00Z",
            termination_grace_seconds=3.0,
        )

        assert cleanup["verified_complete"] is True
        assert descendant_pid in cleanup["descendant_pids"]
        assert cleanup["remaining_pids"] == []
        assert root.poll() is not None
        assert descendant.is_running() is False
        assert blocker.poll() is None
        assert blocker.pid not in cleanup["terminate_sent_pids"]
        assert blocker.pid not in cleanup["kill_sent_pids"]
    finally:
        if root is not None and root.poll() is None:
            root.kill()
            root.wait(timeout=3.0)
        if descendant is not None and descendant.is_running():
            descendant.kill()
            descendant.wait(timeout=3.0)
        if blocker is not None and blocker.poll() is None:
            blocker.kill()
            blocker.wait(timeout=3.0)
