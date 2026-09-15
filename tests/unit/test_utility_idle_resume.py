from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    path = Path(__file__).parents[2] / "scripts/resume_utility_when_idle.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("utility_idle_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(**changes):
    return {
        "foreign_python_pids": [],
        "foreign_gpu_pids": [],
        "gpu_free_mib": 10000,
        "gpu_utilization": 5,
        "available_ram_gib": 16,
        "sample_error": None,
    } | changes


def test_foreign_python_detection_excludes_owned_processes(worker):
    processes = [
        SimpleNamespace(info={"pid": pid, "name": name})
        for pid, name in [
            (1, "python.exe"),
            (2, "python.exe"),
            (3, "pythonw.exe"),
            (4, "torchrun.exe"),
            (5, "chrome.exe"),
        ]
    ]
    assert worker.foreign_python_pids({1, 2}, processes) == [3, 4]


def test_unrelated_desktop_processes_do_not_trigger_expensive_ancestry_queries(worker, monkeypatch):
    queried = []

    def belongs(pid, owned):
        queried.append(pid)
        return pid in owned

    monkeypatch.setattr(worker, "belongs_to_worker", belongs)
    processes = [
        SimpleNamespace(info={"pid": 1, "name": "chrome.exe"}),
        SimpleNamespace(info={"pid": 2, "name": "python.exe"}),
        SimpleNamespace(info={"pid": 3, "name": "explorer.exe"}),
    ]
    assert worker.foreign_python_pids(set(), processes) == [2]
    assert queried == [2]


@pytest.mark.parametrize(
    "change",
    [
        {"foreign_python_pids": [88]},
        {"foreign_gpu_pids": [99]},
        {"sample_error": "unavailable"},
        {"gpu_free_mib": 4000},
        {"available_ram_gib": 4},
        {"gpu_utilization": 90},
    ],
)
def test_start_gate_waits_for_priority_work_and_capacity(worker, change):
    assert worker.can_start(sample())
    assert not worker.can_start(sample(**change))


def test_explicit_first_priority_does_not_yield_to_foreign_processes(worker):
    active = sample(
        first_priority=True, foreign_python_pids=[88], foreign_gpu_pids=[99], gpu_utilization=99
    )
    assert not worker.has_priority_work(active)
    assert worker.can_start(active)
    assert not worker.can_start(active | {"gpu_free_mib": 4000})
    assert not worker.can_start(active | {"available_ram_gib": 4})
    assert worker.has_priority_work(active | {"sample_error": "unavailable"})
    assert worker.has_priority_work(active | {"priority_hold": True})


def test_new_priority_process_only_stops_this_worker_tree(worker, tmp_path):
    child = SimpleNamespace(pid=111, poll=lambda: None, returncode=None)
    tracked = SimpleNamespace(
        pid=111, is_running=lambda: True, children=lambda recursive: [SimpleNamespace(pid=112)]
    )
    stopped = []

    def inspect(owned):
        assert owned == {os.getpid(), 111, 112}
        return sample(foreign_python_pids=[333], foreign_gpu_pids=[333])

    def stop(process, identity, **kwargs):
        stopped.append((process.pid, identity.pid))
        return {"verified_complete": True}

    state = {"stage": "test", "yield_count": 0, "child_pid": 111}
    result = worker.watch_child(
        child,
        tracked,
        state,
        tmp_path / "state.json",
        inspect=inspect,
        stop=stop,
        sleeper=lambda _: pytest.fail("must yield immediately"),
    )
    assert result == "yielded"
    assert stopped == [(111, 111)]
    assert state["child_pid"] is None and state["yield_count"] == 1


def test_completed_stage_is_skipped_but_partial_marker_is_not(worker, tmp_path):
    path = tmp_path / "analysis-forecaster-transfer-v001"
    path.mkdir()
    marker = path / "manifest.json"
    marker.write_text('{"evidence_role":', encoding="utf-8")
    assert len(worker.unfinished_steps(tmp_path, tmp_path / "config.yaml")) == 2
    marker.write_text(json.dumps({"evidence_role": "development"}), encoding="utf-8")
    assert not worker.completed_marker(marker)
    for name in ("summary.csv", "episode_results.csv", "folds.json"):
        (path / name).write_text("data", encoding="utf-8")
    assert [name for name, _ in worker.unfinished_steps(tmp_path, tmp_path / "config.yaml")] == [
        "timesfm_vendor_missing"
    ]


def test_duplicate_idle_worker_is_rejected_and_exit_releases_ownership(worker, tmp_path):
    path = tmp_path / "worker.lock"
    first = worker.acquire_singleton(path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            worker.acquire_singleton(path)
    finally:
        first.close()
    worker.acquire_singleton(path).close()


def test_registered_work_only_skips_matching_completion_status(worker, tmp_path):
    marker = tmp_path / "completion.json"
    jobs = [{"name": "check", "argv": ["python", "check.py"], "completion_marker": str(marker)}]
    marker.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert len(worker.registered_steps(jobs)) == 1
    marker.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert worker.registered_steps(jobs) == []


def test_priority_hold_requires_explicit_release(worker, tmp_path):
    path = tmp_path / "hold.json"
    assert worker.priority_hold(path)
    path.write_text(json.dumps({"hold": True}), encoding="utf-8")
    assert worker.priority_hold(path)
    assert not worker.can_start(sample(priority_hold=True))
    path.write_text(json.dumps({"hold": False}), encoding="utf-8")
    assert not worker.priority_hold(path)


def test_hard_timeout_cleans_only_the_owned_child(worker, tmp_path):
    child = SimpleNamespace(pid=111, poll=lambda: None, returncode=None)
    tracked = SimpleNamespace(pid=111, is_running=lambda: True, children=lambda recursive: [])
    stopped = []

    def stop(process, identity, **kwargs):
        stopped.append((process.pid, identity.pid))
        return {"verified_complete": True}

    times = iter([0.0, 2.0])
    state = {"stage": "test", "yield_count": 0, "child_pid": 111}
    with pytest.raises(TimeoutError, match="exceeded"):
        worker.watch_child(
            child,
            tracked,
            state,
            tmp_path / "state.json",
            timeout_seconds=1,
            clock=lambda: next(times),
            stop=stop,
        )
    assert stopped == [(111, 111)]
    assert state["status"] == "timed_out"


def test_desktop_gpu_spike_does_not_erase_a_genuine_compute_idle_period(worker):
    desktop = sample(gpu_utilization=31)
    assert worker.quiet_capacity(desktop)
    assert not worker.can_start(desktop)
    assert not worker.quiet_capacity(sample(foreign_python_pids=[99]))
    assert not worker.quiet_capacity(sample(priority_hold=True))


def test_desktop_utilization_allowance_never_overrides_other_experiment_priority(worker):
    assert worker.can_start(sample(gpu_utilization=53), max_gpu_utilization=60)
    assert not worker.can_start(sample(gpu_utilization=61), max_gpu_utilization=60)
    for change in (
        {"foreign_python_pids": [99]},
        {"foreign_gpu_pids": [99]},
        {"priority_hold": True},
        {"gpu_free_mib": 4000},
        {"available_ram_gib": 4},
        {"sample_error": "unavailable"},
    ):
        assert not worker.can_start(sample(**change), max_gpu_utilization=60)


def test_newly_spawned_own_descendant_is_not_mistaken_for_another_project(worker, monkeypatch):
    processes = [
        SimpleNamespace(info={"pid": 333, "name": "python.exe"}),
        SimpleNamespace(info={"pid": 444, "name": "python.exe"}),
    ]
    monkeypatch.setattr(
        worker.psutil,
        "Process",
        lambda pid: SimpleNamespace(
            parents=lambda: [SimpleNamespace(pid=111)] if pid == 333 else []
        ),
    )
    assert worker.foreign_python_pids({111}, processes) == [444]


@pytest.mark.parametrize(
    ("last_check", "expected_elapsed"),
    [
        ({"gpu_utilization": 50}, 40),
        ({"foreign_python_pids": [99]}, 70),
        ({"priority_hold": True}, 70),
        ({"available_ram_gib": 4}, 70),
    ],
)
def test_final_resource_recheck_preserves_only_real_compute_idle_time(
    worker, tmp_path, last_check, expected_elapsed
):
    ticks = [0]
    samples = iter([sample(), sample(), sample(), sample(), sample(**last_check)])

    def inspect(_):
        return next(samples, sample())

    def sleep(seconds):
        ticks[0] += seconds
        assert ticks[0] <= 70

    state = {"worker_pid": 111, "child_pid": None}
    worker.wait_for_capacity(
        state,
        tmp_path / "state.json",
        "probe",
        30,
        inspect=inspect,
        sleeper=sleep,
        clock=lambda: ticks[0],
    )
    assert ticks[0] == expected_elapsed
    assert state["quiet_elapsed_seconds"] >= 30
    assert worker.can_start(state["last_resource_check"])
