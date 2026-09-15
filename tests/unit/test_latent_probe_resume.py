import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from probe_frozen_latent_interfaces import cached_probe_call  # noqa: E402


def test_resume_reuses_a_finished_call_but_keeps_repeat_requests_independent(tmp_path):
    calls = []
    values = np.ones((4, 2))

    def invoke():
        calls.append(True)
        return values * 2, [np.ones((2, 3, 5))]

    first = cached_probe_call(tmp_path, "reference", "study", "parameters", values, invoke)
    resumed = cached_probe_call(tmp_path, "reference", "study", "parameters", values, invoke)
    cached_probe_call(tmp_path, "repeat", "study", "parameters", values, invoke)
    assert len(calls) == 2
    np.testing.assert_array_equal(first[0], resumed[0])
    np.testing.assert_array_equal(first[1][0], resumed[1][0])
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_interrupted_call_leaves_evidence_and_can_resume(tmp_path):
    values = np.ones((4, 2))

    def fail():
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        cached_probe_call(tmp_path, "capture", "study", "parameters", values, fail)
    assert not (tmp_path / "calls/capture.npz").exists()
    cached_probe_call(tmp_path, "capture", "study", "parameters", values, lambda: (values, []))
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_resume_rejects_changed_parameters_or_inputs(tmp_path):
    values = np.ones((4, 2))
    cached_probe_call(tmp_path, "reference", "study", "parameters", values, lambda: (values, []))
    with pytest.raises(ValueError, match="another run"):
        cached_probe_call(tmp_path, "reference", "study", "changed", values, lambda: (values, []))
    with pytest.raises(AssertionError):
        cached_probe_call(
            tmp_path, "reference", "study", "parameters", values + 1, lambda: (values, [])
        )
