import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import shared_context_inference as shared  # noqa: E402


def test_shared_context_is_evaluated_once_and_reused(monkeypatch):
    original = np.arange(3 * 7 * 97, dtype=np.float32).reshape(3, 7, 97)
    requested = []

    def predict(_state, values):
        requested.append(values.copy())
        return np.arange(len(values) * 7).reshape(len(values), 7)

    monkeypatch.setattr(shared, "probability_from_state", predict)
    result = shared.broadcast_context_weights({}, np.repeat(original, 2, axis=0))
    assert len(requested) == 1
    np.testing.assert_array_equal(requested[0], original)
    np.testing.assert_array_equal(result[::2], result[1::2])


def test_shared_context_rejects_different_target_features():
    values = np.zeros((2, 7, 97), np.float32)
    values[1, 0, 0] = 1
    with pytest.raises(AssertionError):
        shared.broadcast_context_weights({}, values)
