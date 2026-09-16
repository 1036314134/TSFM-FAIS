import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.dataset  # noqa: F401
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_provenance_pilot import verify_fields  # noqa: E402
from provenance_marker import ProvenanceMarker  # noqa: E402


def test_chronos_changes_only_historical_marker_features_and_restores_module():
    backbone = SimpleNamespace(input_patch_embedding=torch.nn.Identity())
    historical = torch.arange(2 * 6 * 48, dtype=torch.float32).reshape(2, 6, 48)
    historical[..., -16:] = 1
    future = historical + 4
    original = historical.clone()
    observed = np.ones((96, 2), bool)
    observed[10:25, 0] = False
    observed[-6:, 1] = False
    with ProvenanceMarker(backbone, "chronos2", observed, True) as hook:
        altered = backbone.input_patch_embedding(historical)
        unaltered_future = backbone.input_patch_embedding(future)
    assert torch.equal(historical, original)
    assert torch.equal(altered[..., :32], original[..., :32])
    assert torch.equal(unaltered_future, future)
    np.testing.assert_array_equal(altered[..., -16:].numpy(), observed.T.reshape(2, 6, 16))
    assert (
        verify_fields(original.numpy(), altered.numpy(), observed, "chronos2", "provenance", 0)
        == 21
    )
    assert len(hook.calls) == 2
    assert torch.equal(backbone.input_patch_embedding(historical), original)


def test_timesfm_keeps_padding_samples_and_numerical_values_for_both_flip_calls():
    backbone = SimpleNamespace(tokenizer=torch.nn.Identity())
    value = torch.ones((8, 3, 64))
    value[0, :, -32:] = 0
    observed = np.ones((96, 1), bool)
    observed[40:53, 0] = False
    original = value.clone()
    with ProvenanceMarker(backbone, "timesfm2p5", observed, True):
        positive = backbone.tokenizer(value)
        negative_input = value.clone()
        negative_input[..., :32] *= -1
        negative = backbone.tokenizer(negative_input)
    assert torch.equal(value, original)
    assert torch.equal(positive[1:], original[1:])
    assert torch.equal(negative[1:], negative_input[1:])
    assert torch.equal(positive[..., :32], original[..., :32])
    assert (
        verify_fields(original.numpy(), positive.numpy(), observed, "timesfm2p5", "provenance", 0)
        == 13
    )
    assert (
        verify_fields(
            negative_input.numpy(), negative.numpy(), observed, "timesfm2p5", "provenance", 1
        )
        == 13
    )
    damaged = positive.numpy().copy()
    damaged[0, 0, 0] = 99
    with pytest.raises(AssertionError):
        verify_fields(original.numpy(), damaged, observed, "timesfm2p5", "provenance", 0)


def test_complete_markers_are_identity_and_hooks_removed_after_failure():
    backbone = SimpleNamespace(tokenizer=torch.nn.Identity())
    value = torch.zeros((8, 3, 64))
    observed = np.ones((96, 1), bool)
    with ProvenanceMarker(backbone, "timesfm2p5", observed, True):
        assert torch.equal(backbone.tokenizer(value), value)
        assert torch.equal(backbone.tokenizer(value), value)
    with pytest.raises(RuntimeError):
        with ProvenanceMarker(backbone, "timesfm2p5", ~observed, True):
            raise RuntimeError("simulated forecasting failure")
    assert torch.equal(backbone.tokenizer(value), value)
    assert not backbone.tokenizer._forward_pre_hooks
