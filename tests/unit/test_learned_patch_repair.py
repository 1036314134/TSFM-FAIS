import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_patch_repair import independent_repair  # noqa: E402
from learned_patch_repair import PatchRepair  # noqa: E402


def test_zero_initialization_is_identity_for_every_mask():
    torch.manual_seed(12)
    module = PatchRepair(12, 4, rank=3)
    embedding = torch.randn(2, 3, 12)
    missing = torch.randint(0, 2, (2, 3, 4)).float()
    for condition in ("pattern", "fraction"):
        repaired, penalty = module(embedding, missing, condition)
        assert torch.equal(repaired, embedding)
        assert float(penalty) == 0


def test_active_repairs_preserve_complete_tokens_and_have_an_independent_matrix_reconstruction():
    torch.manual_seed(15)
    module = PatchRepair(12, 4, rank=3)
    torch.nn.init.normal_(module.up.weight, std=0.1)
    embedding = torch.randn(2, 3, 12)
    missing = torch.zeros(2, 3, 4)
    missing[0, 1, 0] = 1
    missing[1, 2, 1:] = 1
    state = {
        name: value.detach().numpy().astype(float) for name, value in module.state_dict().items()
    }
    for condition in ("pattern", "fraction"):
        repaired, penalty = module(embedding, missing, condition)
        active = missing.bool().any(-1)
        assert torch.equal(repaired[~active], embedding[~active])
        assert torch.isfinite(penalty) and float(penalty) > 0
        expected = independent_repair(
            embedding.numpy(), missing.numpy().astype(bool), state, condition
        )
        np.testing.assert_allclose(repaired.detach().numpy(), expected, rtol=1e-5, atol=1e-6)
        clean, clean_penalty = module(embedding, torch.zeros_like(missing), condition)
        assert torch.equal(clean, embedding) and float(clean_penalty) == 0


def test_fraction_control_is_invariant_to_within_patch_positions():
    torch.manual_seed(19)
    module = PatchRepair(12, 4, rank=3)
    torch.nn.init.normal_(module.up.weight, std=0.1)
    embedding = torch.randn(1, 2, 12)
    first = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    second = first.flip(-1)
    a, _ = module(embedding, first, "fraction")
    b, _ = module(embedding, second, "fraction")
    assert torch.equal(a, b)
    a, _ = module(embedding, first, "pattern")
    b, _ = module(embedding, second, "pattern")
    assert not torch.equal(a, b)
