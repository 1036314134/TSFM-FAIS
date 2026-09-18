import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from future_query_core import role_masks  # noqa: E402


def masks():
    low = torch.finfo(torch.float32).min
    time = torch.zeros((4, 1, 1, 8)) * low
    group = torch.zeros((4, 4, 8)).permute(2, 0, 1).unsqueeze(1) * low
    time[0, 0, 0, 1] = low
    group[1, 0, :, 0] = low
    return time, group


def test_auxiliary_future_rule_preserves_past_reg_and_target_future_keys():
    time, group = masks()
    t, g = role_masks(time, group, 2, 6, "aux_both")
    torch.testing.assert_close(t[..., :6], time[..., :6], rtol=0, atol=0)
    torch.testing.assert_close(g[:6], group[:6], rtol=0, atol=0)
    assert torch.equal(t[:2], time[:2]) and torch.equal(g[6:, ..., :2], group[6:, ..., :2])
    assert (t[2:, ..., 6:] == torch.finfo(t.dtype).min).all()
    assert (g[6:, ..., 2:] == torch.finfo(g.dtype).min).all()


def test_time_and_group_controls_change_only_their_registered_axis():
    time, group = masks()
    t, g = role_masks(time, group, 2, 6, "aux_time")
    assert g is group and not torch.equal(t, time)
    t, g = role_masks(time, group, 2, 6, "aux_group")
    assert t is time and not torch.equal(g, group)


def test_readonly_future_keeps_each_queries_group_self_transformation():
    time, group = masks()
    t, g = role_masks(time, group, 2, 6, "readonly")
    assert (t[..., 6:] == torch.finfo(t.dtype).min).all()
    torch.testing.assert_close(g[:6], group[:6], rtol=0, atol=0)
    for i in range(4):
        assert torch.equal(g[6:, 0, i, i], group[6:, 0, i, i])
        for j in range(4):
            if i != j:
                assert (g[6:, 0, i, j] == torch.finfo(g.dtype).min).all()


def test_neutral_policy_and_layout_do_not_change_native_numerics():
    time, group = masks()
    for policy in ("aux_both", "aux_time", "aux_group"):
        t, g = role_masks(time, group, 4, 6, policy)
        assert t is time and g is group
    t, g = role_masks(time, group, 2, 6, "readonly")
    assert t.stride() == time.stride() and g.stride() == group.stride()
    torch.testing.assert_close(t[..., :6], time[..., :6], rtol=0, atol=0)


def test_invalid_role_boundaries_are_rejected():
    time, group = masks()
    for targets, boundary in ((0, 6), (5, 6), (2, 8)):
        with pytest.raises(ValueError):
            role_masks(time, group, targets, boundary, "aux_both")
