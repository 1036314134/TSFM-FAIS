from __future__ import annotations

import torch

from tsfm_fais.forecasting.timesfm_differentiable import running_stats, timesfm_median


def test_running_statistics_match_native_forward_and_have_finite_constant_gradients():
    from timesfm.torch.util import update_running_stats

    values = torch.tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 3.0, 5.0, 7.0]], requires_grad=True)
    mask = torch.zeros_like(values, dtype=torch.bool)
    zero = torch.zeros(2)
    actual = running_stats(zero, zero, zero, values, mask)
    expected, _ = update_running_stats(zero, zero, zero, values, mask)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    (actual[1] + actual[2]).sum().backward()
    assert torch.isfinite(values.grad).all()


class LinearCore:
    p, o, q, aridx = 2, 4, 10, 5

    def __call__(self, inputs, masks, decode_caches=None):
        output = (2 * inputs[..., -1:]).expand(-1, -1, self.o * self.q)
        return (inputs, inputs, output, torch.zeros_like(output)), None


def test_prefill_median_preserves_target_order_and_reaches_input_gradients():
    context = torch.tensor(
        [[1.0, -1.0, 11.0], [2.0, -2.0, 12.0], [3.0, -3.0, 13.0], [4.0, -4.0, 14.0]],
        requires_grad=True,
    )
    result = timesfm_median(LinearCore(), context, 3, [1, 0])
    torch.testing.assert_close(result, torch.tensor([[-5.5, 5.5]]).expand(3, -1))
    result[:, 1].mean().backward()
    torch.testing.assert_close(context.grad[:, 0], torch.tensor([-0.25, -0.25, -0.25, 1.75]))
    assert torch.equal(context.grad[:, 2], torch.zeros(4))


def test_constant_completed_series_does_not_produce_invalid_gradients():
    context = torch.ones(4, 1, requires_grad=True)
    result = timesfm_median(LinearCore(), context, 3, [0])
    torch.testing.assert_close(result, torch.ones(3, 1))
    result.mean().backward()
    assert torch.isfinite(context.grad).all()
