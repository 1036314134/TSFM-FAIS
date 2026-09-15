from types import SimpleNamespace

import torch

from tsfm_fais.forecasting.chronos_differentiable import chronos_median


def test_canonical_layout_preserves_values_target_order_and_input_gradients():
    class Model:
        device = torch.device("cpu")

        def __call__(self, *, context, group_ids, num_output_patches):
            assert context.is_contiguous()
            assert context.dtype == torch.float32
            assert group_ids.tolist() == [0, 0, 0]
            middle = context[:, -2:]
            return SimpleNamespace(
                quantile_preds=torch.stack([middle - 1, middle, middle + 1], dim=1)
            )

    pipeline = SimpleNamespace(
        model=Model(), model_output_patch_size=2, max_output_patches=2, quantiles=[0.1, 0.5, 0.9]
    )
    context = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    result = chronos_median(pipeline, context, 2, [2, 0])
    torch.testing.assert_close(result, context[-2:, [2, 0]].float())
    result.sum().backward()
    expected = torch.zeros_like(context)
    expected[-2:, [2, 0]] = 1
    torch.testing.assert_close(context.grad, expected)
