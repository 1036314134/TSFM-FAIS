from __future__ import annotations

import pytest
import torch

from tsfm_fais.routing.differentiable import (
    ContextBlockComposer,
    FixedBlockComposer,
    missing_block_ids,
)


def example():
    context = torch.tensor(
        [
            [1.0, float("nan")],
            [float("nan"), float("nan")],
            [float("nan"), 4.0],
            [5.0, float("nan")],
        ]
    )
    first = torch.nan_to_num(context, nan=2.0)
    second = torch.nan_to_num(context, nan=8.0)
    return context, torch.stack([first, second])


def test_block_identifiers_never_merge_variables_or_separated_runs():
    context, _ = example()
    ids, variables = missing_block_ids(context)
    assert torch.equal(ids, torch.tensor([[-1, 0, 0, -1], [1, 1, -1, 2]]))
    assert variables.tolist() == [0, 1, 1]


def test_initial_context_composer_equals_its_fixed_source_prior():
    context, candidates = example()
    fixed = FixedBlockComposer([0.8, 0.2])(candidates, context)
    adaptive = ContextBlockComposer([0.8, 0.2])(candidates, context)
    torch.testing.assert_close(adaptive.values, fixed.values)
    torch.testing.assert_close(adaptive.weights, torch.tensor([[0.8] * 3, [0.2] * 3]))
    observed = torch.isfinite(context)
    assert torch.equal(adaptive.values[observed], context[observed])


def test_composition_is_variable_permutation_equivariant_after_learning():
    torch.manual_seed(91)
    context, candidates = example()
    model = ContextBlockComposer([0.5, 0.5])
    with torch.no_grad():
        model.score[-1].weight.normal_(0, 0.2)
    original = model(candidates, context)
    permuted = model(candidates[:, :, [1, 0]], context[:, [1, 0]])
    torch.testing.assert_close(permuted.values, original.values[:, [1, 0]], rtol=1e-5, atol=1e-6)


def test_gradient_reaches_context_model_and_values_stay_in_candidate_range():
    context, candidates = example()
    model = ContextBlockComposer([0.5, 0.5])
    output = model(candidates, context)
    output.values.square().mean().backward()
    assert model.score[-1].weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert (output.values >= candidates.min(0).values).all()
    assert (output.values <= candidates.max(0).values).all()
    torch.testing.assert_close(output.weights.sum(0), torch.ones(3))


def test_complete_context_is_an_exact_noop():
    context = torch.arange(12.0).reshape(4, 3)
    candidates = context[None].expand(2, -1, -1)
    result = ContextBlockComposer([0.6, 0.4])(candidates, context)
    assert result.block_count == 0
    assert torch.equal(result.values, context)


def test_identical_imputations_do_not_propagate_an_undefined_input_gradient():
    context = torch.tensor([[float("nan"), float("nan")], [1.0, 2.0]])
    candidates = torch.tensor([[[0.0, 3.0], [1.0, 2.0]], [[0.0, 7.0], [1.0, 2.0]]])
    model = FixedBlockComposer([0.5, 0.5])
    result = model(candidates, context)
    external_gradient = torch.ones_like(context)
    external_gradient[0, 0] = float("nan")
    result.values.backward(external_gradient)
    assert torch.isfinite(model.base_logits.grad).all()
    assert model.base_logits.grad.abs().sum() > 0


def test_frozen_predictor_features_train_the_projection_without_updating_the_features():
    torch.manual_seed(52)
    context, candidates = example()
    model = ContextBlockComposer([0.5, 0.5], forecaster_dim=12)
    features = torch.randn(2, 2, 2, 12, requires_grad=True)
    initial = model(candidates, context, forecaster_patches=features)
    torch.testing.assert_close(
        initial.values, FixedBlockComposer([0.5, 0.5])(candidates, context).values
    )
    with torch.no_grad():
        model.score[-1].weight.normal_(0, 0.2)
    result = model(candidates, context, forecaster_patches=features)
    result.values.square().mean().backward()
    assert features.grad is None
    assert model.forecaster_projection[1].weight.grad.abs().sum() > 0
    swapped = model(
        candidates[:, :, [1, 0]], context[:, [1, 0]], forecaster_patches=features[:, [1, 0]]
    )
    torch.testing.assert_close(swapped.values, result.values[:, [1, 0]], rtol=1e-5, atol=1e-6)


def test_target_conditioning_tracks_variable_permutations_and_requires_the_task():
    torch.manual_seed(109)
    context, candidates = example()
    model = ContextBlockComposer([0.5, 0.5], target_conditioned=True)
    with torch.no_grad():
        model.score[-1].weight.normal_(0, 0.2)
    first = model(candidates, context, targets=[0])
    swapped = model(candidates[:, :, [1, 0]], context[:, [1, 0]], targets=[1])
    torch.testing.assert_close(swapped.values, first.values[:, [1, 0]], rtol=1e-5, atol=1e-6)
    changed_task = model(candidates, context, targets=[1])
    assert not torch.allclose(first.weights, changed_task.weights)
    with pytest.raises(ValueError, match="target indices"):
        model(candidates, context)
