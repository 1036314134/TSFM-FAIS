"""Instance-level input mixtures fitted to observable forecast consensus."""

from __future__ import annotations

import torch

from .differentiable import missing_block_ids, mix_blocks


def project_consensus(
    candidates,
    context,
    candidate_predictions,
    native_prediction,
    forecast,
    *,
    steps=20,
    learning_rate=0.05,
):
    """Return forecast checkpoints without accepting any actual future labels."""
    if steps < 1 or learning_rate <= 0 or candidates.ndim != 3 or len(candidates) < 2:
        raise ValueError("positive optimization settings and [A,L,D] candidates are required")
    if candidates.shape[1:] != context.shape or candidate_predictions.shape[0] != len(candidates):
        raise ValueError("candidate contexts and predictions must align")
    if (
        not bool(torch.isfinite(candidates).all())
        or not bool(torch.isfinite(candidate_predictions).all())
        or not bool(torch.isfinite(native_prediction).all())
    ):
        raise ValueError("the finite candidate bank and teacher forecasts must be valid")
    teacher = (
        torch.cat([candidate_predictions, native_prediction[None]]).quantile(0.5, dim=0).detach()
    )
    distances = ((candidate_predictions - teacher) ** 2).mean(dim=(1, 2))
    finite_choice = int(distances.argmin())
    native_distance = float(((native_prediction - teacher) ** 2).mean())
    best_prediction = native_prediction.detach().clone()
    best_distance, kind, best_weights = native_distance, "native", None
    if float(distances[finite_choice]) < best_distance:
        best_prediction, best_distance, kind = (
            candidate_predictions[finite_choice].detach().clone(),
            float(distances[finite_choice]),
            f"candidate_{finite_choice}",
        )
        best_weights = candidates.new_zeros((len(candidates), context.shape[1]))
        best_weights[finite_choice] = 1
    calls, backwards = 0, 0

    def snapshot():
        return {
            "prediction": best_prediction.clone(),
            "teacher_mse": best_distance,
            "kind": kind,
            "weights": best_weights.clone() if best_weights is not None else None,
            "forward_calls": calls,
            "backward_calls": backwards,
        }

    checkpoints = {0: snapshot()}
    identifiers, variables = missing_block_ids(context)
    if not len(variables) or best_distance <= 1e-14:
        return {
            "teacher": teacher,
            "checkpoints": {step: checkpoints[0] for step in (0, 1, 5, steps)},
            "stop_reason": "complete_context" if not len(variables) else "exact_anchor",
            "forward_calls": 0,
            "backward_calls": 0,
            "weights": best_weights,
        }
    prior = candidates.new_full((len(candidates), context.shape[1]), 0.1 / (len(candidates) - 1))
    prior[finite_choice] = 0.9
    logits = prior.log().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    reason = "step_limit"
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        weights = logits.softmax(0)
        completed = mix_blocks(candidates, context, identifiers, weights[:, variables])
        known = torch.isfinite(context)
        if not torch.equal(completed[known], context[known]):
            raise ValueError("input projection changed observed values")
        prediction = forecast(completed)
        calls += 1
        loss = ((prediction - teacher) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            reason = "nonfinite_objective"
            break
        if float(loss.detach()) < best_distance:
            best_prediction, best_distance, kind = (
                prediction.detach().clone(),
                float(loss.detach()),
                "input_mixture",
            )
            best_weights = weights.detach().clone()
        if step in (1, 5, steps):
            checkpoints[step] = snapshot()
        if step == steps:
            break
        if not loss.requires_grad:
            reason = "no_gradient_path"
            break
        loss.backward()
        backwards += 1
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            reason = "nonfinite_gradient"
            break
        if float(logits.grad.norm()) <= 1e-12:
            reason = "zero_gradient"
            break
        torch.nn.utils.clip_grad_norm_([logits], 5.0)
        optimizer.step()
    final = snapshot()
    for step in (1, 5, steps):
        checkpoints.setdefault(step, final)
    return {
        "teacher": teacher,
        "checkpoints": checkpoints,
        "stop_reason": reason,
        "forward_calls": calls,
        "backward_calls": backwards,
        "weights": best_weights,
    }
