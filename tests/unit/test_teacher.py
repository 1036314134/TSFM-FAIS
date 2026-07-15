from __future__ import annotations

import numpy as np

from tsfm_fais.contracts import CandidateResult, ForecastResult, ForecastSpec, MissingBlock
from tsfm_fais.routing.teacher import TeacherBuilder, replace_block


def _nonlinear_forecast(context: np.ndarray, spec: ForecastSpec) -> ForecastResult:
    values = np.asarray(context, dtype=float)
    targets = spec.target_indices or tuple(range(values.shape[2]))
    level = np.sum(values, axis=(1, 2)) ** 2
    point = np.broadcast_to(
        level[:, None, None], (values.shape[0], spec.horizon, len(targets))
    ).copy()
    return ForecastResult(point=point, target_indices=tuple(targets))


def test_teacher_builds_single_block_labels_and_pair_interaction_formula() -> None:
    clean_context = np.arange(1, 13, dtype=float).reshape(1, 6, 2)
    clean_future = np.zeros((1, 2, 2), dtype=float)
    anchor = clean_context.copy()
    anchor[:, 1:3, 0] = 0.0
    anchor[:, 3:5, 1] = 0.0
    left_block = MissingBlock("left", 0, 0, 1, 3)
    right_block = MissingBlock("right", 0, 1, 3, 5)
    left_values = anchor.copy()
    right_values = anchor.copy()
    left_values[:, 1:3, 0] = clean_context[:, 1:3, 0]
    right_values[:, 3:5, 1] = clean_context[:, 3:5, 1]
    valid = np.ones_like(anchor, dtype=bool)
    candidates = {
        "left_candidate": CandidateResult("left_candidate", left_values, valid),
        "right_candidate": CandidateResult("right_candidate", right_values, valid),
    }
    spec = ForecastSpec(
        "mock",
        "joint_multivariate",
        horizon=2,
        target_indices=(0, 1),
    )
    builder = TeacherBuilder(_nonlinear_forecast)
    labels = builder.unary_labels(
        "episode",
        clean_context,
        clean_future,
        anchor,
        (left_block, right_block),
        candidates,
        spec,
    )
    assert len(labels) == 4
    assert all(label.clean_loss >= 0 for label in labels)
    batched_labels, clean_loss, anchor_loss = builder.unary_labels_batched(
        "episode",
        clean_context,
        clean_future,
        anchor,
        (left_block, right_block),
        candidates,
        spec,
    )
    expected_losses = {
        (label.block_id, label.candidate_id): label.forecast_loss for label in labels
    }
    observed_losses = {
        (label.block_id, label.candidate_id): label.forecast_loss
        for label in batched_labels
    }
    assert observed_losses.keys() == expected_losses.keys()
    for key in expected_losses:
        assert np.isclose(observed_losses[key], expected_losses[key])
    assert np.isclose(clean_loss, builder._loss(clean_context, clean_future, spec))
    assert np.isclose(anchor_loss, builder._loss(anchor, clean_future, spec, builder._scales(clean_context, spec.target_indices)))
    filtered = builder.unary_labels(
        "episode",
        clean_context,
        clean_future,
        anchor,
        (left_block, right_block),
        candidates,
        spec,
        candidate_filter=lambda block, candidate_id, _result: (
            block.block_id == "left" and candidate_id == "left_candidate"
        ),
    )
    assert [(label.block_id, label.candidate_id) for label in filtered] == [
        ("left", "left_candidate")
    ]

    observed_interaction = builder.pair_interaction(
        clean_future,
        anchor,
        left_block,
        right_block,
        candidates["left_candidate"],
        candidates["right_candidate"],
        spec,
        scale_context=clean_context,
    )
    left = replace_block(anchor, left_values, left_block)
    right = replace_block(anchor, right_values, right_block)
    both = replace_block(left, right_values, right_block)
    scales = builder._scales(clean_context, spec.target_indices)
    expected = (
        builder._loss(both, clean_future, spec, scales)
        - builder._loss(left, clean_future, spec, scales)
        - builder._loss(right, clean_future, spec, scales)
        + builder._loss(anchor, clean_future, spec, scales)
    )
    assert np.isclose(observed_interaction, expected)
    batched_interaction = builder.pair_interactions_batched(
        clean_future,
        anchor,
        (
            (
                left_block,
                right_block,
                candidates["left_candidate"],
                candidates["right_candidate"],
            ),
        ),
        spec,
        anchor_loss=builder._loss(anchor, clean_future, spec, scales),
        scale_context=clean_context,
    )
    assert len(batched_interaction) == 1
    assert np.isclose(batched_interaction[0], observed_interaction)


def test_mase_scale_falls_back_to_lag_one_when_period_exceeds_context():
    context = np.asarray([[[0.0], [1.0], [3.0], [6.0]]])
    scale = TeacherBuilder(_nonlinear_forecast, seasonality=24)._scales(context, (0,))
    assert np.isclose(scale[0], np.mean([1.0, 2.0, 3.0]))
