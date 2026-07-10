"""Deterministic forecaster folds for router transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ForecastModelFold:
    held_out_model: str
    train_models: tuple[str, ...]
    test_models: tuple[str, ...]


def leave_model_out_folds(model_ids: Sequence[str]) -> tuple[ForecastModelFold, ...]:
    provided = tuple(model_ids)
    unique = tuple(sorted(set(provided)))
    if len(unique) != len(provided):
        raise ValueError("model_ids must be unique")
    if len(unique) < 2:
        raise ValueError("leave-model-out evaluation requires at least two models")
    return tuple(
        ForecastModelFold(
            held_out_model=model_id,
            train_models=tuple(candidate for candidate in unique if candidate != model_id),
            test_models=(model_id,),
        )
        for model_id in unique
    )


__all__ = ["ForecastModelFold", "leave_model_out_folds"]
