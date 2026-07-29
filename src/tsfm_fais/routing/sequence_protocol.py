"""Shared identity rules for forecaster-independent selector artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

SELECTOR_INDEPENDENT_FORECASTER_ID: Final = "imputation"
SELECTOR_INDEPENDENT_FORECAST_MODE: Final = "selector_independent"
SEQUENCE_SELECTOR_METHODS: Final = frozenset(
    {
        "metaod",
        "dselect1",
        "neuralucb",
        "alors",
        "hybrid_lstm",
        "random_valid_block",
        "random_valid_series",
    }
)


def normalize_configured_selector_method(method: Any) -> str:
    """Map persisted runtime aliases back to RouterConfig selector IDs."""

    normalized = str(method).strip()
    if normalized == "b_fais":
        return "block_fais"
    if normalized == "random_valid_series":
        return "random_valid_block"
    return normalized


def _common_independent_metadata(metadata: Mapping[str, Any]) -> bool:
    return (
        str(metadata.get("selector_method", "")) in SEQUENCE_SELECTOR_METHODS
        and metadata.get("forecaster_independent_selection") is True
        and metadata.get("uses_missing_block_graph") is False
        and metadata.get("requires_pseudo_candidates") is False
    )


def is_independent_sequence_router_metadata(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether trained-router metadata proves the sequence protocol."""

    return bool(
        isinstance(metadata, Mapping)
        and _common_independent_metadata(metadata)
        and metadata.get("routing_target_protocol") == "sequence_imputation_quality_v1"
        and metadata.get("selector_training_target") == "imputation_loss"
    )


def is_independent_sequence_routing_metadata(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether one committed routing record proves the sequence protocol."""

    if not isinstance(metadata, Mapping) or not _common_independent_metadata(metadata):
        return False
    method = str(metadata["selector_method"])
    expected_scope = "fixed_univariate_window" if method == "hybrid_lstm" else "whole_series"
    return (
        metadata.get("selection_scope") == expected_scope
        and metadata.get("routing_target_protocol") == "sequence_imputation_quality_v1"
        and metadata.get("selector_training_target") == "imputation_loss"
    )


__all__ = [
    "SELECTOR_INDEPENDENT_FORECASTER_ID",
    "SELECTOR_INDEPENDENT_FORECAST_MODE",
    "SEQUENCE_SELECTOR_METHODS",
    "is_independent_sequence_router_metadata",
    "is_independent_sequence_routing_metadata",
    "normalize_configured_selector_method",
]
