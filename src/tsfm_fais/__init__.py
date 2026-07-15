"""TSFM-FAIS public package."""

from .contracts import (
    BudgetSpec,
    CandidateResult,
    ForecastResult,
    ForecastSpec,
    ImputerSpec,
    MissingBlock,
    RoutingResult,
    SeriesBatch,
    TimeSeriesItem,
)
from .pipeline import BlockwiseFAIS, FAISResult, RoutePlan

__all__ = [
    "BudgetSpec",
    "BlockwiseFAIS",
    "CandidateResult",
    "ForecastResult",
    "ForecastSpec",
    "FAISResult",
    "ImputerSpec",
    "MissingBlock",
    "RoutingResult",
    "RoutePlan",
    "SeriesBatch",
    "TimeSeriesItem",
]

__version__ = "0.1.0"
