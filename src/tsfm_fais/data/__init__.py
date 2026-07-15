"""Multivariate data loading, auditing, and episode construction."""

from .audit import AuditIssue, DatasetAudit, audit_dataset
from .catalog import DatasetManifest, DatasetSpec, load_manifest
from .episodes import Episode, build_episode, fit_prefix_end, rolling_origins
from .loaders import load_dataset
from .masking import (
    MaskedSeries,
    MaskingSpec,
    extract_missing_blocks,
    mask_time_series,
    stable_seed,
)
from .splits import FamilyFold, family_folds

__all__ = [
    "AuditIssue",
    "DatasetAudit",
    "DatasetManifest",
    "DatasetSpec",
    "Episode",
    "FamilyFold",
    "MaskingSpec",
    "MaskedSeries",
    "audit_dataset",
    "build_episode",
    "extract_missing_blocks",
    "family_folds",
    "fit_prefix_end",
    "load_dataset",
    "load_manifest",
    "mask_time_series",
    "rolling_origins",
    "stable_seed",
]
