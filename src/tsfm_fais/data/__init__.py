"""Multivariate data loading, auditing, and episode construction."""

from .audit import AuditIssue, DatasetAudit, audit_dataset
from .catalog import DatasetManifest, DatasetSpec, load_manifest
from .episodes import Episode, build_episode, rolling_origins
from .loaders import load_dataset
from .masking import MaskingSpec, inject_missing, stable_seed
from .splits import FamilyFold, family_folds

__all__ = [
    "AuditIssue",
    "DatasetAudit",
    "DatasetManifest",
    "DatasetSpec",
    "Episode",
    "FamilyFold",
    "MaskingSpec",
    "audit_dataset",
    "build_episode",
    "family_folds",
    "inject_missing",
    "load_dataset",
    "load_manifest",
    "rolling_origins",
    "stable_seed",
]
