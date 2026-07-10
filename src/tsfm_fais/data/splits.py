"""Dataset-family splits that keep related resolutions in the same fold."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import DatasetManifest, DatasetSpec


@dataclass(frozen=True)
class FamilyFold:
    family_id: str
    train: tuple[DatasetSpec, ...]
    test: tuple[DatasetSpec, ...]

    def __post_init__(self) -> None:
        train_families = {dataset.family_id for dataset in self.train}
        test_families = {dataset.family_id for dataset in self.test}
        if not self.test or test_families != {self.family_id}:
            raise ValueError("test datasets must all belong to the held-out family")
        if train_families & test_families:
            raise ValueError("a dataset family cannot occur in both train and test")


def family_folds(manifest: DatasetManifest) -> tuple[FamilyFold, ...]:
    """Build deterministic leave-family-out folds over enabled datasets."""

    enabled = tuple(dataset for dataset in manifest.datasets if dataset.enabled)
    families = tuple(sorted({dataset.family_id for dataset in enabled}))
    if len(families) < 2:
        raise ValueError("family-aware evaluation requires at least two enabled families")
    return tuple(
        FamilyFold(
            family_id=family_id,
            train=tuple(dataset for dataset in enabled if dataset.family_id != family_id),
            test=tuple(dataset for dataset in enabled if dataset.family_id == family_id),
        )
        for family_id in families
    )


__all__ = ["FamilyFold", "family_folds"]
