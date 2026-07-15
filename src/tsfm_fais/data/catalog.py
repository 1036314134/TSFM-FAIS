"""Dataset manifest contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_fais.config import load_yaml


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    family_id: str
    format: Literal["csv", "arrow"]
    path: Path
    frequency: str
    period: int = Field(ge=1)
    domain: str = "unknown"
    timestamp_column: str | None = None
    item_id_column: str | None = None
    value_columns: tuple[str, ...] | None = None
    target_columns: tuple[str, ...] | Literal["all"] = "all"
    sentinel_values: tuple[float, ...] = (-9999.0,)
    allow_implicit_regular_time: bool = False
    variate_name_normalization: Literal["none", "strip_bracket_suffix"] = "none"
    expected_num_variates: int | None = Field(default=None, ge=2)
    provenance: Literal["source_complete", "release_complete", "snapshot_complete"] = (
        "snapshot_complete"
    )
    enabled: bool = True

    @model_validator(mode="after")
    def validate_schema(self) -> DatasetSpec:
        if not self.dataset_id.strip() or not self.family_id.strip():
            raise ValueError("dataset_id and family_id cannot be empty")
        if not self.frequency.strip():
            raise ValueError("frequency cannot be empty")
        if self.variate_name_normalization != "none" and self.format != "arrow":
            raise ValueError("variate_name_normalization is supported only for Arrow data")
        if self.value_columns is not None:
            if len(self.value_columns) < 2:
                raise ValueError("value_columns must contain at least two variates")
            if len(set(self.value_columns)) != len(self.value_columns):
                raise ValueError("value_columns must be unique")
        if self.target_columns != "all":
            if not self.target_columns:
                raise ValueError("target_columns cannot be empty")
            if len(set(self.target_columns)) != len(self.target_columns):
                raise ValueError("target_columns must be unique")
            if self.value_columns is not None:
                unknown = set(self.target_columns) - set(self.value_columns)
                if unknown:
                    raise ValueError(
                        f"target_columns are not present in value_columns: {sorted(unknown)}"
                    )
        if any(not math.isfinite(value) for value in self.sentinel_values):
            raise ValueError("sentinel_values must be finite")
        if len(set(self.sentinel_values)) != len(self.sentinel_values):
            raise ValueError("sentinel_values must be unique")
        return self


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    data_root: Path
    datasets: tuple[DatasetSpec, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> DatasetManifest:
        if not self.datasets:
            raise ValueError("datasets cannot be empty")
        ids = [spec.dataset_id for spec in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset_id values must be unique")
        return self

    def get(self, dataset_id: str) -> DatasetSpec:
        for spec in self.datasets:
            if spec.dataset_id == dataset_id:
                return spec
        raise KeyError(dataset_id)


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path).resolve()
    manifest = DatasetManifest.model_validate(load_yaml(manifest_path))
    base = manifest_path.parent
    data_root = manifest.data_root
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    specs = []
    for spec in manifest.datasets:
        resolved = spec.path if spec.path.is_absolute() else data_root / spec.path
        specs.append(spec.model_copy(update={"path": resolved.resolve()}))
    return manifest.model_copy(update={"data_root": data_root, "datasets": tuple(specs)})
