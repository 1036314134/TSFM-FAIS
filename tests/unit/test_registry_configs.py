from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tsfm_fais.config import load_config
from tsfm_fais.registry_configs import (
    ForecasterPoolConfig,
    ImputerPoolConfig,
    validate_project_configuration,
)


def test_checked_in_registry_configuration_is_cross_file_valid() -> None:
    validated = validate_project_configuration(load_config("configs/smoke.yaml"))
    assert len(validated["datasets"].datasets) == 32
    assert len(validated["imputers"].imputers) == 20
    assert len(validated["forecasters"].forecasters) == 5


def test_registry_schemas_reject_duplicate_ids_and_unknown_keys() -> None:
    entry = {
        "id": "x",
        "family": "test",
        "mode": "per_channel",
        "fit_scope": "none",
        "supports_tail": True,
        "requires_period": False,
        "stochastic": False,
        "device": "cpu",
        "cost_tier": 1,
    }
    with pytest.raises(ValidationError, match="unique"):
        ImputerPoolConfig.model_validate(
            {"schema_version": 1, "imputers": [entry, entry]}
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ForecasterPoolConfig.model_validate(
            {
                "schema_version": 1,
                "forecasters": [
                    {
                        "id": "mock",
                        "mode": "independent_univariate",
                        "model_name": "mock",
                        "max_context": 16,
                        "output_type": "point",
                        "optional_extra": "none",
                        "unexpected": True,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("registry_field", "filename", "collection", "expected"),
    (
        ("imputer_registry", "imputers.yaml", "imputers", "imputer YAML IDs"),
        (
            "forecaster_registry",
            "forecasters.yaml",
            "forecasters",
            "forecaster YAML IDs",
        ),
    ),
)
def test_project_validation_rejects_yaml_runtime_registry_drift(
    tmp_path,
    registry_field,
    filename,
    collection,
    expected,
):
    config = load_config("configs/smoke.yaml")
    source = getattr(config.registries, registry_field)
    payload = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    payload[collection][0]["id"] = "not_executable"
    target = tmp_path / filename
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    registries = config.registries.model_copy(update={registry_field: target})

    with pytest.raises(ValueError, match=expected):
        validate_project_configuration(config.model_copy(update={"registries": registries}))
