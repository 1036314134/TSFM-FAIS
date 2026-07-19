from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tsfm_fais.config import load_config
from tsfm_fais.registry_configs import (
    ForecasterPoolConfig,
    ImputerPoolConfig,
    RouterConfig,
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
        ImputerPoolConfig.model_validate({"schema_version": 1, "imputers": [entry, entry]})
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


def test_router_rejects_non_convex_evidence_weights() -> None:
    payload = yaml.safe_load(Path("configs/router/block_fais.yaml").read_text(encoding="utf-8"))
    payload["evidence_blend"]["chronos2"]["proxy"] = 0.5

    with pytest.raises(ValidationError, match="sum to one"):
        RouterConfig.model_validate(payload)


def test_forecast_consensus_router_config_is_valid_and_budgeted() -> None:
    payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus.yaml").read_text(encoding="utf-8")
    )

    config = RouterConfig.model_validate(payload)

    assert config.forecast_consensus.mode == "medoid"
    assert config.forecast_consensus.candidates == (
        "locf",
        "linear_interp",
        "seasonal_lag",
        "kalman_ar",
    )
    assert config.forecast_consensus.dataset_prior_candidates == 0
    assert config.forecast_consensus.dataset_prior_candidates_by_model == {}
    assert config.candidate_prior_min_support == 4

    model_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_model_prior.yaml").read_text(encoding="utf-8")
    )
    model_config = RouterConfig.model_validate(model_payload)
    assert model_config.forecast_consensus.dataset_prior_candidates_by_model == {
        "chronos2": 2,
        "timesfm2p5": 0,
    }
    assert model_config.forecast_consensus.context_mode == "targets_with_correlates"
    assert model_config.forecast_consensus.max_context_variates == 8
    assert model_config.forecast_consensus.prior_weight_by_model == {
        "chronos2": 0.0,
        "timesfm2p5": 0.8,
    }
    assert model_config.forecast_consensus.selection_granularity_by_model == {
        "chronos2": "episode",
        "timesfm2p5": "episode",
    }
    assert model_config.candidate_prior_min_support == 4
    topk_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_top2_mean.yaml").read_text(
            encoding="utf-8"
        )
    )
    topk_config = RouterConfig.model_validate(topk_payload)
    assert topk_config.forecast_consensus.mode == "value_topk_mean"
    assert topk_config.forecast_consensus.ensemble_top_k == 2
    target_topk_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_times_targetwise_top2.yaml").read_text(
            encoding="utf-8"
        )
    )
    target_topk_config = RouterConfig.model_validate(target_topk_payload)
    assert target_topk_config.forecast_consensus.mode == "value_topk_mean"
    assert target_topk_config.forecast_consensus.selection_granularity_by_model == {
        "chronos2": "episode",
        "timesfm2p5": "target",
    }
    target_topk_payload["forecast_consensus"]["mode"] = "router_risk"
    with pytest.raises(ValidationError, match="target-level forecast consensus"):
        RouterConfig.model_validate(target_topk_payload)
    proxy_weighted_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_top2_proxy_weighted.yaml").read_text(
            encoding="utf-8"
        )
    )
    proxy_weighted_config = RouterConfig.model_validate(proxy_weighted_payload)
    assert proxy_weighted_config.forecast_consensus.ensemble_proxy_weight_power == 0.5
    proxy_weighted_payload["forecast_consensus"]["mode"] = "medoid"
    with pytest.raises(ValidationError, match="requires value_topk_mean"):
        RouterConfig.model_validate(proxy_weighted_payload)
    convex_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_top2_pseudo_convex.yaml").read_text(
            encoding="utf-8"
        )
    )
    convex_config = RouterConfig.model_validate(convex_payload)
    assert convex_config.forecast_consensus.pseudo_weight_calibration == "convex_l2"
    assert convex_config.forecast_consensus.pseudo_weight_prior_strength == 8.0
    assert convex_config.forecast_consensus.pseudo_weight_min_points == 4
    convex_payload["forecast_consensus"]["ensemble_top_k"] = 3
    with pytest.raises(ValidationError, match="proxy blend or fixed top-two"):
        RouterConfig.model_validate(convex_payload)
    times_convex_payload = yaml.safe_load(
        Path(
            "configs/router/"
            "block_fais_consensus_times_targetwise_proxy050_margin005_seasonal010.yaml"
        ).read_text(encoding="utf-8")
    )
    times_convex_payload["forecast_consensus"].update(
        {
            "mode": "value_topk_mean",
            "ensemble_top_k": 2,
            "pseudo_weight_calibration": "convex_l2",
            "pseudo_weight_prior_strength": 8.0,
            "pseudo_weight_min_points": 4,
        }
    )
    times_convex_config = RouterConfig.model_validate(times_convex_payload)
    assert times_convex_config.forecast_consensus.pseudo_weight_calibration == "convex_l2"
    gap_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_top23_gap010.yaml").read_text(
            encoding="utf-8"
        )
    )
    gap_config = RouterConfig.model_validate(gap_payload)
    assert gap_config.forecast_consensus.ensemble_third_relative_gap == 0.1
    gap_payload["forecast_consensus"]["ensemble_top_k"] = 3
    with pytest.raises(ValidationError, match="top-k two"):
        RouterConfig.model_validate(gap_payload)
    margin_payload = yaml.safe_load(
        Path(
            "configs/router/block_fais_consensus_times_targetwise_proxy050_margin005.yaml"
        ).read_text(encoding="utf-8")
    )
    margin_config = RouterConfig.model_validate(margin_payload)
    assert margin_config.forecast_consensus.proxy_blend_min_relative_margin_by_model == {
        "timesfm2p5": 0.05,
    }
    shrinkage_payload = yaml.safe_load(
        Path(
            "configs/router/"
            "block_fais_consensus_times_targetwise_proxy050_margin005_seasonal010.yaml"
        ).read_text(encoding="utf-8")
    )
    shrinkage_config = RouterConfig.model_validate(shrinkage_payload)
    assert shrinkage_config.forecast_consensus.candidate_shrinkage_id == "seasonal_lag"
    assert shrinkage_config.forecast_consensus.candidate_shrinkage_weight == 0.1
    shrinkage_payload["forecast_consensus"]["candidate_shrinkage_id"] = "saits"
    with pytest.raises(ValidationError, match="must be included"):
        RouterConfig.model_validate(shrinkage_payload)
    chronos_shrinkage_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_top2_mean_linear010.yaml").read_text(
            encoding="utf-8"
        )
    )
    chronos_shrinkage_config = RouterConfig.model_validate(chronos_shrinkage_payload)
    assert chronos_shrinkage_config.forecast_consensus.mode == "value_topk_mean"
    assert chronos_shrinkage_config.forecast_consensus.candidate_shrinkage_id == "linear_interp"
    assert chronos_shrinkage_config.forecast_consensus.candidate_shrinkage_weight == 0.1
    fallback_payload = yaml.safe_load(
        Path(
            "configs/router/block_fais_consensus_chronos_top2_seasonal090_linear075.yaml"
        ).read_text(encoding="utf-8")
    )
    fallback_config = RouterConfig.model_validate(fallback_payload)
    assert fallback_config.forecast_consensus.candidate_shrinkage_id == "seasonal_lag"
    assert fallback_config.forecast_consensus.candidate_shrinkage_weight == 0.9
    assert fallback_config.forecast_consensus.candidate_shrinkage_fallback_id == "linear_interp"
    assert fallback_config.forecast_consensus.candidate_shrinkage_fallback_weight == 0.75
    fallback_payload["forecast_consensus"]["candidate_shrinkage_fallback_id"] = "seasonal_lag"
    with pytest.raises(ValidationError, match="must differ"):
        RouterConfig.model_validate(fallback_payload)
    soft_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_soft075.yaml").read_text(encoding="utf-8")
    )
    soft_config = RouterConfig.model_validate(soft_payload)
    assert soft_config.ranker_target == "full_candidate_loss"
    assert soft_config.forecast_consensus.anchor_weight_by_model == {
        "chronos2": 1.0,
        "timesfm2p5": 0.75,
    }
    safe_payload = yaml.safe_load(
        Path("configs/router/block_fais_consensus_chronos_prior_safe.yaml").read_text(
            encoding="utf-8"
        )
    )
    safe_config = RouterConfig.model_validate(safe_payload)
    assert safe_config.ranker_target == "full_candidate_loss"
    assert safe_config.forecast_consensus.prior_override_max_medoid_penalty_by_model == {
        "chronos2": 0.8,
    }
    assert safe_config.forecast_consensus.prior_override_min_margin_by_model == {
        "chronos2": 0.005,
    }
    payload["forecast_consensus"]["candidates"].extend(["mice", "softimpute", "saits"])
    expanded = RouterConfig.model_validate(payload)
    assert len(expanded.forecast_consensus.candidates) > expanded.shortlist_size

    payload["shortlist_size"] = 1
    with pytest.raises(ValidationError, match="forced_candidates cannot exceed"):
        RouterConfig.model_validate(payload)


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
