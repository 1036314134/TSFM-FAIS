from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm_fais.config import load_config, load_yaml
from tsfm_fais.pipeline import _independent_block_search, _sequence_candidate_search
from tsfm_fais.registry_configs import RouterConfig
from tsfm_fais.stage_execution import (
    _fit_router_bundle,
    _join_reconstruction_targets,
    _router_ranker_targets,
    _validate_reconstruction_label_lineage,
)


def _forecast_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index, block_id in enumerate(("block-a", "block-b")):
        for candidate_index, candidate_id in enumerate(("locf", "linear_interp")):
            rows.append(
                {
                    "episode_id": "dataset__item__12__independent_block__0.2__7",
                    "dataset_id": "dataset",
                    "family_id": "family",
                    "item_id": "item",
                    "forecast_origin": 12,
                    "forecaster_id": "chronos2",
                    "group_id": f"group-{group_index}",
                    "block_id": block_id,
                    "candidate_id": candidate_id,
                    "prior_features": {
                        "candidate_index": float(candidate_index),
                        "block_index": float(group_index),
                    },
                    "unary_features": {
                        "candidate_index": float(candidate_index),
                        "block_index": float(group_index),
                        "runtime_seconds": 0.1 + candidate_index,
                        "peak_memory_mb": 1.0 + candidate_index,
                    },
                    "forecast_loss": 1.0 + candidate_index,
                    "full_candidate_loss": 2.0 + candidate_index,
                }
            )
    return rows


def test_internal_control_router_config_rejects_multifactor_combinations() -> None:
    payload = load_yaml("configs/router/block_fais.yaml")
    sequence = RouterConfig.model_validate(
        {
            **payload,
            "selection_granularity": "sequence",
            "routing_structure": "independent",
        }
    )
    assert sequence.selection_granularity == "sequence"
    assert sequence.routing_structure == "independent"

    with pytest.raises(ValueError, match="sequence selection requires"):
        RouterConfig.model_validate(
            {
                **payload,
                "selection_granularity": "sequence",
                "routing_structure": "structured",
            }
        )
    with pytest.raises(ValueError, match="reconstruction supervision"):
        RouterConfig.model_validate(
            {
                **payload,
                "ranker_target": "imputation_loss",
                "selection_granularity": "block",
                "routing_structure": "independent",
            }
        )


def test_internal_control_configs_change_one_declared_factor_per_edge() -> None:
    root = Path("configs/iclr27-r2/router/internal-controls")
    configs = {path.stem: load_yaml(path) for path in root.glob("*.yaml")}

    def changed(left: str, right: str) -> set[str]:
        return {
            key
            for key in configs[left] | configs[right]
            if configs[left].get(key) != configs[right].get(key)
        }

    assert changed("seq_recon", "seq_forecast") == {"ranker_target"}
    assert changed("seq_forecast", "independent_block_forecast") == {"selection_granularity"}
    assert changed("independent_block_forecast", "structured_block_forecast") == {
        "routing_structure"
    }


def test_r2_followup_configs_change_only_preregistered_factors() -> None:
    def payload(path: str) -> dict[str, object]:
        return load_config(path).model_dump(mode="json")

    for main_path, sequence_path in (
        (
            "configs/iclr27-r2/main_lofo_train.yaml",
            "configs/iclr27-r2/internal-controls/seq_forecast_lofo17_train.yaml",
        ),
        (
            "configs/iclr27-r2/main_lofo_eval.yaml",
            "configs/iclr27-r2/internal-controls/seq_forecast_lofo17_eval.yaml",
        ),
    ):
        main_lofo = payload(main_path)
        sequence_lofo = payload(sequence_path)
        sequence_lofo["registries"]["router_config"] = main_lofo["registries"]["router_config"]
        sequence_lofo["protocol"]["protocol_id"] = main_lofo["protocol"]["protocol_id"]
        assert sequence_lofo == main_lofo

    for deployment_path, identity_path in (
        (
            "configs/iclr27-r2/internal-controls/seq_forecast_train.yaml",
            "configs/iclr27-r2/internal-controls/seq_forecast_identity_free_train.yaml",
        ),
        (
            "configs/iclr27-r2/internal-controls/seq_forecast_confirmation.yaml",
            "configs/iclr27-r2/internal-controls/seq_forecast_identity_free_confirmation.yaml",
        ),
        (
            "configs/iclr27-r2/prefix-sensitivity/no_complete_prefix_train.yaml",
            "configs/iclr27-r2/prefix-sensitivity/no_complete_prefix_identity_free_train.yaml",
        ),
        (
            "configs/iclr27-r2/prefix-sensitivity/no_complete_prefix_confirmation.yaml",
            "configs/iclr27-r2/prefix-sensitivity/"
            "no_complete_prefix_identity_free_confirmation.yaml",
        ),
        (
            "configs/iclr27-r2/native-missing/uci_beijing.yaml",
            "configs/iclr27-r2/native-missing/uci_beijing_identity_free.yaml",
        ),
    ):
        deployment = payload(deployment_path)
        identity = payload(identity_path)
        assert identity["experiment"]["feature_policy"] == "identity_free"
        identity["experiment"]["feature_policy"] = deployment["experiment"]["feature_policy"]
        identity["protocol"]["protocol_id"] = deployment["protocol"]["protocol_id"]
        assert identity == deployment


def test_reconstruction_join_is_strict_and_one_to_one() -> None:
    forecast_rows = _forecast_rows()
    reconstruction_rows = [
        {
            "dataset_id": "dataset",
            "episode_id": "dataset__item__12__independent_block__0.2__7",
            "candidate_id": candidate_id,
            "forecaster_id": "imputation",
            "label_scope": "whole_series",
            "imputation_loss": loss,
        }
        for candidate_id, loss in (("locf", 0.2), ("linear_interp", 0.1))
    ]

    joined, manifest = _join_reconstruction_targets(
        forecast_rows,
        reconstruction_rows,
    )

    assert [row["reconstruction_loss"] for row in joined] == [0.2, 0.1, 0.2, 0.1]
    assert manifest["forecast_row_count"] == 4
    assert manifest["forecast_key_count"] == 2
    assert manifest["reconstruction_key_count"] == 2
    with pytest.raises(ValueError, match="duplicate reconstruction label"):
        _join_reconstruction_targets(
            forecast_rows,
            [*reconstruction_rows, dict(reconstruction_rows[0])],
        )
    with pytest.raises(ValueError, match="identities differ"):
        _join_reconstruction_targets(forecast_rows, reconstruction_rows[:1])
    superset, superset_manifest = _join_reconstruction_targets(
        forecast_rows,
        [
            *reconstruction_rows,
            {
                **reconstruction_rows[0],
                "candidate_id": "additional_native_candidate",
            },
        ],
    )
    assert len(superset) == len(forecast_rows)
    assert superset_manifest["unused_reconstruction_key_count"] == 1


def test_reconstruction_target_uses_the_fixed_protocol() -> None:
    targets, protocol = _router_ranker_targets(
        [{"reconstruction_loss": 0.4}, {"reconstruction_loss": 0.1}],
        "reconstruction_loss",
    )

    assert targets.tolist() == [0.4, 0.1]
    assert protocol == "masked_context_reconstruction_asmape_v1"


def test_reconstruction_lineage_requires_matching_data_fits_and_forecast_candidate_coverage() -> (
    None
):
    binding = {
        "label_mask_seeds": [1101, 1102, 1103],
        "label_dataset_ids": ["a", "b"],
        "label_episode_count": 180,
        "selected_candidates": ["locf", "linear_interp"],
        "data_manifest": {"resolved": "data.yaml", "sha256": "data", "size_bytes": 1},
        "imputer_registry": {
            "resolved": "imputers.yaml",
            "sha256": "registry",
            "size_bytes": 2,
        },
        "imputer_artifact_manifest": {
            "resolved": "manifest.json",
            "sha256": "fit",
            "size_bytes": 3,
        },
    }

    manifest = _validate_reconstruction_label_lineage(binding, dict(binding))

    assert manifest["protocol"] == "same_data_candidate_fit_forecast_subset_v2"
    assert manifest["dataset_count"] == 2
    assert manifest["candidate_count"] == 2
    assert manifest["reconstruction_extra_candidate_count"] == 0
    superset = _validate_reconstruction_label_lineage(
        binding,
        {
            **binding,
            "selected_candidates": ["locf", "linear_interp", "mean"],
        },
    )
    assert superset["forecast_candidate_count"] == 2
    assert superset["reconstruction_candidate_count"] == 3
    assert superset["reconstruction_extra_candidates"] == ["mean"]
    with pytest.raises(ValueError, match="selected_candidates"):
        _validate_reconstruction_label_lineage(
            binding,
            {**binding, "selected_candidates": ["locf"]},
        )
    with pytest.raises(ValueError, match="imputer_artifact_manifest"):
        _validate_reconstruction_label_lineage(
            binding,
            {**binding, "imputer_artifact_manifest": None},
        )


def test_sequence_and_independent_controls_share_identical_unary_model_file(
    tmp_path: Path,
) -> None:
    rows = _forecast_rows()
    sequence = _fit_router_bundle(
        rows,
        [],
        tmp_path / "sequence",
        {
            "split": "rolling_origin",
            "ranker_target": "full_candidate_loss",
            "selection_granularity": "sequence",
            "routing_structure": "independent",
        },
        seed=4101,
    )
    independent = _fit_router_bundle(
        rows,
        [],
        tmp_path / "independent",
        {
            "split": "rolling_origin",
            "ranker_target": "full_candidate_loss",
            "selection_granularity": "block",
            "routing_structure": "independent",
        },
        seed=4101,
    )

    assert (tmp_path / "sequence" / "unary_model.txt").read_bytes() == (
        tmp_path / "independent" / "unary_model.txt"
    ).read_bytes()
    assert not (tmp_path / "sequence" / "pairwise_model.txt").exists()
    assert not (tmp_path / "independent" / "pairwise_model.txt").exists()
    assert sequence.metadata["uses_pairwise_model"] is False
    assert independent.metadata["uses_pairwise_model"] is False
    assert sequence.metadata["selector_training_target"] == "full_candidate_loss"
    assert independent.metadata["selector_training_target"] == "full_candidate_loss"


def test_sequence_and_independent_solvers_change_only_assignment_granularity() -> None:
    blocks = (SimpleNamespace(block_id="a"), SimpleNamespace(block_id="b"))
    candidates = ("first", "second")
    unary = {
        ("a", "first"): 0.0,
        ("a", "second"): 2.0,
        ("b", "first"): 4.0,
        ("b", "second"): 1.0,
    }
    costs = {"first": 1.0, "second": 1.0}

    sequence = _sequence_candidate_search(
        blocks,
        candidates,
        unary,
        costs,
        0.0,
        set(),
    )
    independent = _independent_block_search(
        blocks,
        candidates,
        unary,
        costs,
        0.0,
        set(),
    )

    assert sequence.assignments == {"a": "second", "b": "second"}
    assert sequence.metadata["solver"] == "sequence_candidate_mean"
    assert independent.assignments == {"a": "first", "b": "second"}
    assert independent.metadata["solver"] == "independent_block_argmin"
    assert sequence.predicted_pairwise == {}
    assert independent.predicted_pairwise == {}
