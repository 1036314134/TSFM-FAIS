from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tsfm_fais.artifacts import RunArtifactStore
from tsfm_fais.config import load_config, load_yaml
from tsfm_fais.registry_configs import RouterConfig
from tsfm_fais.routing.baselines import BASELINE_SELECTOR_METHODS
from tsfm_fais.routing.models import RouterBundle
from tsfm_fais.stage_execution import (
    _fit_router_bundle,
    _router_ranker_targets,
    execute_train_router,
)
from tsfm_fais.stages import StageInputs, StagePreparation


@dataclass(frozen=True)
class _DeterministicScoreModel:
    width: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.width:
            raise ValueError("unexpected feature matrix")
        return np.arange(len(matrix), dtype=float)


def _teacher_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = ("locf", "linear_interp")
    for block_index in range(2):
        episode_id = "toy__item__12__independent_block__0.2__22"
        block_id = f"n0:d0:{block_index}-{block_index + 1}"
        for candidate_index, candidate_id in enumerate(candidates):
            loss = float(block_index + candidate_index + 1)
            prior_features = {
                "length": 1.0,
                "start_ratio": float(block_index) / 2.0,
                f"candidate_id::{candidate_id}": 1.0,
            }
            rows.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": "toy",
                    "family_id": "toy-family",
                    "forecast_origin": 12,
                    "forecaster_id": "chronos2",
                    "group_id": f"chronos2::{episode_id}::{block_id}",
                    "block_id": block_id,
                    "candidate_id": candidate_id,
                    "prior_features": prior_features,
                    "unary_features": {
                        **prior_features,
                        "proxy_mae": loss,
                        "runtime_seconds": float(candidate_index),
                        "peak_memory_mb": 0.0,
                    },
                    "forecast_loss": loss,
                    "clean_loss": 0.5,
                    "anchor_loss": 1.0,
                    "degradation": loss - 0.5,
                    "local_marginal": loss - 1.0,
                    "full_candidate_loss": loss + 0.25,
                    "global_marginal_per_block": loss / 2.0,
                    "coherence_adjustment": 0.0,
                    "routing_target": loss - 1.0,
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_baseline_suite_configuration_is_strict_and_backward_compatible() -> None:
    legacy = RouterConfig.model_validate(load_yaml("configs/router/block_fais.yaml"))
    suite_payload = load_yaml("configs/router/baseline_selector_suite.yaml")
    suite = RouterConfig.model_validate(suite_payload)

    assert legacy.selector_methods == ("block_fais",)
    assert legacy.selector_params == {}
    assert suite.selector_methods == BASELINE_SELECTOR_METHODS
    assert suite.ranker_target == "imputation_loss"
    assert suite.shortlist_size == 20

    with pytest.raises(ValueError, match="unsupported selector"):
        RouterConfig.model_validate({**suite_payload, "selector_methods": ["unknown"]})
    with pytest.raises(ValueError, match="cannot be mixed"):
        RouterConfig.model_validate(
            {
                **suite_payload,
                "selector_methods": ["block_fais", "metaod"],
                "selector_params": {"metaod": {}},
            }
        )
    with pytest.raises(ValueError, match="unselected methods"):
        RouterConfig.model_validate(
            {
                **suite_payload,
                "selector_methods": ["metaod"],
                "selector_params": {"alors": {}},
            }
        )
    with pytest.raises(ValueError, match="must be finite"):
        RouterConfig.model_validate(
            {
                **suite_payload,
                "selector_methods": ["metaod"],
                "selector_params": {"metaod": {"learning_rate": float("nan")}},
            }
        )
    with pytest.raises(ValueError, match="unsupported parameters"):
        RouterConfig.model_validate(
            {
                **suite_payload,
                "selector_methods": ["metaod"],
                "selector_params": {"metaod": {"epohs": 2}},
            }
        )
    with pytest.raises(ValueError, match="disabled forecast_consensus"):
        RouterConfig.model_validate(
            {
                **suite_payload,
                "selector_methods": ["metaod"],
                "selector_params": {"metaod": {}},
                "forecast_consensus": {
                    "mode": "medoid",
                    "candidates": ["locf", "linear_interp"],
                },
            }
        )


@pytest.mark.parametrize(
    ("method", "supported_params", "removed_param"),
    (
        ("metaod", {"min_samples_split": 2}, "temperature"),
        ("dselect1", {"reachable_mass_weight": 0.5}, "max_depth"),
        ("neuralucb", {"nu": 0.25}, "alpha"),
        ("alors", {"ndcg_cutoff": 10}, "margin"),
        ("hybrid_lstm", {"multilabel_threshold": 0.02}, "near_optimal_tolerance"),
        ("random_valid_block", {}, "seed"),
    ),
)
def test_selector_parameter_whitelists_match_sequence_implementations(
    method: str,
    supported_params: dict[str, float],
    removed_param: str,
) -> None:
    suite_payload = load_yaml("configs/router/baseline_selector_suite.yaml")
    selected_payload = {
        **suite_payload,
        "selector_methods": [method],
        "selector_params": {method: supported_params},
    }

    config = RouterConfig.model_validate(selected_payload)

    assert config.selector_params[method] == supported_params
    with pytest.raises(ValueError, match="unsupported parameters"):
        RouterConfig.model_validate(
            {
                **selected_payload,
                "selector_params": {method: {removed_param: 1}},
            }
        )


def test_forecast_loss_is_an_explicit_router_target() -> None:
    rows = _teacher_rows()

    losses, protocol = _router_ranker_targets(rows, "forecast_loss")

    np.testing.assert_allclose(losses, [1.0, 2.0, 2.0, 3.0])
    assert protocol == "single_block_counterfactual_forecast_loss_v1"


def test_baseline_bundle_uses_only_prior_features_and_no_pair_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_fit_baseline_selector(**kwargs: object) -> _DeterministicScoreModel:
        calls.append(dict(kwargs))
        return _DeterministicScoreModel(np.asarray(kwargs["features"]).shape[1])

    monkeypatch.setattr(
        "tsfm_fais.routing.baselines.fit_baseline_selector",
        fake_fit_baseline_selector,
    )
    bundle = _fit_router_bundle(
        _teacher_rows(),
        [],
        tmp_path / "metaod",
        {"split": "rolling_origin", "ranker_target": "forecast_loss", "beta": 1.0},
        selector_method="metaod",
        selector_params={"latent_dim": 2},
        seed=7,
    )

    assert len(calls) == 1
    assert calls[0]["method"] == "metaod"
    assert "proxy_mae" not in bundle.feature_names
    assert bundle.pair_feature_names == ()
    assert bundle.pairwise.model is None
    assert bundle.metadata["selector_method"] == "metaod"
    assert bundle.metadata["requires_pseudo_candidates"] is False
    assert bundle.metadata["beta"] == 0.0
    assert bundle.metadata["selector_training_target"] == "forecast_loss"
    assert RouterBundle.load(tmp_path / "metaod").metadata["selector_seed"] == 7


def test_execute_train_router_writes_all_suite_artifacts_without_pair_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fit_baseline_selector(**kwargs: object) -> _DeterministicScoreModel:
        return _DeterministicScoreModel(np.asarray(kwargs["features"]).shape[1])

    monkeypatch.setattr(
        "tsfm_fais.routing.baselines.fit_baseline_selector",
        fake_fit_baseline_selector,
    )
    labels_path = tmp_path / "teacher_labels.jsonl"
    _write_jsonl(labels_path, _teacher_rows())
    config = load_config("configs/main_rolling_train_baselines.yaml")
    store = RunArtifactStore.create(tmp_path / "artifacts", "baseline-suite")
    preparation = StagePreparation("train-router", store, {})

    result = execute_train_router(
        preparation,
        config,
        StageInputs(labels_artifact=labels_path),
    )

    assert tuple(result["selector_methods"]) == BASELINE_SELECTOR_METHODS
    manifest_path = Path(str(result["suite_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert tuple(manifest["selector_methods"]) == BASELINE_SELECTOR_METHODS
    assert set(manifest["routers"]) == set(BASELINE_SELECTOR_METHODS)
    for method in BASELINE_SELECTOR_METHODS:
        artifact = store.root / "routers" / method
        assert (artifact / "router_bundle.joblib").is_file()
        bundle = RouterBundle.load(artifact)
        assert bundle.metadata["selector_method"] == method
        assert bundle.metadata["requires_pseudo_candidates"] is False
