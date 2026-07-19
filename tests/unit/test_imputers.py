from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.contracts import CandidateStatus, SeriesBatch
from tsfm_fais.imputers import (
    IMPUTER_IDS,
    IMPUTER_SPECS,
    CandidateRunner,
    KNNMultivariateImputer,
    MICEImputer,
    MissForestArtifact,
    MissForestImputer,
    SoftImputeImputer,
    create_imputer,
)

EXPECTED_IDS = (
    "locf",
    "linear_interp",
    "seasonal_lag",
    "kalman_local_trend",
    "kalman_ar",
    "stl_kalman",
    "gp_rbf",
    "knn_multivariate",
    "mice",
    "missforest",
    "softimpute",
    "trmf",
    "brits",
    "gpvae",
    "saits",
    "csdi",
    "imputeformer",
    "helix",
    "timemixerpp",
    "totem",
)


def make_batch(*, empty_channel: bool = False) -> SeriesBatch:
    time = np.arange(36, dtype=float)
    values = np.stack(
        [
            np.sin(2 * np.pi * time / 12),
            0.5 * np.cos(2 * np.pi * time / 12) + time / 50,
            np.sin(2 * np.pi * time / 6) - 0.2 * time,
        ],
        axis=1,
    )[None, :, :]
    values = np.concatenate([values, values + np.array([[[0.2, -0.1, 0.3]]])], axis=0)
    mask = np.ones_like(values, dtype=bool)
    mask[0, :3, 0] = False
    mask[0, 10:15, 1] = False
    mask[0, 31:, 2] = False
    mask[1, 18:22, :] = False
    if empty_channel:
        mask[:, :, 2] = False
    return SeriesBatch(values, mask, item_ids=("a", "b"), metadata={"period": 12})


def assert_valid_result(result, batch: SeriesBatch) -> None:
    assert result.values.shape == batch.shape
    assert result.native_valid_mask.shape == batch.shape
    assert np.isfinite(result.values).all()
    np.testing.assert_array_equal(
        result.values[batch.observed_mask], batch.values[batch.observed_mask]
    )
    result.validate_against(batch)


def test_default_pool_has_exactly_the_twenty_published_candidate_ids() -> None:
    assert IMPUTER_IDS == EXPECTED_IDS
    assert len(IMPUTER_SPECS) == len(set(IMPUTER_IDS)) == 20
    for spec in IMPUTER_SPECS:
        assert spec.factory
        assert spec.family
        assert spec.mode in {"per_channel", "joint_multivariate"}
        assert spec.dependencies
        assert spec.cost_tier >= 1


@pytest.mark.parametrize("imputer_id", EXPECTED_IDS[:7])
def test_lightweight_candidates_are_finite_and_preserve_observations(imputer_id: str) -> None:
    batch = make_batch()
    params = {"period": 12} if imputer_id in {"seasonal_lag", "stl_kalman"} else {}
    imputer = create_imputer(imputer_id, **params)
    artifact = imputer.fit(batch, {"period": 12})
    result = imputer.impute(batch, artifact, seed=9)
    assert_valid_result(result, batch)
    assert result.status is CandidateStatus.SUCCESS


@pytest.mark.parametrize(
    "imputer",
    [
        KNNMultivariateImputer(n_neighbors=3),
        MICEImputer(max_iter=3),
        MissForestImputer(n_estimators=12, max_iter=3),
        SoftImputeImputer(max_iter=20),
    ],
    ids=lambda imputer: imputer.imputer_id,
)
def test_structured_candidates_fit_then_impute_synchronous_blocks(imputer) -> None:
    batch = make_batch()
    artifact = imputer.fit(batch, {})
    result = imputer.impute(batch, artifact, seed=4)
    assert_valid_result(result, batch)
    assert result.status in {CandidateStatus.SUCCESS, CandidateStatus.PARTIAL}


@pytest.mark.parametrize(
    "imputer",
    [
        KNNMultivariateImputer(n_neighbors=3),
        MICEImputer(max_iter=2),
        MissForestImputer(n_estimators=8, max_iter=2),
        SoftImputeImputer(max_iter=10),
    ],
    ids=lambda imputer: imputer.imputer_id,
)
def test_all_missing_training_channel_is_safe_but_not_native(imputer) -> None:
    batch = make_batch(empty_channel=True)
    artifact = imputer.fit(batch, {})
    result = imputer.impute(batch, artifact, seed=1)
    assert_valid_result(result, batch)
    assert not result.native_valid_mask[:, :, 2].any()
    assert result.status in {CandidateStatus.PARTIAL, CandidateStatus.FAILED}


def test_runner_converts_missing_artifact_exception_to_failed_result() -> None:
    batch = make_batch()
    result = CandidateRunner().run("mice", batch)
    assert_valid_result(result, batch)
    assert result.status is CandidateStatus.FAILED
    assert "fitted artifact" in (result.failure_reason or "")


def test_missforest_trains_feature_models_on_complete_training_fold() -> None:
    complete_values = np.arange(48, dtype=float).reshape(1, 16, 3)
    complete_values[:, :, 1] = 2.0 * complete_values[:, :, 0] + 1.0
    complete_values[:, :, 2] = -complete_values[:, :, 0]
    training = SeriesBatch(
        complete_values,
        np.ones_like(complete_values, dtype=bool),
    )
    mask = np.ones_like(complete_values, dtype=bool)
    mask[:, 5:9, 1] = False
    evaluation = SeriesBatch(complete_values, mask)
    imputer = MissForestImputer(n_estimators=12, max_iter=2, random_state=3)
    artifact = imputer.fit(training, {})
    assert set(artifact.models) == {0, 1, 2}
    result = imputer.impute(evaluation, artifact, seed=3)
    assert result.status is CandidateStatus.SUCCESS
    assert result.native_valid_mask[~evaluation.observed_mask].all()


def test_missforest_predicts_single_threaded_without_mutating_artifact() -> None:
    class RecordingForest:
        def __init__(self) -> None:
            self.n_jobs = 8
            self.seen_n_jobs: list[int] = []

        def predict(self, values: np.ndarray) -> np.ndarray:
            self.seen_n_jobs.append(self.n_jobs)
            return np.mean(values, axis=1)

    values = np.arange(24, dtype=float).reshape(1, 8, 3)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2:6, 1] = False
    batch = SeriesBatch(values, mask)
    model = RecordingForest()
    artifact = MissForestArtifact(
        medians=np.median(values[0], axis=0),
        models={1: model},
        order=(1,),
        observed_features=np.ones(3, dtype=bool),
        training_deltas=(),
    )

    result = MissForestImputer(max_iter=1).impute(batch, artifact, seed=3)

    assert result.status is CandidateStatus.SUCCESS
    assert model.seen_n_jobs == [1]
    assert model.n_jobs == 8


def test_registry_creation_of_deep_adapter_does_not_import_pypots(monkeypatch) -> None:
    import tsfm_fais.imputers.pypots as adapter_module

    real_import = adapter_module.importlib.import_module

    def guarded_import(name: str, *args, **kwargs):
        if name.startswith("pypots"):
            raise AssertionError("constructor imported PyPOTS eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(adapter_module.importlib, "import_module", guarded_import)
    assert create_imputer("saits").imputer_id == "saits"
