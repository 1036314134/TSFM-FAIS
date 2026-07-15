from __future__ import annotations

import numpy as np

from tsfm_fais.contracts import (
    CandidateResult,
    ForecastSpec,
    MissingBlock,
    SeriesBatch,
)
from tsfm_fais.imputers import DEFAULT_REGISTRY
from tsfm_fais.routing.features import block_features, candidate_features, proxy_features


def test_proxy_features_penalize_nonfinite_uncertainty_with_finite_values() -> None:
    truth = np.arange(12, dtype=float).reshape(1, 6, 2)
    pseudo_mask = np.ones_like(truth, dtype=bool)
    pseudo_mask[:, 2:4, :] = False
    uncertainty = np.zeros_like(truth)
    uncertainty[:, 2, :] = np.nan
    uncertainty[:, 3, :] = np.inf
    candidate = CandidateResult(
        imputer_id="csdi",
        values=truth.copy(),
        native_valid_mask=np.ones_like(truth, dtype=bool),
        uncertainty=uncertainty,
    )

    features = proxy_features(candidate, truth, pseudo_mask, source=truth)

    assert all(np.isfinite(value) for value in features.values())
    assert features["mean_uncertainty"] == 1e12
    assert features["proxy_mae"] == 0.0
    assert features["proxy_rmse"] == 0.0


def test_proxy_features_clip_overflowing_error_and_covariance() -> None:
    truth = np.ones((1, 4, 2), dtype=float)
    pseudo_mask = np.ones_like(truth, dtype=bool)
    pseudo_mask[:, 1:3, :] = False
    values = truth.copy()
    values[:, 1:3, :] = np.finfo(float).max
    candidate = CandidateResult(
        imputer_id="finite_extreme",
        values=values,
        native_valid_mask=np.ones_like(values, dtype=bool),
    )

    features = proxy_features(candidate, truth, pseudo_mask, source=truth)

    assert all(np.isfinite(value) for value in features.values())
    assert features["proxy_mae"] == 1e12
    assert features["proxy_rmse"] == 1e12
    assert features["covariance_drift"] == 1e12


def test_router_features_distinguish_forecasters_and_sequence_missing_rates() -> None:
    values = np.arange(24, dtype=float).reshape(1, 8, 3)
    mask = np.ones_like(values, dtype=bool)
    mask[:, 2:5, 1] = False
    batch = SeriesBatch(
        values,
        mask,
        metadata={
            "missing_mechanism": "independent_block",
            "target_missing_rate": 0.4,
            "global_missing_rate": 0.399,
            "local_missing_rate": 0.125,
        },
    )
    block = MissingBlock("b0", 0, 1, 2, 5, "independent_block")
    block_row = block_features(batch, block, period=4)
    spec = DEFAULT_REGISTRY.get_spec("locf")
    timesfm = candidate_features(
        spec,
        ForecastSpec("timesfm2p5", "independent_univariate", 96, 96, (0, 1)),
    )
    tirex = candidate_features(
        spec,
        ForecastSpec("tirex", "independent_univariate", 96, 96, (0, 1)),
    )

    assert block_row["target_missing_rate"] == 0.4
    assert block_row["global_missing_rate"] == 0.399
    assert block_row["local_missing_rate"] == 0.125
    assert block_row["missing_mechanism::independent_block"] == 1.0
    assert timesfm["forecast_model::timesfm2p5"] == 1.0
    assert "forecast_model::tirex" not in timesfm
    assert tirex["forecast_model::tirex"] == 1.0
