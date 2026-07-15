from __future__ import annotations

import numpy as np

from tsfm_fais.contracts import CandidateResult
from tsfm_fais.routing.features import proxy_features


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
