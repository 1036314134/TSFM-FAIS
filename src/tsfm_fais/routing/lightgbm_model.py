"""LightGBM wrapper that imports the optional backend only when fitted."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


class LazyLightGBMRegressor:
    """Small sklearn-style regressor wrapper with dependency injection for tests."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        defaults: dict[str, Any] = {
            "objective": "regression_l1",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1,
        }
        defaults.update(dict(params or {}))
        self.params = defaults
        self._model = model
        self._model_factory = model_factory
        self.feature_names_: tuple[str, ...] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Any:
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                try:
                    lightgbm = importlib.import_module("lightgbm")
                except ImportError as exc:
                    raise ImportError("routing model requires the lightgbm dependency") from exc
                factory = lightgbm.LGBMRegressor
            self._model = factory(**self.params)
        return self._model

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "LazyLightGBMRegressor":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [R,F] and align with one-dimensional targets")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training arrays must be finite")
        kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(weights) != len(y):
                raise ValueError("sample_weight must align with targets")
            kwargs["sample_weight"] = weights
        self.model.fit(x, y, **kwargs)
        if feature_names is not None:
            if len(feature_names) != x.shape[1]:
                raise ValueError("feature_names must align with feature columns")
            self.feature_names_ = tuple(feature_names)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("routing model has not been fitted or injected")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("prediction features must be a finite [R,F] array")
        prediction = np.asarray(self._model.predict(x), dtype=float).reshape(-1)
        if len(prediction) != len(x) or not np.all(np.isfinite(prediction)):
            raise ValueError("routing model returned invalid predictions")
        return prediction


LightGBMUnaryModel = LazyLightGBMRegressor
