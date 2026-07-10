"""Joint multivariate statistical imputers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from tsfm_fais.contracts import SeriesBatch

from .base import BaseImputer


def _matrix(batch: SeriesBatch) -> np.ndarray:
    return np.asarray(batch.values, dtype=float).reshape(-1, batch.shape[2])


def _restore(matrix: np.ndarray, batch: SeriesBatch) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    expected = (batch.shape[0] * batch.shape[1], batch.shape[2])
    if array.shape != expected:
        raise ValueError(f"structured imputer returned {array.shape}; expected {expected}")
    return array.reshape(batch.shape)


@dataclass
class SklearnArtifact:
    estimator: Any
    observed_features: np.ndarray


class KNNMultivariateImputer(BaseImputer):
    imputer_id = "knn_multivariate"

    def __init__(self, n_neighbors: int = 5, weights: str = "distance") -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights

    def _fit(
        self, train_batch: SeriesBatch, metadata: Mapping[str, Any]
    ) -> SklearnArtifact:
        from sklearn.impute import KNNImputer

        values = _matrix(train_batch)
        estimator = KNNImputer(
            n_neighbors=min(self.n_neighbors, max(1, values.shape[0])),
            weights=self.weights,
            keep_empty_features=True,
        )
        estimator.fit(values)
        return SklearnArtifact(estimator, np.isfinite(values).any(axis=0))

    def _impute_native(
        self, batch: SeriesBatch, artifact: SklearnArtifact | None, seed: int
    ) -> np.ndarray:
        if artifact is None:
            raise ValueError("knn_multivariate requires a fitted artifact")
        original = _matrix(batch)
        output = np.asarray(artifact.estimator.transform(original), dtype=float)
        empty = ~np.asarray(artifact.observed_features, dtype=bool)
        if empty.any():
            output[np.isnan(original) & empty[None, :]] = np.nan
        return _restore(output, batch)


class MICEImputer(BaseImputer):
    imputer_id = "mice"

    def __init__(
        self,
        max_iter: int = 10,
        tolerance: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        if max_iter < 1 or tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)

    def _fit(
        self, train_batch: SeriesBatch, metadata: Mapping[str, Any]
    ) -> SklearnArtifact:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        values = _matrix(train_batch)
        estimator = IterativeImputer(
            max_iter=self.max_iter,
            tol=self.tolerance,
            random_state=self.random_state,
            sample_posterior=False,
            initial_strategy="median",
            skip_complete=False,
            keep_empty_features=True,
        )
        estimator.fit(values)
        return SklearnArtifact(estimator, np.isfinite(values).any(axis=0))

    def _impute_native(
        self, batch: SeriesBatch, artifact: SklearnArtifact | None, seed: int
    ) -> np.ndarray:
        if artifact is None:
            raise ValueError("mice requires a fitted artifact")
        original = _matrix(batch)
        output = np.asarray(artifact.estimator.transform(original), dtype=float)
        empty = ~np.asarray(artifact.observed_features, dtype=bool)
        if empty.any():
            output[np.isnan(original) & empty[None, :]] = np.nan
        return _restore(output, batch)


@dataclass
class MissForestArtifact:
    medians: np.ndarray
    models: dict[int, Any]
    order: tuple[int, ...]
    observed_features: np.ndarray
    training_deltas: tuple[float, ...]


class MissForestImputer(BaseImputer):
    imputer_id = "missforest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
        n_jobs: int | None = 1,
    ) -> None:
        if n_estimators < 1 or max_iter < 1 or min_samples_leaf < 1:
            raise ValueError("forest size, iterations, and leaf size must be positive")
        self.n_estimators = int(n_estimators)
        self.max_iter = int(max_iter)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    @staticmethod
    def _initial(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed_features = np.isfinite(values).any(axis=0)
        medians = np.zeros(values.shape[1], dtype=float)
        for feature in range(values.shape[1]):
            finite = values[np.isfinite(values[:, feature]), feature]
            if finite.size:
                medians[feature] = float(np.median(finite))
        filled = np.where(np.isfinite(values), values, medians[None, :])
        return filled, medians, observed_features

    def _new_model(self, feature: int) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state + feature,
            n_jobs=self.n_jobs,
        )

    def _fit(
        self, train_batch: SeriesBatch, metadata: Mapping[str, Any]
    ) -> MissForestArtifact:
        values = _matrix(train_batch)
        filled, medians, observed_features = self._initial(values)
        missing = ~np.isfinite(values)
        missing_fraction = missing.mean(axis=0)
        order = tuple(int(index) for index in np.argsort(missing_fraction))
        previous_delta = np.inf
        deltas: list[float] = []
        models: dict[int, Any] = {}

        for _ in range(self.max_iter):
            previous = filled.copy()
            iteration_models: dict[int, Any] = {}
            for feature in order:
                missing_rows = missing[:, feature]
                observed_rows = ~missing_rows
                predictors = [index for index in range(values.shape[1]) if index != feature]
                if (
                    observed_rows.sum() < 2
                    or not predictors
                    or not observed_features[feature]
                ):
                    continue
                model = self._new_model(feature)
                model.fit(filled[observed_rows][:, predictors], values[observed_rows, feature])
                if missing_rows.any():
                    filled[missing_rows, feature] = model.predict(
                        filled[missing_rows][:, predictors]
                    )
                iteration_models[feature] = model
            denominator = max(float(np.sum(filled * filled)), 1e-12)
            delta = float(np.sum((filled - previous) ** 2) / denominator)
            deltas.append(delta)
            if delta >= previous_delta:
                break
            previous_delta = delta
            models.update(iteration_models)
            if delta <= 1e-7:
                break
        return MissForestArtifact(
            medians=medians,
            models=models,
            order=order,
            observed_features=observed_features,
            training_deltas=tuple(deltas),
        )

    def _impute_native(
        self, batch: SeriesBatch, artifact: MissForestArtifact | None, seed: int
    ) -> np.ndarray:
        if artifact is None:
            raise ValueError("missforest requires a fitted artifact")
        values = _matrix(batch)
        missing = ~np.isfinite(values)
        filled = np.where(np.isfinite(values), values, artifact.medians[None, :])
        previous_delta = np.inf
        for _ in range(self.max_iter):
            previous = filled.copy()
            for feature in artifact.order:
                rows = missing[:, feature]
                model = artifact.models.get(feature)
                if not rows.any() or model is None:
                    continue
                predictors = [index for index in range(values.shape[1]) if index != feature]
                filled[rows, feature] = model.predict(filled[rows][:, predictors])
            denominator = max(float(np.sum(filled * filled)), 1e-12)
            delta = float(np.sum((filled - previous) ** 2) / denominator)
            if delta >= previous_delta or delta <= 1e-7:
                break
            previous_delta = delta
        empty = ~np.asarray(artifact.observed_features, dtype=bool)
        if empty.any():
            filled[missing & empty[None, :]] = np.nan
        return _restore(filled, batch)


@dataclass(frozen=True)
class SoftImputeArtifact:
    medians: np.ndarray
    shrinkage: float
    observed_features: np.ndarray


class SoftImputeImputer(BaseImputer):
    imputer_id = "softimpute"

    def __init__(
        self,
        shrinkage: float | None = None,
        max_rank: int | None = None,
        max_iter: int = 100,
        tolerance: float = 1e-5,
    ) -> None:
        if shrinkage is not None and shrinkage < 0:
            raise ValueError("shrinkage cannot be negative")
        if max_rank is not None and max_rank < 1:
            raise ValueError("max_rank must be positive")
        if max_iter < 1 or tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        self.shrinkage = shrinkage
        self.max_rank = max_rank
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    @staticmethod
    def _initial(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed_features = np.isfinite(values).any(axis=0)
        medians = np.zeros(values.shape[1], dtype=float)
        for feature in range(values.shape[1]):
            finite = values[np.isfinite(values[:, feature]), feature]
            if finite.size:
                medians[feature] = float(np.median(finite))
        return (
            np.where(np.isfinite(values), values, medians[None, :]),
            medians,
            observed_features,
        )

    def _fit(
        self, train_batch: SeriesBatch, metadata: Mapping[str, Any]
    ) -> SoftImputeArtifact:
        values = _matrix(train_batch)
        filled, medians, observed_features = self._initial(values)
        if self.shrinkage is None:
            from scipy.linalg import svd

            singular_values = svd(filled, full_matrices=False, compute_uv=False)
            shrinkage = float(singular_values[0] * 0.1) if singular_values.size else 0.0
        else:
            shrinkage = float(self.shrinkage)
        return SoftImputeArtifact(medians, shrinkage, observed_features)

    def _impute_native(
        self, batch: SeriesBatch, artifact: SoftImputeArtifact | None, seed: int
    ) -> np.ndarray:
        if artifact is None:
            raise ValueError("softimpute requires a fitted artifact")
        from scipy.linalg import svd

        values = _matrix(batch)
        observed = np.isfinite(values)
        missing = ~observed
        filled = np.where(observed, values, artifact.medians[None, :])
        for _ in range(self.max_iter):
            previous_missing = filled[missing].copy()
            left, singular_values, right = svd(filled, full_matrices=False)
            shrunk = np.maximum(singular_values - artifact.shrinkage, 0.0)
            rank = int(np.count_nonzero(shrunk > 0))
            if self.max_rank is not None:
                rank = min(rank, self.max_rank)
            if rank == 0:
                reconstruction = np.zeros_like(filled)
            else:
                reconstruction = (left[:, :rank] * shrunk[:rank]) @ right[:rank]
            filled[missing] = reconstruction[missing]
            filled[observed] = values[observed]
            if not previous_missing.size:
                break
            difference = np.linalg.norm(filled[missing] - previous_missing)
            scale = max(float(np.linalg.norm(previous_missing)), 1e-12)
            if difference / scale <= self.tolerance:
                break
        empty = ~np.asarray(artifact.observed_features, dtype=bool)
        if empty.any():
            filled[missing & empty[None, :]] = np.nan
        return _restore(filled, batch)


__all__ = [
    "KNNMultivariateImputer",
    "MICEImputer",
    "MissForestArtifact",
    "MissForestImputer",
    "SklearnArtifact",
    "SoftImputeArtifact",
    "SoftImputeImputer",
]
