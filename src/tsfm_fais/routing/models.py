"""Lazy LightGBM wrappers and a serializable router bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _serialized_booster(component: Any) -> str | None:
    estimator = getattr(component, "model", None)
    booster = getattr(estimator, "booster_", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        return None
    return str(booster.model_to_string(num_iteration=-1))


def _lightgbm_classes():
    try:
        from lightgbm import LGBMRanker, LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError("routing requires `pip install lightgbm`") from exc
    return LGBMRanker, LGBMRegressor


def losses_to_relevance(
    losses: np.ndarray,
    groups: Sequence[int],
    *,
    relative_tie_tolerance: float = 1e-5,
) -> np.ndarray:
    """Convert losses to relevance while suppressing numerical near-ties."""

    values = np.asarray(losses, dtype=float).reshape(-1)
    group_sizes = tuple(int(size) for size in groups)
    if any(size < 1 for size in group_sizes) or sum(group_sizes) != len(values):
        raise ValueError("groups must be positive and sum to the number of labels")
    if not np.isfinite(values).all():
        raise ValueError("ranking losses must be finite")
    if not np.isfinite(relative_tie_tolerance) or relative_tie_tolerance < 0:
        raise ValueError("relative_tie_tolerance must be finite and non-negative")
    relevance = np.empty(len(values), dtype=int)
    offset = 0
    for size in group_sizes:
        group = values[offset : offset + size]
        order = np.argsort(group, kind="stable")
        cluster = np.zeros(size, dtype=int)
        cluster_count = 1
        for rank in range(1, size):
            previous = float(group[order[rank - 1]])
            current = float(group[order[rank]])
            tolerance = relative_tie_tolerance * max(
                1.0,
                abs(previous),
                abs(current),
            )
            if current - previous > tolerance:
                cluster_count += 1
            cluster[order[rank]] = cluster_count - 1
        relevance[offset : offset + size] = cluster_count - cluster - 1
        offset += size
    return relevance


def _robust_scale(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        return 1.0
    centered = np.abs(array - np.median(array))
    return max(float(np.quantile(centered, 0.75)), 1e-6)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "tsfm-fais",
        "numpy",
        "lightgbm",
        "scikit-learn",
        "joblib",
        "torch",
    ):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


@dataclass
class RankerModel:
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: Sequence[int],
    ) -> RankerModel:
        LGBMRanker, _ = _lightgbm_classes()
        defaults = {
            "objective": "lambdarank",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 20260710,
            "n_jobs": 1,
            "verbosity": -1,
            "min_child_samples": max(2, min(20, len(features) // 8)),
        }
        defaults.update(self.params)
        self.model = LGBMRanker(**defaults)
        self.model.fit(np.asarray(features), np.asarray(labels), group=list(groups))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker has not been fitted")
        return np.asarray(
            self.model.predict(np.asarray(features), num_threads=1),
            dtype=float,
        )


@dataclass
class PairwiseRiskModel:
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> PairwiseRiskModel:
        _, LGBMRegressor = _lightgbm_classes()
        defaults = {
            "objective": "huber",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 20260710,
            "n_jobs": 1,
            "verbosity": -1,
            "min_child_samples": max(2, min(20, len(features) // 8)),
        }
        defaults.update(self.params)
        self.model = LGBMRegressor(**defaults)
        self.model.fit(np.asarray(features), np.asarray(labels))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("pairwise model has not been fitted")
        return np.asarray(
            self.model.predict(np.asarray(features), num_threads=1),
            dtype=float,
        )


@dataclass
class RouterBundle:
    prior: Any
    unary: Any
    pairwise: Any
    feature_names: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    pair_feature_names: tuple[str, ...] = ()
    categorical_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target / "router_bundle.joblib")
        component_files: dict[str, dict[str, Any]] = {}
        for name, component in (
            ("prior", self.prior),
            ("unary", self.unary),
            ("pairwise", self.pairwise),
        ):
            serialized = _serialized_booster(component)
            if serialized is None:
                continue
            payload = serialized.encode("utf-8")
            component_path = target / f"{name}_model.txt"
            temporary = component_path.with_suffix(component_path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(component_path)
            component_files[name] = {
                "path": component_path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        manifest = {
            "schema_version": 1,
            "selector_method": self.metadata.get("selector_method", "b_fais"),
            "feature_names": list(self.feature_names),
            "pair_feature_names": list(self.pair_feature_names),
            "candidate_ids": list(self.candidate_ids),
            "categorical_maps": self.categorical_maps,
            "dependency_versions": _dependency_versions(),
            "component_models": component_files,
            "metadata": self.metadata,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> RouterBundle:
        source = Path(path)
        if source.is_dir():
            source = source / "router_bundle.joblib"
        bundle = joblib.load(source)
        if not isinstance(bundle, cls):
            raise TypeError(f"unexpected router artifact type: {type(bundle)!r}")
        return bundle


@dataclass
class RouterTrainer:
    prior_params: dict[str, Any] = field(default_factory=dict)
    unary_params: dict[str, Any] = field(default_factory=dict)
    pairwise_params: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        prior_features: np.ndarray,
        unary_features: np.ndarray,
        unary_labels: np.ndarray,
        groups: Sequence[int],
        pair_features: np.ndarray,
        pair_labels: np.ndarray,
        feature_names: Sequence[str],
        candidate_ids: Sequence[str],
        pair_feature_names: Sequence[str] | None = None,
        *,
        train_pairwise: bool = True,
    ) -> RouterBundle:
        labels = np.asarray(unary_labels, dtype=float).reshape(-1)
        relevance = losses_to_relevance(labels, groups)
        prior = RankerModel(self.prior_params).fit(prior_features, relevance, groups)
        unary = RankerModel(self.unary_params).fit(unary_features, relevance, groups)
        pairwise = (
            PairwiseRiskModel(self.pairwise_params).fit(pair_features, pair_labels)
            if train_pairwise
            else PairwiseRiskModel()
        )
        candidate_tuple = tuple(candidate_ids)
        return RouterBundle(
            prior=prior,
            unary=unary,
            pairwise=pairwise,
            feature_names=tuple(feature_names),
            candidate_ids=candidate_tuple,
            pair_feature_names=(tuple(pair_feature_names or ()) if train_pairwise else ()),
            categorical_maps={
                "candidate_id": {
                    candidate_id: index for index, candidate_id in enumerate(candidate_tuple)
                }
            },
            metadata={
                "dependency_versions": _dependency_versions(),
                "unary_risk_scale": _robust_scale(labels),
                "pair_risk_scale": (
                    _robust_scale(np.asarray(pair_labels, dtype=float)) if train_pairwise else 1.0
                ),
            },
        )
