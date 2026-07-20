"""Reproducible algorithm-selection baselines for teacher-label rows.

The selectors in this module share the row-wise scoring contract used by the
learned router: ``predict(X)`` returns one utility per block/candidate row and
larger values are preferred.  Training may use group structure, while stored
models contain only scikit-learn objects and CPU NumPy arrays so they remain
portable through :mod:`joblib`.

PyTorch is imported lazily and is needed only while fitting neural selectors.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

BASELINE_SELECTOR_METHODS = (
    "metaod",
    "dselect1",
    "neuralucb",
    "alors",
    "hybrid_lstm",
    "random_valid_block",
)

BASELINE_SELECTOR_PARAM_NAMES: dict[str, frozenset[str]] = {
    "metaod": frozenset(
        {
            "batch_size",
            "epochs",
            "latent_dim",
            "learning_rate",
            "max_depth",
            "min_samples_leaf",
            "n_estimators",
            "torch_threads",
            "weight_decay",
        }
    ),
    "alors": frozenset(
        {
            "epochs",
            "latent_dim",
            "max_depth",
            "min_samples_leaf",
            "n_estimators",
            "regularization",
        }
    ),
    "dselect1": frozenset(
        {
            "batch_size",
            "entropy_weight",
            "epochs",
            "gamma",
            "init_scale",
            "learning_rate",
            "padding_penalty",
            "torch_threads",
            "weight_decay",
        }
    ),
    "neuralucb": frozenset(
        {
            "alpha",
            "batch_size",
            "epochs",
            "hidden_size",
            "learning_rate",
            "replay_size",
            "retrain_interval",
            "ridge",
            "torch_threads",
            "update_epochs",
            "weight_decay",
        }
    ),
    "hybrid_lstm": frozenset(
        {
            "batch_size",
            "epochs",
            "hidden_size",
            "learning_rate",
            "multilabel_weight",
            "near_optimal_tolerance",
            "torch_threads",
            "weight_decay",
        }
    ),
    "random_valid_block": frozenset(),
}

_METHOD_ALIASES = {
    "dselect_1": "dselect1",
    "hybridlstm": "hybrid_lstm",
    "neural_ucb": "neuralucb",
    "random": "random_valid_block",
    "random_valid": "random_valid_block",
}


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "neural selector training requires `pip install -e .[selector-baselines]` "
            "or `pip install torch`"
        ) from error
    return torch


def _configure_torch(torch: Any, seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(max(1, int(threads)))
    except RuntimeError:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):  # pragma: no cover - old/unsupported torch
        pass


@dataclass(frozen=True)
class _ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> _ArrayScaler:
        matrix = np.asarray(values, dtype=float)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=np.asarray(mean, dtype=float), scale=np.asarray(scale, dtype=float))

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        return (matrix - self.mean) / self.scale


@dataclass(frozen=True)
class _PreparedTraining:
    features: np.ndarray
    losses: np.ndarray
    group_sizes: tuple[int, ...]
    group_slices: tuple[slice, ...]
    feature_names: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    row_candidate_indices: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    context_indices: np.ndarray
    candidate_feature_indices: tuple[int, ...]
    group_contexts: np.ndarray


def _group_slices(groups: Sequence[int], row_count: int) -> tuple[slice, ...]:
    sizes = tuple(int(size) for size in groups)
    if not sizes or any(size < 1 for size in sizes) or sum(sizes) != row_count:
        raise ValueError("groups must be positive and sum to the number of rows")
    result: list[slice] = []
    offset = 0
    for size in sizes:
        result.append(slice(offset, offset + size))
        offset += size
    return tuple(result)


def _context_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    indices = [
        index for index, name in enumerate(feature_names) if not str(name).startswith("candidate_")
    ]
    if not indices:
        indices = [
            index
            for index, name in enumerate(feature_names)
            if not str(name).startswith("candidate_id::")
        ]
    if not indices:
        indices = list(range(len(feature_names)))
    return np.asarray(indices, dtype=int)


def _candidate_feature_indices(
    feature_names: Sequence[str], candidate_ids: Sequence[str]
) -> tuple[int, ...]:
    positions = {str(name): index for index, name in enumerate(feature_names)}
    return tuple(
        positions.get(f"candidate_id::{candidate_id}", -1) for candidate_id in candidate_ids
    )


def _infer_candidate_indices(
    features: np.ndarray,
    candidate_feature_indices: Sequence[int],
    candidate_count: int,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    available = [
        (candidate_index, feature_index)
        for candidate_index, feature_index in enumerate(candidate_feature_indices)
        if feature_index >= 0
    ]
    if available:
        values = np.stack([matrix[:, feature_index] for _, feature_index in available], axis=1)
        selected = np.argmax(values, axis=1)
        mapped = np.asarray([available[index][0] for index in selected], dtype=int)
        # A row without an active one-hot value is uncommon but can occur in a
        # hand-built smoke test. Preserve deterministic block-major behavior.
        inactive = np.max(values, axis=1) <= 0.0
        mapped[inactive] = np.arange(len(matrix), dtype=int)[inactive] % candidate_count
        return mapped
    return np.arange(len(matrix), dtype=int) % candidate_count


def _candidate_indices_from_keys(
    keys: Sequence[Any], candidate_lookup: Mapping[str, int]
) -> np.ndarray | None:
    result: list[int] = []
    for key in keys:
        candidate: Any = None
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            candidate = key[-1]
        elif isinstance(key, Mapping):
            candidate = key.get("candidate_id")
        if not isinstance(candidate, str) or candidate not in candidate_lookup:
            return None
        result.append(candidate_lookup[candidate])
    return np.asarray(result, dtype=int)


def _prepare_training(
    features: np.ndarray,
    losses: np.ndarray,
    groups: Sequence[int],
    feature_names: Sequence[str],
    candidate_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> _PreparedTraining:
    matrix = np.asarray(features, dtype=float)
    targets = np.asarray(losses, dtype=float).reshape(-1)
    names = tuple(str(name) for name in feature_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("features must be a 2D matrix matching feature_names")
    if len(matrix) != len(targets) or len(matrix) != len(rows):
        raise ValueError("features, losses, and rows must have the same length")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("selector training data must be finite")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    universe = tuple(dict.fromkeys(str(candidate_id) for candidate_id in candidate_ids))
    if not universe:
        raise ValueError("candidate_ids must be non-empty")
    lookup = {candidate_id: index for index, candidate_id in enumerate(universe)}
    row_candidates: list[int] = []
    normalized_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} is not a mapping")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in lookup:
            raise ValueError(f"row {index} has an unknown candidate_id")
        row_candidates.append(lookup[candidate_id])
        normalized_rows.append(row)
    slices = _group_slices(groups, len(matrix))
    for group_slice in slices:
        group_candidates = np.asarray(row_candidates[group_slice], dtype=int)
        if len(set(group_candidates.tolist())) != len(group_candidates):
            raise ValueError("each group must contain at most one row per candidate")
    context_indices = _context_feature_indices(names)
    group_contexts = np.stack(
        [matrix[group_slice.start, context_indices] for group_slice in slices], axis=0
    )
    return _PreparedTraining(
        features=matrix,
        losses=targets,
        group_sizes=tuple(int(value) for value in groups),
        group_slices=slices,
        feature_names=names,
        candidate_ids=universe,
        row_candidate_indices=np.asarray(row_candidates, dtype=int),
        rows=tuple(normalized_rows),
        context_indices=context_indices,
        candidate_feature_indices=_candidate_feature_indices(names, universe),
        group_contexts=group_contexts,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and math.isclose(
            float(array[order[start]]),
            float(array[order[end]]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _group_utilities(prepared: _PreparedTraining) -> tuple[np.ndarray, np.ndarray]:
    group_count = len(prepared.group_slices)
    candidate_count = len(prepared.candidate_ids)
    utilities = np.zeros((group_count, candidate_count), dtype=float)
    observed = np.zeros((group_count, candidate_count), dtype=bool)
    for group_index, group_slice in enumerate(prepared.group_slices):
        candidate_indices = prepared.row_candidate_indices[group_slice]
        ranks = _average_ranks(prepared.losses[group_slice])
        if len(ranks) == 1:
            values = np.ones(1, dtype=float)
        else:
            values = 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)
        utilities[group_index, candidate_indices] = values
        observed[group_index, candidate_indices] = True
    return utilities, observed


def _normalized_group_rewards(prepared: _PreparedTraining) -> np.ndarray:
    rewards = np.empty(len(prepared.losses), dtype=float)
    for group_slice in prepared.group_slices:
        values = prepared.losses[group_slice]
        minimum = float(np.min(values))
        span = float(np.max(values) - minimum)
        if span <= 1e-12:
            rewards[group_slice] = 1.0
        else:
            rewards[group_slice] = 1.0 - (values - minimum) / span
    return rewards


def _fit_regressor(
    contexts: np.ndarray,
    latent: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
) -> RandomForestRegressor:
    regressor = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 64)),
        max_depth=(
            None if params.get("max_depth", 12) is None else int(params.get("max_depth", 12))
        ),
        min_samples_leaf=max(1, int(params.get("min_samples_leaf", 1))),
        random_state=seed,
        n_jobs=1,
    )
    regressor.fit(contexts, latent)
    return regressor


@dataclass
class _LatentColdStartSelector:
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    context_indices: np.ndarray
    candidate_feature_indices: tuple[int, ...]
    scaler: _ArrayScaler
    regressor: Any
    candidate_factors: np.ndarray
    pca: Any | None = None
    method: str = "latent"

    def _contexts(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        contexts = self.scaler.transform(matrix[:, self.context_indices])
        return self.pca.transform(contexts) if self.pca is not None else contexts

    def _candidate_indices(self, features: np.ndarray, keys: Sequence[Any] | None) -> np.ndarray:
        if keys is not None:
            if len(keys) != len(features):
                raise ValueError("keys must match the number of prediction rows")
            keyed = _candidate_indices_from_keys(
                keys, {candidate_id: index for index, candidate_id in enumerate(self.candidate_ids)}
            )
            if keyed is not None:
                return keyed
        return _infer_candidate_indices(
            features, self.candidate_feature_indices, len(self.candidate_ids)
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_context(features, keys=None, seed=None)

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: Sequence[Any] | None,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        matrix = np.asarray(features, dtype=float)
        latent = np.asarray(self.regressor.predict(self._contexts(matrix)), dtype=float)
        if latent.ndim == 1:
            latent = latent[:, None]
        candidate_indices = self._candidate_indices(matrix, keys)
        scores = np.einsum(
            "nk,nk->n", latent, self.candidate_factors[candidate_indices], optimize=True
        )
        return np.asarray(scores, dtype=float)


def _fit_metaod(
    prepared: _PreparedTraining,
    params: Mapping[str, Any],
    seed: int,
) -> _LatentColdStartSelector:
    torch = _require_torch()
    _configure_torch(torch, seed, int(params.get("torch_threads", 1)))
    scaler = _ArrayScaler.fit(prepared.group_contexts)
    contexts = scaler.transform(prepared.group_contexts)
    latent_dim = max(
        1,
        min(
            int(params.get("latent_dim", 8)),
            contexts.shape[0],
            contexts.shape[1],
        ),
    )
    pca = PCA(n_components=latent_dim, random_state=seed)
    initial_u = pca.fit_transform(contexts)
    rng = np.random.default_rng(seed)
    initial_v = rng.normal(
        0.0, 1.0 / math.sqrt(latent_dim), size=(len(prepared.candidate_ids), latent_dim)
    )

    pair_groups: list[int] = []
    pair_better: list[int] = []
    pair_worse: list[int] = []
    pair_weights: list[float] = []
    for group_index, group_slice in enumerate(prepared.group_slices):
        candidates = prepared.row_candidate_indices[group_slice]
        losses = prepared.losses[group_slice]
        ranks = _average_ranks(losses)
        denominator = max(1.0, len(candidates) - 1.0)
        relevance = 1.0 - (ranks - 1.0) / denominator
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if math.isclose(float(losses[left]), float(losses[right]), rel_tol=1e-10):
                    continue
                if losses[left] < losses[right]:
                    better, worse = left, right
                else:
                    better, worse = right, left
                pair_groups.append(group_index)
                pair_better.append(int(candidates[better]))
                pair_worse.append(int(candidates[worse]))
                pair_weights.append(max(abs(float(relevance[better] - relevance[worse])), 1e-3))

    u = torch.nn.Parameter(torch.as_tensor(initial_u, dtype=torch.float32))
    v = torch.nn.Parameter(torch.as_tensor(initial_v, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        (u, v),
        lr=float(params.get("learning_rate", 0.02)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    if pair_groups:
        group_tensor = torch.as_tensor(pair_groups, dtype=torch.long)
        better_tensor = torch.as_tensor(pair_better, dtype=torch.long)
        worse_tensor = torch.as_tensor(pair_worse, dtype=torch.long)
        weight_tensor = torch.as_tensor(pair_weights, dtype=torch.float32)
        batch_size = max(1, int(params.get("batch_size", 4096)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(max(1, int(params.get("epochs", 30)))):
            permutation = torch.randperm(len(pair_groups), generator=generator)
            for start in range(0, len(pair_groups), batch_size):
                indices = permutation[start : start + batch_size]
                difference = v[better_tensor[indices]] - v[worse_tensor[indices]]
                margin = torch.sum(u[group_tensor[indices]] * difference, dim=1)
                loss = torch.mean(torch.nn.functional.softplus(-margin) * weight_tensor[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    latent = u.detach().cpu().numpy().astype(float)
    factors = v.detach().cpu().numpy().astype(float)
    regressor = _fit_regressor(pca.transform(contexts), latent, params, seed)
    return _LatentColdStartSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        context_indices=prepared.context_indices,
        candidate_feature_indices=prepared.candidate_feature_indices,
        scaler=scaler,
        pca=pca,
        regressor=regressor,
        candidate_factors=factors,
        method="metaod",
    )


def _fit_alors(
    prepared: _PreparedTraining,
    params: Mapping[str, Any],
    seed: int,
) -> _LatentColdStartSelector:
    utilities, observed = _group_utilities(prepared)
    group_count, candidate_count = utilities.shape
    latent_dim = max(1, min(int(params.get("latent_dim", 8)), group_count, candidate_count))
    regularization = max(float(params.get("regularization", 0.1)), 1e-8)
    rng = np.random.default_rng(seed)
    candidate_factors = rng.normal(0.0, 0.1, size=(candidate_count, latent_dim))
    group_factors = np.zeros((group_count, latent_dim), dtype=float)
    identity = np.eye(latent_dim, dtype=float)
    for _ in range(max(1, int(params.get("epochs", 15)))):
        for group_index in range(group_count):
            candidates = np.flatnonzero(observed[group_index])
            design = candidate_factors[candidates]
            group_factors[group_index] = np.linalg.solve(
                design.T @ design + regularization * identity,
                design.T @ utilities[group_index, candidates],
            )
        for candidate_index in range(candidate_count):
            groups_for_candidate = np.flatnonzero(observed[:, candidate_index])
            if not len(groups_for_candidate):
                continue
            design = group_factors[groups_for_candidate]
            candidate_factors[candidate_index] = np.linalg.solve(
                design.T @ design + regularization * identity,
                design.T @ utilities[groups_for_candidate, candidate_index],
            )
    scaler = _ArrayScaler.fit(prepared.group_contexts)
    contexts = scaler.transform(prepared.group_contexts)
    regressor = _fit_regressor(contexts, group_factors, params, seed)
    return _LatentColdStartSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        context_indices=prepared.context_indices,
        candidate_feature_indices=prepared.candidate_feature_indices,
        scaler=scaler,
        regressor=regressor,
        candidate_factors=candidate_factors,
        method="alors",
    )


def _smooth_step_numpy(values: np.ndarray, gamma: float) -> np.ndarray:
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    array = np.asarray(values, dtype=float)
    middle = -2.0 * np.power(array, 3) / gamma**3 + 3.0 * array / (2.0 * gamma) + 0.5
    return np.where(
        array <= -gamma / 2.0,
        0.0,
        np.where(array >= gamma / 2.0, 1.0, middle),
    )


def _binary_codes(candidate_count: int) -> np.ndarray:
    bit_count = max(1, int(math.ceil(math.log2(max(2, candidate_count)))))
    padded = 1 << bit_count
    indices = np.arange(padded, dtype=np.int64)[:, None]
    shifts = np.arange(bit_count, dtype=np.int64)[None, :]
    return ((indices >> shifts) & 1).astype(float)


@dataclass
class DSelectOneSelector:
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    context_indices: np.ndarray
    candidate_feature_indices: tuple[int, ...]
    scaler: _ArrayScaler
    weight: np.ndarray
    bias: np.ndarray
    gamma: float

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        contexts = self.scaler.transform(matrix[:, self.context_indices])
        smooth = _smooth_step_numpy(contexts @ self.weight.T + self.bias, self.gamma)
        codes = _binary_codes(len(self.candidate_ids))
        factors = np.where(codes[None, :, :] > 0.5, smooth[:, None, :], 1.0 - smooth[:, None, :])
        probabilities = np.prod(factors, axis=2)
        real_probabilities = probabilities[:, : len(self.candidate_ids)]
        real_mass = real_probabilities.sum(axis=1, keepdims=True)
        return np.divide(
            real_probabilities,
            real_mass,
            out=np.full_like(real_probabilities, 1.0 / len(self.candidate_ids)),
            where=real_mass > 1e-15,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_context(features, keys=None, seed=None)

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: Sequence[Any] | None,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        matrix = np.asarray(features, dtype=float)
        if keys is not None and len(keys) != len(matrix):
            raise ValueError("keys must match the number of prediction rows")
        candidate_indices = (
            _candidate_indices_from_keys(
                keys, {value: index for index, value in enumerate(self.candidate_ids)}
            )
            if keys is not None
            else None
        )
        if candidate_indices is None:
            candidate_indices = _infer_candidate_indices(
                matrix, self.candidate_feature_indices, len(self.candidate_ids)
            )
        probabilities = self._probabilities(matrix)
        selected = probabilities[np.arange(len(matrix)), candidate_indices]
        return np.log(np.clip(selected, 1e-15, 1.0))


def _fit_dselect1(
    prepared: _PreparedTraining,
    params: Mapping[str, Any],
    seed: int,
) -> DSelectOneSelector:
    torch = _require_torch()
    _configure_torch(torch, seed, int(params.get("torch_threads", 1)))
    scaler = _ArrayScaler.fit(prepared.group_contexts)
    contexts = scaler.transform(prepared.group_contexts)
    utilities, observed = _group_utilities(prepared)
    normalized_losses = np.zeros_like(utilities)
    for group_index in range(len(prepared.group_slices)):
        candidates = np.flatnonzero(observed[group_index])
        # Utility is one for the best candidate and zero for the worst.
        normalized_losses[group_index, candidates] = 1.0 - utilities[group_index, candidates]

    gamma = float(params.get("gamma", 1.0))
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("dselect1 gamma must be finite and positive")
    codes = _binary_codes(len(prepared.candidate_ids))
    bit_count = codes.shape[1]
    rng = np.random.default_rng(seed)
    weight = torch.nn.Parameter(
        torch.as_tensor(
            rng.normal(
                0.0, float(params.get("init_scale", 0.01)), size=(bit_count, contexts.shape[1])
            ),
            dtype=torch.float32,
        )
    )
    bias = torch.nn.Parameter(torch.zeros(bit_count, dtype=torch.float32))
    optimizer = torch.optim.Adam(
        (weight, bias),
        lr=float(params.get("learning_rate", 0.01)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
    )
    context_tensor = torch.as_tensor(contexts, dtype=torch.float32)
    loss_tensor = torch.as_tensor(normalized_losses, dtype=torch.float32)
    mask_tensor = torch.as_tensor(observed, dtype=torch.float32)
    code_tensor = torch.as_tensor(codes, dtype=torch.float32)
    batch_size = max(1, int(params.get("batch_size", 512)))
    entropy_weight = max(0.0, float(params.get("entropy_weight", 0.01)))
    padding_penalty = max(0.0, float(params.get("padding_penalty", 1.0)))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def smooth_step(values: Any) -> Any:
        middle = -2.0 * values.pow(3) / gamma**3 + 3.0 * values / (2.0 * gamma) + 0.5
        return torch.where(
            values <= -gamma / 2.0,
            torch.zeros_like(values),
            torch.where(values >= gamma / 2.0, torch.ones_like(values), middle),
        )

    for _ in range(max(1, int(params.get("epochs", 100)))):
        permutation = torch.randperm(len(context_tensor), generator=generator)
        for start in range(0, len(context_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            smooth = smooth_step(context_tensor[indices] @ weight.T + bias)
            factors = torch.where(
                code_tensor[None, :, :] > 0.5,
                smooth[:, None, :],
                1.0 - smooth[:, None, :],
            )
            probabilities = torch.prod(factors, dim=2)
            real_probabilities = probabilities[:, : len(prepared.candidate_ids)]
            masked = real_probabilities * mask_tensor[indices]
            normalized = masked / torch.clamp(masked.sum(dim=1, keepdim=True), min=1e-8)
            expected = torch.sum(normalized * loss_tensor[indices], dim=1).mean()
            entropy = -torch.sum(
                probabilities * torch.log(torch.clamp(probabilities, min=1e-8)), dim=1
            ).mean()
            valid_code_mass = real_probabilities.sum(dim=1).mean()
            loss = expected + entropy_weight * entropy - padding_penalty * valid_code_mass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return DSelectOneSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        context_indices=prepared.context_indices,
        candidate_feature_indices=prepared.candidate_feature_indices,
        scaler=scaler,
        weight=weight.detach().cpu().numpy().astype(float),
        bias=bias.detach().cpu().numpy().astype(float),
        gamma=gamma,
    )


@dataclass
class NeuralUCBSelector:
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    candidate_feature_indices: tuple[int, ...]
    scaler: _ArrayScaler
    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: float
    precision_weight1: np.ndarray
    precision_bias1: np.ndarray
    precision_weight2: np.ndarray
    precision_bias2: float
    alpha: float
    offline_replay: bool = True
    replayed_group_count: int = 0
    observed_row_count: int = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_context(features, keys=None, seed=None)

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: Sequence[Any] | None,
        seed: int | None = None,
    ) -> np.ndarray:
        del keys, seed
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        scaled = self.scaler.transform(matrix)
        preactivation = scaled @ self.weight1.T + self.bias1
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        mean = hidden @ self.weight2 + self.bias2
        weight2_squared = np.square(self.weight2)
        first_layer = np.einsum(
            "nd,dh,nh->n",
            np.square(scaled),
            weight2_squared[None, :] / self.precision_weight1,
            active,
            optimize=True,
        )
        first_bias = np.sum(
            active * weight2_squared[None, :] / self.precision_bias1[None, :], axis=1
        )
        second_layer = np.sum(np.square(hidden) / self.precision_weight2[None, :], axis=1)
        uncertainty = np.sqrt(
            np.maximum(
                (first_layer + first_bias + second_layer + 1.0 / self.precision_bias2)
                / max(1, self.weight1.shape[0]),
                0.0,
            )
        )
        return np.asarray(mean + self.alpha * uncertainty, dtype=float)


def _fit_neuralucb(
    prepared: _PreparedTraining,
    params: Mapping[str, Any],
    seed: int,
) -> NeuralUCBSelector:
    torch = _require_torch()
    _configure_torch(torch, seed, int(params.get("torch_threads", 1)))
    scaler = _ArrayScaler.fit(prepared.features)
    features = scaler.transform(prepared.features)
    center = float(np.median(prepared.losses))
    lower, upper = np.quantile(prepared.losses, (0.25, 0.75))
    reward_scale = max(float((upper - lower) / 1.349), 1e-6)
    reward_argument = np.clip(-(prepared.losses - center) / reward_scale, -40.0, 40.0)
    rewards = 1.0 / (1.0 + np.exp(-reward_argument))
    hidden_size = max(2, int(params.get("hidden_size", 32)))
    network = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    batch_size = max(1, int(params.get("batch_size", 512)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ridge = max(float(params.get("ridge", 1.0)), 1e-8)
    normalization = float(max(1, hidden_size))
    precision_weight1 = np.full((features.shape[1], hidden_size), ridge, dtype=float)
    precision_bias1 = np.full(hidden_size, ridge, dtype=float)
    precision_weight2 = np.full(hidden_size, ridge, dtype=float)
    precision_bias2 = float(ridge)
    alpha = max(0.0, float(params.get("alpha", 1.0)))
    replay_size = max(1, int(params.get("replay_size", 2048)))
    retrain_interval = max(1, int(params.get("retrain_interval", 64)))
    update_epochs = max(1, int(params.get("update_epochs", 1)))
    observed_rows: list[int] = []

    def numpy_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            network[0].weight.detach().cpu().numpy().astype(float),
            network[0].bias.detach().cpu().numpy().astype(float),
            network[2].weight.detach().cpu().numpy().reshape(-1).astype(float),
            float(network[2].bias.detach().cpu().item()),
        )

    def score_rows(indices: np.ndarray) -> np.ndarray:
        weight1, bias1, weight2, bias2 = numpy_parameters()
        selected_features = features[indices]
        preactivation = selected_features @ weight1.T + bias1
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        mean = hidden @ weight2 + bias2
        weight2_squared = np.square(weight2)
        first_layer = np.einsum(
            "nd,dh,nh->n",
            np.square(selected_features),
            weight2_squared[None, :] / precision_weight1,
            active,
            optimize=True,
        )
        first_bias = np.sum(active * weight2_squared[None, :] / precision_bias1[None, :], axis=1)
        second_layer = np.sum(np.square(hidden) / precision_weight2[None, :], axis=1)
        bonus = np.sqrt(
            np.maximum(
                (first_layer + first_bias + second_layer + 1.0 / precision_bias2) / normalization,
                0.0,
            )
        )
        return mean + alpha * bonus

    def update_precision(row_index: int) -> None:
        nonlocal precision_bias2
        weight1, bias1, weight2, _ = numpy_parameters()
        row = features[row_index]
        preactivation = row @ weight1.T + bias1
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        local = weight2 * active
        precision_weight1[:] += np.square(row[:, None] * local[None, :]) / normalization
        precision_bias1[:] += np.square(local) / normalization
        precision_weight2[:] += np.square(hidden) / normalization
        precision_bias2 += 1.0 / normalization

    def train_selected(indices: Sequence[int], epochs: int) -> None:
        if not indices:
            return
        selected = torch.as_tensor(tuple(indices)[-replay_size:], dtype=torch.long)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(params.get("learning_rate", 0.005)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        network.train()
        for _ in range(epochs):
            permutation = selected[torch.randperm(len(selected), generator=generator)]
            for start in range(0, len(permutation), batch_size):
                batch_indices = permutation[start : start + batch_size]
                predicted = network(feature_tensor[batch_indices]).reshape(-1)
                loss = torch.nn.functional.mse_loss(predicted, reward_tensor[batch_indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    for group_index, group_slice in enumerate(prepared.group_slices):
        row_indices = np.arange(group_slice.start, group_slice.stop, dtype=int)
        selected_row = int(row_indices[int(np.argmax(score_rows(row_indices)))])
        observed_rows.append(selected_row)
        update_precision(selected_row)
        if (group_index + 1) % retrain_interval == 0:
            train_selected(observed_rows, update_epochs)
    train_selected(observed_rows, max(1, int(params.get("epochs", 10))))
    weight1, bias1, weight2, bias2 = numpy_parameters()
    return NeuralUCBSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        candidate_feature_indices=prepared.candidate_feature_indices,
        scaler=scaler,
        weight1=weight1,
        bias1=bias1,
        weight2=weight2,
        bias2=bias2,
        precision_weight1=np.asarray(precision_weight1, dtype=float),
        precision_bias1=np.asarray(precision_bias1, dtype=float),
        precision_weight2=np.asarray(precision_weight2, dtype=float),
        precision_bias2=precision_bias2,
        alpha=alpha,
        offline_replay=True,
        replayed_group_count=len(prepared.group_slices),
        observed_row_count=len(observed_rows),
    )


def _episode_sequences(prepared: _PreparedTraining) -> tuple[tuple[int, ...], ...]:
    episodes: OrderedDict[str, list[int]] = OrderedDict()
    for group_index, group_slice in enumerate(prepared.group_slices):
        row = prepared.rows[group_slice.start]
        episode_id = str(row.get("episode_id", row.get("group_id", group_index)))
        forecaster_id = str(row.get("forecaster_id", ""))
        episodes.setdefault(f"{forecaster_id}::{episode_id}", []).append(group_index)

    def ordering(group_index: int) -> tuple[float, float, str]:
        row = prepared.rows[prepared.group_slices[group_index].start]
        prior = row.get("prior_features", {})
        if not isinstance(prior, Mapping):
            prior = {}
        try:
            start = float(prior.get("start_ratio", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        try:
            channel = float(prior.get("channel_ratio", 0.0))
        except (TypeError, ValueError):
            channel = 0.0
        return start, channel, str(row.get("block_id", group_index))

    return tuple(tuple(sorted(indices, key=ordering)) for indices in episodes.values())


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass
class HybridLSTMSelector:
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    context_indices: np.ndarray
    candidate_feature_indices: tuple[int, ...]
    scaler: _ArrayScaler
    static_weight: np.ndarray
    static_bias: np.ndarray
    lstm_weight_ih: np.ndarray
    lstm_weight_hh: np.ndarray
    lstm_bias_ih: np.ndarray
    lstm_bias_hh: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def _sequence_logits(self, contexts: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(contexts, dtype=float))
        static = np.maximum(scaled @ self.static_weight.T + self.static_bias, 0.0)
        hidden_size = self.static_weight.shape[0]
        hidden = np.zeros(hidden_size, dtype=float)
        cell = np.zeros(hidden_size, dtype=float)
        recurrent: list[np.ndarray] = []
        for row in scaled:
            gates = (
                self.lstm_weight_ih @ row
                + self.lstm_bias_ih
                + self.lstm_weight_hh @ hidden
                + self.lstm_bias_hh
            )
            input_gate = _sigmoid_numpy(gates[:hidden_size])
            forget_gate = _sigmoid_numpy(gates[hidden_size : 2 * hidden_size])
            proposal = np.tanh(gates[2 * hidden_size : 3 * hidden_size])
            output_gate = _sigmoid_numpy(gates[3 * hidden_size :])
            cell = forget_gate * cell + input_gate * proposal
            hidden = output_gate * np.tanh(cell)
            recurrent.append(hidden.copy())
        combined = np.concatenate((static, np.stack(recurrent, axis=0)), axis=1)
        return combined @ self.output_weight.T + self.output_bias

    def _groups_from_keys(
        self, features: np.ndarray, keys: Sequence[Any] | None
    ) -> tuple[list[list[int]], np.ndarray]:
        matrix = np.asarray(features, dtype=float)
        candidate_indices = (
            _candidate_indices_from_keys(
                keys, {value: index for index, value in enumerate(self.candidate_ids)}
            )
            if keys is not None
            else None
        )
        if candidate_indices is None:
            candidate_indices = _infer_candidate_indices(
                matrix, self.candidate_feature_indices, len(self.candidate_ids)
            )
        groups: list[list[int]] = []
        if keys is not None:
            if len(keys) != len(matrix):
                raise ValueError("keys must match the number of prediction rows")
            by_block: OrderedDict[str, list[int]] = OrderedDict()
            for index, key in enumerate(keys):
                if isinstance(key, (tuple, list)) and key:
                    block_id = str(key[0])
                elif isinstance(key, Mapping):
                    block_id = str(key.get("block_id", index))
                else:
                    block_id = str(index)
                by_block.setdefault(block_id, []).append(index)
            groups = list(by_block.values())
        else:
            context = matrix[:, self.context_indices]
            current: list[int] = []
            seen: set[int] = set()
            previous: np.ndarray | None = None
            for index, candidate_index in enumerate(candidate_indices):
                changed = previous is not None and not np.array_equal(context[index], previous)
                if current and (changed or int(candidate_index) in seen):
                    groups.append(current)
                    current = []
                    seen = set()
                current.append(index)
                seen.add(int(candidate_index))
                previous = context[index]
            if current:
                groups.append(current)
        return groups, candidate_indices

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_context(features, keys=None, seed=None)

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: Sequence[Any] | None,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        groups, candidate_indices = self._groups_from_keys(matrix, keys)

        feature_lookup = {name: index for index, name in enumerate(self.feature_names)}
        start_index = feature_lookup.get("start_ratio")
        channel_index = feature_lookup.get("channel_ratio")

        def ordering(group: list[int]) -> tuple[float, float, str]:
            row_index = group[0]
            start = float(matrix[row_index, start_index]) if start_index is not None else 0.0
            channel = float(matrix[row_index, channel_index]) if channel_index is not None else 0.0
            key = keys[row_index] if keys is not None else row_index
            if isinstance(key, (tuple, list)) and key:
                block_id = str(key[0])
            elif isinstance(key, Mapping):
                block_id = str(key.get("block_id", row_index))
            else:
                block_id = str(row_index)
            return start, channel, block_id

        groups.sort(key=ordering)
        contexts = np.stack([matrix[group[0], self.context_indices] for group in groups], axis=0)
        logits = self._sequence_logits(contexts)
        scores = np.empty(len(matrix), dtype=float)
        for group_index, row_indices in enumerate(groups):
            for row_index in row_indices:
                scores[row_index] = logits[group_index, candidate_indices[row_index]]
        return scores


def _fit_hybrid_lstm(
    prepared: _PreparedTraining,
    params: Mapping[str, Any],
    seed: int,
) -> HybridLSTMSelector:
    torch = _require_torch()
    _configure_torch(torch, seed, int(params.get("torch_threads", 1)))
    scaler = _ArrayScaler.fit(prepared.group_contexts)
    contexts = scaler.transform(prepared.group_contexts)
    utilities, observed = _group_utilities(prepared)
    candidate_count = len(prepared.candidate_ids)
    targets = np.argmax(utilities, axis=1).astype(np.int64)
    near_optimal = np.zeros_like(observed, dtype=float)
    tolerance = max(0.0, float(params.get("near_optimal_tolerance", 0.05)))
    for group_index, group_slice in enumerate(prepared.group_slices):
        candidates = prepared.row_candidate_indices[group_slice]
        losses = prepared.losses[group_slice]
        minimum = float(np.min(losses))
        span = float(np.max(losses) - minimum)
        threshold = minimum + tolerance * span
        near_optimal[group_index, candidates] = (losses <= threshold + 1e-12).astype(float)

    hidden_size = max(2, int(params.get("hidden_size", 32)))

    class HybridNetwork(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.static = torch.nn.Linear(contexts.shape[1], hidden_size)
            self.lstm = torch.nn.LSTM(contexts.shape[1], hidden_size, batch_first=True)
            self.output = torch.nn.Linear(2 * hidden_size, candidate_count)

        def forward(self, values: Any) -> Any:
            static = torch.relu(self.static(values))
            recurrent, _ = self.lstm(values)
            return self.output(torch.cat((static, recurrent), dim=-1))

    network = HybridNetwork()
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=float(params.get("learning_rate", 0.003)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    sequences = _episode_sequences(prepared)
    batch_size = max(1, int(params.get("batch_size", 64)))
    multilabel_weight = max(0.0, float(params.get("multilabel_weight", 0.5)))
    rng = np.random.default_rng(seed)
    network.train()
    for _ in range(max(1, int(params.get("epochs", 30)))):
        order = rng.permutation(len(sequences))
        for start in range(0, len(sequences), batch_size):
            episode_batch = [sequences[index] for index in order[start : start + batch_size]]
            maximum = max(len(sequence) for sequence in episode_batch)
            padded = np.zeros((len(episode_batch), maximum, contexts.shape[1]), dtype=np.float32)
            time_mask = np.zeros((len(episode_batch), maximum), dtype=bool)
            candidate_mask = np.zeros((len(episode_batch), maximum, candidate_count), dtype=bool)
            target_batch = np.zeros((len(episode_batch), maximum), dtype=np.int64)
            multi_batch = np.zeros((len(episode_batch), maximum, candidate_count), dtype=np.float32)
            for batch_index, sequence in enumerate(episode_batch):
                length = len(sequence)
                padded[batch_index, :length] = contexts[list(sequence)]
                time_mask[batch_index, :length] = True
                candidate_mask[batch_index, :length] = observed[list(sequence)]
                target_batch[batch_index, :length] = targets[list(sequence)]
                multi_batch[batch_index, :length] = near_optimal[list(sequence)]
            padded_tensor = torch.as_tensor(padded, dtype=torch.float32)
            time_tensor = torch.as_tensor(time_mask, dtype=torch.bool)
            candidate_tensor = torch.as_tensor(candidate_mask, dtype=torch.bool)
            target_tensor = torch.as_tensor(target_batch, dtype=torch.long)
            multi_tensor = torch.as_tensor(multi_batch, dtype=torch.float32)
            logits = network(padded_tensor)
            masked_logits = logits.masked_fill(~candidate_tensor, -1e9)
            classification = torch.nn.functional.cross_entropy(
                masked_logits[time_tensor], target_tensor[time_tensor]
            )
            valid_logits = logits[candidate_tensor & time_tensor[:, :, None]]
            valid_multi = multi_tensor[candidate_tensor & time_tensor[:, :, None]]
            multilabel = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_logits, valid_multi
            )
            loss = classification + multilabel_weight * multilabel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    state = {
        name: value.detach().cpu().numpy().astype(float)
        for name, value in network.state_dict().items()
    }
    return HybridLSTMSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        context_indices=prepared.context_indices,
        candidate_feature_indices=prepared.candidate_feature_indices,
        scaler=scaler,
        static_weight=state["static.weight"],
        static_bias=state["static.bias"],
        lstm_weight_ih=state["lstm.weight_ih_l0"],
        lstm_weight_hh=state["lstm.weight_hh_l0"],
        lstm_bias_ih=state["lstm.bias_ih_l0"],
        lstm_bias_hh=state["lstm.bias_hh_l0"],
        output_weight=state["output.weight"],
        output_bias=state["output.bias"],
    )


def _stable_uniform(parts: Sequence[Any]) -> float:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    integer = int.from_bytes(digest.digest()[:8], "little", signed=False)
    return integer / float(1 << 64)


@dataclass
class RandomValidBlockSelector:
    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    candidate_feature_indices: tuple[int, ...]
    root_seed: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        candidate_indices = _infer_candidate_indices(
            matrix, self.candidate_feature_indices, len(self.candidate_ids)
        )
        scores = []
        for row, candidate_index in zip(matrix, candidate_indices, strict=True):
            row_bytes = np.asarray(row, dtype="<f8").tobytes().hex()
            scores.append(
                _stable_uniform((self.root_seed, row_bytes, self.candidate_ids[candidate_index]))
            )
        return np.asarray(scores, dtype=float)

    def predict_with_context(
        self,
        features: np.ndarray,
        *,
        keys: Sequence[Any],
        seed: int | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("prediction features do not match the fitted selector schema")
        if len(keys) != len(matrix):
            raise ValueError("keys must match the number of prediction rows")
        candidate_indices = _candidate_indices_from_keys(
            keys, {value: index for index, value in enumerate(self.candidate_ids)}
        )
        if candidate_indices is None:
            candidate_indices = _infer_candidate_indices(
                matrix, self.candidate_feature_indices, len(self.candidate_ids)
            )
        replicate = self.root_seed if seed is None else int(seed)

        def block_id(key: Any, row_index: int) -> Any:
            if isinstance(key, (tuple, list)) and key:
                return key[0]
            if isinstance(key, Mapping):
                return key.get("block_id", row_index)
            return key

        return np.asarray(
            [
                _stable_uniform(
                    (
                        self.root_seed,
                        replicate,
                        block_id(key, row_index),
                        self.candidate_ids[candidate_index],
                    )
                )
                for row_index, (key, candidate_index) in enumerate(
                    zip(keys, candidate_indices, strict=True)
                )
            ],
            dtype=float,
        )


def fit_baseline_selector(
    method: str,
    features: np.ndarray,
    losses: np.ndarray,
    groups: Sequence[int],
    feature_names: Sequence[str],
    candidate_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any] | None = None,
    seed: int = 20260710,
) -> Any:
    """Fit one selector baseline and return a joblib-serializable model.

    Parameters follow the existing router-training layout: rows are contiguous
    ranking groups, and ``candidate_ids`` is the global candidate universe.
    """

    normalized_method = str(method).strip().lower().replace("-", "_")
    normalized_method = _METHOD_ALIASES.get(normalized_method, normalized_method)
    if normalized_method not in BASELINE_SELECTOR_METHODS:
        raise ValueError(
            f"unknown baseline selector {method!r}; expected one of "
            + ", ".join(BASELINE_SELECTOR_METHODS)
        )
    prepared = _prepare_training(
        features,
        losses,
        groups,
        feature_names,
        candidate_ids,
        rows,
    )
    resolved_params = dict(params or {})
    unknown_params = set(resolved_params).difference(
        BASELINE_SELECTOR_PARAM_NAMES[normalized_method]
    )
    if unknown_params:
        raise ValueError(
            f"unsupported parameters for {normalized_method}: " + ", ".join(sorted(unknown_params))
        )
    resolved_seed = int(seed)
    if normalized_method == "metaod":
        return _fit_metaod(prepared, resolved_params, resolved_seed)
    if normalized_method == "alors":
        return _fit_alors(prepared, resolved_params, resolved_seed)
    if normalized_method == "dselect1":
        return _fit_dselect1(prepared, resolved_params, resolved_seed)
    if normalized_method == "neuralucb":
        return _fit_neuralucb(prepared, resolved_params, resolved_seed)
    if normalized_method == "hybrid_lstm":
        return _fit_hybrid_lstm(prepared, resolved_params, resolved_seed)
    return RandomValidBlockSelector(
        candidate_ids=prepared.candidate_ids,
        feature_names=prepared.feature_names,
        candidate_feature_indices=prepared.candidate_feature_indices,
        root_seed=resolved_seed,
    )


__all__ = [
    "ALORSSelector",
    "BASELINE_SELECTOR_PARAM_NAMES",
    "BASELINE_SELECTOR_METHODS",
    "DSelectOneSelector",
    "HybridLSTMSelector",
    "MetaODSelector",
    "NeuralUCBSelector",
    "RandomValidBlockSelector",
    "fit_baseline_selector",
]


# Public aliases keep artifact class names descriptive without duplicating the
# shared cold-start implementation.
MetaODSelector = _LatentColdStartSelector
ALORSSelector = _LatentColdStartSelector
