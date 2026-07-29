"""Whole-sequence cold-start selectors inspired by MetaOD and ALORS.

Both selectors treat one incomplete sequence as one task and return one score
per imputation candidate.  Training losses are lower-is-better reconstruction
losses; ``NaN`` entries denote candidate/task pairs that were not evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .sequence_features import SEQUENCE_FEATURE_NAMES


def _validate_training_data(
    contexts: np.ndarray,
    loss_matrix: np.ndarray,
    candidate_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    context_array = np.asarray(contexts, dtype=float)
    losses = np.asarray(loss_matrix, dtype=float)
    ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    if context_array.ndim != 2 or context_array.shape[0] < 1 or context_array.shape[1] < 1:
        raise ValueError("contexts must have non-empty shape [tasks, features]")
    if not np.isfinite(context_array).all():
        raise ValueError("contexts must be finite")
    if losses.ndim != 2 or losses.shape != (len(context_array), len(ids)):
        raise ValueError("loss_matrix must have shape [tasks, candidates]")
    if np.isinf(losses).any():
        raise ValueError("loss_matrix may contain finite values or NaN, but not infinity")
    if not ids or any(not candidate_id.strip() for candidate_id in ids):
        raise ValueError("candidate_ids must contain non-empty identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")

    minimum_observations = 1 if len(ids) == 1 else 2
    useful_rows = np.count_nonzero(np.isfinite(losses), axis=1) >= minimum_observations
    if not np.any(useful_rows):
        raise ValueError("training requires at least one task with comparable candidates")
    context_array = context_array[useful_rows]
    losses = losses[useful_rows]
    unsupported = [
        ids[index] for index in range(len(ids)) if not np.any(np.isfinite(losses[:, index]))
    ]
    if unsupported:
        raise ValueError(f"candidates have no observed training loss: {unsupported}")
    return context_array, losses, ids


def _positive_int(params: Mapping[str, Any], name: str, default: int) -> int:
    value = int(params.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(params: Mapping[str, Any], name: str, default: float) -> float:
    value = float(params.get(name, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_float(params: Mapping[str, Any], name: str, default: float) -> float:
    value = float(params.get(name, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _fit_random_forest(
    contexts: np.ndarray,
    targets: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    *,
    default_n_estimators: int,
    default_max_depth: int | None,
    independent_outputs: bool,
) -> RandomForestRegressor | MultiOutputRegressor:
    max_depth_value = params.get("max_depth", default_max_depth)
    max_depth = None if max_depth_value is None else int(max_depth_value)
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be positive or None")
    base_regressor = RandomForestRegressor(
        n_estimators=_positive_int(params, "n_estimators", default_n_estimators),
        max_depth=max_depth,
        min_samples_split=_positive_int(params, "min_samples_split", 2),
        min_samples_leaf=_positive_int(params, "min_samples_leaf", 1),
        random_state=seed,
        n_jobs=1,
    )
    regressor: RandomForestRegressor | MultiOutputRegressor
    if independent_outputs:
        # MetaOD's reference implementation fits one forest per latent target.
        regressor = MultiOutputRegressor(base_regressor, n_jobs=1)
    else:
        regressor = base_regressor
    regressor.fit(contexts, targets)
    return regressor


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with lower values receiving better ranks."""

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and np.isclose(
            array[order[end]], array[order[start]], rtol=1e-10, atol=1e-12
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _loss_gains(losses: np.ndarray) -> np.ndarray | None:
    """Map lower-is-better reconstruction losses to MetaOD DCG gains.

    MetaOD consumes a higher-is-better performance matrix and its reference
    code uses ``10 ** performance - 1``.  Sequence-label generation defines
    performance (``imputation_reward``) as ``1 / (1 + loss)``, so the same
    mapping is reconstructed here from the loss matrix passed to the selector.
    """

    row = np.asarray(losses, dtype=float)
    if np.any(row < 0.0):
        raise ValueError("MetaOD reconstruction losses must be non-negative")
    if float(np.max(row) - np.min(row)) <= 1e-12:
        return None
    performance = 1.0 / (1.0 + row)
    return np.power(10.0, performance) - 1.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _smooth_ndcg_and_gradient(
    scores: np.ndarray,
    loss_matrix: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray, int]:
    """Compute MetaOD's masked, unnormalized sDCG and score gradient.

    The historical helper name is retained for artifact/test compatibility.
    The paper's self-comparison contributes ``sigmoid(0) == 0.5``; after
    removing the diagonal from the pair matrix this yields the explicit 1.5
    offset below.
    """

    score_array = np.asarray(scores, dtype=float)
    losses = np.asarray(loss_matrix, dtype=float)
    if score_array.shape != losses.shape or score_array.ndim != 2:
        raise ValueError("scores and loss_matrix must have the same 2-D shape")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    gradient = np.zeros_like(score_array)
    objective = 0.0
    informative_rows = 0
    for row_index in range(len(score_array)):
        observed = np.flatnonzero(np.isfinite(losses[row_index]))
        if len(observed) < 2:
            continue
        gains = _loss_gains(losses[row_index, observed])
        if gains is None:
            continue
        row_scores = score_array[row_index, observed]
        differences = (row_scores[None, :] - row_scores[:, None]) / temperature
        pair_probabilities = _sigmoid(differences)
        np.fill_diagonal(pair_probabilities, 0.0)
        beta = 1.5 + np.sum(pair_probabilities, axis=1)
        log_beta = np.log(beta)
        row_dcg = float(np.sum(np.log(2.0) * gains / log_beta))
        objective += row_dcg

        dcg_rank_gradient = -np.log(2.0) * gains / (beta * np.square(log_beta))
        rank_derivative = pair_probabilities * (1.0 - pair_probabilities) / temperature
        row_gradient = rank_derivative.T @ dcg_rank_gradient - dcg_rank_gradient * np.sum(
            rank_derivative, axis=1
        )
        gradient[row_index, observed] = row_gradient
        informative_rows += 1

    return objective, gradient, informative_rows


def _clip_gradient(gradient: np.ndarray, maximum_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(gradient))
    if norm > maximum_norm:
        return gradient * (maximum_norm / norm)
    return gradient


def _adam_ascent(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta1 = 0.9
    beta2 = 0.999
    first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    second_moment = beta2 * second_moment + (1.0 - beta2) * np.square(gradient)
    corrected_first = first_moment / (1.0 - beta1**step)
    corrected_second = second_moment / (1.0 - beta2**step)
    parameter = parameter + learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return parameter, first_moment, second_moment


def _prepare_context(
    context: np.ndarray | Mapping[str, float],
    expected_dimension: int,
) -> np.ndarray:
    if isinstance(context, Mapping):
        if expected_dimension != len(SEQUENCE_FEATURE_NAMES):
            raise ValueError(
                "dictionary contexts require a selector fitted with the sequence feature schema"
            )
        missing = set(SEQUENCE_FEATURE_NAMES) - set(context)
        extra = set(context) - set(SEQUENCE_FEATURE_NAMES)
        if missing or extra:
            raise ValueError(
                f"context feature dictionary schema mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        array = np.asarray([context[name] for name in SEQUENCE_FEATURE_NAMES], dtype=float)
    else:
        array = np.asarray(context, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape != (1, expected_dimension):
        raise ValueError(
            f"context must have shape [{expected_dimension}] or [1,{expected_dimension}]"
        )
    if not np.isfinite(array).all():
        raise ValueError("context must be finite")
    return array


def _latent_prediction(
    regressor: RandomForestRegressor | MultiOutputRegressor,
    contexts: np.ndarray,
) -> np.ndarray:
    latent = np.asarray(regressor.predict(contexts), dtype=float)
    if latent.ndim == 1:
        latent = latent[:, None]
    return latent


@dataclass
class MetaODSequenceSelector:
    """MetaOD-style smooth-DCG selector at whole-sequence granularity."""

    params: Mapping[str, Any] | None = None
    seed: int = 0
    candidate_ids: tuple[str, ...] = field(default=(), init=False)
    context_dimension: int = field(default=0, init=False)
    candidate_factors: np.ndarray | None = field(default=None, init=False)
    pca: PCA | None = field(default=None, init=False)
    embedding_scaler: StandardScaler | None = field(default=None, init=False)
    regressor: RandomForestRegressor | MultiOutputRegressor | None = field(
        default=None,
        init=False,
    )
    training_objective: float = field(default=0.0, init=False)
    method: str = field(default="metaod", init=False)

    def fit(
        self,
        contexts: np.ndarray,
        loss_matrix: np.ndarray,
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> MetaODSequenceSelector:
        """Fit latent task/candidate factors and the cold-start random forest."""

        settings = dict((self.params or {}) if params is None else params)
        training_seed = self.seed if seed is None else int(seed)
        context_array, losses, ids = _validate_training_data(contexts, loss_matrix, candidate_ids)
        rng = np.random.default_rng(training_seed)
        requested_dimension = _positive_int(settings, "latent_dim", 8)
        latent_dimension = min(
            requested_dimension,
            context_array.shape[0],
            context_array.shape[1],
        )
        pca = PCA(n_components=latent_dimension, random_state=training_seed)
        embedded_contexts = pca.fit_transform(context_array)
        embedding_scaler = StandardScaler()
        encoded_contexts = embedding_scaler.fit_transform(embedded_contexts)
        encoded_contexts = np.nan_to_num(encoded_contexts, nan=0.0)

        group_factors = np.asarray(encoded_contexts, dtype=float).copy()
        if float(np.linalg.norm(group_factors)) <= 1e-12:
            group_factors = np.asarray(
                rng.normal(0.0, 0.05, size=group_factors.shape),
                dtype=float,
            )
        candidate_factors = rng.normal(
            0.0,
            1.0 / latent_dimension,
            size=(len(ids), latent_dimension),
        )
        epochs = _positive_int(settings, "epochs", 10)
        learning_rate = _positive_float(settings, "learning_rate", 0.03)
        regularization = _nonnegative_float(settings, "regularization", 0.0)
        maximum_gradient_norm = _positive_float(settings, "max_gradient_norm", 10.0)
        objective, _, informative_rows = _smooth_ndcg_and_gradient(
            group_factors @ candidate_factors.T,
            losses,
            1.0,
        )
        if informative_rows == 0:
            group_factors.fill(0.0)
            candidate_factors.fill(0.0)
        else:
            # Eq. (6) is optimized one task at a time.  Each epoch first holds
            # U fixed while updating V, then holds the resulting V fixed while
            # updating every U_i.  This preserves MetaOD's alternating update
            # rather than applying one simultaneous matrix gradient.
            for _ in range(epochs):
                task_order = rng.permutation(len(group_factors))
                for task_index in task_order:
                    _, score_gradient, row_count = _smooth_ndcg_and_gradient(
                        group_factors[task_index : task_index + 1] @ candidate_factors.T,
                        losses[task_index : task_index + 1],
                        1.0,
                    )
                    if row_count:
                        candidate_gradient = np.outer(
                            score_gradient[0],
                            group_factors[task_index],
                        )
                        candidate_factors += learning_rate * _clip_gradient(
                            candidate_gradient,
                            maximum_gradient_norm,
                        )
                if regularization:
                    candidate_factors -= learning_rate * regularization * candidate_factors

                for task_index in task_order:
                    _, score_gradient, row_count = _smooth_ndcg_and_gradient(
                        group_factors[task_index : task_index + 1] @ candidate_factors.T,
                        losses[task_index : task_index + 1],
                        1.0,
                    )
                    if row_count:
                        group_gradient = score_gradient[0] @ candidate_factors
                        group_factors[task_index] += learning_rate * _clip_gradient(
                            group_gradient,
                            maximum_gradient_norm,
                        )
                if regularization:
                    group_factors -= learning_rate * regularization * group_factors

            objective, _, _ = _smooth_ndcg_and_gradient(
                group_factors @ candidate_factors.T,
                losses,
                1.0,
            )

        regressor = _fit_random_forest(
            encoded_contexts,
            group_factors,
            settings,
            training_seed,
            default_n_estimators=100,
            default_max_depth=10,
            independent_outputs=True,
        )
        self.candidate_ids = ids
        self.context_dimension = context_array.shape[1]
        self.candidate_factors = np.asarray(candidate_factors, dtype=float)
        self.pca = pca
        self.embedding_scaler = embedding_scaler
        self.regressor = regressor
        self.params = settings
        self.seed = training_seed
        self.training_objective = float(objective)
        return self

    def score(self, context: np.ndarray | Mapping[str, float]) -> np.ndarray:
        """Return higher-is-better scores in ``candidate_ids`` order."""

        candidate_factors = self.candidate_factors
        pca = self.pca
        embedding_scaler = self.embedding_scaler
        regressor = self.regressor
        if (
            candidate_factors is None
            or pca is None
            or embedding_scaler is None
            or regressor is None
        ):
            raise RuntimeError("MetaODSequenceSelector must be fitted before scoring")
        context_array = _prepare_context(context, self.context_dimension)
        embedded = embedding_scaler.transform(pca.transform(context_array))
        latent = _latent_prediction(regressor, embedded)[0]
        scores = np.asarray(candidate_factors @ latent, dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("MetaOD produced non-finite candidate scores")
        return scores

    def rank(self, context: np.ndarray | Mapping[str, float]) -> tuple[str, ...]:
        scores = self.score(context)
        order = np.argsort(-scores, kind="stable")
        return tuple(self.candidate_ids[index] for index in order)


def _cofirank_gains(losses: np.ndarray) -> np.ndarray | None:
    """Convert lower-is-better losses to bounded rank relevance gains."""

    ranks = _average_ranks(np.asarray(losses, dtype=float))
    if float(np.max(ranks) - np.min(ranks)) <= 1e-12:
        return None
    relevance = (len(ranks) - ranks) / max(1.0, len(ranks) - 1.0)
    return np.exp2(relevance) - 1.0


def _cofirank_loss_and_gradient(
    scores: np.ndarray,
    loss_matrix: np.ndarray,
    ndcg_cutoff: int,
) -> tuple[float, np.ndarray, int]:
    """Return CoFiRank's structured NDCG upper bound and score subgradient.

    For each task, loss-augmented inference solves the linear assignment from
    Weimer et al. (2007).  The position feature is
    ``c_i = (i + 1) ** -0.25`` and the task loss is ``1 - NDCG@L``.  Missing
    performance entries are excluded from that task's assignment.
    """

    score_array = np.asarray(scores, dtype=float)
    losses = np.asarray(loss_matrix, dtype=float)
    if score_array.shape != losses.shape or score_array.ndim != 2:
        raise ValueError("scores and loss_matrix must have the same 2-D shape")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if np.isinf(losses).any():
        raise ValueError("loss_matrix may contain finite values or NaN, but not infinity")
    if ndcg_cutoff < 1:
        raise ValueError("ndcg_cutoff must be positive")

    objective = 0.0
    gradient = np.zeros_like(score_array)
    informative_rows = 0
    for task_index, row in enumerate(losses):
        observed = np.flatnonzero(np.isfinite(row))
        if len(observed) < 2:
            continue
        gains = _cofirank_gains(row[observed])
        if gains is None:
            continue

        row_scores = score_array[task_index, observed]
        item_count = len(observed)
        cutoff = min(ndcg_cutoff, item_count)
        positions = np.arange(item_count, dtype=float)
        position_weights = np.power(positions + 1.0, -0.25)
        discounts = np.zeros(item_count, dtype=float)
        discounts[:cutoff] = 1.0 / np.log2(positions[:cutoff] + 2.0)
        ideal_permutation = np.argsort(-gains, kind="stable")
        ideal_dcg = float(np.dot(discounts, gains[ideal_permutation]))
        if ideal_dcg <= 1e-12:
            continue

        # Dropping constants from
        #   Delta(pi,y) + <c, f_pi - f_pi_y>
        # leaves the following assignment utility.  Hungarian minimization on
        # its negative returns the loss-augmented permutation.
        utility = (
            position_weights[:, None] * row_scores[None, :]
            - discounts[:, None] * gains[None, :] / ideal_dcg
        )
        assignment_rows, assignment_items = linear_sum_assignment(-utility)
        permutation = np.empty(item_count, dtype=np.int64)
        permutation[assignment_rows] = assignment_items

        ndcg = float(np.dot(discounts, gains[permutation]) / ideal_dcg)
        structured_loss = (
            1.0
            - ndcg
            + float(np.dot(position_weights, row_scores[permutation]))
            - float(np.dot(position_weights, row_scores[ideal_permutation]))
        )
        informative_rows += 1
        if structured_loss <= 1e-12:
            continue
        objective += structured_loss
        row_gradient = np.zeros(item_count, dtype=float)
        np.add.at(row_gradient, permutation, position_weights)
        np.add.at(row_gradient, ideal_permutation, -position_weights)
        gradient[task_index, observed] = row_gradient

    return objective, gradient, informative_rows


def _comparable_pair_count(loss_matrix: np.ndarray) -> int:
    """Retain the existing diagnostic without using pairwise training."""

    count = 0
    for row in np.asarray(loss_matrix, dtype=float):
        observed = row[np.isfinite(row)]
        for left in range(len(observed)):
            for right in range(left + 1, len(observed)):
                if not np.isclose(
                    observed[left],
                    observed[right],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    count += 1
    return count


@dataclass
class ALORSSequenceSelector:
    """ALORS CoFiRank-NDCG selector with random-forest cold start."""

    params: Mapping[str, Any] | None = None
    seed: int = 0
    candidate_ids: tuple[str, ...] = field(default=(), init=False)
    context_dimension: int = field(default=0, init=False)
    candidate_factors: np.ndarray | None = field(default=None, init=False)
    context_scaler: StandardScaler | None = field(default=None, init=False)
    regressor: RandomForestRegressor | MultiOutputRegressor | None = field(
        default=None,
        init=False,
    )
    observed_pair_count: int = field(default=0, init=False)
    training_objective: float = field(default=0.0, init=False)
    method: str = field(default="alors", init=False)

    def fit(
        self,
        contexts: np.ndarray,
        loss_matrix: np.ndarray,
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> ALORSSequenceSelector:
        """Fit CoFiRank task/candidate factors and the cold-start RF."""

        settings = dict((self.params or {}) if params is None else params)
        training_seed = self.seed if seed is None else int(seed)
        context_array, losses, ids = _validate_training_data(contexts, loss_matrix, candidate_ids)
        context_scaler = StandardScaler()
        scaled_contexts = context_scaler.fit_transform(context_array)
        latent_dimension = min(
            _positive_int(settings, "latent_dim", 10),
            len(ids),
        )
        rng = np.random.default_rng(training_seed)
        group_factors = rng.normal(
            0.0,
            1.0 / np.sqrt(latent_dimension),
            size=(len(context_array), latent_dimension),
        )
        candidate_factors = rng.normal(
            0.0,
            1.0 / np.sqrt(latent_dimension),
            size=(len(ids), latent_dimension),
        )
        ndcg_cutoff = _positive_int(settings, "ndcg_cutoff", 10)
        objective, _, structured_task_count = _cofirank_loss_and_gradient(
            group_factors @ candidate_factors.T,
            losses,
            ndcg_cutoff,
        )
        if not structured_task_count:
            group_factors.fill(0.0)
            candidate_factors.fill(0.0)
        else:
            epochs = _positive_int(settings, "epochs", 10)
            learning_rate = _positive_float(settings, "learning_rate", 0.03)
            regularization = _nonnegative_float(settings, "regularization", 10.0)
            maximum_gradient_norm = _positive_float(settings, "max_gradient_norm", 10.0)
            first_u = np.zeros_like(group_factors)
            second_u = np.zeros_like(group_factors)
            first_v = np.zeros_like(candidate_factors)
            second_v = np.zeros_like(candidate_factors)
            for step in range(1, epochs + 1):
                objective, score_gradient, _ = _cofirank_loss_and_gradient(
                    group_factors @ candidate_factors.T,
                    losses,
                    ndcg_cutoff,
                )
                group_gradient = score_gradient @ candidate_factors + regularization * group_factors
                group_gradient = _clip_gradient(group_gradient, maximum_gradient_norm)
                group_factors, first_u, second_u = _adam_ascent(
                    group_factors,
                    -group_gradient,
                    first_u,
                    second_u,
                    step,
                    learning_rate,
                )

                objective, score_gradient, _ = _cofirank_loss_and_gradient(
                    group_factors @ candidate_factors.T,
                    losses,
                    ndcg_cutoff,
                )
                candidate_gradient = (
                    score_gradient.T @ group_factors + regularization * candidate_factors
                )
                candidate_gradient = _clip_gradient(candidate_gradient, maximum_gradient_norm)
                candidate_factors, first_v, second_v = _adam_ascent(
                    candidate_factors,
                    -candidate_gradient,
                    first_v,
                    second_v,
                    step,
                    learning_rate,
                )
            objective, _, _ = _cofirank_loss_and_gradient(
                group_factors @ candidate_factors.T,
                losses,
                ndcg_cutoff,
            )
            objective += (
                0.5
                * regularization
                * float(np.sum(np.square(group_factors)) + np.sum(np.square(candidate_factors)))
            )

        regressor = _fit_random_forest(
            scaled_contexts,
            group_factors,
            settings,
            training_seed,
            default_n_estimators=10,
            default_max_depth=None,
            independent_outputs=False,
        )
        self.candidate_ids = ids
        self.context_dimension = context_array.shape[1]
        self.candidate_factors = np.asarray(candidate_factors, dtype=float)
        self.context_scaler = context_scaler
        self.regressor = regressor
        self.params = settings
        self.seed = training_seed
        self.observed_pair_count = _comparable_pair_count(losses)
        self.training_objective = float(objective)
        return self

    def score(self, context: np.ndarray | Mapping[str, float]) -> np.ndarray:
        """Return higher-is-better scores in ``candidate_ids`` order."""

        if self.candidate_factors is None or self.context_scaler is None or self.regressor is None:
            raise RuntimeError("ALORSSequenceSelector must be fitted before scoring")
        context_array = _prepare_context(context, self.context_dimension)
        scaled = self.context_scaler.transform(context_array)
        latent = _latent_prediction(self.regressor, scaled)[0]
        scores = np.asarray(self.candidate_factors @ latent, dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("ALORS produced non-finite candidate scores")
        return scores

    def rank(self, context: np.ndarray | Mapping[str, float]) -> tuple[str, ...]:
        scores = self.score(context)
        order = np.argsort(-scores, kind="stable")
        return tuple(self.candidate_ids[index] for index in order)


__all__ = ["ALORSSequenceSelector", "MetaODSequenceSelector"]
