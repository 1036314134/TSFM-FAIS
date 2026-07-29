"""Neural selectors that choose imputers once for an entire sequence.

The classes in this module deliberately have no dependency on the block-wise
routing pipeline.  DSelect-1 treats each imputer as one expert and produces one
gate vector per sequence.  NeuralUCB treats each imputer as one action and
updates from the bounded reward of the selected action only.

Both fitted objects contain only NumPy arrays and Python containers, making
them portable through :mod:`joblib`.  PyTorch is imported lazily while fitting
the DSelect gate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "DSelect-1 training requires `pip install -e .[selector-baselines]` "
            "or `pip install torch`"
        ) from error
    return torch


def _configure_torch(torch: Any, seed: int, threads: int) -> None:
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


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _candidate_tuple(candidate_ids: Sequence[str], expected: int) -> tuple[str, ...]:
    candidates = tuple(str(candidate_id) for candidate_id in candidate_ids)
    if len(candidates) != expected:
        raise ValueError("candidate_ids must match the candidate dimension")
    if any(not candidate_id for candidate_id in candidates):
        raise ValueError("candidate_ids cannot contain empty values")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_ids must be unique")
    return candidates


def _validated_params(
    params: Mapping[str, Any] | None,
    allowed: frozenset[str],
    *,
    method: str,
) -> dict[str, Any]:
    resolved = dict(params or {})
    unknown = set(resolved).difference(allowed)
    if unknown:
        raise ValueError(f"unsupported parameters for {method}: " + ", ".join(sorted(unknown)))
    return resolved


@dataclass(frozen=True)
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> _Standardizer:
        matrix = _matrix(values, name="contexts")
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean=np.asarray(mean, dtype=float), scale=np.asarray(scale, dtype=float))

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("contexts do not match the fitted feature dimension")
        if not np.isfinite(matrix).all():
            raise ValueError("contexts must contain only finite values")
        return (matrix - self.mean) / self.scale


def smooth_step(values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Evaluate the cubic smooth-step function used by DSelect-k."""

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
    if candidate_count < 2:
        raise ValueError("DSelect-1 requires at least two candidates")
    bit_count = int(math.ceil(math.log2(candidate_count)))
    return np.asarray(
        [
            [float(digit) for digit in np.binary_repr(index, width=bit_count)]
            for index in range(candidate_count)
        ],
        dtype=float,
    )


@dataclass
class DSelectOneSequenceSelector:
    """Example-conditioned DSelect-1 gate over whole-sequence experts.

    ``raw_score`` mirrors the paper's binary-code gate.  For a non-power-of-two
    portfolio its real-expert mass is intentionally not normalized away; the
    reciprocal regularizer drives that mass toward one as in the paper.
    """

    candidate_ids: tuple[str, ...]
    standardizer: _Standardizer
    weight: np.ndarray
    bias: np.ndarray
    gamma: float
    entropy_weight: float = 0.0
    reachable_mass_weight: float = 1.0

    _PARAMS = frozenset(
        {
            "batch_size",
            "entropy_weight",
            "epochs",
            "gamma",
            "init_scale",
            "learning_rate",
            "reachable_mass_weight",
            "padding_penalty",
            "torch_threads",
            "weight_decay",
        }
    )

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        expert_outputs: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        candidate_ids: Sequence[str],
        native_valid: np.ndarray | None = None,
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
    ) -> DSelectOneSequenceSelector:
        """Fit the gate on loss of the mixed expert completion.

        Each entry in ``expert_outputs`` has shape ``[candidates, hidden_points]``.
        Hidden-point counts may differ between sequence episodes.
        """

        context_matrix = _matrix(features, name="features")
        outputs = tuple(np.asarray(value, dtype=float) for value in expert_outputs)
        target_values = tuple(np.asarray(value, dtype=float).reshape(-1) for value in targets)
        if len(outputs) != len(context_matrix) or len(target_values) != len(context_matrix):
            raise ValueError("features, expert_outputs, and targets must have the same length")
        candidate_values = tuple(candidate_ids)
        candidates = _candidate_tuple(candidate_values, len(candidate_values))
        for index, (episode_outputs, episode_targets) in enumerate(
            zip(outputs, target_values, strict=True)
        ):
            expected = (len(candidates), len(episode_targets))
            if episode_outputs.shape != expected or not len(episode_targets):
                raise ValueError(f"expert_outputs[{index}] must have non-empty shape {expected}")
            if not np.isfinite(episode_outputs).all() or not np.isfinite(episode_targets).all():
                raise ValueError("DSelect training completions and targets must be finite")
        validity = (
            np.ones((len(context_matrix), len(candidates)), dtype=bool)
            if native_valid is None
            else np.asarray(native_valid, dtype=bool)
        )
        if validity.shape != (len(context_matrix), len(candidates)):
            raise ValueError("native_valid must have shape [episodes,candidates]")
        if np.any(np.count_nonzero(validity, axis=1) < 1):
            raise ValueError("every DSelect episode requires a valid expert")
        codes = _binary_codes(len(candidates))
        resolved = _validated_params(params, cls._PARAMS, method="dselect1")

        gamma = float(resolved.get("gamma", 1.0))
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive")
        entropy_weight = float(resolved.get("entropy_weight", 1e-3))
        if not np.isfinite(entropy_weight) or entropy_weight < 0.0:
            raise ValueError("entropy_weight must be finite and non-negative")
        reachable_mass_weight = float(
            resolved.get(
                "reachable_mass_weight",
                resolved.get("padding_penalty", 1.0),
            )
        )
        if not np.isfinite(reachable_mass_weight) or reachable_mass_weight < 0.0:
            raise ValueError("reachable_mass_weight must be finite and non-negative")

        torch = _require_torch()
        _configure_torch(torch, int(seed), int(resolved.get("torch_threads", 1)))
        standardizer = _Standardizer.fit(context_matrix)
        scaled = standardizer.transform(context_matrix)
        rng = np.random.default_rng(int(seed))
        bit_count = codes.shape[1]
        init_scale = float(resolved.get("init_scale", gamma / 100.0))
        if not np.isfinite(init_scale) or init_scale < 0.0:
            raise ValueError("init_scale must be finite and non-negative")
        weight = torch.nn.Parameter(
            torch.as_tensor(
                rng.uniform(
                    -init_scale,
                    init_scale,
                    size=(bit_count, scaled.shape[1]),
                ),
                dtype=torch.float32,
            )
        )
        bias = torch.nn.Parameter(
            torch.as_tensor(
                rng.uniform(-init_scale, init_scale, size=bit_count),
                dtype=torch.float32,
            )
        )
        learning_rate = float(resolved.get("learning_rate", 0.01))
        weight_decay = float(resolved.get("weight_decay", 1e-5))
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        optimizer = torch.optim.Adam(
            (weight, bias),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        context_tensor = torch.as_tensor(scaled, dtype=torch.float32)
        output_tensors = tuple(torch.as_tensor(value, dtype=torch.float32) for value in outputs)
        target_tensors = tuple(
            torch.as_tensor(value, dtype=torch.float32) for value in target_values
        )
        validity_tensor = torch.as_tensor(validity, dtype=torch.bool)
        code_tensor = torch.as_tensor(codes, dtype=torch.bool)
        batch_size = max(1, int(resolved.get("batch_size", 256)))
        epochs = max(1, int(resolved.get("epochs", 100)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        power_of_two = len(candidates) == 1 << bit_count

        def torch_smooth_step(values: Any) -> Any:
            middle = -2.0 * values.pow(3) / gamma**3 + 3.0 * values / (2.0 * gamma) + 0.5
            return torch.where(
                values <= -gamma / 2.0,
                torch.zeros_like(values),
                torch.where(values >= gamma / 2.0, torch.ones_like(values), middle),
            )

        for _ in range(epochs):
            permutation = torch.randperm(len(context_tensor), generator=generator)
            for start in range(0, len(context_tensor), batch_size):
                indices = permutation[start : start + batch_size]
                activations = torch_smooth_step(context_tensor[indices] @ weight.T + bias)
                factors = torch.where(
                    code_tensor[None, :, :],
                    activations[:, None, :],
                    1.0 - activations[:, None, :],
                )
                raw_weights = torch.prod(factors, dim=2)
                reachable_mass = torch.sum(raw_weights, dim=1, keepdim=True)
                episode_losses = []
                for local_index, raw_index in enumerate(indices.tolist()):
                    valid = validity_tensor[raw_index]
                    masked = torch.where(valid, raw_weights[local_index], 0.0)
                    masked_mass = torch.sum(masked)
                    # Dynamic native validity is a repository constraint.  Preserve
                    # the paper gate's reachable mass while redistributing only the
                    # mass assigned to unavailable experts.
                    if not bool(torch.all(valid)):
                        masked = masked * (
                            reachable_mass[local_index, 0] / torch.clamp(masked_mass, min=1e-8)
                        )
                    mixed = torch.sum(masked[:, None] * output_tensors[raw_index], dim=0)
                    target = target_tensors[raw_index]
                    denominator = torch.abs(target) + torch.abs(mixed)
                    difference = torch.abs(mixed - target)
                    episode_losses.append(
                        torch.mean(
                            torch.where(
                                denominator > 0.0,
                                difference / torch.clamp(denominator, min=1e-8),
                                torch.zeros_like(difference),
                            )
                        )
                    )
                mixed_output_loss = torch.stack(episode_losses).mean()
                entropy = -torch.sum(
                    raw_weights * torch.log(torch.clamp(raw_weights, min=1e-8)),
                    dim=1,
                ).mean()
                objective = mixed_output_loss + entropy_weight * entropy
                if not power_of_two and reachable_mass_weight > 0.0:
                    objective = objective + reachable_mass_weight * torch.mean(
                        1.0 / torch.clamp(reachable_mass, min=1e-8)
                    )
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()

        return cls(
            candidate_ids=candidates,
            standardizer=standardizer,
            weight=weight.detach().cpu().numpy().astype(float),
            bias=bias.detach().cpu().numpy().astype(float),
            gamma=gamma,
            entropy_weight=entropy_weight,
            reachable_mass_weight=reachable_mass_weight,
        )

    @classmethod
    def fit_from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        feature_names: Sequence[str],
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
    ) -> DSelectOneSequenceSelector:
        """Reconstruct paper training examples from compact sequence-label rows."""

        candidates = tuple(str(value) for value in candidate_ids)
        grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if row.get("label_scope") != "whole_series":
                continue
            group_id = str(row.get("group_id", ""))
            candidate_id = str(row.get("candidate_id", ""))
            if not group_id or candidate_id not in candidates:
                raise ValueError("invalid DSelect sequence-label identity")
            group = grouped.setdefault(group_id, {})
            if candidate_id in group:
                raise ValueError(f"duplicate DSelect candidate {candidate_id!r} in {group_id!r}")
            group[candidate_id] = row
        if not grouped:
            raise ValueError("DSelect training has no whole-series groups")

        contexts: list[list[float]] = []
        outputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        validity: list[list[bool]] = []
        names = tuple(str(value) for value in feature_names)
        for group_id, group in grouped.items():
            feature_rows: list[Mapping[str, Any]] = []
            for row in group.values():
                raw_features = row.get("prior_features")
                if isinstance(raw_features, Mapping):
                    feature_rows.append(raw_features)
            if not feature_rows:
                raise ValueError(f"DSelect group {group_id!r} lacks sequence features")
            features = max(
                feature_rows,
                key=lambda values: sum(name in values for name in names),
            )
            if names and not any(name in features for name in names):
                raise ValueError(f"DSelect group {group_id!r} lacks sequence features")
            contexts.append([float(features.get(name, 0.0)) for name in names])
            raw_target_rows = [
                row.get("dselect_target_values")
                for row in group.values()
                if isinstance(row.get("dselect_target_values"), list)
            ]
            if not raw_target_rows:
                raise ValueError(f"DSelect group {group_id!r} lacks target values")
            target = np.asarray(raw_target_rows[0], dtype=float).reshape(-1)
            if not len(target) or not np.isfinite(target).all():
                raise ValueError(f"DSelect group {group_id!r} has invalid target values")
            for raw_targets in raw_target_rows[1:]:
                duplicate = np.asarray(raw_targets, dtype=float).reshape(-1)
                if duplicate.shape != target.shape or not np.array_equal(duplicate, target):
                    raise ValueError(f"DSelect group {group_id!r} has inconsistent targets")
            episode_outputs: list[np.ndarray] = []
            episode_validity: list[bool] = []
            for candidate_id in candidates:
                candidate_row = group.get(candidate_id)
                native_valid = bool(
                    candidate_row is not None and candidate_row.get("native_valid", False)
                )
                raw_values = (
                    None if candidate_row is None else candidate_row.get("dselect_expert_values")
                )
                values = (
                    np.asarray(raw_values, dtype=float).reshape(-1)
                    if isinstance(raw_values, list)
                    else np.zeros_like(target)
                )
                finite_and_aligned = values.shape == target.shape and np.isfinite(values).all()
                if native_valid and not finite_and_aligned:
                    raise ValueError(
                        f"DSelect group {group_id!r}/{candidate_id!r} values are invalid"
                    )
                if not finite_and_aligned:
                    values = np.zeros_like(target)
                episode_outputs.append(values)
                episode_validity.append(native_valid)
            outputs.append(np.stack(episode_outputs, axis=0))
            targets.append(target)
            validity.append(episode_validity)
        return cls.fit(
            np.asarray(contexts, dtype=float),
            outputs,
            targets,
            candidates,
            native_valid=np.asarray(validity, dtype=bool),
            params=params,
            seed=seed,
        )

    @property
    def input_dim(self) -> int:
        return len(self.standardizer.mean)

    def raw_score_many(self, contexts: np.ndarray) -> np.ndarray:
        matrix = self.standardizer.transform(contexts)
        if self.weight.shape != (len(self.bias), self.input_dim):
            raise ValueError("stored DSelect weight shape is invalid")
        codes = _binary_codes(len(self.candidate_ids))
        if codes.shape[1] != len(self.bias):
            raise ValueError("stored DSelect bit dimension is invalid")
        activations = smooth_step(matrix @ self.weight.T + self.bias, self.gamma)
        factors = np.where(
            codes[None, :, :] > 0.5,
            activations[:, None, :],
            1.0 - activations[:, None, :],
        )
        return np.prod(factors, axis=2)

    def raw_score(self, features_1d: np.ndarray) -> np.ndarray:
        vector = np.asarray(features_1d, dtype=float)
        if vector.ndim != 1:
            raise ValueError("features_1d must be one-dimensional")
        return self.raw_score_many(vector[None, :])[0]

    def score_many(self, contexts: np.ndarray) -> np.ndarray:
        return self.raw_score_many(contexts)

    def score(self, features_1d: np.ndarray) -> np.ndarray:
        vector = np.asarray(features_1d, dtype=float)
        if vector.ndim != 1:
            raise ValueError("features_1d must be one-dimensional")
        return self.score_many(vector[None, :])[0]

    def weights(self, features_1d: np.ndarray) -> np.ndarray:
        """Return the normalized DSelect expert weights for one sequence."""

        return self.score(features_1d)

    def select(self, features_1d: np.ndarray) -> str:
        return self.candidate_ids[int(np.argmax(self.score(features_1d)))]


@dataclass
class NeuralUCBSequenceSelector:
    """Sequence-level NeuralUCB policy with the paper's full confidence matrix.

    Candidate contexts use the disjoint representation from the paper's
    multiclass experiments.  Each context is duplicated and normalized as
    required by the symmetric initialization used in the analysis.  The full
    matrix ``Z = lambda I + sum(g g.T / m)`` is stored exactly through its
    historical gradient factors and a dual inverse, avoiding an infeasible
    dense ``p x p`` allocation for the main experiment.
    """

    input_dim: int
    candidate_ids: tuple[str, ...]
    hidden_size: int
    ridge: float
    nu: float
    learning_rate: float
    training_steps: int
    retrain_interval: int
    regularization: float
    gradient_clip: float
    seed: int
    weight1: np.ndarray
    weight2: np.ndarray
    initial_weight1: np.ndarray
    initial_weight2: np.ndarray
    precision_locals: np.ndarray
    precision_outputs: np.ndarray
    dual_precision_inverse: np.ndarray
    observed_contexts: list[np.ndarray] = field(default_factory=list)
    observed_rewards: list[float] = field(default_factory=list)
    selected_actions: list[int] = field(default_factory=list)
    round_count: int = 0
    last_trained_round: int = 0

    _PARAMS = frozenset(
        {
            "gradient_clip",
            "hidden_size",
            "learning_rate",
            "nu",
            "regularization",
            "retrain_interval",
            "ridge",
            "training_steps",
        }
    )

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
    ) -> NeuralUCBSequenceSelector:
        """Create an unobserved online policy."""

        dimension = int(input_dim)
        if dimension < 1:
            raise ValueError("input_dim must be positive")
        candidate_values = tuple(str(value) for value in candidate_ids)
        if not candidate_values:
            raise ValueError("candidate_ids cannot be empty")
        candidates = _candidate_tuple(candidate_values, len(candidate_values))
        resolved = _validated_params(params, cls._PARAMS, method="neuralucb")
        hidden_size = max(2, int(resolved.get("hidden_size", 100)))
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for symmetric NeuralUCB initialization")
        ridge = float(resolved.get("ridge", 1.0))
        nu = float(resolved.get("nu", 1.0))
        learning_rate = float(resolved.get("learning_rate", 0.01))
        training_steps = max(1, int(resolved.get("training_steps", 100)))
        retrain_interval = max(1, int(resolved.get("retrain_interval", 1)))
        regularization = float(resolved.get("regularization", ridge))
        gradient_clip = float(resolved.get("gradient_clip", 10.0))
        for name, value, positive in (
            ("ridge", ridge, True),
            ("nu", nu, False),
            ("learning_rate", learning_rate, True),
            ("regularization", regularization, False),
            ("gradient_clip", gradient_clip, True),
        ):
            if not np.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")

        rng = np.random.default_rng(int(seed))
        disjoint_dimension = dimension * len(candidates)
        action_dimension = 2 * disjoint_dimension
        half_hidden = hidden_size // 2
        shared_weight = rng.normal(
            loc=0.0,
            scale=math.sqrt(4.0 / hidden_size),
            size=(half_hidden, disjoint_dimension),
        )
        weight1 = np.zeros((hidden_size, action_dimension), dtype=float)
        weight1[:half_hidden, :disjoint_dimension] = shared_weight
        weight1[half_hidden:, disjoint_dimension:] = shared_weight
        shared_output = rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 / hidden_size),
            size=half_hidden,
        )
        weight2 = np.concatenate((shared_output, -shared_output))
        return cls(
            input_dim=dimension,
            candidate_ids=candidates,
            hidden_size=hidden_size,
            ridge=ridge,
            nu=nu,
            learning_rate=learning_rate,
            training_steps=training_steps,
            retrain_interval=retrain_interval,
            regularization=regularization,
            gradient_clip=gradient_clip,
            seed=int(seed),
            weight1=np.asarray(weight1, dtype=float),
            weight2=np.asarray(weight2, dtype=float),
            initial_weight1=np.asarray(weight1, dtype=float).copy(),
            initial_weight2=np.asarray(weight2, dtype=float).copy(),
            precision_locals=np.empty((0, hidden_size), dtype=float),
            precision_outputs=np.empty((0, hidden_size), dtype=float),
            dual_precision_inverse=np.empty((0, 0), dtype=float),
        )

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        loss_matrix: np.ndarray,
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
        *,
        native_valid: np.ndarray | None = None,
    ) -> NeuralUCBSequenceSelector:
        """Replay sequence rounds and observe only the chosen imputer reward.

        Reconstruction losses are converted to bounded rewards with
        ``1 / (1 + loss)``.  The loss values for candidates that were not
        selected in a round are not used for the NeuralUCB update.
        """

        feature_matrix = _matrix(features, name="features")
        losses = np.asarray(loss_matrix, dtype=float)
        if losses.ndim != 2 or not losses.shape[0] or not losses.shape[1]:
            raise ValueError("loss_matrix must be a non-empty two-dimensional array")
        if losses.shape[0] != feature_matrix.shape[0]:
            raise ValueError("features and loss_matrix must have the same row count")
        candidates = _candidate_tuple(candidate_ids, losses.shape[1])
        validity = (
            np.ones(losses.shape, dtype=bool)
            if native_valid is None
            else np.asarray(native_valid, dtype=bool)
        )
        if validity.shape != losses.shape:
            raise ValueError("native_valid must have shape [episodes,candidates]")
        if np.any(np.count_nonzero(validity, axis=1) < 1):
            raise ValueError("every NeuralUCB episode requires a native-valid action")
        valid_losses = losses[validity]
        if not np.isfinite(valid_losses).all() or np.any(valid_losses < 0.0):
            raise ValueError("native-valid NeuralUCB losses must be finite and non-negative")
        selector = cls.initialize(
            feature_matrix.shape[1],
            candidate_ids=candidates,
            params=params,
            seed=seed,
        )
        candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidates)}
        for round_features, round_losses, round_validity in zip(
            feature_matrix,
            losses,
            validity,
            strict=True,
        ):
            valid_candidates = tuple(
                candidate_id
                for candidate_id, is_valid in zip(candidates, round_validity, strict=True)
                if is_valid
            )
            selected = selector.select(round_features, valid_candidates)
            selected_loss = float(round_losses[candidate_index[selected]])
            selector.observe(
                round_features,
                selected,
                reward=1.0 / (1.0 + selected_loss),
            )
        selector.finalize()
        return selector

    @classmethod
    def fit_from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        feature_names: Sequence[str],
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
    ) -> NeuralUCBSequenceSelector:
        """Replay label episodes while retaining each episode's valid actions."""

        candidates = tuple(str(value) for value in candidate_ids)
        grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if row.get("label_scope") != "whole_series":
                continue
            group_id = str(row.get("group_id", ""))
            candidate_id = str(row.get("candidate_id", ""))
            if not group_id or candidate_id not in candidates:
                raise ValueError("invalid NeuralUCB sequence-label identity")
            group = grouped.setdefault(group_id, {})
            if candidate_id in group:
                raise ValueError(f"duplicate NeuralUCB candidate {candidate_id!r} in {group_id!r}")
            group[candidate_id] = row
        if not grouped:
            raise ValueError("NeuralUCB training has no whole-series groups")

        names = tuple(str(value) for value in feature_names)
        expected = set(candidates)
        contexts: list[list[float]] = []
        losses: list[list[float]] = []
        validity: list[list[bool]] = []
        for group_id, group in grouped.items():
            if set(group) != expected:
                raise ValueError(f"NeuralUCB group {group_id!r} has an incomplete action set")
            reference = group[candidates[0]]
            features = reference.get("prior_features")
            if not isinstance(features, Mapping):
                raise ValueError(f"NeuralUCB group {group_id!r} lacks sequence features")
            contexts.append([float(features.get(name, 0.0)) for name in names])
            group_losses: list[float] = []
            group_validity: list[bool] = []
            for candidate_id in candidates:
                candidate_row = group[candidate_id]
                group_validity.append(bool(candidate_row.get("native_valid", False)))
                try:
                    group_losses.append(float(candidate_row["imputation_loss"]))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"NeuralUCB group {group_id!r} has an invalid imputation loss"
                    ) from error
            losses.append(group_losses)
            validity.append(group_validity)
        return cls.fit(
            np.asarray(contexts, dtype=float),
            np.asarray(losses, dtype=float),
            candidates,
            params=params,
            seed=seed,
            native_valid=np.asarray(validity, dtype=bool),
        )

    @classmethod
    def offline_replay(
        cls,
        features: np.ndarray,
        loss_matrix: np.ndarray,
        candidate_ids: Sequence[str],
        params: Mapping[str, Any] | None = None,
        seed: int = 20260710,
        *,
        native_valid: np.ndarray | None = None,
    ) -> NeuralUCBSequenceSelector:
        return cls.fit(
            features,
            loss_matrix,
            candidate_ids,
            params,
            seed,
            native_valid=native_valid,
        )

    @property
    def disjoint_input_dim(self) -> int:
        return self.input_dim * len(self.candidate_ids)

    @property
    def action_input_dim(self) -> int:
        return 2 * self.disjoint_input_dim

    @property
    def parameter_count(self) -> int:
        return self.hidden_size * self.action_input_dim + self.hidden_size

    @property
    def observed_count(self) -> int:
        return len(self.observed_rewards)

    def _feature_vector(self, features_1d: np.ndarray) -> np.ndarray:
        vector = np.asarray(features_1d, dtype=float)
        if vector.ndim != 1 or len(vector) != self.input_dim:
            raise ValueError("features_1d must match the fitted feature dimension")
        if not np.isfinite(vector).all():
            raise ValueError("features_1d must contain only finite values")
        return vector

    def _normalized_feature(self, features_1d: np.ndarray) -> np.ndarray:
        vector = self._feature_vector(features_1d)
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1e-12 else vector / norm

    def _action_context_from_normalized(
        self,
        vector: np.ndarray,
        action_index: int,
    ) -> np.ndarray:
        context = np.zeros(self.action_input_dim, dtype=float)
        start = action_index * self.input_dim
        scaled = vector / math.sqrt(2.0)
        context[start : start + self.input_dim] = scaled
        mirrored_start = self.disjoint_input_dim + start
        context[mirrored_start : mirrored_start + self.input_dim] = scaled
        return context

    def _action_contexts(self, features_1d: np.ndarray) -> np.ndarray:
        """Embed one sequence into a disjoint context for every candidate."""

        vector = self._normalized_feature(features_1d)
        return np.stack(
            [
                self._action_context_from_normalized(vector, action_index)
                for action_index in range(len(self.candidate_ids))
            ],
            axis=0,
        )

    def _validate_precision_state(self) -> None:
        count = len(self.observed_contexts)
        if len(self.observed_rewards) != count or len(self.selected_actions) != count:
            raise ValueError("stored NeuralUCB history lengths are inconsistent")
        if self.precision_locals.shape != (count, self.hidden_size):
            raise ValueError("stored NeuralUCB local-gradient factors are invalid")
        if self.precision_outputs.shape != (count, self.hidden_size):
            raise ValueError("stored NeuralUCB output-gradient factors are invalid")
        if self.dual_precision_inverse.shape != (count, count):
            raise ValueError("stored NeuralUCB dual precision inverse is invalid")
        if count:
            contexts = np.asarray(self.observed_contexts, dtype=float)
            if contexts.shape != (count, self.action_input_dim):
                raise ValueError("stored NeuralUCB contexts are invalid")
            if not all(
                np.isfinite(values).all()
                for values in (
                    contexts,
                    self.precision_locals,
                    self.precision_outputs,
                    self.dual_precision_inverse,
                )
            ):
                raise ValueError("stored NeuralUCB precision state must be finite")

    def _validate_action_contexts(self, action_contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(action_contexts, dtype=float)
        if (
            contexts.ndim != 2
            or not contexts.shape[0]
            or contexts.shape[1] != self.action_input_dim
        ):
            raise ValueError("action_contexts must have shape [action, action_input_dim]")
        if not np.isfinite(contexts).all():
            raise ValueError("action_contexts must contain only finite values")
        if self.weight1.shape != (self.hidden_size, self.action_input_dim):
            raise ValueError("stored NeuralUCB first-layer weights are invalid")
        if self.weight2.shape != (self.hidden_size,):
            raise ValueError("stored NeuralUCB output weights are invalid")
        self._validate_precision_state()
        return contexts

    def _means_and_gradients(self, action_contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        contexts = self._validate_action_contexts(action_contexts)
        preactivation = contexts @ self.weight1.T
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        width_scale = math.sqrt(self.hidden_size)
        means = width_scale * (hidden @ self.weight2)
        local = width_scale * active * self.weight2[None, :]
        gradients = np.concatenate(
            (
                np.einsum("nh,nd->nhd", local, contexts, optimize=True).reshape(len(contexts), -1),
                width_scale * hidden,
            ),
            axis=1,
        )
        if gradients.shape[1] != self.parameter_count:
            raise RuntimeError("NeuralUCB gradient dimension is inconsistent")
        return np.asarray(means, dtype=float), np.asarray(gradients, dtype=float)

    def _means_and_gradient_parts(
        self,
        action_contexts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts = self._validate_action_contexts(action_contexts)
        preactivation = contexts @ self.weight1.T
        active = preactivation > 0.0
        hidden = np.maximum(preactivation, 0.0)
        width_scale = math.sqrt(self.hidden_size)
        means = width_scale * (hidden @ self.weight2)
        locals_ = width_scale * active * self.weight2[None, :]
        outputs = width_scale * hidden
        return means, locals_, outputs

    def _precision_update_matrix(self) -> np.ndarray:
        """Materialize historical gradients for small-matrix verification."""

        self._validate_precision_state()
        if not self.observed_contexts:
            return np.empty((0, self.parameter_count), dtype=float)
        contexts = np.asarray(self.observed_contexts, dtype=float)
        return np.concatenate(
            (
                np.einsum(
                    "nh,nd->nhd",
                    self.precision_locals,
                    contexts,
                    optimize=True,
                ).reshape(len(contexts), -1),
                self.precision_outputs,
            ),
            axis=1,
        )

    def _precision_matrix(self) -> np.ndarray:
        """Materialize ``Z`` for diagnostics; scoring uses the exact dual form."""

        gradients = self._precision_update_matrix()
        return self.ridge * np.eye(self.parameter_count) + gradients.T @ gradients / float(
            self.hidden_size
        )

    def _quadratic_forms(
        self,
        contexts: np.ndarray,
        locals_: np.ndarray,
        outputs: np.ndarray,
    ) -> np.ndarray:
        context_norms = np.sum(np.square(contexts), axis=1)
        gradient_norms = np.sum(np.square(locals_), axis=1) * context_norms + np.sum(
            np.square(outputs), axis=1
        )
        if not self.observed_contexts:
            return gradient_norms / self.ridge
        history_contexts = np.asarray(self.observed_contexts, dtype=float)
        gradient_cross = (self.precision_locals @ locals_.T) * (
            history_contexts @ contexts.T
        ) + self.precision_outputs @ outputs.T
        factor_cross = gradient_cross / math.sqrt(self.hidden_size)
        correction = np.sum(
            factor_cross * (self.dual_precision_inverse @ factor_cross),
            axis=0,
        )
        quadratic = gradient_norms / self.ridge - correction / self.ridge**2
        return np.maximum(quadratic, 0.0)

    def _score_action_contexts(self, action_contexts: np.ndarray) -> np.ndarray:
        contexts = self._validate_action_contexts(action_contexts)
        means, locals_, outputs = self._means_and_gradient_parts(contexts)
        quadratic = self._quadratic_forms(contexts, locals_, outputs)
        bonus = self.nu * np.sqrt(quadratic / float(self.hidden_size))
        scores = means + bonus
        if not np.isfinite(scores).all():
            raise ValueError("NeuralUCB produced non-finite scores")
        return np.asarray(scores, dtype=float)

    def score(self, features_1d: np.ndarray) -> np.ndarray:
        """Return one UCB value per configured candidate without changing state."""

        return self._score_action_contexts(self._action_contexts(features_1d))

    def select(
        self,
        features_1d: np.ndarray,
        valid_candidate_ids: Sequence[str] | None = None,
    ) -> str:
        """Select the highest-UCB candidate from the supplied valid subset."""

        requested = (
            self.candidate_ids
            if valid_candidate_ids is None
            else tuple(str(candidate_id) for candidate_id in valid_candidate_ids)
        )
        if not requested:
            raise ValueError("valid_candidate_ids cannot be empty")
        unknown = set(requested).difference(self.candidate_ids)
        if unknown:
            raise ValueError(
                "valid_candidate_ids contain unknown candidates: " + ", ".join(sorted(unknown))
            )
        requested_set = set(requested)
        valid_indices = [
            index
            for index, candidate_id in enumerate(self.candidate_ids)
            if candidate_id in requested_set
        ]
        scores = self.score(features_1d)
        selected_index = max(valid_indices, key=lambda index: scores[index])
        return self.candidate_ids[selected_index]

    def observe(
        self,
        features_1d: np.ndarray,
        candidate_id: str,
        reward: float,
    ) -> None:
        """Update NeuralUCB from one selected candidate's bounded reward."""

        observed = float(reward)
        if not np.isfinite(observed) or not 0.0 <= observed <= 1.0:
            raise ValueError("reward must be finite and lie in [0, 1]")
        selected_id = str(candidate_id)
        if selected_id not in self.candidate_ids:
            raise ValueError(f"unknown candidate_id: {selected_id}")
        action_index = self.candidate_ids.index(selected_id)
        selected_context = self._action_contexts(features_1d)[action_index]
        _, locals_, outputs = self._means_and_gradient_parts(selected_context[None, :])
        selected_local = locals_[0]
        selected_output = outputs[0]
        gradient_norm = float(
            np.dot(selected_local, selected_local) * np.dot(selected_context, selected_context)
            + np.dot(selected_output, selected_output)
        )
        count = len(self.observed_contexts)
        if count:
            contexts = np.asarray(self.observed_contexts, dtype=float)
            gradient_cross = (self.precision_locals @ selected_local) * (
                contexts @ selected_context
            ) + self.precision_outputs @ selected_output
            border = gradient_cross / (self.hidden_size * self.ridge)
            projected = self.dual_precision_inverse @ border
            schur = (
                1.0 + gradient_norm / (self.hidden_size * self.ridge) - float(border @ projected)
            )
            if not np.isfinite(schur) or schur <= 1e-12:
                raise ValueError("NeuralUCB precision update is numerically singular")
            updated_dual = np.empty((count + 1, count + 1), dtype=float)
            updated_dual[:count, :count] = (
                self.dual_precision_inverse + np.outer(projected, projected) / schur
            )
            updated_dual[:count, count] = -projected / schur
            updated_dual[count, :count] = -projected / schur
            updated_dual[count, count] = 1.0 / schur
        else:
            diagonal = 1.0 + gradient_norm / (self.hidden_size * self.ridge)
            updated_dual = np.asarray([[1.0 / diagonal]], dtype=float)
        self.precision_locals = np.concatenate(
            (self.precision_locals, selected_local[None, :]),
            axis=0,
        )
        self.precision_outputs = np.concatenate(
            (self.precision_outputs, selected_output[None, :]),
            axis=0,
        )
        self.dual_precision_inverse = updated_dual
        self.observed_contexts.append(selected_context.copy())
        self.observed_rewards.append(observed)
        self.selected_actions.append(action_index)
        self.round_count += 1
        if self.round_count % self.retrain_interval == 0:
            self._train_from_history()

    def finalize(self) -> None:
        """Train on feedback accumulated since the last scheduled update."""

        if self.round_count > self.last_trained_round:
            self._train_from_history()

    def _train_from_history(self) -> None:
        if not self.observed_contexts:
            return
        contexts = np.stack(self.observed_contexts, axis=0)
        rewards = np.asarray(self.observed_rewards, dtype=float)
        # Algorithm 2 trains against the fixed initial parameter theta_0.
        self.weight1 = self.initial_weight1.copy()
        self.weight2 = self.initial_weight2.copy()
        count = float(len(contexts))
        # Algorithm 2 minimizes a sum of squared errors plus
        # m * lambda * ||theta-theta_0||^2 / 2.  The data gradients below are
        # expressed as a mean, so the matching regularizer coefficient is
        # m * lambda / t rather than a history-size-independent lambda.
        regularization_scale = self.hidden_size * self.regularization / count

        for _ in range(self.training_steps):
            preactivation = contexts @ self.weight1.T
            active = preactivation > 0.0
            hidden = np.maximum(preactivation, 0.0)
            width_scale = math.sqrt(self.hidden_size)
            predictions = width_scale * (hidden @ self.weight2)
            error = predictions - rewards
            hidden_gradient = width_scale * error[:, None] * self.weight2[None, :] * active / count
            gradient_weight2 = width_scale * hidden.T @ error / count
            gradient_weight1 = hidden_gradient.T @ contexts

            gradient_weight1 += regularization_scale * (self.weight1 - self.initial_weight1)
            gradient_weight2 += regularization_scale * (self.weight2 - self.initial_weight2)
            flattened = np.concatenate(
                (
                    gradient_weight1.reshape(-1),
                    gradient_weight2,
                )
            )
            norm = float(np.linalg.norm(flattened))
            factor = min(1.0, self.gradient_clip / max(norm, 1e-12))
            self.weight1 -= self.learning_rate * factor * gradient_weight1
            self.weight2 -= self.learning_rate * factor * gradient_weight2
        self.last_trained_round = self.round_count


__all__ = [
    "DSelectOneSequenceSelector",
    "NeuralUCBSequenceSelector",
    "smooth_step",
]
