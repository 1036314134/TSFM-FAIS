"""Pairwise cost-sensitive classification and matched regression controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .utility import _family_weights, family_macro


def hard_pairwise_choice(preferences, pairs, candidates, preferred):
    """Positive preferences vote for the left candidate; exact ties split a vote."""
    preferences = np.asarray(preferences, dtype=float)
    if (
        preferences.ndim != 2
        or preferences.shape[1] != len(pairs)
        or not np.isfinite(preferences).all()
    ):
        raise ValueError("finite pairwise preferences with aligned columns are required")
    votes = np.zeros((len(preferences), len(candidates)))
    for column, (left, right) in enumerate(pairs):
        vote = (preferences[:, column] > 0).astype(float) + 0.5 * (preferences[:, column] == 0)
        votes[:, left] += vote
        votes[:, right] += 1 - vote
    order = sorted(
        range(len(candidates)),
        key=lambda index: (candidates[index] != preferred, candidates[index]),
    )
    winners = np.asarray(order)[votes[:, order].argmax(axis=1)]
    return winners, votes


@dataclass
class PairwiseUtilitySelector:
    mode: str = "classification"
    seed: int = 5101
    n_estimators: int = 160
    n_jobs: int = 1
    feature_prefixes: tuple[str, ...] = ("static.", "response.")
    candidate_ids: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    baseline_id: str | None = None
    models: dict[tuple[int, int], Any] = field(default_factory=dict)
    fit_records: list[dict] = field(default_factory=list)

    def _pack(self, frame):
        if frame.empty or frame.duplicated(["episode_id", "candidate_id"]).any():
            raise ValueError("each episode-candidate row must be unique")
        if set(frame.candidate_id) != set(self.candidate_ids):
            raise ValueError("the input must contain the fitted candidate pool")
        episodes = frame.episode_id.drop_duplicates().tolist()
        features, ordered = [], []
        for candidate in self.candidate_ids:
            rows = frame[frame.candidate_id == candidate].set_index("episode_id")
            if len(rows) != len(episodes):
                raise ValueError("every episode must contain every candidate")
            rows = rows.loc[episodes]
            values = rows.loc[:, self.feature_names].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("candidate features must be finite")
            features.append(values)
            ordered.append(rows)
        return episodes, features, ordered

    def _pair_matrix(self, features, left, right):
        return pd.DataFrame(
            np.concatenate([features[left], features[right]], axis=1),
            columns=[f"{side}.{name}" for side in ("left", "right") for name in self.feature_names],
        )

    def fit(self, frame):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if self.mode not in {"classification", "regression"}:
            raise ValueError("unsupported pairwise learning mode")
        self.candidate_ids = tuple(sorted(frame.candidate_id.unique()))
        self.feature_names = tuple(
            sorted(name for name in frame if name.startswith(self.feature_prefixes))
        )
        if len(self.candidate_ids) < 2 or not self.feature_names:
            raise ValueError("pairwise selection requires candidates and observable features")
        episodes, features, ordered = self._pack(frame)
        cost = np.stack([rows.loss.to_numpy(dtype=float) for rows in ordered], axis=1)
        if not np.isfinite(cost).all():
            raise ValueError("training costs must be finite")
        self.baseline_id = min(
            self.candidate_ids,
            key=lambda action: (family_macro(frame[frame.candidate_id == action]), action),
        )
        base_weights = _family_weights(ordered[0].reset_index())
        self.models, self.fit_records = {}, []
        settings = dict(
            n_estimators=self.n_estimators,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=25,
            reg_lambda=5.0,
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=self.n_jobs,
            verbosity=-1,
        )
        for left, right in combinations(range(len(self.candidate_ids)), 2):
            difference = cost[:, left] - cost[:, right]
            matrix = self._pair_matrix(features, left, right)
            if self.mode == "classification":
                weights = base_weights * np.abs(difference)
                positive = weights > 0
                labels = (difference < 0).astype(int)
                if not positive.any():
                    model = 0.5
                elif np.unique(labels[positive]).size == 1:
                    model = float(labels[positive][0])
                else:
                    # A pair-wide rescaling preserves relative mistake costs.
                    weights /= weights[positive].mean()
                    model = LGBMClassifier(objective="binary", **settings).fit(
                        matrix[positive], labels[positive], sample_weight=weights[positive]
                    )
            else:
                positive = np.ones(len(episodes), dtype=bool)
                if np.ptp(difference) == 0:
                    model = float(difference[0])
                else:
                    model = LGBMRegressor(objective="regression", **settings).fit(
                        matrix, difference, sample_weight=base_weights
                    )
            self.models[(left, right)] = model
            self.fit_records.append(
                {
                    "left": self.candidate_ids[left],
                    "right": self.candidate_ids[right],
                    "examples": len(episodes),
                    "positive_weight_examples": int(positive.sum()),
                    "constant": isinstance(model, float),
                }
            )
        return self

    def pair_scores(self, frame):
        if not self.models or self.baseline_id is None:
            raise ValueError("the pairwise selector has not been fitted")
        episodes, features, _ = self._pack(frame)
        scores, pairs = [], []
        for (left, right), model in self.models.items():
            matrix = self._pair_matrix(features, left, right)
            if isinstance(model, float):
                value = np.full(len(episodes), model)
            else:
                value = (
                    model.predict_proba(matrix)[:, 1]
                    if self.mode == "classification"
                    else model.predict(matrix)
                )
            scores.append(value - 0.5 if self.mode == "classification" else -value)
            pairs.append((left, right))
        return episodes, np.stack(scores, axis=1), pairs

    def select(self, frame):
        episodes, scores, pairs = self.pair_scores(frame)
        winners, votes = hard_pairwise_choice(scores, pairs, self.candidate_ids, self.baseline_id)
        choices = pd.DataFrame(
            {
                "episode_id": episodes,
                "candidate_id": np.asarray(self.candidate_ids)[winners],
                "pairwise_wins": votes[np.arange(len(winners)), winners],
            }
        )
        return choices.merge(frame, on=["episode_id", "candidate_id"], validate="one_to_one")
