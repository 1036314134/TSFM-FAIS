"""Deployment-available sequence features and forecast-utility selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 0.0, 0.0
    return float(np.mean(finite)), float(np.std(finite)), float(np.max(np.abs(finite)))


def sequence_features(
    context: np.ndarray,
    completed: np.ndarray,
    anchor: np.ndarray,
    scales: np.ndarray,
    targets: Sequence[int],
    *,
    period: int,
    native_coverage: float,
) -> dict[str, float]:
    """Use only the observed context, candidate output, and frozen prefix scale."""

    values = np.asarray(context, dtype=float)
    candidate = np.asarray(completed, dtype=float)
    baseline = np.asarray(anchor, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if values.ndim != 2 or candidate.shape != values.shape or baseline.shape != values.shape:
        raise ValueError("sequence inputs must share a [L,D] shape")
    if scale.shape != (values.shape[1],) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("sequence features require positive finite per-variate scales")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("feature completions must be finite")
    selected = np.asarray(tuple(targets), dtype=int)
    missing = ~np.isfinite(values)
    target_missing = missing[:, selected]
    lengths: list[int] = []
    for channel in range(target_missing.shape[1]):
        changes = np.diff(np.r_[False, target_missing[:, channel], False].astype(int))
        lengths.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist())
    scaled = candidate[:, selected] / scale[selected]
    delta = (candidate - baseline)[:, selected] / scale[selected]
    observed = values[:, selected] / scale[selected]
    observed_mean, observed_std, _ = _summary(observed)
    result = {
        "static.missing_fraction": float(missing.mean()),
        "static.target_missing_fraction": float(target_missing.mean()),
        "static.tail_missing_fraction": float(target_missing[-1].mean()),
        "static.recent_missing_fraction": float(target_missing[-max(1, len(values) // 4) :].mean()),
        "static.gap_mean_ratio": float(np.mean(lengths) / len(values)) if lengths else 0.0,
        "static.gap_max_ratio": float(max(lengths) / len(values)) if lengths else 0.0,
        "static.dimension_log": float(np.log1p(values.shape[1])),
        "static.period_context_ratio": float(period / len(values)),
        "static.native_coverage": float(native_coverage),
        "static.observed_mean": observed_mean,
        "static.observed_std": observed_std,
        "static.mean_fill_change": float(np.mean(np.abs(delta))),
        "static.max_fill_change": float(np.max(np.abs(delta))),
        "static.recent_fill_change": float(np.mean(np.abs(delta[-max(1, len(delta) // 4) :]))),
        "static.variation": float(np.mean(np.abs(np.diff(scaled, axis=0)))),
        "static.curvature": float(np.mean(np.abs(np.diff(scaled, n=2, axis=0))))
        if len(values) > 2
        else 0.0,
        "static.slope": float(np.mean(scaled[-1] - scaled[0])),
    }
    return {key: float(np.clip(value, -1e8, 1e8)) for key, value in result.items()}


def response_features(
    point: np.ndarray,
    anchor_point: np.ndarray,
    pool_point: np.ndarray,
    last_observed: np.ndarray,
    target_scales: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> dict[str, float]:
    """Describe actual forecasts without consulting future observations."""

    prediction = np.asarray(point, dtype=float)
    reference = np.asarray(anchor_point, dtype=float)
    pool = np.asarray(pool_point, dtype=float)
    scales = np.asarray(target_scales, dtype=float)
    if (
        prediction.ndim != 2
        or reference.shape != prediction.shape
        or pool.shape != prediction.shape
    ):
        raise ValueError("response forecasts must share a [H,K] shape")
    if scales.shape != (prediction.shape[1],) or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("response scales must be positive and cover all targets")
    if not all(np.isfinite(array).all() for array in (prediction, reference, pool, last_observed)):
        raise ValueError("response inputs must be finite")
    relative = (prediction - reference) / scales
    disagreement = (prediction - pool) / scales
    departure = (prediction - last_observed) / scales
    horizon = prediction.shape[0]
    result = {
        "response.mean_change": float(np.mean(np.abs(relative))),
        "response.signed_change": float(np.mean(relative)),
        "response.max_change": float(np.max(np.abs(relative))),
        "response.early_change": float(np.mean(np.abs(relative[: max(1, horizon // 4)]))),
        "response.late_change": float(np.mean(np.abs(relative[-max(1, horizon // 4) :]))),
        "response.pool_distance": float(np.mean(np.abs(disagreement))),
        "response.departure": float(np.mean(np.abs(departure))),
        "response.signed_departure": float(np.mean(departure)),
        "response.slope_change": float(np.mean(relative[-1] - relative[0])),
        "response.variation": float(np.mean(np.abs(np.diff(prediction / scales, axis=0))))
        if horizon > 1
        else 0.0,
        "response.has_quantiles": float(quantiles is not None),
        "response.interval_width": 0.0,
    }
    if quantiles is not None:
        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 3 or q.shape[:2] != prediction.shape or not np.isfinite(q).all():
            raise ValueError("quantiles must have finite [H,K,Q] values")
        result["response.interval_width"] = float(np.mean((q[..., -1] - q[..., 0]) / scales))
    return {key: float(np.clip(value, -1e8, 1e8)) for key, value in result.items()}


def family_macro(frame: pd.DataFrame, column: str = "loss") -> float:
    """Equal datasets within each family, then equal families."""
    means = frame.groupby(["family_id", "dataset_id"], observed=True)[column].mean()
    return float(means.groupby(level="family_id").mean().mean())


def _family_weights(frame: pd.DataFrame) -> np.ndarray:
    dataset_counts = frame.groupby("family_id")["dataset_id"].transform("nunique")
    row_counts = frame.groupby(["family_id", "dataset_id"])["episode_id"].transform("size")
    weights = 1.0 / (dataset_counts.to_numpy() * row_counts.to_numpy())
    return weights / np.mean(weights)


@dataclass
class UtilitySelector:
    """A small regression baseline with empirical calibration of selected actions.

    Calibration controls observed development behavior; no distribution-shift or
    finite-sample safety guarantee is claimed. All masks of an origin remain in
    the same fitting, calibration, or evaluation group.
    """

    use_response: bool = True
    seed: int = 5101
    n_estimators: int = 160
    baseline_id: str | None = None
    feature_names: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    model: Any = None
    switch_margin: float = float("inf")
    calibration: Mapping[str, Any] | None = None
    training_groups: tuple[str, ...] = ()

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        identity = np.column_stack(
            [
                (frame["candidate_id"] == candidate_id).to_numpy(dtype=float)
                for candidate_id in self.candidate_ids
            ]
        )
        matrix = np.column_stack([numeric, identity])
        if not np.isfinite(matrix).all():
            raise ValueError("selector features must be finite")
        names = [*self.feature_names, *(f"action.{candidate}" for candidate in self.candidate_ids)]
        return pd.DataFrame(matrix, columns=names, index=frame.index)

    def fit(self, frame: pd.DataFrame) -> UtilitySelector:
        from lightgbm import LGBMRegressor

        if frame.empty or frame.duplicated(["episode_id", "candidate_id"]).any():
            raise ValueError("training requires unique episode-candidate rows")
        counts = frame.groupby("candidate_id")["episode_id"].nunique()
        if (counts != frame.episode_id.nunique()).any():
            raise ValueError("every training episode must contain the same complete action pool")
        self.candidate_ids = tuple(sorted(counts.index))
        self.training_groups = tuple(sorted(frame.get("origin_id", frame.episode_id).unique()))
        risks = {
            candidate: family_macro(group) for candidate, group in frame.groupby("candidate_id")
        }
        self.baseline_id = min(risks, key=lambda candidate: (risks[candidate], candidate))
        baseline = frame[frame.candidate_id == self.baseline_id].set_index("episode_id")["loss"]
        target = frame.loss.to_numpy() - frame.episode_id.map(baseline).to_numpy()
        prefixes = ("static.", "response.") if self.use_response else ("static.",)
        self.feature_names = tuple(
            sorted(column for column in frame if column.startswith(prefixes))
        )
        if not self.feature_names or not np.isfinite(target).all():
            raise ValueError("training needs deployment features and finite utility labels")
        self.model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.n_estimators,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=25,
            reg_lambda=5.0,
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=4,
            verbosity=-1,
        )
        self.model.fit(self._matrix(frame), target, sample_weight=_family_weights(frame))
        self.switch_margin = float("inf")
        self.calibration = None
        return self

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.baseline_id is None:
            raise ValueError("selector has not been fitted")
        if set(frame.candidate_id).difference(self.candidate_ids):
            raise ValueError("unknown action in selector input")
        if (
            frame.duplicated(["episode_id", "candidate_id"]).any()
            or not (
                frame.groupby("episode_id").candidate_id.nunique() == len(self.candidate_ids)
            ).all()
        ):
            raise ValueError("each episode must contain the full unique candidate pool")
        scores = np.asarray(self.model.predict(self._matrix(frame)), dtype=float)
        scores[frame.candidate_id.to_numpy() == self.baseline_id] = 0.0
        return scores

    def select(self, frame: pd.DataFrame, *, gated: bool = False) -> pd.DataFrame:
        """Select using features only; outcome columns are carried through, never read."""
        scored = frame.copy()
        scored["predicted_delta"] = self.scores(frame)
        ranked = scored.sort_values(["episode_id", "predicted_delta", "candidate_id"])
        selected = ranked.drop_duplicates("episode_id").set_index("episode_id")
        baseline = scored[scored.candidate_id == self.baseline_id].set_index("episode_id")
        if len(baseline) != frame.episode_id.nunique():
            raise ValueError("each episode must include the fitted baseline action")
        # Zero estimated advantage always retains the baseline, including ties.
        threshold = self.switch_margin if gated else 0.0
        retain = selected.predicted_delta >= -threshold
        selected.loc[retain] = baseline.loc[selected.index[retain]]
        return selected.reset_index()

    def calibrate(self, frame: pd.DataFrame) -> UtilitySelector:
        if frame.empty:
            raise ValueError("calibration requires held-out episodes")
        if set(frame.get("origin_id", frame.episode_id)).intersection(self.training_groups):
            raise ValueError("calibration origins overlap selector training")
        raw = self.select(frame)
        baseline = frame[frame.candidate_id == self.baseline_id].set_index("episode_id")
        baseline = baseline.assign(predicted_delta=0.0)
        advantages = -raw.predicted_delta.to_numpy()
        # A small fixed grid is chosen on calibration data only. Infinity is the
        # always-baseline control; ties prefer the more conservative threshold.
        margins = np.unique(np.r_[0.0, np.quantile(advantages, [0.25, 0.5, 0.75, 0.9]), np.inf])
        trials = []
        for margin in margins:
            selected = raw.set_index("episode_id").copy()
            retain = selected.predicted_delta >= -margin
            selected.loc[retain] = baseline.loc[selected.index[retain]]
            trials.append((family_macro(selected.reset_index()), float(margin)))
        _, self.switch_margin = min(trials, key=lambda item: (item[0], -item[1]))
        self.calibration = {
            "episode_count": int(frame.episode_id.nunique()),
            "family_ids": sorted(frame.family_id.unique().tolist()),
            "trials": [
                {"macro_mase": risk, "margin": None if np.isinf(margin) else margin}
                for risk, margin in trials
            ],
            "always_baseline": bool(np.isinf(self.switch_margin)),
        }
        return self
