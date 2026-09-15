"""Score the predictions actually returned by imputation portfolios."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .forecast_projection import NORM_EPS, ForecastProjectionRegressor, forecast_offsets
from .forecast_response import FORECAST_FEATURES, RESPONSE_FEATURES
from .preforecast import STATIC_FEATURES
from .utility import response_features


def option_catalog(actions):
    if (
        len(actions) != 7
        or len(set(actions)) != 7
        or "locf" not in actions
        or "guarded_direct" not in actions
    ):
        raise ValueError("use the original seven distinct supported candidates")
    members = [(index,) for index in range(7)] + list(combinations(range(7), 3)) + [tuple(range(7))]
    names = tuple("median:" + "+".join(actions[index] for index in group) for group in members)
    return names, tuple(members)


def option_vectors(candidate_vectors, members):
    values = np.asarray(candidate_vectors, float)
    if values.ndim != 3 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("finite seven-candidate forecast vectors are required")
    if len(members) != 43 or members[-1] != tuple(range(7)):
        raise ValueError("the registered 43-option catalog changed")
    return np.stack([np.median(values[:, group], axis=1) for group in members], axis=1)


def build_option_features(candidate_vectors, static, last_locf, actions, *, horizon, targets):
    """Use current forecasts and imputer descriptors; no teacher or future is an input."""
    candidate_vectors = np.asarray(candidate_vectors, float)
    names, members = option_catalog(actions)
    vectors = option_vectors(candidate_vectors, members)
    static, last_locf = np.asarray(static, float), np.asarray(last_locf, float)
    n = len(vectors)
    if static.shape != (n, 7, len(STATIC_FEATURES)) or last_locf.shape != (n, targets):
        raise ValueError("aligned static descriptors and last-LOCF values are required")
    if (
        vectors.shape[2] != horizon * targets
        or not np.isfinite(static).all()
        or not np.isfinite(last_locf).all()
    ):
        raise ValueError("feature dimensions or finite-value requirements failed")
    reference = candidate_vectors[:, actions.index("locf")].reshape(n, horizon, targets)
    finite = [index for index, name in enumerate(actions) if name != "guarded_direct"]
    pool = np.median(candidate_vectors[:, finite], axis=1).reshape(n, horizon, targets)
    features = np.empty((n, 43, len(FORECAST_FEATURES) + 7), dtype=np.float32)
    for option, group in enumerate(members):
        features[:, option, : len(STATIC_FEATURES)] = static[:, group].mean(axis=1)
        features[:, option, -7:] = np.isin(np.arange(7), group)
        forecasts = vectors[:, option].reshape(n, horizon, targets)
        for row in range(n):
            response = response_features(
                forecasts[row], reference[row], pool[row], last_locf[row], np.ones(targets), None
            )
            features[row, option, len(STATIC_FEATURES) : len(FORECAST_FEATURES)] = [
                response[name] for name in RESPONSE_FEATURES
            ]
    return features, vectors, names


def option_scores(vectors, estimates, *, target_kind):
    """The final option is the fixed seven-candidate median reference."""
    _, _, energy = forecast_offsets(vectors, anchor=vectors[:, -1])
    estimates = np.asarray(estimates, float)
    if estimates.shape != energy.shape or not np.isfinite(estimates).all():
        raise ValueError("aligned finite option estimates are required")
    if target_kind == "unit_projection":
        scores = energy - 2 * np.sqrt(energy) * estimates
    elif target_kind == "direct_risk":
        scores = estimates.copy()
    else:
        raise ValueError("unsupported aligned portfolio target")
    scores[energy <= NORM_EPS**2] = 0.0
    return scores


def choose_option(scores, menu):
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] != 43 or not np.isfinite(scores).all():
        raise ValueError("the complete finite 43-option score vector is required")
    choices = {
        "single": list(range(7)),
        "triple": list(range(7, 42)),
        "mixed": list(range(42)),
        "full": [42, *range(42)],
    }
    if menu not in choices:
        raise ValueError("unknown registered option menu")
    indices = np.asarray(choices[menu])
    return indices[scores[:, indices].argmin(axis=1)]


@dataclass
class AlignedPortfolioRegressor(ForecastProjectionRegressor):
    member_names: tuple[str, ...] = ()

    def _matrix(self, frame):
        columns = (*FORECAST_FEATURES, *self.member_names)
        observed = {name for name in frame if name.startswith(("static.", "response.", "member."))}
        if len(self.member_names) != 7 or observed != set(columns):
            raise ValueError("aligned portfolio inputs require the exact feature whitelist")
        if set(frame.candidate_id) != set(self.candidate_ids):
            raise ValueError("the fitted option catalog changed")
        result = frame[list(columns)].astype(np.float32)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("option features must be finite")
        return result
