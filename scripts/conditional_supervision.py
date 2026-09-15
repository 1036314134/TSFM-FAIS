"""Known-process source populations and conditional forecasting supervision."""

import numpy as np
from conditional_future import future_moments, sample_futures, seasonal


def supervision_scenarios():
    result = []
    for group in ("diagonal_ar3", "coupled_ar3", "seasonal_ar7"):
        dimension = 7 if group == "seasonal_ar7" else 3
        for variant in range(4):
            phi = ([0.2, 0.5, 0.8, 0.95] if group == "diagonal_ar3" else [0.2, 0.5, 0.75, 0.9])[
                variant
            ]
            coupling = 0.0 if group == "diagonal_ar3" else 0.04 if group == "seasonal_ar7" else 0.03
            a = phi * np.eye(dimension) + coupling * np.roll(np.eye(dimension), 1, axis=1)
            correlation = [0.0, 0.2, 0.4, 0.6][variant]
            q = 0.04 * (
                (1 - correlation) * np.eye(dimension)
                + correlation * np.ones((dimension, dimension))
            )
            period = [12, 24, 48, 96][variant] if group == "seasonal_ar7" else 24
            amplitude = (
                [0.5, 1.0, 1.5, 2.0][variant] * np.linspace(1.0, 2.0, dimension)
                if group == "seasonal_ar7"
                else np.zeros(dimension)
            )
            if max(abs(np.linalg.eigvals(a))) >= 1 or np.linalg.eigvalsh(q).min() <= 0:
                raise ValueError("source dynamics must be stable with positive innovations")
            result.append(
                {
                    "name": f"r16_{group}_{variant}",
                    "group": group,
                    "a": a,
                    "q": q,
                    "period": period,
                    "amplitude": amplitude,
                }
            )
    return result


def supervision_seed(stream, generator, history=0, condition=0):
    if not (0 <= stream < 6 and 0 <= generator < 12 and 0 <= history < 64 and 0 <= condition < 10):
        raise ValueError("unregistered source random stream")
    return 16000000 + 100000 * stream + 1000 * generator + 10 * history + condition


def factual_future(model, clean, phase, seed):
    state = clean[-1] - seasonal(model, [phase + len(clean) - 1])[0]
    return sample_futures(
        model, state, np.zeros_like(model["q"]), phase + len(clean), 96, 1.0, 1, seed
    )[0]


def conditional_labels(model, posterior_mean, posterior_covariance, phase, center, scale):
    mean, missing, innovation = future_moments(
        model, posterior_mean, posterior_covariance, phase + 96, 96, 1.0
    )
    return (mean[:, :2] - center[:2]) / scale[:2], (missing + innovation)[:, :2] / scale[:2] ** 2
