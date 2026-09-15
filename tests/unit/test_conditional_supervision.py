import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_conditional_supervision import independent_labels  # noqa: E402
from collect_conditional_supervision import visible_inputs  # noqa: E402
from conditional_future import condition_history, draw_history  # noqa: E402
from conditional_supervision import (  # noqa: E402
    conditional_labels,
    factual_future,
    supervision_scenarios,
    supervision_seed,
)


def test_processes_are_stable_and_random_roles_are_disjoint():
    models = supervision_scenarios()
    assert len(models) == 12 and len({model["name"] for model in models}) == 12
    for model in models:
        assert max(abs(np.linalg.eigvals(model["a"]))) < 1
        assert np.linalg.eigvalsh(model["q"]).min() > 0
    seeds = [
        supervision_seed(stream, generator, history, condition)
        for stream in range(6)
        for generator in range(12)
        for history in range(64)
        for condition in range(10)
    ]
    assert len(set(seeds)) == len(seeds)


def test_conditional_labels_match_independent_recursion_and_share_factual_future():
    for generator, model in enumerate(supervision_scenarios()):
        phase = 5
        clean = draw_history(model, 96, phase, supervision_seed(2, generator, 0))
        center, scale = clean.mean(0), clean.std(0)
        seed = supervision_seed(5, generator, 0)
        factual = (factual_future(model, clean, phase, seed)[:, :2] - center[:2]) / scale[:2]
        for mechanism in ("complete", "point", "tail"):
            context = clean.copy()
            if mechanism == "point":
                context[::3] = np.nan
            elif mechanism == "tail":
                context[-29:] = np.nan
            mean, covariance = condition_history(model, context, phase)
            mu, variance = conditional_labels(model, mean, covariance, phase, center, scale)
            actual, independent_mu, independent_var = independent_labels(
                model, clean, context, phase, seed, center, scale
            )
            np.testing.assert_array_equal(factual, actual)
            np.testing.assert_allclose(mu, independent_mu, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(variance, independent_var, rtol=1e-10, atol=1e-10)
            assert np.isfinite(variance).all() and (variance > 0).all()


def test_visible_inputs_ignore_privileged_labels_and_process_parameters():
    rng = np.random.default_rng(160001)
    context = rng.normal(size=(96, 3))
    context[-12:] = np.nan
    actions = ["knn_multivariate", "linear_interp", "locf", "saits", "seasonal_lag", "timemixerpp"]
    candidates = np.repeat(context[None], 6, axis=0)
    for i in range(6):
        candidates[i, -12:] = context[-13] + i * 0.05
    motm = candidates[2].copy()
    names = sorted([*actions, "motm_reference", "guarded_direct"])
    points = rng.normal(size=(8, 96, 2))
    row = {
        "episode_id": "source|0|tail",
        "origin_id": "source|0",
        "family_id": "source",
        "dataset_id": "source0",
        "item_id": "series",
        "split": "train",
    }
    hidden = {
        **row,
        "posterior_mean": np.ones(3) * 1e9,
        "future": np.ones((96, 2)) * 1e12,
        "generator": 99,
        "a": np.eye(3) * 9,
    }
    for model_id, count in (("chronos2", 1), ("timesfm2p5", 2)):

        def build(metadata, model_id=model_id):
            return visible_inputs(
                metadata,
                0,
                context,
                candidates,
                actions,
                np.ones(6),
                motm,
                {"fallback_columns": []},
                points,
                names,
                np.zeros(3),
                np.ones(3),
                model_id,
                24,
            )

        frame, x, p = build(row)
        changed_frame, changed_x, changed_p = build(hidden)
        assert frame.equals(changed_frame)
        np.testing.assert_array_equal(x, changed_x)
        np.testing.assert_array_equal(p, changed_p)
        assert x.shape == (count, 8, 97)
        np.testing.assert_array_equal(x[:, :, 33:], 0)
        expected = points.reshape(1, 8, 192) if count == 1 else points.transpose(2, 0, 1)
        np.testing.assert_array_equal(p, expected)
