from __future__ import annotations

import numpy as np
import pytest

from tsfm_fais.contracts import BudgetSpec, CandidateResult, SeriesBatch
from tsfm_fais.routing import (
    CandidateShortlister,
    InferenceRouterBundle,
    LazyLightGBMRegressor,
    extract_missing_blocks,
)


class FakeRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = 0.0

    def fit(self, x, y, **kwargs):
        del x, kwargs
        self.mean = float(np.mean(y))

    def predict(self, x):
        return np.full(len(x), self.mean)


def make_inputs():
    clean = np.arange(8, dtype=float).reshape(1, 8, 1)
    observed = np.ones_like(clean, dtype=bool)
    observed[0, 2:4, 0] = False
    observed[0, 6:8, 0] = False
    masked = clean.copy()
    masked[~observed] = np.nan
    batch = SeriesBatch(masked, observed)
    a = clean.copy()
    b = clean.copy()
    a[0, 6:8, 0] += 2.0
    b[0, 2:4, 0] += 2.0
    candidates = {
        "a": CandidateResult("a", a, np.ones_like(observed)),
        "b": CandidateResult("b", b, np.ones_like(observed)),
    }
    return clean, batch, candidates


def test_lazy_lightgbm_wrapper_uses_injected_factory_only_on_fit():
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeRegressor(**kwargs)

    model = LazyLightGBMRegressor(model_factory=factory)
    assert not model.is_loaded
    model.fit(np.ones((3, 2)), np.asarray([1.0, 2.0, 3.0]))
    assert model.is_loaded and len(calls) == 1
    np.testing.assert_allclose(model.predict(np.ones((2, 2))), 2.0)


def test_router_bundle_routes_different_blocks_to_different_candidates():
    clean, batch, candidates = make_inputs()
    blocks = extract_missing_blocks(batch)
    unary = {
        (blocks[0].block_id, "a"): 0.0,
        (blocks[0].block_id, "b"): 3.0,
        (blocks[1].block_id, "a"): 3.0,
        (blocks[1].block_id, "b"): 0.0,
    }
    bundle = InferenceRouterBundle(pairwise_weight=0.0)
    result = bundle.route(
        batch,
        candidates,
        blocks=blocks,
        budget=BudgetSpec(max_candidates=2),
        predicted_unary=unary,
        predicted_pairwise={},
    )
    assert result.assignments == {
        blocks[0].block_id: "a",
        blocks[1].block_id: "b",
    }
    np.testing.assert_allclose(bundle.assemble(batch, blocks, candidates, result), clean)


def test_shortlist_respects_candidate_count_and_score_order():
    shortlist = CandidateShortlister().select(
        ("c", "a", "b"),
        {"a": 1.0, "b": 2.0, "c": 0.0},
        BudgetSpec(max_candidates=2),
    )
    assert shortlist.selected == ("c", "a")


def test_inference_bundle_never_routes_safe_completion_as_native_output():
    _, batch, candidates = make_inputs()
    blocks = extract_missing_blocks(batch)
    for candidate in candidates.values():
        selector = (0, slice(blocks[0].start, blocks[0].end), blocks[0].channel)
        candidate.native_valid_mask[selector] = False
    with pytest.raises(RuntimeError, match="no feasible"):
        InferenceRouterBundle(pairwise_weight=0.0).route(
            batch,
            candidates,
            blocks=blocks,
            budget=BudgetSpec(max_candidates=2),
        )
