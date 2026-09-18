import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from hdb_generalization_core import (  # noqa: E402
    build_case,
    choose_peers,
    combine,
    fit_dynamics,
    population,
    queries,
)


def data():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(672, 4)).cumsum(0)
    values[350:356, 0] = np.nan
    values[410:416, 1] = np.nan
    return values


def test_population_does_not_use_outcome_magnitudes_and_rejects_reserve_rows():
    original = data()
    changed = original.copy()
    changed[336:] *= -500
    first, metadata = population(original, list("ABCD"))
    second, updated = population(changed, list("ABCD"))
    assert first == second and metadata == updated
    assert any(r["panel"] == "natural_outage_h24" for r in first)
    with pytest.raises(ValueError, match="first 672"):
        population(np.vstack([original, original]), list("ABCD"))


def test_peer_selection_has_prefix_boundary_and_excludes_self():
    original = data()
    columns, decisions = choose_peers(original[:336], 0, list(range(4)), list("ABCD"))
    assert columns[0] == 0 and len(set(columns)) == len(columns) == 4
    assert all(r["column"] != 0 for r in decisions)
    with pytest.raises(ValueError, match="prefix"):
        choose_peers(original, 0, list(range(4)), list("ABCD"))


def test_input_construction_preserves_observations_and_ignores_current_future():
    original = data()
    row = {"origin": 400, "panel": "synthetic_outage_h24"}
    model = fit_dynamics(original[:336])
    first = build_case(original, list(range(4)), row, model)
    changed = original.copy()
    changed[400:] = 123456
    second = build_case(changed, list(range(4)), row, model)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    observed = np.isfinite(first["context"])
    for fill in first["fills"]:
        np.testing.assert_array_equal(fill[observed], first["context"][observed])
    assert np.isnan(first["context"][-24:, 0]).all()


def test_all_queries_use_the_same_hidden_target_tail_and_declared_information_scope():
    original = data()
    prepared = build_case(
        original,
        list(range(4)),
        {"origin": 400, "panel": "synthetic_outage_h24"},
        fit_dynamics(original[:336]),
    )
    requests = queries(prepared)
    assert len(requests) == 11
    assert requests["native_target192"].shape == (1, 192)
    assert requests["native_peer336"].shape == (4, 336)
    for name in ("native_target192", "native_peer192", "native_peer336"):
        assert np.isnan(requests[name][0, -24:]).all()
    assert all(v.dtype == np.float32 and v.flags.c_contiguous for v in requests.values())
    combined = combine({name: np.zeros(24) for name in requests}, prepared)
    assert len(combined) == 31
    np.testing.assert_array_equal(
        combined["half_var_full_gaussian"], 0.5 * combined["linear_var_direct"]
    )
