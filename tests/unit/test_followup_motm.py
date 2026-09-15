import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluate_followup_motm import matching_candidate


def test_forecast_reuse_respects_covariate_scope_and_float32_inputs():
    actions = ["locf", "linear_interp", "seasonal_lag", "knn_multivariate", "saits", "timemixerpp"]
    context = np.array([[1, 2, 3], [np.nan, 3, np.nan], [2, 4, 5]], float)
    base = np.array([[1, 2, 3], [1, 3, 3], [2, 4, 5]], float)
    candidates = np.repeat(base[None], 6, axis=0)
    completed = base.copy()
    completed[1, 2] = 30
    assert matching_candidate(context, candidates, actions, completed, joint=False) == 0
    assert matching_candidate(context, candidates, actions, completed, joint=True) is None
    completed = base.copy()
    completed[1, 0] += 1e-10
    assert matching_candidate(context, candidates, actions, completed, joint=True) == 0
    completed[1, 0] += 1
    assert matching_candidate(context, candidates, actions, completed, joint=False) is None
