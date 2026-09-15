import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from apply_followup_policies import ranked_prediction, result_panels


def test_old_ranked_control_keeps_joint_and_independent_target_choices():
    actions = ["a", "b", "c", "d"]
    points = np.array([[[1, 10]], [[2, 20]], [[3, 30]], [[4, 40]]], float)
    rankings = {"shared": ["a", "b", "c"], "t0": ["a", "b", "c"], "t1": ["b", "c", "d"]}
    np.testing.assert_array_equal(
        ranked_prediction(points, actions, rankings, ["shared"], joint=True), [[2, 20]]
    )
    np.testing.assert_array_equal(
        ranked_prediction(points, actions, rankings, ["t0", "t1"], joint=False), [[2, 30]]
    )


def test_panels_keep_new_sources_distinct_from_unused_items_and_synthetic_masks():
    frame = pd.DataFrame(
        {
            "panel": ["held_items", "held_items", "new_native", "new_native", "new_synthetic"],
            "native_missing_context": [False, True, False, True, False],
            "episode_id": list("abcde"),
        }
    )
    groups = {name: set(rows.episode_id) for name, rows in result_panels(frame)}
    assert groups["held_items_missing"] == {"b"}
    assert groups["new_native_missing"] == {"d"}
    assert groups["new_synthetic_all"] == {"e"}
    assert "new_synthetic_complete" not in groups
