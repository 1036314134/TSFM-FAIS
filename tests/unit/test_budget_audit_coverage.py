from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import audit_imputer_budget_study

    return audit_imputer_budget_study


@pytest.mark.parametrize("count", [18, 54])
def test_budget_audit_uses_the_registered_panel_count_and_ids(audit, count):
    ids = [f"episode-{index}" for index in range(count)]
    prepared = {"evaluation_episode_count": count, "cases": [{"episode_id": name} for name in ids]}
    frame = pd.DataFrame(
        {"episode_id": ids, "model_id": "model", "budget": "budget", "method": "method"}
    )
    audit.verify_evaluation_coverage(frame, prepared)
    frame.loc[0, "episode_id"] = "outside-the-registered-panel"
    with pytest.raises(ValueError, match="evaluation coverage"):
        audit.verify_evaluation_coverage(frame, prepared)
    frame.loc[0, "episode_id"] = ids[1]
    with pytest.raises(ValueError, match="evaluation coverage"):
        audit.verify_evaluation_coverage(frame, prepared)


def test_budget_audit_rejects_a_declared_count_without_the_expected_ids(audit):
    prepared = {"evaluation_episode_count": 54, "cases": [{"episode_id": "only-one"}]}
    frame = pd.DataFrame(
        {"episode_id": ["only-one"], "model_id": "model", "budget": "budget", "method": "method"}
    )
    with pytest.raises(ValueError, match="declared episode count"):
        audit.verify_evaluation_coverage(frame, prepared)
