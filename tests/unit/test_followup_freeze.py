from pathlib import Path

import pytest


def test_followup_freeze_rejects_forecasts_but_allows_preparation(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from freeze_followup_portfolios import assert_unstarted_confirmation

    prepared = tmp_path / "native" / "prepared" / "example.npz"
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"preparation only")
    assert_unstarted_confirmation(tmp_path)
    predicted = tmp_path / "native" / "chronos2" / "predictions" / "example.npz"
    predicted.parent.mkdir(parents=True)
    predicted.write_bytes(b"a forecast exists")
    with pytest.raises(ValueError, match="already exist"):
        assert_unstarted_confirmation(tmp_path)
