from tsfm_fais.cli import main


def test_cli_smoke(capsys):
    code = main(["smoke", "--config", "configs/smoke.yaml"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "SMOKE PASS" in captured.out

