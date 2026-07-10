from __future__ import annotations

from tsfm_fais.forecasting import leave_model_out_folds


def test_leave_model_out_folds_never_train_on_held_out_model() -> None:
    folds = leave_model_out_folds(("chronos2", "timesfm2p5", "sundial"))
    assert len(folds) == 3
    for fold in folds:
        assert fold.test_models == (fold.held_out_model,)
        assert fold.held_out_model not in fold.train_models
