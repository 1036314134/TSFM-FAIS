from __future__ import annotations

from pathlib import Path

import numpy as np

from tsfm_fais.contracts import CandidateStatus, SeriesBatch
from tsfm_fais.imputers.pypots import (
    CSDIImputer,
    SAITSImputer,
    TimeMixerPPImputer,
    TRMFImputer,
)


def make_batch() -> SeriesBatch:
    time = np.arange(12, dtype=float)
    values = np.stack([time, time * 2 + 1, np.sin(time)], axis=1)[None, :, :]
    mask = np.ones_like(values, dtype=bool)
    mask[:, 3:6, 0] = False
    mask[:, 7:9, 1:] = False
    return SeriesBatch(values, mask)


class FakeModel:
    fit_calls = 0
    last_params = None

    def __init__(self, n_steps: int, n_features: int, epochs: int, batch_size: int, **kwargs):
        self.params = {
            "n_steps": n_steps,
            "n_features": n_features,
            "epochs": epochs,
            "batch_size": batch_size,
            **kwargs,
        }
        type(self).last_params = self.params
        self.loaded = False

    def fit(self, train_set):
        type(self).fit_calls += 1
        assert train_set["X"].ndim == 3

    def impute(self, test_set):
        return np.nan_to_num(test_set["X"], nan=0.25)

    def save(self, path: str):
        Path(path).write_text("fake model", encoding="utf-8")

    def load(self, path: str):
        assert Path(path).read_text(encoding="utf-8") == "fake model"
        self.loaded = True


class FakeCSDI(FakeModel):
    def predict(self, test_set, n_sampling_times: int):
        base = np.nan_to_num(test_set["X"], nan=0.0)
        offsets = np.linspace(-1.0, 1.0, n_sampling_times)
        samples = np.stack([base + offset for offset in offsets], axis=1)
        return {"imputation": samples}


class FakeTRMF:
    last_max_iter = None

    def __init__(
        self,
        lags,
        K,
        lambda_f,
        lambda_x,
        lambda_w,
        alpha,
        eta,
        max_iter=1000,
        **kwargs,
    ):
        del lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, kwargs
        type(self).last_max_iter = max_iter

    def fit(self, train_set):
        assert train_set["X"].ndim == 3

    def impute(self, test_set):
        return np.nan_to_num(test_set["X"], nan=0.0)


class FakeTimeMixerPP(FakeModel):
    def __init__(
        self,
        n_steps,
        n_features,
        n_layers,
        d_model,
        d_ffn,
        top_k,
        n_heads,
        n_kernels,
        **kwargs,
    ):
        assert n_heads > 0 and n_kernels > 0
        super().__init__(
            n_steps,
            n_features,
            kwargs.pop("epochs"),
            kwargs.pop("batch_size"),
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            top_k=top_k,
            n_heads=n_heads,
            n_kernels=n_kernels,
            **kwargs,
        )


def test_pypots_fit_and_inference_are_separate_and_joint(monkeypatch) -> None:
    import tsfm_fais.imputers.pypots as module

    FakeModel.fit_calls = 0
    monkeypatch.setattr(module, "_load_model_class", lambda _: FakeModel)
    batch = make_batch()
    imputer = SAITSImputer(epochs=2, batch_size=8, model_params={"d_model": 16})
    artifact = imputer.fit(batch, {})
    assert FakeModel.fit_calls == 1
    assert FakeModel.last_params["n_steps"] == batch.shape[1]
    assert FakeModel.last_params["n_features"] == batch.shape[2]
    assert FakeModel.last_params["d_model"] == 16
    result = imputer.impute(batch, artifact, seed=7)
    assert FakeModel.fit_calls == 1
    assert result.status is CandidateStatus.SUCCESS
    assert np.isfinite(result.values).all()
    np.testing.assert_array_equal(
        result.values[batch.observed_mask], batch.values[batch.observed_mask]
    )


def test_csdi_uses_sample_median_and_variance(monkeypatch) -> None:
    import tsfm_fais.imputers.pypots as module

    monkeypatch.setattr(module, "_load_model_class", lambda _: FakeCSDI)
    batch = make_batch()
    imputer = CSDIImputer(epochs=1, num_samples=5)
    artifact = imputer.fit(batch, {})
    result = imputer.impute(batch, artifact, seed=3)
    assert result.uncertainty is not None
    assert result.uncertainty.shape == batch.shape
    assert np.all(result.uncertainty[~batch.observed_mask] > 0)
    assert np.all(result.uncertainty[batch.observed_mask] == 0)


def test_pypots_artifact_save_and_load_does_not_refit(monkeypatch, tmp_path) -> None:
    import tsfm_fais.imputers.pypots as module

    FakeModel.fit_calls = 0
    monkeypatch.setattr(module, "_load_model_class", lambda _: FakeModel)
    batch = make_batch()
    imputer = SAITSImputer(epochs=1)
    artifact = imputer.fit(batch, {})
    directory = imputer.save_artifact(artifact, tmp_path / "artifact")
    restored = imputer.load_artifact(directory)
    assert restored.model.loaded
    assert FakeModel.fit_calls == 1
    result = imputer.impute(batch, restored, seed=2)
    assert result.status is CandidateStatus.SUCCESS


def test_trmf_and_timemixerpp_explicit_required_parameters(monkeypatch) -> None:
    import tsfm_fais.imputers.pypots as module

    batch = make_batch()
    monkeypatch.setattr(
        module,
        "_load_model_class",
        lambda name: FakeTRMF if name == "TRMF" else FakeTimeMixerPP,
    )
    trmf = TRMFImputer(epochs=7)
    trmf_artifact = trmf.fit(batch, {})
    assert FakeTRMF.last_max_iter == 7
    assert trmf.impute(batch, trmf_artifact).status is CandidateStatus.SUCCESS

    mixer = TimeMixerPPImputer(epochs=1)
    mixer_artifact = mixer.fit(batch, {})
    assert mixer_artifact.constructor_params["n_heads"] == 4
    assert mixer_artifact.constructor_params["n_kernels"] == 6
