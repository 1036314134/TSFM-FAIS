from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tsfm_fais.contracts import ForecastSpec
from tsfm_fais.forecasting.adapters import (
    Chronos2Adapter,
    ChronosBoltAdapter,
    SundialAdapter,
    TimesFM2p5Adapter,
    TiRexAdapter,
)


class FakeChronos2:
    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels, **kwargs):
        del kwargs
        outputs = []
        for entry in inputs:
            target = np.asarray(entry["target"])
            n_targets = 1 if target.ndim == 1 else target.shape[0]
            value = np.arange(n_targets, dtype=float)
            output = np.empty((len(quantile_levels), prediction_length, n_targets))
            for q_index in range(len(quantile_levels)):
                output[q_index] = value + q_index
            outputs.append(output)
        return outputs, None


class FakeChronosBolt:
    def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
        values = np.empty((len(inputs), len(quantile_levels), prediction_length))
        for q_index in range(len(quantile_levels)):
            values[:, q_index, :] = q_index
        return values, None


class FakeTimesFM:
    def forecast(self, *, horizon, inputs):
        point = np.repeat(np.asarray([values[-1] for values in inputs])[:, None], horizon, 1)
        full = np.empty((len(inputs), horizon, 10))
        full[:, :, 0] = point
        for q_index in range(9):
            full[:, :, q_index + 1] = q_index
        return point, full


class FakeSundial:
    def forecast(self, *, inputs, prediction_length, num_samples):
        output = np.empty((len(inputs), num_samples, prediction_length))
        for sample in range(num_samples):
            output[:, sample, :] = sample
        return output


class FakeTiRex:
    def forecast(self, *, context, prediction_length, **kwargs):
        del kwargs
        output = np.empty((len(context), prediction_length, 9))
        for q_index in range(9):
            output[:, :, q_index] = q_index
        return output, None


def test_chronos2_joint_multivariate_normalization_and_target_selection():
    contexts = np.ones((2, 8, 3))
    spec = ForecastSpec(
        "chronos2",
        "joint_multivariate",
        horizon=4,
        target_indices=(2, 0),
    )
    result = Chronos2Adapter(backend=FakeChronos2()).predict(contexts, spec)
    assert result.point.shape == (2, 4, 2)
    np.testing.assert_allclose(result.point[0, 0], [3.0, 1.0])


def test_chronos_bolt_mock_quantiles():
    contexts = np.ones((2, 8, 1))
    spec = ForecastSpec("chronosbolt", "independent_univariate", horizon=3)
    result = ChronosBoltAdapter(backend=FakeChronosBolt()).predict(contexts, spec)
    assert result.quantiles.shape == (2, 3, 1, 3)
    np.testing.assert_allclose(result.point, 1.0)


def test_timesfm_mock_native_quantiles_are_selected():
    contexts = np.arange(16, dtype=float).reshape(2, 8, 1)
    spec = ForecastSpec("timesfm2p5", "independent_univariate", horizon=3)
    result = TimesFM2p5Adapter(backend=FakeTimesFM()).predict(contexts, spec)
    assert result.quantiles.shape == (2, 3, 1, 3)
    np.testing.assert_allclose(result.quantiles[0, 0, 0], [0.0, 4.0, 8.0])


def test_timesfm_falls_back_when_hub_mixin_forwards_proxies(monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            raise TypeError("__init__() got an unexpected keyword argument 'proxies'")

        @classmethod
        def _from_pretrained(cls, **kwargs):
            del cls
            calls.update(kwargs)
            return "loaded"

    timesfm_module = SimpleNamespace(TimesFM_2p5_200M_torch=Model)
    configs_module = SimpleNamespace()

    def fake_import(name):
        return timesfm_module if name == "timesfm" else configs_module

    monkeypatch.setattr(
        "tsfm_fais.forecasting.adapters.timesfm.importlib.import_module",
        fake_import,
    )
    adapter = TimesFM2p5Adapter("local-snapshot", torch_compile=False)
    assert adapter._load_backend() == "loaded"
    assert calls["model_id"] == "local-snapshot"
    assert calls["local_files_only"] is True
    assert calls["torch_compile"] is False


def test_sundial_mock_samples_are_normalized():
    contexts = np.ones((2, 8, 1))
    spec = ForecastSpec(
        "sundial",
        "independent_univariate",
        horizon=3,
        num_samples=5,
    )
    result = SundialAdapter(backend=FakeSundial()).predict(contexts, spec)
    assert result.samples.shape == (2, 5, 3, 1)
    np.testing.assert_allclose(result.point, 2.0)


def test_tirex_mock_native_quantiles_are_selected():
    contexts = np.ones((2, 8, 1))
    spec = ForecastSpec("tirex", "independent_univariate", horizon=3)
    result = TiRexAdapter(backend=FakeTiRex()).predict(contexts, spec)
    np.testing.assert_allclose(result.quantiles[0, 0, 0], [0.0, 4.0, 8.0])


def test_tirex_loads_a_local_snapshot_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            del cls
            calls.update(path=path, **kwargs)
            return "loaded"

    module = SimpleNamespace(
        load_model=lambda *args, **kwargs: None,
        base=SimpleNamespace(
            PretrainedModel=SimpleNamespace(REGISTRY={"TiRex": Model})
        ),
    )
    monkeypatch.setattr(
        "tsfm_fais.forecasting.adapters.tirex.importlib.import_module",
        lambda _name: module,
    )
    adapter = TiRexAdapter(str(tmp_path), device="cpu")
    assert adapter._load_backend() == "loaded"
    assert calls["path"] == str(checkpoint)
    assert calls["backend"] == "torch"
    assert calls["device"] == "cpu"
