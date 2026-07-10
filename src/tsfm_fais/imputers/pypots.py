"""Lazy PyPOTS 1.5-series adapters with a strict fit/impute lifecycle."""

from __future__ import annotations

import importlib
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np

from tsfm_fais.contracts import SeriesBatch

from .base import BaseImputer, ImputerDependencyError, NativeImputation

PYPOTS_CLASS_NAMES: dict[str, str] = {
    "trmf": "TRMF",
    "brits": "BRITS",
    "gpvae": "GPVAE",
    "saits": "SAITS",
    "csdi": "CSDI",
    "imputeformer": "ImputeFormer",
    "helix": "HELIX",
    "timemixerpp": "TimeMixerPP",
    "totem": "TOTEM",
}


MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "trmf": {
        "lags": [1, 2, 3],
        "K": 10,
        "lambda_f": 1.0,
        "lambda_x": 1.0,
        "lambda_w": 1.0,
        "alpha": 1.0,
        "eta": 1.0,
    },
    "brits": {"rnn_hidden_size": 64},
    "gpvae": {
        "latent_size": 16,
        "encoder_sizes": (64,),
        "decoder_sizes": (64,),
        "beta": 0.2,
        "M": 1,
        "K": 1,
    },
    "saits": {
        "n_layers": 2,
        "d_model": 64,
        "d_ffn": 128,
        "n_heads": 4,
        "d_k": 16,
        "d_v": 16,
        "dropout": 0.0,
        "ORT_weight": 1.0,
        "MIT_weight": 1.0,
    },
    "csdi": {
        "n_layers": 4,
        "n_channels": 64,
        "n_heads": 8,
        "d_time_embedding": 64,
        "d_feature_embedding": 16,
        "d_diffusion_embedding": 128,
        "n_diffusion_steps": 50,
        "schedule": "quad",
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "is_unconditional": False,
    },
    "imputeformer": {
        "n_layers": 2,
        "d_input_embed": 32,
        "d_learnable_embed": 32,
        "d_proj": 32,
        "d_ffn": 64,
        "n_temporal_heads": 4,
        "dropout": 0.0,
    },
    "helix": {
        "d_pe": 8,
        "d_feature_embed": 1,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "ORT_weight": 1.0,
        "MIT_weight": 1.0,
    },
    "timemixerpp": {
        "n_layers": 1,
        "d_model": 32,
        "d_ffn": 64,
        "top_k": 3,
        "n_heads": 4,
        "n_kernels": 6,
        "dropout": 0.0,
        "channel_independence": False,
        "decomp_method": "moving_avg",
        "moving_avg": 3,
        "downsampling_layers": 1,
        "downsampling_window": 2,
        "apply_nonstationary_norm": False,
    },
    "totem": {
        "d_block_hidden": 32,
        "n_residual_layers": 1,
        "d_residual_hidden": 32,
        "d_embedding": 16,
        "n_embeddings": 32,
        "commitment_cost": 0.25,
        "compression_factor": 4,
    },
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_class(class_name: str) -> type[Any]:
    try:
        module = importlib.import_module("pypots.imputation")
    except ImportError as error:
        raise ImputerDependencyError(
            "PyPOTS candidates require the optional 'deep-imputers' extra "
            "(pypots>=1.5,<1.6 and torch)."
        ) from error
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImputerDependencyError(
            f"The installed PyPOTS package does not expose pypots.imputation.{class_name}."
        )
    return model_class


def _accepted_constructor_params(
    model_class: type[Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    signature = inspect.signature(model_class)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(params)
    return {key: value for key, value in params.items() if key in signature.parameters}


def _normalization(batch: SeriesBatch) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(batch.shape[2], dtype=float)
    scale = np.ones(batch.shape[2], dtype=float)
    for channel in range(batch.shape[2]):
        values = batch.values[:, :, channel]
        finite = values[np.isfinite(values)]
        if finite.size:
            mean[channel] = float(np.mean(finite))
            standard_deviation = float(np.std(finite))
            if np.isfinite(standard_deviation) and standard_deviation > 1e-8:
                scale[channel] = standard_deviation
    return mean, scale


@dataclass
class PyPOTSArtifact:
    model: Any
    mean: np.ndarray
    scale: np.ndarray
    n_steps: int
    n_features: int
    model_class_name: str
    constructor_params: dict[str, Any]


class PyPOTSImputer(BaseImputer):
    """Base adapter. Concrete classes declare one public candidate ID."""

    model_class_name: ClassVar[str] = ""
    model_defaults: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int | None = None,
        device: str | None = "cpu",
        random_state: int = 0,
        num_samples: int = 20,
        model_params: Mapping[str, Any] | None = None,
        **model_overrides: Any,
    ) -> None:
        if epochs < 1 or batch_size < 1 or num_samples < 1:
            raise ValueError("epochs, batch_size, and num_samples must be positive")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = patience
        self.device = device
        self.random_state = int(random_state)
        self.num_samples = int(num_samples)
        self.model_params = {
            **dict(self.model_defaults),
            **dict(model_params or {}),
            **model_overrides,
        }

    def _constructor_params(self, batch: SeriesBatch) -> dict[str, Any]:
        params: dict[str, Any] = {
            "n_steps": batch.shape[1],
            "n_features": batch.shape[2],
            "epochs": self.epochs,
            "batch_size": min(self.batch_size, batch.shape[0]),
            "patience": self.patience,
            "device": self.device,
            "verbose": False,
            "model_saving_strategy": None,
            **self.model_params,
        }
        if self.imputer_id == "trmf":
            params["max_iter"] = self.epochs
        return params

    def _fit(
        self, train_batch: SeriesBatch, metadata: Mapping[str, Any]
    ) -> PyPOTSArtifact:
        _set_seed(self.random_state)
        mean, scale = _normalization(train_batch)
        standardized = (train_batch.values - mean[None, None, :]) / scale[None, None, :]
        model_class = _load_model_class(self.model_class_name)
        params = _accepted_constructor_params(
            model_class, self._constructor_params(train_batch)
        )
        model = model_class(**params)
        model.fit({"X": standardized.astype(np.float32)})
        return PyPOTSArtifact(
            model=model,
            mean=mean,
            scale=scale,
            n_steps=train_batch.shape[1],
            n_features=train_batch.shape[2],
            model_class_name=self.model_class_name,
            constructor_params=params,
        )

    @staticmethod
    def _extract_prediction(
        raw: Any, batch_shape: tuple[int, int, int]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(raw, Mapping):
            for key in ("imputation", "imputed_data", "samples", "imputation_samples", "X"):
                if key in raw:
                    raw = raw[key]
                    break
            else:
                raise ValueError("PyPOTS output does not contain an imputation array")
        array = np.asarray(raw, dtype=float)
        if array.ndim == 3:
            if array.shape != batch_shape:
                raise ValueError(
                    f"PyPOTS returned {array.shape}; expected {batch_shape}"
                )
            return array, None
        if array.ndim != 4:
            raise ValueError("PyPOTS output must have shape [N,L,D] or sampled [N,S,L,D]")
        if array.shape[0] == batch_shape[0] and array.shape[2:] == batch_shape[1:]:
            samples = array
        elif array.shape[1] == batch_shape[0] and array.shape[2:] == batch_shape[1:]:
            samples = np.transpose(array, (1, 0, 2, 3))
        else:
            raise ValueError(
                f"sampled PyPOTS output {array.shape} is incompatible with {batch_shape}"
            )
        return np.median(samples, axis=1), np.var(samples, axis=1)

    def _impute_native(
        self, batch: SeriesBatch, artifact: PyPOTSArtifact | None, seed: int
    ) -> NativeImputation:
        if artifact is None:
            raise ValueError(f"{self.imputer_id} requires a fitted PyPOTS artifact")
        if (batch.shape[1], batch.shape[2]) != (artifact.n_steps, artifact.n_features):
            raise ValueError(
                "PyPOTS inference windows must match the fitted n_steps and n_features"
            )
        _set_seed(seed)
        standardized = (batch.values - artifact.mean[None, None, :]) / artifact.scale[
            None, None, :
        ]
        test_set = {"X": standardized.astype(np.float32)}
        if self.imputer_id == "csdi" and hasattr(artifact.model, "predict"):
            raw = artifact.model.predict(test_set, n_sampling_times=self.num_samples)
        else:
            raw = artifact.model.impute(test_set)
        point, uncertainty = self._extract_prediction(raw, batch.shape)
        point = artifact.mean[None, None, :] + artifact.scale[None, None, :] * point
        if uncertainty is not None:
            uncertainty = uncertainty * (artifact.scale[None, None, :] ** 2)
        return NativeImputation(
            point,
            uncertainty,
            {"seed": int(seed), "model_class": artifact.model_class_name},
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): PyPOTSImputer._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [PyPOTSImputer._json_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def save_artifact(self, artifact: PyPOTSArtifact, path: str | Path) -> Path:
        """Serialize normalization state and the fitted PyPOTS model."""

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "normalization.npz",
            mean=artifact.mean,
            scale=artifact.scale,
        )
        metadata = {
            "imputer_id": self.imputer_id,
            "n_steps": artifact.n_steps,
            "n_features": artifact.n_features,
            "model_class_name": artifact.model_class_name,
            "constructor_params": self._json_value(artifact.constructor_params),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        if hasattr(artifact.model, "save"):
            artifact.model.save(str(directory / "model.pypots"))
            (directory / "serializer.txt").write_text("pypots", encoding="utf-8")
        else:
            import joblib

            joblib.dump(artifact.model, directory / "model.joblib")
            (directory / "serializer.txt").write_text("joblib", encoding="utf-8")
        return directory

    def load_artifact(self, path: str | Path) -> PyPOTSArtifact:
        """Load an artifact without fitting the model again."""

        directory = Path(path)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["imputer_id"] != self.imputer_id:
            raise ValueError(
                f"artifact belongs to {metadata['imputer_id']}, not {self.imputer_id}"
            )
        normalization = np.load(directory / "normalization.npz")
        serializer = (directory / "serializer.txt").read_text(encoding="utf-8").strip()
        if serializer == "joblib":
            import joblib

            model = joblib.load(directory / "model.joblib")
        elif serializer == "pypots":
            model_class = _load_model_class(metadata["model_class_name"])
            constructor_params = _accepted_constructor_params(
                model_class, metadata["constructor_params"]
            )
            model = model_class(**constructor_params)
            if not hasattr(model, "load"):
                raise TypeError("the PyPOTS model does not implement load()")
            model.load(str(directory / "model.pypots"))
        else:
            raise ValueError(f"unsupported artifact serializer: {serializer}")
        return PyPOTSArtifact(
            model=model,
            mean=np.asarray(normalization["mean"], dtype=float),
            scale=np.asarray(normalization["scale"], dtype=float),
            n_steps=int(metadata["n_steps"]),
            n_features=int(metadata["n_features"]),
            model_class_name=str(metadata["model_class_name"]),
            constructor_params=dict(metadata["constructor_params"]),
        )

    save = save_artifact
    load = load_artifact


class TRMFImputer(PyPOTSImputer):
    imputer_id = "trmf"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class BRITSImputer(PyPOTSImputer):
    imputer_id = "brits"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class GPVAEImputer(PyPOTSImputer):
    imputer_id = "gpvae"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class SAITSImputer(PyPOTSImputer):
    imputer_id = "saits"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class CSDIImputer(PyPOTSImputer):
    imputer_id = "csdi"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class ImputeFormerImputer(PyPOTSImputer):
    imputer_id = "imputeformer"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class HELIXImputer(PyPOTSImputer):
    imputer_id = "helix"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class TimeMixerPPImputer(PyPOTSImputer):
    imputer_id = "timemixerpp"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


class TOTEMImputer(PyPOTSImputer):
    imputer_id = "totem"
    model_class_name = PYPOTS_CLASS_NAMES[imputer_id]
    model_defaults = MODEL_DEFAULTS[imputer_id]


__all__ = [
    "BRITSImputer",
    "CSDIImputer",
    "GPVAEImputer",
    "HELIXImputer",
    "ImputeFormerImputer",
    "MODEL_DEFAULTS",
    "PYPOTS_CLASS_NAMES",
    "PyPOTSArtifact",
    "PyPOTSImputer",
    "SAITSImputer",
    "TOTEMImputer",
    "TRMFImputer",
    "TimeMixerPPImputer",
]
