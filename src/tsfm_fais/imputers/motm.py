"""Pinned MoTM reference using its released networks and observed-context fitting."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def state_digest(models):
    result = hashlib.sha256()
    for number, model in enumerate(models):
        for name, value in model.state_dict().items():
            result.update(f"{number}:{name}:{tuple(value.shape)}:{value.dtype}".encode())
            result.update(
                value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            )
    return result.hexdigest()


def prepare_context(context, seed=42):
    """Reproduce the reference's observed-point duplication with a local RNG."""
    values = torch.as_tensor(
        np.asarray(context).T.copy(), dtype=torch.float32, device="cpu"
    ).unsqueeze(-1)
    if values.ndim != 3 or values.shape[1] < 2 or bool(torch.isinf(values).any()):
        raise ValueError("context must be [time, variable], at least two steps, with NaN gaps")
    grid = (
        torch.linspace(0, 1, values.shape[1], device="cpu")
        .reshape(1, -1, 1)
        .expand_as(values)
        .clone()
    )
    filled, coordinates = values.clone(), grid.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    available = torch.isfinite(values[..., 0]).any(1)
    for channel in range(len(values)):
        observed = torch.where(torch.isfinite(values[channel, :, 0]))[0]
        missing = torch.where(torch.isnan(values[channel, :, 0]))[0]
        if len(observed) and len(missing):
            replacement = observed[
                torch.randint(len(observed), (len(missing),), generator=generator)
            ]
            filled[channel, missing, 0] = values[channel, replacement, 0]
            coordinates[channel, missing, 0] = grid[channel, replacement, 0]
    return filled, coordinates, grid, available


class MOTMReference:
    def __init__(self, reference_root, runtime_root, *, device="cuda", ridge=0.5, batch_size=32):
        self.reference_root = Path(reference_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.device, self.ridge, self.batch_size = device, float(ridge), int(batch_size)
        if self.ridge <= 0 or self.batch_size < 1:
            raise ValueError("positive ridge regularization and batch size are required")
        source = json.loads((self.reference_root / "manifest.json").read_text(encoding="utf-8"))
        runtime = json.loads((self.runtime_root / "manifest.json").read_text(encoding="utf-8"))
        source_sha = hashlib.sha256(
            (self.reference_root / "manifest.json").read_bytes()
        ).hexdigest()
        if (
            source["status"] != "completed"
            or runtime["identity"]["reference_manifest_sha256"] != source_sha
        ):
            raise ValueError("reference code and runtime metadata do not match")
        for record in source["files"]:
            if (
                hashlib.sha256((self.reference_root / record["path"]).read_bytes()).hexdigest()
                != record["sha256"]
            ):
                raise ValueError("pinned MoTM source or weights changed")
        sys.path.insert(0, str(self.runtime_root / "dependencies"))
        sys.path.insert(0, str(self.reference_root))
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        from src.data.scaler import CustomStandardScaler
        from src.metalearning.metalearning import inner_loop
        from src.modules.inr import ModulatedFourierFeatures
        from src.modules.ridge.ridge import RidgeRegressor

        self.inner_loop, self.scaler_class = inner_loop, CustomStandardScaler
        self.ridge_model = RidgeRegressor(lambda_regu=self.ridge).to(device)
        self.models, self.settings = [], []
        allowed = [
            DictConfig,
            ListConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            Any,
            defaultdict,
            dict,
            list,
            int,
        ]
        for record in runtime["checkpoints"]:
            path = self.reference_root / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("published checkpoint changed")
            with torch.serialization.safe_globals(allowed):
                saved = torch.load(path, map_location="cpu", weights_only=True)
            settings = record["saved_optim_config"]
            if not settings or not all(
                key in settings for key in ("inner_steps", "lr_code", "loss_type")
            ):
                raise ValueError("original context-fitting settings were not recovered")
            config = dict(record["inr_config"])
            if config.pop("_target_") != "src.modules.inr.ModulatedFourierFeatures":
                raise ValueError("unsupported published model class")
            with torch.random.fork_rng(devices=[]):
                model = ModulatedFourierFeatures(**config)
            model.load_state_dict(saved["inr"], strict=True)
            self.models.append(model.to(device).eval().requires_grad_(False))
            self.settings.append(
                {
                    "inner_steps": settings["inner_steps"],
                    "inner_lr": settings["lr_code"],
                    "loss_type": settings["loss_type"],
                }
            )
        self.initial_digest = state_digest(self.models)

    def predict_prepared(self, values, coordinates, grid):
        values, coordinates, grid = [value.to(self.device) for value in (values, coordinates, grid)]
        outputs = []
        for begin in range(0, len(values), self.batch_size):
            observed = values[begin : begin + self.batch_size]
            context_grid = coordinates[begin : begin + self.batch_size]
            target_grid = grid[begin : begin + self.batch_size]
            scaler = self.scaler_class(dim=1, epsilon=1e-7)
            scaler.fit(observed)
            normalized = scaler.transform(observed)
            hidden_context, hidden_target = [], []
            for model, settings in zip(self.models, self.settings, strict=True):
                latent = torch.zeros(
                    (len(observed), model.latent_dim), device=self.device, requires_grad=True
                )
                latent = self.inner_loop(
                    model, latent, context_grid, normalized, is_train=False, **settings
                )
                with torch.no_grad():
                    hidden_context.append(
                        model.modulated_forward_mixture(
                            context_grid, latent.detach(), n_last_layers=1
                        ).flatten(2)
                    )
                    hidden_target.append(
                        model.modulated_forward_mixture(
                            target_grid, latent.detach(), n_last_layers=1
                        ).flatten(2)
                    )
            with torch.no_grad():
                context_features = torch.cat(hidden_context, dim=2)
                target_features = torch.cat(hidden_target, dim=2)
                weights, bias = self.ridge_model.get_weights(
                    context_features, normalized, share_weights=False
                )
                output = scaler.inv_transform(target_features @ weights + bias)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("MoTM returned a nonfinite value")
                outputs.append(output.cpu())
        return torch.cat(outputs) if outputs else torch.empty_like(values, device="cpu")

    def impute(self, context, fallback_values):
        context, fallback = (
            np.asarray(context, dtype=float),
            np.asarray(fallback_values, dtype=float),
        )
        if context.ndim != 2 or fallback.shape != context.shape or not np.isfinite(fallback).all():
            raise ValueError("a finite, matching fallback context is required")
        observed = np.isfinite(context)
        if observed.all():
            return context.copy(), {"fallback_columns": [], "fitted_variables": 0}
        values, coordinates, grid, available = prepare_context(context)
        prediction = self.predict_prepared(
            values[available], coordinates[available], grid[available]
        )
        completed = fallback.copy()
        completed[:, available.numpy()] = prediction[..., 0].numpy().T
        completed[observed] = context[observed]
        return completed, {
            "fallback_columns": torch.where(~available)[0].tolist(),
            "fitted_variables": int(available.sum()),
        }

    def verify_frozen(self):
        unchanged = state_digest(self.models) == self.initial_digest
        if not unchanged or any(
            parameter.requires_grad or parameter.grad is not None
            for model in self.models
            for parameter in model.parameters()
        ):
            raise ValueError("the released MoTM networks were modified")
        return unchanged
