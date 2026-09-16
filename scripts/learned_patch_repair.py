"""Low-rank repairs on incomplete historical tokens; the forecasting backbone is separate."""

import numpy as np
import torch
from torch import nn


class PatchRepair(nn.Module):
    def __init__(self, width, patch_size, rank=8):
        super().__init__()
        self.width, self.patch_size, self.rank = width, patch_size, rank
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.mask = nn.Linear(patch_size, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, embedding, missing, condition):
        if condition not in ("pattern", "fraction"):
            raise ValueError("the registered repair condition is unknown")
        active = missing.bool().any(-1)
        if not bool(active.any()):
            return embedding, embedding.new_zeros(())
        descriptor = (
            missing if condition == "pattern" else missing.mean(-1, keepdim=True).expand_as(missing)
        )
        change = self.up(
            torch.nn.functional.gelu(self.down(self.norm(embedding)) + self.mask(descriptor))
        )
        repaired = torch.where(active[..., None], embedding + change, embedding)
        energy = change.square().mean(-1) / embedding.detach().square().mean(-1).clamp_min(1e-6)
        return repaired, energy[active].mean()


def repair_dimensions(backbone, model_id):
    if model_id == "chronos2":
        module = backbone.input_patch_embedding
        return module.output_layer.out_features, backbone.chronos_config.input_patch_size
    module = backbone.tokenizer
    return module.output_layer.out_features, backbone.p


class RepairHook:
    def __init__(self, backbone, model_id, observed, module, condition, targets=(0, 1)):
        self.backbone, self.model_id = backbone, model_id
        self.observed = np.asarray(observed, bool)
        self.module, self.condition, self.targets = module, condition, targets
        self.calls, self.penalties, self.changed_tokens = 0, [], 0

    def __enter__(self):
        host = (
            self.backbone.input_patch_embedding
            if self.model_id == "chronos2"
            else self.backbone.tokenizer
        )
        self.handle = host.register_forward_hook(self.after)
        return self

    def after(self, host, args, embedding):
        index = self.calls
        self.calls += 1
        if self.model_id == "chronos2" and index != 0:
            return embedding
        if self.calls > 2:
            raise ValueError("this repair contract supports the H96 history encoding only")
        patch = self.module.patch_size
        observed = (
            self.observed if self.model_id == "chronos2" else self.observed[:, list(self.targets)]
        )
        if len(observed) % patch or embedding.shape[1] != len(observed) // patch:
            raise ValueError("repair markers are not aligned to the historical tokens")
        real = observed.shape[1]
        if real > embedding.shape[0]:
            raise ValueError("the real target count exceeds the inference batch")
        mask = embedding.new_zeros((*embedding.shape[:2], patch))
        mask[:real] = torch.as_tensor(
            (~observed).T.copy(), device=embedding.device, dtype=embedding.dtype
        ).reshape(real, -1, patch)
        repaired, penalty = self.module(embedding, mask, self.condition)
        self.penalties.append(penalty)
        self.changed_tokens += int(mask.bool().any(-1).sum())
        return repaired

    @property
    def penalty(self):
        return torch.stack(self.penalties).mean()

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        if exc_type is None and self.calls != 2:
            raise ValueError(
                "the expected history/future or positive/negative encoder calls changed"
            )
