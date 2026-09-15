"""Learn blockwise convex compositions of fixed imputer outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BlockComposition:
    values: torch.Tensor
    weights: torch.Tensor
    block_ids: torch.Tensor
    penalty: torch.Tensor

    @property
    def block_count(self):
        return self.weights.shape[1]


def missing_block_ids(context):
    """Number maximal missing runs by variable, then by time; observed IDs are -1."""
    missing = ~torch.isfinite(context.T)
    preceding = torch.cat([torch.zeros_like(missing[:, :1]), missing[:, :-1]], dim=1)
    starts = missing & ~preceding
    counts = starts.sum(dim=1)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    identifiers = starts.long().cumsum(1) - 1 + offsets[:, None]
    identifiers = torch.where(missing, identifiers, -1)
    return identifiers, starts.nonzero(as_tuple=False)[:, 0]


def mix_blocks(candidates, context, block_ids, weights):
    """Assign one simplex vector to every missing block and restore observations."""
    flat_ids = block_ids.reshape(-1)
    missing = flat_ids >= 0
    if not bool(missing.any()):
        return context.clone()
    flattened = candidates.transpose(1, 2).reshape(candidates.shape[0], -1)
    available = flattened[:, missing]
    filled = (available * weights[:, flat_ids[missing]]).sum(0)
    identical = (available == available[:1]).all(0)
    filled = torch.where(identical, available[0], filled)
    output = torch.nan_to_num(context.T).reshape(-1).clone()
    output[missing] = filled
    return output.reshape(context.shape[1], context.shape[0]).T


class FixedBlockComposer(nn.Module):
    def __init__(self, prior):
        super().__init__()
        probability = torch.as_tensor(prior, dtype=torch.float32)
        if probability.ndim != 1 or len(probability) < 2 or not bool((probability > 0).all()):
            raise ValueError("a positive prior over at least two imputers is required")
        initial = (probability / probability.sum()).log()
        self.register_buffer("initial_logits", initial)
        self.base_logits = nn.Parameter(initial.clone())

    def _validate(self, candidates, context):
        if (
            candidates.ndim != 3
            or candidates.shape[1:] != context.shape
            or len(candidates) != len(self.base_logits)
        ):
            raise ValueError("candidate tensors must be [A,L,D] and match the context and prior")
        if not bool(torch.isfinite(candidates).all()) or bool(torch.isinf(context).any()):
            raise ValueError("candidates must be finite and unknown context values must be NaN")

    def forward(self, candidates, context, period=1):
        self._validate(candidates, context)
        identifiers, variables = missing_block_ids(context)
        weights = self.base_logits.softmax(0)[:, None].expand(-1, len(variables))
        penalty = (self.base_logits - self.initial_logits).square().mean()
        return BlockComposition(
            mix_blocks(candidates, context, identifiers, weights), weights, identifiers, penalty
        )


class ContextBlockComposer(FixedBlockComposer):
    """Shared temporal features and variable pooling; no dataset identity or future labels."""

    def __init__(self, prior, hidden=16, forecaster_dim=None, target_conditioned=False):
        super().__init__(prior)
        self.hidden = hidden
        self.target_conditioned = target_conditioned
        self.encoder = nn.Sequential(
            nn.Conv1d(5, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.action_embedding = nn.Embedding(len(prior), 8)
        self.score = nn.Sequential(
            nn.Linear(7 * hidden + 12 + int(target_conditioned), 32), nn.GELU(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.forecaster_projection = (
            nn.Sequential(
                nn.LayerNorm(forecaster_dim), nn.Linear(forecaster_dim, hidden), nn.GELU()
            )
            if forecaster_dim is not None
            else None
        )

    def forward(self, candidates, context, period=1, forecaster_patches=None, targets=None):
        self._validate(candidates, context)
        identifiers, variables = missing_block_ids(context)
        n_actions, length, dimensions = candidates.shape
        n_blocks = len(variables)
        if not n_blocks:
            return BlockComposition(
                context.clone(),
                candidates.new_empty(n_actions, 0),
                identifiers,
                self.base_logits.sum() * 0,
            )
        observed = torch.isfinite(context)
        consensus = candidates.quantile(0.5, dim=0)
        time = torch.linspace(0, 1, length, device=context.device, dtype=context.dtype)[
            :, None
        ].expand(-1, dimensions)
        channels = torch.stack(
            [
                candidates.asinh(),
                torch.nan_to_num(context).asinh().expand_as(candidates),
                observed.to(context.dtype).expand_as(candidates),
                (candidates - consensus).asinh(),
                time.expand_as(candidates),
            ],
            dim=2,
        )
        encoded = self.encoder(
            channels.permute(0, 3, 2, 1).reshape(n_actions * dimensions, 5, length)
        )
        encoded = encoded.reshape(n_actions, dimensions, self.hidden, length)
        if self.forecaster_projection is not None:
            if (
                forecaster_patches is None
                or forecaster_patches.ndim != 4
                or forecaster_patches.shape[:2] != (n_actions, dimensions)
                or forecaster_patches.shape[2] < 1
                or length % forecaster_patches.shape[2]
            ):
                raise ValueError(
                    "frozen patch features must be [A,D,P,E] and align with the context"
                )
            projected = self.forecaster_projection(forecaster_patches.detach())
            projected = projected.repeat_interleave(length // projected.shape[2], dim=2)
            encoded = encoded + projected.permute(0, 1, 3, 2)
        elif forecaster_patches is not None:
            raise ValueError("this composer was not configured for predictor features")
        variable_summary = torch.cat(
            [encoded.mean(-1), encoded[..., -max(1, length // 4) :].mean(-1)], dim=-1
        )
        variable_consensus = variable_summary.mean(0)
        group_summary = variable_consensus.mean(0)
        target_role = None
        if self.target_conditioned:
            if targets is None or not len(targets) or len(set(targets)) != len(targets):
                raise ValueError("target-conditioned composition requires distinct target indices")
            target_indices = torch.as_tensor(targets, dtype=torch.long, device=context.device)
            if bool(((target_indices < 0) | (target_indices >= dimensions)).any()):
                raise ValueError("forecast target is outside the context variables")
            group_summary = variable_consensus[target_indices].mean(0)
            role = candidates.new_zeros(dimensions)
            role[target_indices] = 1
            target_role = role[variables][None, :, None].expand(n_actions, -1, -1)
        flat_ids = identifiers.reshape(-1)
        present = flat_ids >= 0
        ids = flat_ids[present]
        block_lengths = torch.bincount(ids, minlength=n_blocks).to(context.dtype)
        local = encoded.permute(0, 1, 3, 2).reshape(n_actions, -1, self.hidden)[:, present]
        pooled = local.new_zeros(n_actions, n_blocks, self.hidden)
        pooled.index_add_(1, ids, local)
        pooled = pooled / block_lengths[None, :, None]
        center = time.T.reshape(-1)[present]
        block_center = center.new_zeros(n_blocks).index_add_(0, ids, center) / block_lengths
        tail = center.new_zeros(n_blocks)
        tail_ids = identifiers[:, -1]
        tail[tail_ids[tail_ids >= 0]] = 1
        geometry = torch.stack(
            [
                block_lengths / length,
                block_center,
                tail,
                torch.full_like(tail, math.log1p(max(1, period)) / math.log1p(length)),
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                pooled,
                variable_summary[:, variables],
                variable_consensus[variables][None].expand(n_actions, -1, -1),
                group_summary[None, None].expand(n_actions, n_blocks, -1),
                self.action_embedding.weight[:, None].expand(-1, n_blocks, -1),
                geometry[None].expand(n_actions, -1, -1),
            ],
            dim=-1,
        )
        if target_role is not None:
            features = torch.cat([features, target_role], dim=-1)
        adjustment = self.score(features).squeeze(-1)
        weights = (self.base_logits[:, None] + adjustment).softmax(0)
        penalty = (
            adjustment.square().mean() + (self.base_logits - self.initial_logits).square().mean()
        )
        return BlockComposition(
            mix_blocks(candidates, context, identifiers, weights), weights, identifiers, penalty
        )
