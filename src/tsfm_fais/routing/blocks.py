"""Atomic missing blocks and their sparse relationship graph."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tsfm_fais.contracts import CandidateResult, MissingBlock, SeriesBatch

from .graph import BlockGraph
from .graph import build_block_graph as _build_block_graph


def detect_missing_blocks(observed_mask: np.ndarray, pattern: str = "observed") -> tuple[MissingBlock, ...]:
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3:
        raise ValueError("observed_mask must have shape [N,L,D] or [L,D]")
    blocks: list[MissingBlock] = []
    for batch_index in range(mask.shape[0]):
        for channel in range(mask.shape[2]):
            missing = ~mask[batch_index, :, channel]
            transitions = np.diff(np.pad(missing.astype(np.int8), (1, 1)))
            starts = np.flatnonzero(transitions == 1)
            ends = np.flatnonzero(transitions == -1)
            for start, end in zip(starts, ends, strict=True):
                blocks.append(
                    MissingBlock(
                        block_id=f"n{batch_index}:d{channel}:{int(start)}-{int(end)}",
                        batch_index=batch_index,
                        channel=channel,
                        start=int(start),
                        end=int(end),
                        pattern=pattern,
                    )
                )
    return tuple(blocks)


def build_block_graph(
    blocks: tuple[MissingBlock, ...] | list[MissingBlock],
    correlation: np.ndarray | None = None,
    top_k_correlated: int = 3,
) -> BlockGraph:
    return _build_block_graph(
        tuple(blocks),
        correlation,
        top_k_correlated=top_k_correlated,
    )


def extract_missing_blocks(batch: SeriesBatch, pattern: str = "observed") -> tuple[MissingBlock, ...]:
    return detect_missing_blocks(batch.observed_mask, pattern)


def validate_blocks(blocks: tuple[MissingBlock, ...] | list[MissingBlock], batch: SeriesBatch) -> None:
    seen: set[tuple[int, int, int]] = set()
    for block in blocks:
        if block.batch_index >= batch.shape[0] or block.channel >= batch.shape[2] or block.end > batch.shape[1]:
            raise ValueError(f"block {block.block_id} is outside the batch")
        for step in range(block.start, block.end):
            coordinate = (block.batch_index, step, block.channel)
            if coordinate in seen:
                raise ValueError("missing blocks overlap at the same coordinate")
            seen.add(coordinate)
            if batch.observed_mask[coordinate]:
                raise ValueError(f"block {block.block_id} covers an observed value")


def block_mask(blocks: tuple[MissingBlock, ...] | list[MissingBlock], shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for block in blocks:
        mask[block.batch_index, block.start : block.end, block.channel] = True
    return mask


def assemble_routed_values(
    batch: SeriesBatch,
    blocks: tuple[MissingBlock, ...] | list[MissingBlock],
    candidates: dict[str, CandidateResult] | Mapping[str, CandidateResult],
    assignments: Mapping[str, str],
) -> np.ndarray:
    validate_blocks(blocks, batch)
    completed = batch.values.copy()
    for block in blocks:
        if block.block_id not in assignments:
            raise ValueError(f"missing assignment for {block.block_id}")
        candidate_id = assignments[block.block_id]
        if candidate_id not in candidates:
            raise ValueError(f"unknown assigned candidate: {candidate_id}")
        candidate = candidates[candidate_id]
        completed[block.batch_index, block.start : block.end, block.channel] = candidate.values[
            block.batch_index, block.start : block.end, block.channel
        ]
    completed[batch.observed_mask] = batch.values[batch.observed_mask]
    if not np.isfinite(completed).all():
        raise ValueError("assembled values contain non-finite entries")
    return completed
