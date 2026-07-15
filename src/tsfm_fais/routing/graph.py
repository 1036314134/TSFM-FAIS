"""Sparse graph connecting related missing blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tsfm_fais.contracts import MissingBlock


@dataclass(frozen=True, order=True)
class BlockEdge:
    left: str
    right: str
    weight: float = 1.0
    kind: str = "temporal"

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValueError("self edges are not allowed")
        if self.weight < 0:
            raise ValueError("edge weight must be non-negative")


@dataclass(frozen=True)
class BlockGraph:
    blocks: tuple[MissingBlock, ...]
    edges: tuple[BlockEdge, ...]

    def __post_init__(self) -> None:
        block_ids = {block.block_id for block in self.blocks}
        if len(block_ids) != len(self.blocks):
            raise ValueError("block ids must be unique")
        for edge in self.edges:
            if edge.left not in block_ids or edge.right not in block_ids:
                raise ValueError("edge refers to an unknown block")

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.block_id for block in self.blocks)

    def neighbors(self, block_id: str) -> tuple[str, ...]:
        values: list[str] = []
        for edge in self.edges:
            if edge.left == block_id:
                values.append(edge.right)
            elif edge.right == block_id:
                values.append(edge.left)
        return tuple(values)


def _interval_gap(left: MissingBlock, right: MissingBlock) -> int:
    if left.end < right.start:
        return right.start - left.end
    if right.end < left.start:
        return left.start - right.end
    return 0


def _overlap(left: MissingBlock, right: MissingBlock) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def build_block_graph(
    blocks: Sequence[MissingBlock],
    correlation: np.ndarray | None = None,
    *,
    cross_channel_max_gap: int = 0,
    connect_channel_neighbors: bool = True,
    top_k_correlated: int = 3,
    max_cross_channel_neighbors: int = 3,
) -> BlockGraph:
    """Connect same-channel neighbors, overlaps, and nearby correlated channels."""

    if (
        cross_channel_max_gap < 0
        or top_k_correlated < 0
        or max_cross_channel_neighbors < 0
    ):
        raise ValueError("graph neighbor limits must be non-negative")

    ordered = tuple(sorted(blocks, key=lambda b: (b.batch_index, b.channel, b.start, b.end)))
    edges: dict[tuple[str, str], BlockEdge] = {}
    groups: dict[tuple[int, int], list[MissingBlock]] = {}
    for block in ordered:
        groups.setdefault((block.batch_index, block.channel), []).append(block)

    if connect_channel_neighbors:
        for group in groups.values():
            for left, right in zip(group, group[1:]):
                gap = _interval_gap(left, right)
                weight = 1.0 / (1.0 + gap)
                key = _edge_key(left.block_id, right.block_id)
                edges[key] = BlockEdge(key[0], key[1], weight=weight, kind="same_channel")

    matrix: np.ndarray | None = None
    if correlation is not None:
        matrix = np.asarray(correlation, dtype=float)
        max_channel = max((block.channel for block in ordered), default=-1)
        if matrix.ndim != 2 or matrix.shape[0] <= max_channel or matrix.shape[1] <= max_channel:
            raise ValueError("correlation matrix does not cover all block channels")

    # Index intervals by time step.  Each block retains only its strongest
    # overlapping cross-channel neighbors, avoiding the dense clique produced
    # by synchronous or high-dimensional point missingness.
    time_index: dict[tuple[int, int], list[int]] = {}
    for index, block in enumerate(ordered):
        # Include one boundary step on each side. MissingBlock intervals are
        # half-open, while ``_interval_gap`` assigns zero gap to [a,b) and
        # [b,c); without the boundary step the indexed implementation drops
        # those cross-channel edges that the original pairwise scan retained.
        start = max(0, block.start - cross_channel_max_gap - 1)
        end = block.end + cross_channel_max_gap + 1
        for step in range(start, end):
            time_index.setdefault((block.batch_index, step), []).append(index)
    for left_index, left in enumerate(ordered):
        candidate_indices: set[int] = set()
        for step in range(left.start, left.end):
            candidate_indices.update(time_index.get((left.batch_index, step), ()))
        scored: list[tuple[float, float, str, MissingBlock]] = []
        for right_index in candidate_indices:
            if right_index == left_index:
                continue
            right = ordered[right_index]
            if left.channel == right.channel:
                continue
            gap = _interval_gap(left, right)
            overlap = _overlap(left, right)
            if not overlap and gap > cross_channel_max_gap:
                continue
            weight = (
                overlap / max(left.length, right.length)
                if overlap
                else 1.0 / (1.0 + gap)
            )
            correlation_strength = (
                0.0 if matrix is None else float(abs(matrix[left.channel, right.channel]))
            )
            scored.append((weight, correlation_strength, right.block_id, right))
        for weight, _, _, right in sorted(scored, reverse=True)[
            :max_cross_channel_neighbors
        ]:
            key = _edge_key(left.block_id, right.block_id)
            existing = edges.get(key)
            if existing is None or weight > existing.weight:
                edges[key] = BlockEdge(
                    key[0], key[1], weight=weight, kind="cross_channel"
                )

    if matrix is not None and top_k_correlated:
        for left in ordered:
            row = np.abs(matrix[left.channel]).copy()
            row[left.channel] = -np.inf
            related_channels = [
                int(channel)
                for channel in np.argsort(row)[::-1]
                if np.isfinite(row[channel]) and row[channel] > 0
            ][:top_k_correlated]
            for channel in related_channels:
                candidates = groups.get((left.batch_index, channel), ())
                scored_candidates = []
                for right in candidates:
                    gap = _interval_gap(left, right)
                    if gap == 0 or gap > max(left.length, right.length):
                        continue
                    weight = float(abs(matrix[left.channel, right.channel])) / (
                        1.0 + gap
                    )
                    scored_candidates.append((weight, right.block_id, right))
                if not scored_candidates:
                    continue
                weight, _, right = max(scored_candidates)
                key = _edge_key(left.block_id, right.block_id)
                existing = edges.get(key)
                if existing is None or weight > existing.weight:
                    edges[key] = BlockEdge(
                        key[0],
                        key[1],
                        weight=weight,
                        kind="correlated_channel",
                    )

    return BlockGraph(blocks=ordered, edges=tuple(sorted(edges.values())))
