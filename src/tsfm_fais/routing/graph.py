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
) -> BlockGraph:
    """Connect same-channel neighbors, overlaps, and nearby correlated channels."""

    ordered = tuple(sorted(blocks, key=lambda b: (b.batch_index, b.channel, b.start, b.end)))
    edges: dict[tuple[str, str], BlockEdge] = {}

    if connect_channel_neighbors:
        groups: dict[tuple[int, int], list[MissingBlock]] = {}
        for block in ordered:
            groups.setdefault((block.batch_index, block.channel), []).append(block)
        for group in groups.values():
            for left, right in zip(group, group[1:]):
                gap = _interval_gap(left, right)
                weight = 1.0 / (1.0 + gap)
                key = _edge_key(left.block_id, right.block_id)
                edges[key] = BlockEdge(key[0], key[1], weight=weight, kind="same_channel")

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.batch_index != right.batch_index or left.channel == right.channel:
                continue
            gap = _interval_gap(left, right)
            overlap = _overlap(left, right)
            kind = "cross_channel"
            if overlap or gap <= cross_channel_max_gap:
                weight = (
                    overlap / max(left.length, right.length)
                    if overlap
                    else 1.0 / (1.0 + gap)
                )
            else:
                if correlation is None:
                    continue
                matrix = np.asarray(correlation, dtype=float)
                if matrix.shape[0] <= max(left.channel, right.channel):
                    raise ValueError("correlation matrix does not cover all block channels")
                row = np.abs(matrix[left.channel]).copy()
                row[left.channel] = -np.inf
                ranked_related = [
                    int(index)
                    for index in np.argsort(row)[::-1]
                    if np.isfinite(row[index]) and row[index] > 0
                ]
                related = set(ranked_related[:top_k_correlated])
                if right.channel not in related or gap > max(left.length, right.length):
                    continue
                weight = float(abs(matrix[left.channel, right.channel])) / (1.0 + gap)
                kind = "correlated_channel"
            key = _edge_key(left.block_id, right.block_id)
            existing = edges.get(key)
            if existing is None or weight > existing.weight:
                edges[key] = BlockEdge(key[0], key[1], weight=weight, kind=kind)

    return BlockGraph(blocks=ordered, edges=tuple(sorted(edges.values())))
