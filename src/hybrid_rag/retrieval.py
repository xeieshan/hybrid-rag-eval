"""Retrievers (BM25, dense, hybrid) and the on-disk index format."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embeddings import Embedder, load_embedder, save_embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids: score(id) = sum_i w_i / (k + rank_i(id)), ranks start at 1."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class DenseIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

