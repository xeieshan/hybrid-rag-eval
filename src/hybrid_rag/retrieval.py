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

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        sims = self.vectors @ np.asarray(query_vec, dtype=np.float32).ravel()
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top]


class IndexBundle:
    """Chunks + BM25 + dense vectors + the embedder used to make them."""

    def __init__(self, chunks: list[Chunk], bm25: BM25Index, dense: DenseIndex, embedder: Embedder):
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> IndexBundle:
        texts = [c.indexed_text for c in chunks]
        bm25 = BM25Index().fit(texts)
        embedder.fit(texts)
        dense = DenseIndex(embedder.embed(texts))
        return cls(chunks, bm25, dense, embedder)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        with (directory / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f)
        np.save(directory / "dense.npy", self.dense.vectors)
        save_embedder(self.embedder, directory / "embedder.pkl")
        meta = {"n_chunks": len(self.chunks), "embedder": self.embedder.name}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> IndexBundle:
        directory = Path(directory)
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        with (directory / "bm25.pkl").open("rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 - local artifact written by build-index
        dense = DenseIndex(np.load(directory / "dense.npy"))
        return cls(chunks, bm25, dense, load_embedder(directory / "embedder.pkl"))


class Retriever:
    """``mode`` is ``bm25``, ``dense`` or ``hybrid`` (RRF over both)."""

    def __init__(self, index: IndexBundle, mode: str = "hybrid", rrf_k: int = 60, candidates: int = 50):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        self.index = index
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidates = candidates

    def _bm25_ids(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.index.bm25.search(query, k)

    def _dense_ids(self, query: str, k: int) -> list[tuple[int, float]]:
        qvec = self.index.embedder.embed([query])[0]
        return self.index.dense.search(qvec, k)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        if self.mode == "bm25":
            ranked = self._bm25_ids(query, k)
        elif self.mode == "dense":
            ranked = self._dense_ids(query, k)
        else:
            n = max(k, self.candidates)
            lexical = [i for i, _ in self._bm25_ids(query, n)]
            semantic = [i for i, _ in self._dense_ids(query, n)]
            ranked = reciprocal_rank_fusion([lexical, semantic], k=self.rrf_k)[:k]
        return [Hit(chunk=self.index.chunks[i], score=score, rank=r) for r, (i, score) in enumerate(ranked, start=1)]
