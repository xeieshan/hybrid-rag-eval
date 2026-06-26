"""Embedding backends.

* ``LSAEmbedder``     TF-IDF + truncated SVD. Runs locally, no API key; a useful
                      dense baseline that shows how much an API model actually adds.
* ``LiteLLMEmbedder`` Any embedding model LiteLLM supports (OpenAI, Cohere,
                      Voyage, Mistral, Ollama, ...). Results are cached on disk.
* ``HashingEmbedder`` Deterministic, dependency-free; used by the tests.

All embedders return L2-normalised float32 matrices, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


def _field(obj: object, key: str):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    def __init__(self, n_components: int = 256, random_state: int = 0) -> None:
        self.n_components = n_components
        self.name = f"lsa-{n_components}"
        self._tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, stop_words="english", ngram_range=(1, 2))
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, texts: list[str]) -> LSAEmbedder:
        x = self._tfidf.fit_transform(texts)
        n = min(self.n_components, x.shape[1] - 1, x.shape[0] - 1)
        if n < self.n_components:
            self._svd = TruncatedSVD(n_components=max(n, 1), random_state=self._svd.random_state)
        self._svd.fit(x)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._svd.transform(self._tfidf.transform(texts)))


class HashingEmbedder:
    def __init__(self, n_features: int = 512) -> None:
        self.name = f"hashing-{n_features}"
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None)

    def fit(self, texts: list[str]) -> HashingEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._vec.transform(texts).toarray())


class LiteLLMEmbedder:
    """Embeds through LiteLLM, e.g. ``text-embedding-3-small`` or ``ollama/nomic-embed-text``."""

    def __init__(self, model: str, batch_size: int = 64, cache_dir: str | Path | None = None) -> None:
        self.model = model
        self.name = f"litellm:{model}"
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = self.cache_dir / f"{hashlib.sha256(model.encode()).hexdigest()[:12]}.jsonl"
            if self._cache_file.exists():
                for line in self._cache_file.read_text(encoding="utf-8").splitlines():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["vec"]

    def fit(self, texts: list[str]) -> LiteLLMEmbedder:
        return self

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        import litellm

        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start : start + self.batch_size]
            resp = litellm.embedding(model=self.model, input=[texts[i] for i in idx])
            rows = sorted(resp.data, key=lambda d: _field(d, "index"))
            new = {keys[i]: _field(row, "embedding") for i, row in zip(idx, rows, strict=True)}
            self._cache.update(new)
            if self.cache_dir:
                with self._cache_file.open("a", encoding="utf-8") as f:
                    for k, v in new.items():
                        f.write(json.dumps({"key": k, "vec": v}) + "\n")
        return _normalize(np.array([self._cache[k] for k in keys]))

    def __getstate__(self) -> dict:
        # Don't pickle the in-memory cache; it lives in cache_dir.
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state


def make_embedder(spec: str, cache_dir: str | Path | None = None) -> Embedder:
    """``lsa`` | ``lsa:<dims>`` | ``hashing`` | ``litellm:<model>``."""
    kind, _, arg = spec.partition(":")
    if kind == "lsa":
        return LSAEmbedder(n_components=int(arg) if arg else 256)
    if kind == "hashing":
        return HashingEmbedder(n_features=int(arg) if arg else 512)
    if kind == "litellm":
        if not arg:
            raise ValueError("litellm embedder needs a model, e.g. litellm:text-embedding-3-small")
        return LiteLLMEmbedder(model=arg, cache_dir=cache_dir)
    raise ValueError(f"Unknown embedder spec: {spec!r}")


def save_embedder(embedder: Embedder, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(embedder, f)


def load_embedder(path: Path) -> Embedder:
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 - local artifact written by build-index

# Maintenance pass: 2026-06-26 13:32
