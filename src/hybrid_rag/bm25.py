"""Okapi BM25 over a sparse term-frequency matrix.

Written from scratch (numpy + scipy.sparse) rather than pulled from a library
so the scoring formula is visible and testable:

    score(q, d) = sum_t idf(t) * tf(t, d) * (k1 + 1)
                  / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it
    its me my not of on or so that the their then there these this to was what when where
    which while who why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self._weights: sparse.csc_matrix | None = None
        self.n_docs = 0

    def fit(self, texts: list[str]) -> BM25Index:
        rows, cols, vals = [], [], []
        doc_lens = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_lens[i] = sum(counts.values())
            for term, tf in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        self.n_docs = len(texts)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)),
            shape=(self.n_docs, len(self.vocab)),
        )
        df = np.bincount(tf.indices, minlength=len(self.vocab)).astype(np.float64)
        self.idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))

        avgdl = doc_lens.mean() if self.n_docs else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * doc_lens / max(avgdl, 1e-9))
        # Precompute the per-(doc, term) BM25 weight so queries are a sparse column sum.
        tf = tf.tocoo()
        weights = tf.data * (self.k1 + 1.0) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self._weights = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Index is not fitted")
        term_ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not term_ids:
            return np.zeros(self.n_docs)
        return np.asarray(self._weights[:, term_ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        if not s.any():
            return []
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

# Maintenance pass: 2026-06-10 17:00
