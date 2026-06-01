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
