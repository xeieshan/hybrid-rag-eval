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
