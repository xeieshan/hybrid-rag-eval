import math

import numpy as np
import pytest

from hybrid_rag.bm25 import BM25Index, tokenize
from hybrid_rag.retrieval import IndexBundle, Retriever, reciprocal_rank_fusion

DOCS = [
    "the walrus operator assigns inside expressions",
    "style guide limits line length to 79 characters",
    "zen of python aphorisms",
    "line length line length line length",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Walrus IS an Operator!") == ["walrus", "operator"]


def test_bm25_ranks_matching_doc_first():
    idx = BM25Index().fit(DOCS)
    top = idx.search("walrus operator", k=2)
    assert top[0][0] == 0
    assert all(score > 0 for _, score in top)


def test_bm25_idf_matches_formula():
    idx = BM25Index().fit(DOCS)
    n, df = 4, 2  # "length" appears in docs 1 and 3
    expected = math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert idx.idf[idx.vocab["length"]] == pytest.approx(expected)
