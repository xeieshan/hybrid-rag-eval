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


def test_bm25_saturates_term_frequency():
    idx = BM25Index().fit(DOCS)
    s = idx.scores("length")
    # doc 3 repeats the term 3x but BM25's tf saturation keeps the gap bounded
    assert s[3] > s[1]
    assert s[3] < 3 * s[1]


def test_bm25_unknown_query_returns_nothing():
    idx = BM25Index().fit(DOCS)
    assert idx.search("kubernetes", k=3) == []
    assert not idx.scores("").any()


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == [1, 2] or ids[:2] == [2, 1]
    assert set(ids) == {1, 2, 3, 4}
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] < scores[2]


def test_rrf_weights_length_mismatch():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retriever_modes_find_walrus(bundle, mode):
    hits = Retriever(bundle, mode=mode).retrieve("walrus operator assignment expression", k=3)
    assert hits[0].chunk.number == 572
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_index_roundtrip(bundle, tmp_path):
    bundle.save(tmp_path / "idx")
    loaded = IndexBundle.load(tmp_path / "idx")
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in bundle.chunks]
    np.testing.assert_allclose(loaded.dense.vectors, bundle.dense.vectors)
    q = "maximum line length"
    assert [h.chunk.chunk_id for h in Retriever(loaded).retrieve(q, 3)] == [
        h.chunk.chunk_id for h in Retriever(bundle).retrieve(q, 3)
    ]


def test_unknown_mode_rejected(bundle):
    with pytest.raises(ValueError):
        Retriever(bundle, mode="magic")

# Maintenance pass: 2026-06-22 13:55
