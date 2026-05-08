from hybrid_rag.chunking import chunk_document, chunk_text, split_sections
from hybrid_rag.corpus import load_corpus, parse_header


def test_parse_header_joins_continuation_lines():
    meta, body = parse_header("PEP: 1\nAuthor: A,\n        B\nStatus: Final\n\nBody text")
    assert meta["Author"] == "A, B"
    assert meta["Status"] == "Final"
    assert body == "Body text"
