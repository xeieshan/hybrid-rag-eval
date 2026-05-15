from hybrid_rag.chunking import chunk_document, chunk_text, split_sections
from hybrid_rag.corpus import load_corpus, parse_header


def test_parse_header_joins_continuation_lines():
    meta, body = parse_header("PEP: 1\nAuthor: A,\n        B\nStatus: Final\n\nBody text")
    assert meta["Author"] == "A, B"
    assert meta["Status"] == "Final"
    assert body == "Body text"


def test_load_corpus_reads_only_pep_files(corpus_dir):
    docs = load_corpus(corpus_dir)
    assert [d.number for d in docs] == [8, 20, 572]
    pep572 = docs[2]
    assert pep572.title == "Assignment Expressions"
    assert pep572.python_version == "3.8"
    assert pep572.doc_id == "pep-0572"


def test_split_sections_finds_rst_headings():
    body = "Intro line\n\nAbstract\n========\n\nText A\n\nSub part\n--------\n\nText B"
    sections = split_sections(body)
    assert [h for h, _ in sections] == ["Preamble", "Abstract", "Sub part"]
    assert sections[1][1] == "Text A"


def test_split_sections_ignores_short_underlines():
    body = "A long heading line\n===\n\ntext"
    assert [h for h, _ in split_sections(body)] == ["Preamble"]

