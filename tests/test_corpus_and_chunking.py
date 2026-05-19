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


def test_chunk_text_respects_budget_and_overlap():
    paras = [" ".join(f"w{i}_{j}" for j in range(30)) for i in range(10)]
    chunks = chunk_text("\n\n".join(paras), max_words=100, overlap_words=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 100 + 20
    # consecutive chunks share the overlap words
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_long_paragraph_is_windowed():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, max_words=200, overlap_words=50)
    words = set()
    for c in chunks:
        words.update(c.split())
    assert len(words) == 500


def test_chunks_keep_section_and_header(corpus_dir):
    doc = [d for d in load_corpus(corpus_dir) if d.number == 8][0]
    chunks = chunk_document(doc, max_words=60, overlap_words=10)
    assert {c.section for c in chunks} == {"Maximum Line Length", "Programming Recommendations"}
